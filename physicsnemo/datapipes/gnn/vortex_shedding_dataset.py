# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.datapipes.gnn.utils import load_json, save_json

# Lazy imports for optional dependencies
pyg = OptionalImport("torch_geometric")
tfrecord_torch = OptionalImport("tfrecord.torch.dataset")


class VortexSheddingDataset(Dataset):
    """In-memory MeshGraphNet Dataset for stationary mesh
    Notes:
        - This dataset prepares and processes the data available in MeshGraphNet's repo:
            https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
        - A single adj matrix is used for each transient simulation.
            Do not use with adaptive mesh or remeshing

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    num_samples : int, optional
        Number of samples, by default 1000
    num_steps : int, optional
        Number of time steps in each sample, by default 600
    noise_std : float, optional
        The standard deviation of the noise added to the "train" split, by default 0.02
    """

    def __init__(
        self,
        name="dataset",
        data_dir=None,
        split="train",
        num_samples=1000,
        num_steps=600,
        noise_std=0.02,
    ):
        self.name = name
        self.data_dir = data_dir
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.length = num_samples * (num_steps - 1)

        print(f"Preparing the {split} dataset...")
        # Create the graphs with edge features.
        tfrecord_dataset = self._load_tfrecord_dataset(self.data_dir, self.split)
        self.graphs, self.cells, self.node_type = [], [], []
        noise_mask, self.rollout_mask = [], []
        self.mesh_pos = []
        for i, data_np in enumerate(tfrecord_dataset):
            if i >= self.num_samples:
                break
            # Slice to num_steps for each feature.
            data_np = {key: arr[:num_steps] for key, arr in data_np.items()}
            src, dst = self.cell_to_adj(data_np["cells"][0])  # assuming stationary mesh
            graph = self.create_graph(src, dst, dtype=torch.int32)
            graph = self.add_edge_features(graph, data_np["mesh_pos"][0])
            self.graphs.append(graph)
            node_type = torch.tensor(data_np["node_type"][0], dtype=torch.uint8)
            self.node_type.append(self._one_hot_encode(node_type))
            noise_mask.append(torch.eq(node_type, torch.zeros_like(node_type)))

            if self.split != "train":
                self.mesh_pos.append(torch.tensor(data_np["mesh_pos"][0]))
                self.cells.append(data_np["cells"][0])
                self.rollout_mask.append(self._get_rollout_mask(node_type))

        # compute or load edge data stats
        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json("edge_stats.json")

        # normalize edge features
        for i in range(num_samples):
            self.graphs[i].edge_attr = self.normalize_edge(
                self.graphs[i],
                self.edge_stats["edge_mean"],
                self.edge_stats["edge_std"],
            )

        # Create the node features.
        tfrecord_dataset = self._load_tfrecord_dataset(self.data_dir, self.split)
        self.node_features, self.node_targets = [], []
        for i, data_np in enumerate(tfrecord_dataset):
            if i >= self.num_samples:
                break
            # Slice to num_steps for each feature.
            data_np = {key: arr[:num_steps] for key, arr in data_np.items()}
            features, targets = {}, {}

            # 为了说明数据到底是什么样，我们先明确原始数据的结构。假设一个流体仿真样本包含 $T$ 个时间步，物理网格包含 $N$ 个节点，且这是一个二维流体问题。

            # ### 1. `targets["velocity"]`

            # * **执行代码**：`self._push_forward_diff(data_np["velocity"])`
            # * **数据形态**：它是一个多维数组（张量），形状为 `(T-1, N, 2)`。
            #     * `T-1`：表示时间序列的总长度减少了 1 步。
            #     * `N`：表示网格节点的总数。
            #     * `2`：表示速度有两个分量（X 轴方向和 Y 轴方向）。
            # * **表示的意思**：它存储的是**当前时刻到下一时刻的速度变化量（速度增量）**。
            #     * 具体计算方式是：用第 $t+1$ 时刻的速度坐标减去第 $t$ 时刻的速度坐标。即 $\Delta V = V_{t+1} - V_t$。
            #     * 对于张量中的任意一个时间步索引 $t$ 和节点索引 $n$，里面存储的具体数值是该节点在 $t+1$ 时刻与 $t$ 时刻的 $[X方向速度差, Y方向速度差]$。

            # ### 2. `targets["pressure"]`

            # * **执行代码**：`self._push_forward(data_np["pressure"])`
            # * **数据形态**：它是一个多维数组，形状为 `(T-1, N, 1)`。
            #     * `1`：表示压力是一个单一的标量数值。
            # * **表示的意思**：它存储的是**下一时刻的绝对压力值**。
            #     * 通过将原始数据索引整体向后平移一位（即丢弃第 0 个时刻的数据，保留第 1 到最后一个时刻的数据）得到。
            #     * 对于给定的时间步索引 $t$，里面存储的具体数值是第 $t+1$ 时刻该网格节点的实际压力 $P_{t+1}$。

            # 要完全理解 `targets` 的意义，需要结合它的输入 `features` 来看代码中的时间对齐逻辑：

            # 1.  `features["velocity"] = self._drop_last(...)` 提取了第 $0$ 到 $T-2$ 时刻的数据，作为神经网络的输入。
            # 2.  `targets` 提取并计算了第 $1$ 到 $T-1$ 时刻的数据，作为神经网络预测的目标。
            features["velocity"] = self._drop_last(data_np["velocity"])
            targets["velocity"] = self._push_forward_diff(data_np["velocity"])
            targets["pressure"] = self._push_forward(data_np["pressure"])

            # add noise
            if split == "train":
                features["velocity"], targets["velocity"] = self._add_noise(
                    features["velocity"],
                    targets["velocity"],
                    self.noise_std,
                    noise_mask[i],
                )
            self.node_features.append(features)
            self.node_targets.append(targets)

        # compute or load node data stats
        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            self.node_stats = load_json("node_stats.json")

        # normalize node features
        for i in range(num_samples):
            self.node_features[i]["velocity"] = self.normalize_node(
                self.node_features[i]["velocity"],
                self.node_stats["velocity_mean"],
                self.node_stats["velocity_std"],
            )
            self.node_targets[i]["velocity"] = self.normalize_node(
                self.node_targets[i]["velocity"],
                self.node_stats["velocity_diff_mean"],
                self.node_stats["velocity_diff_std"],
            )
            self.node_targets[i]["pressure"] = self.normalize_node(
                self.node_targets[i]["pressure"],
                self.node_stats["pressure_mean"],
                self.node_stats["pressure_std"],
            )

    def __getitem__(self, idx):
        gidx = idx // (self.num_steps - 1)  # graph index
        tidx = idx % (self.num_steps - 1)  # time step index
        graph = self.graphs[gidx]

        # * **`gidx` 和 `tidx`**：由于这是一个瞬态流体仿真数据集（包含多张图，每张图包含多个时间步），这里通过数学取余和整除，算出当前需要抓取的是“第几个仿真（图）的第几个时间步”。
        # * **`torch.cat(..., dim=-1)`**：将两个不同的张量在特征维度（最后一个维度）上拼接起来，形成最终的节点特征。
        #     * **第一部分 `self.node_features[gidx]["velocity"][tidx]`**：这是节点在当前时间步的**速度特征**。因为这是一个二维卡门涡街流体仿真，速度包含 X 方向和 Y 方向的分量。**（此处贡献 2 个特征维度）**
        #     * **第二部分 `self.node_type[gidx]`**：这正是上面 `_one_hot_encode` 函数输出的结果。**（此处贡献 4 个特征维度）**

        # ### config.yaml 写的是 `num_input_features: 6`
        # 神经网络模型（MeshGraphNet 的节点编码器）在接收每个节点的数据时，看到的是经过 `torch.cat` 拼接后的一个长向量。
        # 这个长向量的构成是：
        # **速度分量 (2维) + 节点独热类型 (4维) = 6 个维度**
        # 例如，一个流体内部节点（类型 0）的某个时刻的数据长这样：
        # `[Vx, Vy, 1, 0, 0, 0]`
        # 一个上方墙壁节点（类型 6 -> 映射为 3）的数据长这样：
        # `[Vx, Vy, 0, 0, 0, 1]`

        node_features = torch.cat(
            (self.node_features[gidx]["velocity"][tidx], self.node_type[gidx]), dim=-1
        )
        node_targets = torch.cat(
            (
                self.node_targets[gidx]["velocity"][tidx],
                self.node_targets[gidx]["pressure"][tidx],
            ),
            dim=-1,
        )
        graph.x = node_features
        graph.y = node_targets
        if self.split == "train":
            return graph
        else:
            graph["mesh_pos"] = self.mesh_pos[gidx]
            cells = torch.tensor(self.cells[gidx])
            rollout_mask = self.rollout_mask[gidx]
            return graph, cells, rollout_mask

    def __len__(self):
        return self.length

    def _get_edge_stats(self):
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edge_attr, dim=0) / self.num_samples
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edge_attr), dim=0)
                / self.num_samples
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")

        # save to file
        save_json(stats, "edge_stats.json")
        return stats

    def _get_node_stats(self):
        stats = {
            "velocity_mean": 0,
            "velocity_meansqr": 0,
            "velocity_diff_mean": 0,
            "velocity_diff_meansqr": 0,
            "pressure_mean": 0,
            "pressure_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["velocity_mean"] += (
                torch.mean(self.node_features[i]["velocity"], dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_meansqr"] += (
                torch.mean(torch.square(self.node_features[i]["velocity"]), dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_mean"] += (
                torch.mean(self.node_targets[i]["pressure"], dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_meansqr"] += (
                torch.mean(torch.square(self.node_targets[i]["pressure"]), dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_diff_mean"] += (
                torch.mean(
                    self.node_targets[i]["velocity"],
                    dim=(0, 1),
                )
                / self.num_samples
            )
            stats["velocity_diff_meansqr"] += (
                torch.mean(
                    torch.square(self.node_targets[i]["velocity"]),
                    dim=(0, 1),
                )
                / self.num_samples
            )
        stats["velocity_std"] = torch.sqrt(
            stats["velocity_meansqr"] - torch.square(stats["velocity_mean"])
        )
        stats["pressure_std"] = torch.sqrt(
            stats["pressure_meansqr"] - torch.square(stats["pressure_mean"])
        )
        stats["velocity_diff_std"] = torch.sqrt(
            stats["velocity_diff_meansqr"] - torch.square(stats["velocity_diff_mean"])
        )
        stats.pop("velocity_meansqr")
        stats.pop("pressure_meansqr")
        stats.pop("velocity_diff_meansqr")

        # save to file
        save_json(stats, "node_stats.json")
        return stats

    def _load_tfrecord_dataset(self, path, split):
        """Load TFRecord dataset using the tfrecord package.

        Utility for loading the .tfrecord dataset in DeepMind's MeshGraphNet repo:
        https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
        Follow the instructions provided in that repo to download the .tfrecord files.

        Parameters
        ----------
        path : str
            Path to the directory containing TFRecord files and meta.json.
        split : str
            Dataset split name (e.g., "train", "valid", "test").

        Returns
        -------
        TFRecordDataset
            An iterable dataset that yields decoded records.
        """
        with open(os.path.join(path, "meta.json"), "r") as fp:
            meta = json.loads(fp.read())

        tfrecord_path = os.path.join(path, split + ".tfrecord")
        # Check for index file (enables multi-worker DataLoader).
        index_path = os.path.join(path, split + ".tfindex")
        if not os.path.exists(index_path):
            index_path = None

        # Define feature description for tfrecord package.
        # All features are stored as raw bytes in the TFRecord.
        description = {k: "byte" for k in meta["field_names"]}

        # Create dataset with transform to decode records.
        dataset = tfrecord_torch.TFRecordDataset(
            tfrecord_path,
            index_path,
            description,
            transform=lambda rec: self._decode_record(rec, meta),
        )
        return dataset

    # 这两段代码的核心作用是将流体仿真中的**物理网格（Mesh）数据**，转换为图神经网络（GNN）可以处理的**图结构（Graph）数据**。

    # ### 1. `cell_to_adj(cells)`：从网格单元提取边关系

    # 这个函数的作用是读取网格中的三角形单元，并提取出顶点与顶点之间的连接关系，输出 COO（Coordinate Format）格式的起点和终点列表。

    # * **输入 `cells`**：是一个二维数组，表示物理网格中的三角形单元（Cells）。数组的形状为 `(num_cells, 3)`。每一行代表一个三角形，包含该三角形 3 个顶点的索引号。例如，某一行的数据是 `[A, B, C]`。
    # * **`num_cells = np.shape(cells)[0]`**：获取网格中三角形单元的总数。
    # * **`src` 的推导式**：`indx` 按照 `[0, 1, 2]` 的顺序取值。对于三角形 `[A, B, C]`，提取出的起点依次为 `A, B, C`。
    # * **`dst` 的推导式**：`indx` 按照 `[1, 2, 0]` 的顺序取值。对于同一个三角形 `[A, B, C]`，提取出的终点依次为 `B, C, A`。
    # * **逻辑结果**：将 `src` 和 `dst` 上下对应来看，这实际上是在三角形的三个顶点之间建立了三条有向边：
    #     * 起点 A $\rightarrow$ 终点 B
    #     * 起点 B $\rightarrow$ 终点 C
    #     * 起点 C $\rightarrow$ 终点 A
    #     这种方式为每一个三角形单元建立了一个内部的闭环有向连接。

    # ### 2. `create_graph(src, dst, dtype)`：构建 PyTorch Geometric 图对象

    # 这个函数接收上一步生成的 `src` 和 `dst` 列表，将其转换为 PyTorch Geometric (PyG) 框架标准的图数据格式，并处理图的无向性。

    # * **`torch.stack([...], dim=0).long()`**：
    #     * 首先将 Python 列表 `src` 和 `dst` 转换为 PyTorch 张量。
    #     * 然后在第 0 维度（上下方向）将它们堆叠起来。这会生成一个形状为 `(2, num_edges)` 的张量。在 PyG 中，这被称为 `edge_index`，第一行代表所有的起点，第二行代表对应的终点。
    #     * `.long()` 将数据类型转换为 64 位整数（int64）。
    # * **`pyg.utils.to_undirected(edges)`**：这一步非常关键，包含两个隐藏操作：
    #     1.  **转换为无向图**：前一个函数 `cell_to_adj` 生成的是单向环（例如只有 $A \rightarrow B$）。此函数会自动补充反向边（添加 $B \rightarrow A$），使得节点之间的连接是双向的（无向图）。
    #     2.  **去重**：在物理网格中，相邻的两个三角形通常会共享一条边。前一步对每个三角形独立处理，会导致共享边被重复计算。此函数会自动识别并剔除重复的边索引，保持图结构的精简。
    # * **`pyg.data.Data(...)`**：将处理完毕的 `edge_index` 传入 PyG 的图数据容器 `Data` 中，返回一个标准的图对象。后续可以在这个对象上继续附加节点特征（如速度、压力）和边特征（如相对距离）。
    @staticmethod
    def cell_to_adj(cells):
        """creates adjancy matrix in COO format from mesh cells"""
        num_cells = np.shape(cells)[0]
        src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
        dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
        return src, dst

    @staticmethod
    def create_graph(src, dst, dtype=torch.int32):
        """
        creates a PyG graph from an adj matrix in COO format.
        torch.int32 can handle graphs with up to 2**31-1 nodes or edges.
        """
        edges = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).long()
        graph = pyg.data.Data(edge_index=pyg.utils.to_undirected(edges))
        return graph

    # 这段代码的作用是为已经构建好的图结构（Graph）计算并添加**边特征（Edge Features）**。

    # 在图神经网络（GNN）处理物理网格数据时，节点通常代表网格点，而边代表网格点之间的连接。为了让神经网络理解物理空间的几何结构，我们需要显式地告诉模型两个相连节点之间的相对位置关系。这就是这段代码的核心目的。

    # 下面逐行拆解代码逻辑，并解释为什么 `config.yaml` 中 `num_edge_features` 是 3。

    # ### 代码逐行解析

    # **1. `row, col = graph.edge_index`**
    # * `graph.edge_index` 是一个形状为 `(2, 边的总数)` 的张量，记录了图中所有的连接关系。
    # * 第一行（索引 0）是所有边的起点索引，赋值给 `row`。
    # * 第二行（索引 1）是所有边的终点索引，赋值给 `col`。

    # **2. `disp = torch.tensor(pos[row] - pos[col])`**
    # * `pos` 是一个张量，存储了所有节点的物理坐标。
    # * `pos[row]` 获取了所有起点的坐标，`pos[col]` 获取了所有终点的坐标。
    # * 两者相减，计算出的是每一条边的**相对位移向量（Relative Displacement）**。它表示从终点指向起点的向量（或者反过来，取决于具体定义，这里是起点坐标减去终点坐标）。

    # **3. `disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)`**
    # * 这一步计算上述位移向量的**范数（Norm）**，在欧几里得空间中，这也就是两个节点之间的**绝对直线距离**。
    # * `dim=-1` 表示在最后一个维度（坐标轴维度）上求范数。
    # * `keepdim=True` 确保输出张量保持原有的维度结构（形状为 `(边的总数, 1)`），方便后续拼接。

    # **4. `graph.edge_attr = torch.cat((disp, disp_norm), dim=1)`**
    # * `torch.cat` 将“相对位移向量”和“绝对距离”在特征维度（`dim=1`）上拼接在一起。
    # * 将拼接后的结果赋值给 `graph.edge_attr`，作为图中所有边的最终特征。

    # ---

    # ### 为什么 `num_edge_features: 3`？

    # 这与你正在处理的数据集（Vortex Shedding，卡门涡街流体仿真）的物理维度直接相关。

    # 卡门涡街仿真通常是一个 **二维（2D）流体问题**。
    # 这意味着 `pos` 中的坐标只包含 X 和 Y 两个方向的值。

    # 根据代码的拼接逻辑 `torch.cat((disp, disp_norm), dim=1)`，我们来计算一下最终边特征的维度构成：

    # 1.  **`disp`（相对位移向量）**：包含 X 方向的位移差 $\Delta x$ 和 Y 方向的位移差 $\Delta y$。**（贡献 2 个特征维度）**
    # 2.  **`disp_norm`（绝对距离）**：由公式 $\sqrt{\Delta x^2 + \Delta y^2}$ 计算得出的单一标量数值。**（贡献 1 个特征维度）**

    # 将它们拼接在一起：
    # **2 (位移) + 1 (距离) = 3 个特征**

    # 因此，每一条边都会被表示为一个长度为 3 的向量 `[∆x, ∆y, 距离]`。`config.yaml` 中的 `num_edge_features: 3` 正是用来告诉神经网络模型的第一层（输入层）：准备接收维度为 3 的边特征数据。
    @staticmethod
    def add_edge_features(graph, pos):
        """
        adds relative displacement & displacement norm as edge features
        """
        row, col = graph.edge_index
        disp = torch.tensor(pos[row] - pos[col])
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edge_attr = torch.cat((disp, disp_norm), dim=1)
        return graph

    @staticmethod
    def normalize_node(invar, mu, std):
        """normalizes a tensor"""
        if (invar.size()[-1] != mu.size()[-1]) or (invar.size()[-1] != std.size()[-1]):
            raise AssertionError("input and stats must have the same size")
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def normalize_edge(graph, mu, std):
        """normalizes a tensor"""
        if (
            graph.edge_attr.size()[-1] != mu.size()[-1]
            or graph.edge_attr.size()[-1] != std.size()[-1]
        ):
            raise AssertionError("Graph edge data must be same size as stats.")
        return (graph.edge_attr - mu) / std

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar

    # ### 1. `_one_hot_encode(node_type)` 详解
    # 这个函数的作用是将原始的、用整数表示的“节点物理类型”，转换为机器学习模型更容易处理的**独热编码（One-Hot Encoding）**向量。

    # 逐行解析：
    # * **`node_type = torch.squeeze(node_type, dim=-1)`**：去除最后一个多余的维度。比如把形状为 `[节点数, 1]` 的张量变成 `[节点数]`。
    # * **`node_type = torch.where(...)`**：这是一个重映射（Remapping）操作。
    #     * 在 DeepMind 开源的流体数据集中，节点类型（Node Type）通常是这样定义的整数：0 表示普通流体节点，而边界节点可能是 4（入口）、5（出口）、6（墙壁）。
    #     * 这段代码的逻辑是：如果节点类型是 0，保持为 0；如果不是 0，则减去 3。
    #     * **结果**：原本的类型 `[0, 4, 5, 6]` 被压缩映射成了 `[0, 1, 2, 3]`。
    # * **`node_type = F.one_hot(node_type.long(), num_classes=4)`**：
    #     * 将上面映射好的 `[0, 1, 2, 3]` 转换为长度为 4 的独热向量。
    #     * 0 变成 `[1, 0, 0, 0]`，1 变成 `[0, 1, 0, 0]`，以此类推。
    #     * **核心结论：经过这一步，每一个节点都被赋予了一个维度为 4 的类型特征向量。**
    @staticmethod
    def _one_hot_encode(node_type):  # TODO generalize
        node_type = torch.squeeze(node_type, dim=-1)
        node_type = torch.where(
            node_type == 0,
            torch.zeros_like(node_type),
            node_type - 3,
        )
        node_type = F.one_hot(node_type.long(), num_classes=4)
        return node_type

    @staticmethod
    def _drop_last(invar):
        return torch.tensor(invar[0:-1], dtype=torch.float)

    @staticmethod
    def _push_forward(invar):
        return torch.tensor(invar[1:], dtype=torch.float)

    @staticmethod
    def _push_forward_diff(invar):
        return torch.tensor(invar[1:] - invar[0:-1], dtype=torch.float)

    @staticmethod
    def _get_rollout_mask(node_type):
        mask = torch.logical_or(
            torch.eq(node_type, torch.zeros_like(node_type)),
            torch.eq(
                node_type,
                torch.zeros_like(node_type) + 5,
            ),
        )
        return mask

    @staticmethod
    def _add_noise(features, targets, noise_std, noise_mask):
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, 2)
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise
        targets -= noise
        return features, targets

    @staticmethod
    def _decode_record(rec_bytes: dict, meta: dict) -> dict:
        """Decode raw bytes from TFRecord into numpy arrays.

        The tfrecord package parses the TFRecord and
        provides raw bytes for each feature, which are decoded using numpy.

        Parameters
        ----------
        rec_bytes : dict
            Dictionary mapping feature names to raw bytes from tfrecord package.
        meta : dict
            Metadata dictionary containing feature specifications (dtype, shape, type).

        Returns
        -------
        dict
            Dictionary mapping feature names to decoded numpy arrays.
        """
        outvar = {}
        for k, v in meta["features"].items():
            # Map TensorFlow dtype names to numpy dtypes.
            dtype_map = {
                "float32": np.float32,
                "float64": np.float64,
                "int32": np.int32,
                "int64": np.int64,
            }
            dtype = dtype_map.get(v["dtype"], getattr(np, v["dtype"]))

            # Decode raw bytes to numpy array.
            # Use .copy() to make array writable (np.frombuffer returns read-only view).
            data = np.frombuffer(rec_bytes[k], dtype=dtype).copy()
            data = data.reshape(v["shape"])

            if v["type"] == "static":
                # Tile static features across trajectory length.
                # np.tile creates a new writable array.
                data = np.tile(data, (meta["trajectory_length"], 1, 1))
            elif v["type"] == "dynamic_varlen":
                # Handle variable-length sequences using row lengths.
                row_len = np.frombuffer(rec_bytes["length_" + k], dtype=np.int32)
                # Convert to list of variable-length arrays (ragged).
                data = np.split(data, np.cumsum(row_len)[:-1])

            outvar[k] = data
        return outvar

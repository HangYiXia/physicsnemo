# AMR-M4GN 开发进度与操作手册（M1 阶段）

**更新日期**：2026年6月10日（按新版 Design Doc 修订）
**当前阶段**：M1 — 离线预处理（模态分解 + 混合分区 + 诊断可视化）✅ 已完成
**配套设计文档**：`E:\phys\physicsnemo\AMR_M4GN_Design_Doc.md`
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`

> **里程碑总进度**（截至 2026-06-12）：**M1 ✅** · M2 ✅ · M3 ✅ · M4 🟢 · M5 🟢（AMR>MGN，全量待算力）· M6 ⬜ · M7 ⬜
> **文档索引**：M1（本文）· M2 `AMR_M4GN_Progress_M2.md` · M3 `..._M3.md` · M4 `..._M4.md` · 设计 `AMR_M4GN_Design_Doc.md`

> 本文档三件事：(1) 说清 M1 做了什么、为什么、对应 Design Doc 哪里；(2) 给出**从零到跑通 M1 的逐步操作**（装环境→下数据→跑脚本→看图验收）；(3) 列出后续开发（M2 起）的注意事项与决策门。**每一步都标注「为什么这么做」。**

---

## 一、M1 做了什么（文件清单）

```
examples/cfd/vortex_shedding_mgn/
├── amr_m4gn/                       ← 模型包（M1 新建）
│   ├── __init__.py                 ✅ 导出 laplacian_eigenmodes / graph_laplacian /
│   │                                  cotangent_laplacian / hybrid_segmentation /
│   │                                  build_partition_tree
│   ├── modal_decomp.py             ✅ 模态分解（Laplacian 特征模）
│   └── segmentation.py             ✅ 混合分区（METIS + SLIC + 递归二层树）
├── visualize_partition.py          ✅ 独立诊断可视化脚本（出 6 张图 + 缓存）
├── train.py                        ⬜ 原 MGN baseline（未改，与本工作共存）
├── inference.py                    ⬜ 原推理脚本（未改）
└── conf/config.yaml                ⬜ 原 MGN 配置（未改）
```

M1 的目标（Design Doc §八 M1）：**确认模态分解与分区在真实网格上合理**，产出可肉眼判读的诊断图。**不涉及任何训练**。

---

## 二、各模块功能与函数签名

### 2.1 `amr_m4gn/modal_decomp.py` — 模态分解

对应 Design Doc **§4.2 模态分解**。提取网格几何-物理结构特征 `f_md`，供分区算法产出「物理一致」的段。

| 函数 | 签名 | 作用 |
| --- | --- | --- |
| `graph_laplacian` | `(edge_index, num_nodes, pos=None) -> (L, M)` | 简单图 Laplacian L=D−A（任意图）|
| `cotangent_laplacian` | `(pos, cells, num_nodes) -> (L, M)` | FEM 余切 Laplacian（对 −∇² 一致离散，物理意义更好）|
| `laplacian_eigenmodes` | `(edge_index, pos, node_type=None, cells=None, num_modes=6, use_cotangent=True, boundary_type="neumann") -> (f_md[N,m], eigvals[m])` | 算前 m 个特征模 |
| `compute_node_area` | `(pos, cells, num_nodes) -> area[N]` | 每节点 Voronoi 面积（后续动量 M 用）|

**关键设计决策（为什么）**：
- 默认 `use_cotangent=True`：余切权重对 −∇² 的离散一致，模态有物理意义；`graph_laplacian` 仅作无 cells 时的退路。
- 默认 `boundary_type="neumann"`：全域求解、跳过常数零模，避免 Dirichlet 在边界条件不当时特征值退化（Design Doc §九 风险表第 1 行）。
- `eigsh` 用 shift-invert（sigma=0）加速小特征值收敛；异常 fallback 宽松容差；每模归一化单位范数。

### 2.2 `amr_m4gn/segmentation.py` — 混合分区

对应 Design Doc **§4.3 混合网格分区**。

| 函数 | 签名 | 作用 |
| --- | --- | --- |
| `metis_partition` | `(edge_index, num_nodes, num_parts) -> assign[N]` | Stage1：METIS 粗分（连通、最小割）|
| `slic_refinement` | `(pos, features, init_assign, num_segments, tau=1.0, max_iter=10, connectivity_constraint=True, edge_index=None) -> assign[N]` | Stage2：SLIC 物理感知精修 |
| `hybrid_segmentation` | `(edge_index, pos, f_md, f_obs=None, num_segments=64, tau=1.0, max_iter=10) -> assign[N]` | METIS+SLIC 合一 |
| `build_partition_tree` | `(edge_index, pos, f_md, f_obs=None, K_list=(64,256), tau=1.0, max_iter=10) -> (levels=[L0,L1], seg_adj=[L0_adj,L1_adj])` | 递归二层树 + 段邻接 |
| `compute_obstacle_distance` | `(pos, node_type, obstacle_type_id=6) -> f_obs[N]` | 到固体壁面距离（SLIC 特征）|

**关键设计决策（为什么）**：
- SLIC 距离 `d(i,Ck)=‖f_obs‖+‖f_md‖+τ‖x‖`（Design Doc §4.3 公式），特征与坐标均 [0,1] 归一化，保证 τ 物理含义一致。
- `connectivity_constraint=True`：节点只能重分配到「邻居所属」段，避免跨空隙连接（修正 same-size k-means 缺陷）。
- 无 `pymetis` 时自动 fallback 谱聚类（会告警），保证 Windows 也能跑。

### 2.3 `visualize_partition.py` — 诊断可视化（独立脚本）

读 1 个 case → 跑模态分解 + 分区 → 出 6 张 PNG + 缓存 `partition_cache.pt` 到 `./partition_vis/`。对应 Design Doc **§7.7 可视化校验 / §8.1 可视化验收清单**。

输出图：`01_mesh.png`、`02_velocity.png`、`03_eigenmodes.png`、`04_obstacle_dist.png`、`05_partition_L0.png`、`06_partition_L1.png`。

---

## 三、M1 相对「旧方案」的修正（本次改动）

旧 M1 按旧方案写，本次对照新 Design Doc 核对，修了 **3 处缺陷**（详见 Design Doc §八 M1 与本节）：

| # | 文件 | 旧 | 新 | 为什么 |
| --- | --- | --- | --- | --- |
| 1 | `segmentation.py` | `compute_obstacle_distance(obstacle_type_id=5)` | `=6` | DeepMind NodeType 约定：**5=OUTFLOW（出口）、6=WALL_BOUNDARY（含圆柱壁）**。旧默认把「出口」当障碍，`f_obs` 全错→污染 SLIC 分区。由数据集 `_one_hot_encode`（4/5/6→1/2/3，num_classes=4）佐证真实 type∈{0,4,5,6}。|
| 2 | `visualize_partition.py` | 探测顺序 `[5,4,3,6]`、`type_names` 5/6 标反 | 探测 `[6,1]`、标签纠正 | 同上，先命中出口是错的；标签错误会误导诊断图 |
| 3 | `visualize_partition.py` | 无 | 打印 `pos` 的 x/y 范围 | 提供 Design Doc **决策门 D2 / 不确定点 U1** 所需数据：确认各 case 是否同坐标系，决定 AMR 阈值用绝对值还是 Top-r 分位 |

> ⚠️ **未实跑验证**：本机无数据集、`base` Python 缺 numpy，仅做了 `ast` 语法校验。上述修正基于 DeepMind 标准约定推断，**需你跑 §六 后用脚本打印的「Node types」计数确认**（见 §七 验收）。

---

## 四、与 Design Doc 的对应关系

| Design Doc 章节 | M1 落地 |
| --- | --- |
| §4.1 总体架构 — OFFLINE ①模态/③METIS/④递归 | `modal_decomp` + `segmentation` |
| §4.2 模态分解 | `modal_decomp.py` |
| §4.3 混合网格分区（METIS+SLIC+递归二层树）| `segmentation.py` |
| §7.3 概览表（modal/segmentation 标 ✅）| 本阶段交付 |
| §7.7 测试策略 — 可视化校验 | `visualize_partition.py` |
| §8.1 可视化验收清单（模态/分区图）| 6 张诊断图 |
| §八 M1 退出标准 | 见 §七 验收清单 |

**M1 未覆盖（属后续里程碑，勿在此实现）**：
- 段重叠 δ=1 → 归 **M4**（聚合阶段用，§4.1⑤/§7.2-C）。
- `l1_to_l0` 父子映射、段质心 `centroid`、`area`、RWSE → 归 **M3（pe.py）/ M4（preprocess_partitions.py）**（§7.4.1/§7.5）。当前 `partition_cache.pt` 只存 M1 诊断所需字段。

---

## 五、环境准备（从零开始）

> **为什么**：M1 只需 numpy/scipy/torch/matplotlib/tfrecord（读数据）；`pymetis` 强烈建议装（否则分区退化为较慢的谱聚类）；`torch_geometric` 在 `visualize` 里用于 `to_undirected`。

### 5.1 进入工作目录

```bash
cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
```

### 5.2 安装依赖

```bash
# 项目自带依赖（含 tfrecord / scipy / torch_geometric / torch_scatter）
pip install -r requirements.txt

# 可视化 + 分区质量（务必装 pymetis；Windows 装不上见下）
pip install matplotlib pymetis
```

- **为什么装 pymetis**：METIS 是分区主力（连通、最小割、快）。装不上时 `segmentation.py` 会打印 `Warning: pymetis not found`，自动用谱聚类退路——能跑但质量/速度差。
- **Windows 装 pymetis 失败**的处理：① 用 conda：`conda install -c conda-forge pymetis`；② 或先在 WSL/Linux 里跑预处理；③ 或直接接受谱聚类退路（M1 可视化阶段够用）。

### 5.3 自检（不依赖数据集）

```bash
python -c "import numpy, scipy, torch, matplotlib; print('core OK')"
python -c "from amr_m4gn import laplacian_eigenmodes, build_partition_tree; print('package OK')"
python -c "import pymetis; print('pymetis OK')"   # 失败=会走谱聚类退路，可继续
```

- **为什么**：先确认包能导入，避免跑脚本到一半才报缺依赖。

---

## 六、数据集下载（从零开始）

> **为什么**：M1 的 `visualize_partition.py` 需要 1 个真实 case（含 `mesh_pos`/`cells`/`node_type`/`velocity`）。数据来自 DeepMind 的 cylinder_flow（COMSOL 仿真，1000 训练/100 验证/100 测试，每 case ~1885 节点、600 时间步）。

### 6.1 直接从 Google Cloud 下载（最稳，推荐）

DeepMind 把数据放在公开 GCS 桶：`https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/`，共 4 个文件。手动建目录并下载：

```bash
cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
mkdir raw_dataset\cylinder_flow\cylinder_flow

# 逐个下载（任选 curl / wget / 浏览器）
curl -o raw_dataset\cylinder_flow\cylinder_flow\meta.json      https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/meta.json
curl -o raw_dataset\cylinder_flow\cylinder_flow\train.tfrecord https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord
curl -o raw_dataset\cylinder_flow\cylinder_flow\valid.tfrecord https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord
curl -o raw_dataset\cylinder_flow\cylinder_flow\test.tfrecord  https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord
```

> **为什么手动**：本仓库的 `vortex_shedding_mgn` 目录**没有自带** `download_dataset.sh`（README 里那条 `sh download_dataset.sh` 实际依赖 DeepMind 的脚本）。M1 只看 1 个 case，**只下 `meta.json` + `test.tfrecord` 即可**，train/valid 可暂时不下（M5 全量训练再补）。

下载后目录应为：
```
raw_dataset/cylinder_flow/cylinder_flow/
├── meta.json
├── test.tfrecord      (M1 必需)
├── train.tfrecord     (M5 才需)
└── valid.tfrecord     (M5 才需)
```

- **为什么是双层 `cylinder_flow/cylinder_flow`**：与 `conf/config.yaml` 的 `data_dir: ./raw_dataset/cylinder_flow/cylinder_flow` 一致，也是 `visualize_partition.py` 的默认路径。

### 6.2 用 DeepMind 官方脚本（需 Git Bash / WSL）

```bash
cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
mkdir raw_dataset && cd raw_dataset
git clone https://github.com/deepmind/deepmind-research.git
sh deepmind-research/meshgraphnets/download_dataset.sh cylinder_flow ./cylinder_flow/cylinder_flow
```

- **为什么**：官方脚本内部就是从 §6.1 的 GCS 桶 `wget` 那 4 个文件，等价但需要 `sh`/`wget`。

### 6.3（可选）生成 tfindex

```bash
python -m tfrecord.tools.tfrecord2idx raw_dataset/cylinder_flow/cylinder_flow/test.tfrecord raw_dataset/cylinder_flow/cylinder_flow/test.tfindex
```

- **为什么可选**：`visualize_partition.py` 在 `test.tfindex` 缺失时把 `index_path` 置 None 仍能顺序读取；有索引则支持随机访问、更快。

---

## 七、运行 M1 与验收（核心）

### 7.1 跑可视化脚本

```bash
cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
python visualize_partition.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test --case_idx 0
```

常用参数（均有默认值）：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--split` | `test` | 用哪个划分（test 才带 mesh_pos/cells）|
| `--case_idx` | `0` | 看第几个 case |
| `--num_modes` | `6` | 模态数（对应 Design Doc m=6）|
| `--K0` / `--K1` | `64` / `256` | 二层分区段数 |
| `--tau` | `1.0` | SLIC 紧凑度 |
| `--boundary_type` | `neumann` | Laplacian 边界条件 |
| `--output_dir` | `./partition_vis` | 出图目录 |

### 7.2 期望终端输出（关键，用于验收）

跑通后终端会打印（节选）：
```
Nodes: ~1885, Cells: ..., Edges: ...
Node types: {0: ..., 4: ..., 5: ..., 6: ...}      ← 验收点 A
pos x-range: [...], y-range: [...]                 ← 验收点 B（决策门 D2）
Eigenvalues: [...]                                 ← 验收点 C
Found wall/obstacle nodes with type=6 (count=...)  ← 验收点 D
Level 0: 64 segments / Level 1: 256 segments       ← 验收点 E
```

### 7.3 验收清单（对应 Design Doc §8.1）

| 验收点 | 看哪里 | 合格判据 | 为什么 |
| --- | --- | --- | --- |
| **A. node type** | 终端 `Node types` | 出现 `{0,4,5,6}`，且 6 的数量是合理的壁面节点数 | 确认 §三 修正#1 的推断正确（圆柱壁=6）。**若实际不是 6，按打印结果改 `obstacle_type_id`** |
| **B. pos 范围** | 终端 `pos x/y-range` | 记录数值；多跑几个 `case_idx` 看是否同范围 | 决策门 D2：同坐标系→AMR 可用绝对阈值；否则用 Top-r 分位 |
| **C. 特征模** | `03_eigenmodes.png` | 6 个模无「死模」（全同色）；中频在圆柱/尾迹有结构 | 模态有意义才能指导分区 |
| **D. 障碍距离** | `04_obstacle_dist.png` + 终端 type=6 | 壁面/圆柱处 ≈0、远场渐大 | f_obs 正确才不会污染 SLIC |
| **E. 分区** | `05_partition_L0.png` / `06_partition_L1.png` | 同色连通；圆柱附近段更密；min/max 段大小不极端悬殊 | 分区连通+物理一致是 Design Doc 核心 |
| **F. 网格** | `01_mesh.png` / `02_velocity.png` | 圆柱后方可见涡街、节点类型分布合理 | 确认数据读对 |

**M1 通过标准**：A~F 全部合格 → M1 关闭，进入 M2。

### 7.3.1 实测结果（2026-06-10，4 个 case）

跑完 `test` 划分 `case_idx ∈ {0, 1, 50, 99}`，肉眼+数值核验全过：

| case | Nodes | Cells | type=0/4/5/6 | pos x-range | pos y-range | λ₁ | λ₆ | L0 段 (min/max/mean/std) | L1 段 (min/max/mean/std) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1923 | 3612 | 1689/17/17/200 | [0, 1.6] | [0, 0.41] | -0.0501 | 61.05 | 16/50/30.0/9.3 | 4/13/7.5/2.4 |
| 1 | 1757 | 3276 | 1519/17/17/204 | [0, 1.6] | [0, 0.41] |  0.0173 | 60.68 | — | — |
| 50 | 1912 | 3590 | 1678/17/17/200 | [0, 1.6] | [0, 0.41] | -0.0352 | 60.15 | — | — |
| 99 | 1976 | 3718 | 1742/17/17/200 | [0, 1.6] | [0, 0.41] |  0.0321 | 60.81 | — | — |

**结论**：
- **A. node type 一致**：4 case 全是 `{0,4,5,6}`，inflow/outflow 恒为 17，type=6 在 200~204（圆柱大小/位置略变）。修正 #1（`obstacle_type_id=6`）确认正确。
- **B. pos 范围 4 case 全等** `x∈[0, 1.6], y∈[0, 0.41]` → **决策门 D2 闭环：AMR 阈值用「绝对值」**，不需 Top-r 分位（M3 阈值实现可直接读 `pos` 比较）。
- **C. 特征值一致性高**：λ₁≈0（数值噪声 ±0.05，Neumann 常数模），λ₂∈[3.60, 3.83]、λ₆∈[60.15, 61.05]，跨 case 谱稳定 → 模态特征 `f_md` 跨 case 可比。
- **D. f_obs 物理正确**：通道中线 ≈ 通道半高（0.205），圆柱位置中线变暗（验收点 D 通过，见 case 0 `04_obstacle_dist.png`）。
- **E. 分区合格**：L0 段大小比 50/16≈3.1×、L1 段大小比 13/4≈3.25×，未极端悬殊；mean=N/K 命中理论值；颜色 x 向自然渐变、连通无飞地。

**M1 验收**：✅ 关闭。

### 7.3.1.1 ⚠️ 关注项（非阻塞，留待 M3/D3 复核）— 分区呈「流向带状」

观察 `05_partition_L0.png` / `06_partition_L1.png`：分区颜色沿 x 方向 red→…→black 平滑渐变，本质是**沿流向（x）的竖条带**，竖直（y）方向细分很弱。原因：域为长通道（1.6×0.41，~4:1），低阶 Laplacian 模（Mode2-4）天然是流向驻波，主导了 SLIC 的 `f_md`；圆柱/尾迹结构要到 Mode5-6 才出现，权重低。

- **为什么 M1 仍可关闭**：分区只提供「候选段」，真正决定 token 粗细的是 **M3 的 AMR Router 用运行时物理量（涡量 ω 等）**判活跃，与分区是否带状无关。圆柱区已被 f_obs 单独隔出（灰色段），满足连通/均衡。
- **潜在风险（M3/D3 须验证）**：若尾迹某个 L1 段同时跨「高涡量中线」与「平静外区」，按「段内 max|ω|」会把整段保细 → 浪费。即**尾迹段的几何隔离度可能不足**。
- **可调杠杆（M3 若 D3 显示 token 浪费再用，勿提前改）**：① 增 `num_modes`（捕捉更多圆柱/尾迹结构）；② 增大 K0/K1；③ 提高 SLIC `τ`（更紧凑、更各向同性的段）；④ 给 f_obs 更高权重。

### 7.3.2 已知非阻塞告警

| 告警 | 触发位置 | 影响 | 处理 |
| --- | --- | --- | --- |
| `M does not have the same type precision as A` (ARPACK) | `modal_decomp.py:285/291` | 无功能影响，仅收敛性提示 | 想消除：在 `eigsh` 调用前 `M = M.astype(A.dtype)`（M2 顺手修）|
| `cm.get_cmap deprecated` (matplotlib 3.7+) | `visualize_partition.py:225` | 出图正常 | 想消除：换 `matplotlib.colormaps.get_cmap('tab20', K)` |

### 7.4 常见问题

- `ModuleNotFoundError: tfrecord` → `pip install tfrecord`。
- `pymetis not found` 告警 → 可继续（谱聚类退路），但建议装好再正式用。
- 图里分区有「飞地」（同色不连通）→ 调大 `--max_iter` 或检查 `connectivity_constraint`。
- `04_obstacle_dist.png` 全场无明显近壁低值 → 说明 type 探测错，回看验收点 A，手动指定正确 `obstacle_type_id`。

---

## 八、后续开发注意事项（M2 起）

### 8.1 数据管线（最先要解决，Design Doc §7.2-A）

- **训练 split 的 `graph` 不带 `pos`/`cells`，`edge_attr` 是归一化的**。M2 起需要坐标：按 **方案 P1**（推荐）新建 `data_amr.py` 子类 `VortexSheddingDatasetAMR`，train 分支补回 `graph.pos`、`graph.gidx`。**这是决策门 D5，M4 开工即定。**

### 8.2 决策门（必须看到数据/结果才能定，勿提前硬编码）

| 门 | 在哪 | 等什么数据 | 决定什么 |
| --- | --- | --- | --- |
| D1 (U4) | M2 | 对比「反归一化 vs 归一化空间」算的涡量图 | physics_ops 是否反归一化速度（影响物理可解释性）|
| D2 (U1) | M1/M2 | §7.2 打印的 pos 范围 | AMR 阈值用绝对值还是 Top-r |✅ **已闭环 (2026-06-10)**：4 case (0/1/50/99) pos 全等 `x∈[0,1.6], y∈[0,0.41]` → **用绝对值阈值**
| D3 | M3 | T（token 数）分布统计 | 最终 K0/K1 与阈值区间 |
| D4 (U2) | M4 | micro_gnn 单测 | MeshGraphNet 旁路 decoder 用路径 (a) 还是 (b) |
| D6 (U5) | M5 | 关 AMR 期 loss 曲线 | AMR warm-up 时长 |
| D7 | M5 | 长程指标 vs MGN | 课题是否成立（Transformer 是否真起作用）|

### 8.3 接口约定（新文件须遵守，Design Doc §7.4）

- 缓存按 `gidx` 命名：`partition_cache_{split}_{gidx}.pt`（每 case 几何固定，只算一次）。
- PyG 批处理时 `kept_assign`/token 须按 `graph.batch` **偏移**，否则跨图节点会被错聚（§7.2-C）。M4 先 `batch_size=1`，M5 再支持多图。
- `physics_ops` 的 `u,v` 来自 `graph.x[:, :2]`（**已归一化**），算物理量前按 D1 决定是否用 `node_stats.json` 反归一化。

### 8.4 测试约定

- 每个新文件配 `pytest` 单测（解析场/合成图），见 Design Doc §7.4 各小节的「单元测试」。
- 集成测试：`model.py` 单 case 单步前向 → 输出 `[N,3]` 无 NaN、全参有梯度。
- Overfit：单 case 多步训练 loss 应降到接近 0。

### 8.5 下一步（M2）速览

新建 `amr_m4gn/physics_ops.py`（G/ω/M/S + 1-ring 最小二乘 + 虚拟步），扩展 `visualize_partition.py` 出四标量场图。退出标准：解析场单测误差 <5%，可视化中尾迹 ω 显著高于来流。**D2 已闭环（绝对阈值），M2 期间需过决策门 D1（涡量反归一化与否）。**

### 8.6 M2 待办清单（具体到文件/函数）

按 Design Doc §7.4.2 / §4.4，M2 一上来要做的事：

1. **新建 `amr_m4gn/physics_ops.py`**：
   - `compute_gradient(u, v, pos, edge_index) -> (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)`：1-ring 加权最小二乘
   - `compute_vorticity(grad_u, grad_v) -> ω = ∂v/∂x - ∂u/∂y`
   - `compute_kh_shear(grad_u, grad_v) -> S = ∂u/∂y - ∂v/∂x`（Kelvin-Helmholtz 剪切，**对齐 Design Doc §4.6 / AMR-Transformer Eq.5**；注意不是应变率幅值 √(2 S_ij S_ij)）
   - `compute_velocity_gradient_mag(grad_u, grad_v) -> G = √(‖∇u‖²+‖∇v‖²)`（速度梯度，§4.6 第一判据）
   - `compute_momentum_indicator(u, v, area) -> M = ρ·|U|·area`（Design Doc §4.6）
   - `virtual_step(u_t, u_prev) -> u_{t+1}^* = u_t + (u_t − u_prev)`（前向欧拉，用于 AMR 阈值的预估场）
2. **扩展 `visualize_partition.py`**：加 `--plot_physics` 开关，多出 4 张图 `07_grad.png` / `08_vorticity.png` / `09_strain.png` / `10_momentum.png`。
3. **决策门 D1 实测**：在 case 0 上同时算「归一化 u/v」和「反归一化 u/v（用 `node_stats.json`）」的涡量场；对比尾迹 ω 量级是否合物理（典型 Re=100 圆柱尾迹 |ω|~O(10) /s）→ 决定 `physics_ops` 默认是否反归一化。
4. **单测**：`tests/test_physics_ops.py`：
   - 解析场 `u=sin(x)cos(y), v=-cos(x)sin(y)` → ω 误差 <5%
   - 均匀流 `u=const` → ω≈0、S≈0
   - 全参有梯度（用合成图反传 dummy loss）
5. **顺手修 M1 告警**（§7.3.2）：ARPACK dtype 一致 + matplotlib colormap API。

**M2 退出标准**：①解析场单测全过 ②case 0 涡量图尾迹结构正确（脱涡可见）③D1 拍板写进 doc。

# AMR-M4GN 已完成工作记录

**更新日期**：2026年5月14日  
**当前阶段**：M1 - 离线预处理模块（模态分解 + 混合分区 + 可视化）

---

## 一、已完成的文件清单

```
physicsnemo/examples/cfd/vortex_shedding_mgn/
├── amr_m4gn/                          ✅ 新建（模型包）
│   ├── __init__.py                    ✅ 包初始化
│   ├── modal_decomp.py                ✅ 模态分解模块
│   └── segmentation.py                ✅ 混合分区模块
├── visualize_partition.py             ✅ 独立可视化脚本
├── train.py                           ⬜ 未修改（原 MGN baseline）
├── conf/config.yaml                   ⬜ 未修改
└── ...
```

---

## 二、各模块功能说明

### 2.1 `amr_m4gn/modal_decomp.py`

**状态**：✅ 已完成

**提供的函数**：

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `graph_laplacian()` | 构建图 Laplacian L=D-A | edge_index, num_nodes | L, M (scipy sparse) |
| `cotangent_laplacian()` | 构建 FEM 余切 Laplacian | pos, cells, num_nodes | L, M (scipy sparse) |
| `laplacian_eigenmodes()` | 计算前 m 个特征模 | edge_index, pos, node_type, cells, num_modes | f_md [N,m], eigvals [m] |
| `compute_node_area()` | 计算每个节点的 Voronoi 面积 | pos, cells, num_nodes | area [N] |

**关键设计决策**：

1. **两种 Laplacian 实现**：
   - `graph_laplacian`：简单的 L=D-A，适用于任意图，不需要三角形信息
   - `cotangent_laplacian`：FEM 余切权重，物理意义更好（对 -∇² 的一致离散化），需要 cells
   - 默认使用 cotangent（`use_cotangent=True`）

2. **边界条件处理**：
   - `neumann`（默认）：在全域求解，跳过第一个零特征值（常数模）
   - `dirichlet`：固定边界节点为 0，只在内部节点上求解，结果 scatter 回全域

3. **鲁棒性**：
   - `eigsh` 使用 shift-invert (`sigma=0`) 加速小特征值收敛
   - 异常时自动 fallback 到宽松容差
   - 每个模归一化到单位范数

### 2.2 `amr_m4gn/segmentation.py`

**状态**：✅ 已完成

**提供的函数**：

| 函数 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `metis_partition()` | METIS 图分区 | edge_index, num_nodes, num_parts | assign [N] |
| `slic_refinement()` | SLIC 超像素精修 | pos, features, init_assign, ... | assign [N] |
| `hybrid_segmentation()` | METIS + SLIC 两阶段 | edge_index, pos, f_md, f_obs, K, tau | assign [N] |
| `build_partition_tree()` | 构建二层层次分区树 | edge_index, pos, f_md, f_obs, K_list | levels, adjacency |
| `compute_obstacle_distance()` | 计算到障碍物的距离 | pos, node_type | f_obs [N] |

**关键设计决策**：

1. **METIS 依赖处理**：
   - 优先使用 `pymetis`（高质量图分区）
   - 自动 fallback 到谱聚类（`scipy.sparse.linalg.eigsh` + KMeans），无需额外依赖
   - fallback 会打印 warning 提示安装 pymetis

2. **SLIC 精修逻辑**（M4GN Algorithm 2）：
   - 距离度量：`d(i, Ck) = ||f_i - f_Ck|| + tau * ||x_i - x_Ck||`
   - 连通性约束：节点只能被重分配到邻居节点所属的 segment（避免跨空隙连接）
   - 特征和坐标都做 [0,1] 归一化，确保 tau 的物理含义一致
   - 收敛检测：无节点变动即停止

3. **层次树构建**：
   - Level 0：对全图运行 `hybrid_segmentation(K=64)`
   - Level 1：对每个 L0 segment 的子图运行 `metis_partition(sub_K=4)`
   - 自动处理过小的 segment（< sub_K 个节点则不再细分）
   - 返回 segment 级邻接矩阵（用于 RWSE 计算和 Transformer 可视化）

4. **障碍物距离计算**：
   - 自动检测 node_type 中的圆柱壁面节点（类型 5）
   - 批量计算避免 N×M 的内存爆炸
   - fallback：无障碍物时使用到网格中心的距离

### 2.3 `visualize_partition.py`

**状态**：✅ 已完成

**功能**：独立脚本，加载一个 case 的原始数据，运行完整预处理流水线并产出 6 张诊断图。

**输出文件（保存到 `./partition_vis/` 目录）**：

| 文件名 | 内容 |
| --- | --- |
| `01_mesh.png` | 网格结构 + 节点类型着色 |
| `02_velocity.png` | 初始时刻速度场幅值 |
| `03_eigenmodes.png` | 前 6 个 Laplacian 特征模（2×3 子图） |
| `04_obstacle_dist.png` | 障碍物距离场 f_obs |
| `05_partition_L0.png` | Level 0 分区（64段）+ segment 邻接线 |
| `06_partition_L1.png` | Level 1 分区（256段） |
| `partition_cache.pt` | 所有预处理数据的 PyTorch 缓存 |

**使用方法**：
```bash
cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
python visualize_partition.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow
```

**可选参数**：
```
--split test          数据集 split（test/train/valid）
--case_idx 0          可视化第几个 case
--num_modes 6         Laplacian 模态数
--K0 64               Level 0 segment 数
--K1 256              Level 1 segment 数
--tau 1.0             SLIC 紧凑性参数
--output_dir ./partition_vis    输出目录
--boundary_type neumann         边界条件类型
```

---

## 三、依赖环境

### 必须安装的包

```bash
pip install pymetis          # METIS 图分区（强烈推荐，否则 fallback 到谱聚类）
pip install scipy            # 稀疏特征值求解
pip install matplotlib       # 可视化
pip install tfrecord         # 读取 vortex_shedding 的 TFRecord 数据
pip install torch-geometric  # PyG 图数据结构
```

### 可选安装

```bash
pip install networkx         # 如果想用 nx 做分区对比
```

### 已假设可用

```
torch >= 2.0
numpy
```

---

## 四、运行前准备

1. **确保数据存在**：  
   数据路径：`./raw_dataset/cylinder_flow/cylinder_flow/`  
   需要包含：`test.tfrecord`（或 `train.tfrecord`）+ `meta.json`

2. **安装依赖**（在目标测试机器上）：
   ```bash
   pip install pymetis scipy matplotlib tfrecord torch-geometric
   ```

3. **运行可视化**：
   ```bash
   cd physicsnemo/examples/cfd/vortex_shedding_mgn
   python visualize_partition.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test
   ```

4. **检查输出**：
   - `partition_vis/` 目录下应有 6 张 PNG + 1 个 `.pt` 缓存文件
   - 重点观察 `03_eigenmodes.png` 和 `05_partition_L0.png`

---

## 五、验收标准

运行成功后，需要确认以下几点（肉眼判断）：

### 5.1 模态分解 (`03_eigenmodes.png`)

- [ ] 低频模（Mode 1-2）应表现为大尺度空间变化（如沿流向/法向的光滑渐变）
- [ ] 中频模（Mode 3-6）应在圆柱附近和尾迹区域有更复杂的结构
- [ ] 边界（壁面）附近的模值应该有可见的梯度变化
- [ ] 不应出现全白或全黑的"死模"（如果出现说明特征值求解有问题）

### 5.2 障碍物距离场 (`04_obstacle_dist.png`)

- [ ] 圆柱表面处距离=0（最深色）
- [ ] 距离向外平滑递增
- [ ] 无跳变或异常值

### 5.3 Level 0 分区 (`05_partition_L0.png`)

- [ ] 圆柱附近应有较多小 segment（网格密度高）
- [ ] 远场应有较大 segment
- [ ] 同一颜色的 segment 应是**连通**的（不应出现跨越空白区域的同色块）
- [ ] segment 大小方差不应过大（stats 行中 std/mean < 0.5 为佳）
- [ ] 红色 adjacency 线应连接相邻 segment 的质心

### 5.4 Level 1 分区 (`06_partition_L1.png`)

- [ ] 每个 L0 segment 内部应被进一步细分为约 4 个子 segment
- [ ] 总 segment 数接近 256
- [ ] 圆柱附近细分更密集

---

## 六、已知限制 & 后续改进点

| 编号 | 限制 | 影响 | 计划在哪个里程碑解决 |
| --- | --- | --- | --- |
| 1 | SLIC 是 Python 循环，对大网格（>10k 节点）较慢 | 预处理慢但只算一次 | M5（可用 Cython/Numba 加速） |
| 2 | spectral fallback 在 K>100 时质量差 | 分区不均匀 | 安装 pymetis 即解决 |
| 3 | 暂不支持 3D 网格 | — | 远期扩展 |
| 4 | 未实现 segment overlap (δ=1) | 分区边界可能有不连续 | M4 |
| 5 | node_type 到障碍物的映射是硬编码的 (type=5) | 换数据集需改 | 加 config 参数 |

---

## 七、下一步计划（M2-M4）

### M2：物理量算子（预计 2 天）
- `amr_m4gn/physics_ops.py`
- 实现 1-ring least-square 梯度估计
- 实现 G, ω, M, S 四个物理量
- 可视化脚本画四个标量场在 mesh 上

### M3：AMR Router（预计 2 天）
- `amr_m4gn/amr_router.py`
- 实现二层 fold/keep 决策
- 单元测试：用已知涡街场验证 router 选择

### M4：端到端训练跑通（预计 3 天）
- `amr_m4gn/micro_gnn.py`
- `amr_m4gn/macro_transformer.py`
- `amr_m4gn/pe.py`
- `amr_m4gn/model.py`
- `train_amr_m4gn.py`
- `conf/config_amr_m4gn.yaml`
- 在单个 case 上验证 loss 下降

---

## 八、Git 提交建议

```bash
git add physicsnemo/examples/cfd/vortex_shedding_mgn/amr_m4gn/
git add physicsnemo/examples/cfd/vortex_shedding_mgn/visualize_partition.py
git add AMR_M4GN_Design_Doc.md
git add AMR_M4GN_Progress.md

git commit -m "feat(amr-m4gn): M1 - modal decomposition + hybrid segmentation + visualization

- Add Laplacian eigenfunctions (graph + cotangent FEM)
- Add METIS + SLIC hybrid mesh segmentation (M4GN style)
- Add 2-level partition tree construction
- Add standalone visualization script for preprocessing validation
- Add design document and progress tracking"
```

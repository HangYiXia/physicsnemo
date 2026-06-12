# AMR-M4GN 开发文档 — M7：EAGLE 大规模扩展（可选）

**更新日期**：2026-06-12
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`
**配套文档**：设计 `AMR_M4GN_Design_Doc.md`（§八 M7 / 决策门 D9）；总览 `README_AMR_M4GN.md`

> **里程碑总进度**：M1 ✅ · M2 ✅ · M3 ✅ · M4 ✅ · M5 🟢 · M6 🟡 · **M7 🟡（EAGLE reader + 并行预处理已写，按官方格式实现，真实数据待验证）**
>
> **状态图例**：✅ 完成并实跑验证 · 🟢 主体完成（仅更大规模待算力）· 🟡 **代码就绪、等待真实数据/算力验证** · ⬜ 未开始

---

## 0. M7 是什么 / 现在到哪了

M7 = 把 AMR-M4GN 扩到 **EAGLE 大规模湍流数据集**（Janny et al., ICLR 2023），验证大规模能力。Design Doc 两个交付物：**① EAGLE 数据 reader**、**② 预处理并行化（U6）**。

| # | 工作 | 交付物 | 状态 |
| --- | --- | --- | --- |
| 1 | EAGLE 数据 reader | `eagle_dataset.py`（`load_eagle_case` + `EagleDatasetAMR`）| 🟡 按官方格式写，合成数据验过管线，真实数据待验证 |
| 2 | 预处理并行化（U6）| `preprocess_partitions.py`: `--workers` + `--source eagle` | 🟡 已实现 |
| 3 | 数据集切换 | `data_amr.make_amr_dataset` + 训练/推理/评估加 `--dataset` | 🟡 已实现 |
| 4 | 单元测试 | `tests/test_eagle_dataset.py`（合成 npz 验形状/格式）| 🟡 已写，待目标机 `pytest` |

**一句话现状**：EAGLE reader 与全套切换、并行预处理**代码已写完**，并用**合成 EAGLE 格式 npz** 验证了「形状/图格式/统计」管线正确。但**未在真实 EAGLE 数据上跑过**（本机无数据/算力），真实数据的字段名、节点类型 id、动态网格行为需到目标机核对。训练/对比结果待算力。

---

## 1. 关键设计与**如实标注的假设**

EAGLE 与圆柱绕流的两大不同 + 处置：

| 差异 | 处置（文档化假设）|
| --- | --- |
| **动态网格**（节点/连通性可能逐帧变）| **A1**：AMR 分区缓存每个 sim 只建一次 → 取 **t=0 网格**（`pos[0]`、`cells[0]`）作为全 sim 的固定连通性/分区，仅让**场（速度/压强）随时间变**。与圆柱绕流「静止网格」假设一致。真正逐帧 remesh 需逐帧分区（未来工作，见 D9）。|
| **节点类型 id 体系不同** | **A3**：`eagle_node_type_to_class()` 把 EAGLE id 映射到 4 类 `{0 normal,1 inflow,2 outflow,3 wall}`，保持 `x` 维 6、4 路 one-hot 不变。默认映射是**最佳猜测**，需按真实数据核对。|
| 磁盘格式不同 | **适配层**：`EAGLE_KEYS` 字典集中 `.npz` 键名（已按官方 VERSION1 页核对：`mesh_pos`/`VX`/`VY`/`PS`(动压)/`PG`(静压)/`node_type`）；**三角网格在单独文件**，由 `EAGLE_TRIANGLE_KEYS`/`EAGLE_TRIANGLE_FILES` 解析（先查 npz 内键 → 再查同名 triangles 文件 → 否则 Delaunay 兜底并告警）。键名不符**只改这里**。|
| 压强双通道 | EAGLE 有 `PS`(动压) 与 `PG`(静压)；模型单压强通道，**默认用 `PS`(动压)**，需要静压改 `EAGLE_KEYS["pressure"]="PG"`。|

其余：2D 场 → `in_nodes=6`/`in_edges=3` 不变（**A2**）；rollout 只更新内部（映射类型 0）节点、边界保 GT（**A4**）。`node_type` 官方描述为「是否边界」的整数，默认映射 `0→normal`、`非0→wall`（如区分 inflow/outflow，在适配层 `_EAGLE_INFLOW/_OUTFLOW` 补 id）。

> **诚实边界**：键名已按官方页核对，但 **triangles 单独文件的确切布局**与 **node_type id 语义**仍需用真实文件核对（见 §8）。`eagle_dataset.py` 顶部「ADAPTER」段集中了所有需核对处。

---

## 2. 数据集复用方式（与圆柱绕流同一套代码）

`EagleDatasetAMR` 复用 `VortexSheddingDataset` 的静态方法（`cell_to_adj`/`create_graph`/`add_edge_features`/`normalize_*`/`_drop_last`/`_push_forward*`/`_add_noise`），产出**完全相同**的图格式：

- `graph.x` `[N,6]` = 归一化速度(2) + 节点类型 one-hot(4)
- `graph.y` `[N,3]` = 速度增量(2) + 压强(1)
- `graph.edge_attr` `[E,3]`、`graph.pos`、`graph.gidx`、`graph.x_prev`
- `node_stats`/`edge_stats`（train 时计算并存 `*_eagle.json`，避免覆盖圆柱绕流的 `node_stats.json`）
- `cells` / `rollout_mask` / `get_cache`

因此 `AMRM4GN`、`preprocess_partitions`、`train_amr_m4gn_full`、`inference_amr_m4gn`、`compare_baselines`、`eval_rollout` **无需改模型逻辑**，仅通过 `--dataset eagle` 切换。

---

## 3. 一键复现（待 EAGLE 数据，命令已就绪）

> **下载数据**：官方主页 `https://eagle-dataset.github.io/`（`Download from: FTP / HTTPS`）。数据**按 3 种几何类型拆成 3 个文件**，**子集 = 只下其中 1 个几何类型**；三角网格是**单独文件**，需一并下载。也可用官方仓库 `github.com/eagle-dataset/EagleMeshTransformer` / `eagle-dataset` pip 包。
>
> **前置**：把 EAGLE `.npz` 放到 `--data_dir`，每个 sim 一个文件；triangles 单独文件放好（让 `EAGLE_TRIANGLE_FILES` 能找到，或先用 Delaunay 兜底）；可选 `{split}.txt` 列出 id（否则按 `*.npz` 排序）。先到 `eagle_dataset.py` 顶部核对 `EAGLE_KEYS` / triangles 路径 / 节点类型映射。

```bash
cd examples/cfd/vortex_shedding_mgn

# 0) 并行预处理 EAGLE（U6；--workers 按 CPU 核数设）
python preprocess_partitions.py --source eagle --data_dir <EAGLE_DIR> --split train --num_cases 50 --out_dir ./eagle_cache --workers 8
python preprocess_partitions.py --source eagle --data_dir <EAGLE_DIR> --split test  --num_cases 10 --out_dir ./eagle_cache --workers 8

# 1) 训练（注意 EAGLE 网格更大，可调小 batch / processor_size，或减 num_cases）
python train_amr_m4gn_full.py --dataset eagle --data_dir <EAGLE_DIR> --cache_dir ./eagle_cache --num_cases 50 --num_steps 100 --batch_size 1 --epochs 100 --noise_std 0.02 --omega_thresh 8.9 --tag amr_eagle

# 2) 多 case 平均评估（如有 MGN-EAGLE 基线则带 --mgn_ckpt）
python eval_rollout.py --dataset eagle --data_dir <EAGLE_DIR> --cache_dir ./eagle_cache --amr_ckpt ./checkpoints_amr/amr_eagle_epoch99.pt --num_cases 10 --num_steps 90 --rollout 80 --omega_thresh 8.9

# 3) 场 GIF
python inference_amr_m4gn.py --dataset eagle --data_dir <EAGLE_DIR> --cache_dir ./eagle_cache --ckpt ./checkpoints_amr/amr_eagle_epoch99.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9 --gif

# (现在就能跑) 合成 npz 的 reader 单测
pytest tests/test_eagle_dataset.py -v
```

> **D9（显存）**：EAGLE 单图可能远大于圆柱绕流。若单图 OOM → 减 `--batch_size`/`--processor_size`/`num_steps`；仍不行则评估引入 X-MGN 式 Halo 分块（Design Doc §十一-2，与本架构正交）。`omega_thresh` 需在 EAGLE 上用 `calibrate_thresholds.py --dataset`（待加）或手动重标。

---

## 4. 结果记录（**待真实 EAGLE 数据后填入**）

| 项 | 命令（见 §3）| 回传内容 |
| --- | --- | --- |
| EAGLE 字段核对 | 打开任一 `.npz` 看 keys/shape | 若与 `EAGLE_KEYS` 不符，回传真实键名 → 本人改适配层 |
| 节点类型核对 | `np.unique(mask)` | 真实 id 集合 → 本人改 `eagle_node_type_to_class` |
| 预处理产出 | §3 步骤 0 | 缓存生成数 + 是否 OOM/报错 |
| 训练收敛 | §3 步骤 1 | NMSE 末值 + checkpoint 名 |
| rollout 评估 | §3 步骤 2/3 | mean RMSE + `14_eval_multicase.*` + GIF |
| reader 单测 | `pytest tests/test_eagle_dataset.py -v` | 通过数（应 2/2）|

---

## 5. 代码改动一览

| 文件 | 改动 | 说明 |
| --- | --- | --- |
| `eagle_dataset.py` | **新建**：`EAGLE_KEYS` 适配层 + `load_eagle_case` + `EagleDatasetAMR` | 复用基类静态方法，图格式一致 |
| `preprocess_partitions.py` | + `--source {tfrecord,eagle}`、`--workers`（`ProcessPoolExecutor` 并行 U6）、`build_cache(source=)` | 并行 + EAGLE 缓存 |
| `data_amr.py` | + `make_amr_dataset(dataset, **kw)` 工厂 | vortex/eagle 切换 |
| `train_amr_m4gn_full.py` / `inference_amr_m4gn.py` / `eval_rollout.py` | + `--dataset {vortex,eagle}` | 走工厂 |
| `tests/test_eagle_dataset.py` | **新建**：合成 npz 验形状/格式 | 2 测试 |

---

## 6. 参数大全（M7 新增）

### 6.1 `preprocess_partitions.py` 新增
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--source` | `tfrecord` | 数据源：`tfrecord`（圆柱绕流）/ `eagle` |
| `--workers` | 1 | 并行进程数（U6）；>1 用多进程，1 串行 |

### 6.2 训练/推理/评估新增
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--dataset` | `vortex` | 数据集：`vortex` / `eagle`（`train_amr_m4gn_full.py`/`inference_amr_m4gn.py`/`eval_rollout.py`）|

### 6.3 `eagle_dataset.py` 适配层（**核对处**）
| 符号 | 含义 |
| --- | --- |
| `EAGLE_KEYS` | `.npz` 键名（已核对：`mesh_pos`/`VX`/`VY`/`PS`/`node_type`；压强默认 `PS` 动压）|
| `EAGLE_TRIANGLE_KEYS` / `EAGLE_TRIANGLE_FILES` | 三角网格解析（npz 内键 → 同名 triangles 文件 → Delaunay 兜底）|
| `eagle_node_type_to_class()` | node_type → `{0,1,2,3}`（默认 0→normal、非0→wall）|
| `stats_prefix`（构造参数）| 统计文件名后缀（默认 `eagle`，避免覆盖圆柱绕流统计）|

---

## 7. M7 退出标准 与 待办

**退出标准（Design Doc §八 M7）**：在 EAGLE 上跑通 + 与 MGN/X-MGN 对比 + 展示 AMR token 节省与长程优势。

**当前状态**：
- ✅ EAGLE reader / 并行预处理 / 数据集切换 / 合成数据单测**全部就绪**。
- 🟡 **未在真实 EAGLE 上验证**：字段名、节点类型 id、动态网格行为、显存（D9）均待目标机核对。
- ⬜ 未做：EAGLE 上的 MGN/X-MGN 基线、token 节省统计、长程优势量化（需数据+算力）。

**下一步（拿到 EAGLE 数据后）**：
1. 核对 `EAGLE_KEYS` 与节点类型 → 必要时改适配层；
2. 跑 §3 步骤 0–3 → 回传 §4；
3. 若 OOM → 触发 D9（减规模 / 评估 Halo 分块）；
4. 补 EAGLE 阈值标定与基线对比。

---

## 8. 结果回传记录区（待真实数据后填入）

**最重要**：先回传一个 EAGLE `.npz` 的 `keys` 与各数组 `shape`、以及 `np.unique(mask)`，本人据此校正 `eagle_dataset.py` 适配层，再推进训练/对比。之后按 §4 表回传，本人原样录入、不改数字。

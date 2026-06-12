# AMR-M4GN 开发进度与操作手册（M4 阶段）

**更新日期**：2026年6月11日
**当前阶段**：M4 — 端到端模型（micro GNN + 段编码 + Macro Transformer + dispatch + 统一 decoder）+ 数据管线 + overfit 单 case
**M4 状态**：🟢 **端到端管线已跑通**。
- 第一批（`micro_gnn`/`macro_transformer`）：**单测 9/9 实跑通过**。
- 第二批（`model`/`data_amr`/`preprocess`/`train`）：**集成测试 3/3 通过、预处理成功、overfit 已实跑**——单 case NMSE **0.92 → 0.013**（降约 70×，趋势向下）。
- **诚实结论**：M4 的核心目标「端到端管线正确 + 可反向学习」**达成**；但 NMSE 后期在 1e-2~6e-2 **震荡、未压到 ≈0**，严格意义的「overfit 到 loss≈0」**只算部分达成**（见 §五步骤 4 评估）。
**前置**：M1/M2/M3 已关闭。
**配套设计文档**：`AMR_M4GN_Design_Doc.md`（§4.3/4.8/4.9、§7.2、§7.4.4~7.4.6、§八 M4）
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`（你的运行机：`C:\GitHub\physicsnemo\...`）

> **里程碑总进度**（截至 2026-06-12）：M1 ✅ · M2 ✅ · M3 ✅ · **M4 🟢 管线跑通** · M5 🟢（AMR>MGN，全量待算力）· M6 ⬜ · M7 ⬜
> **文档索引**：M1 `AMR_M4GN_Progress_M1.md` · M2 `..._M2.md` · M3 `..._M3.md` · M4（本文）· 设计 `AMR_M4GN_Design_Doc.md`

> 本文档面向「拿到代码后怎么跑」：§五 每一步都写清楚 **做什么 → 为什么 → 应该得到什么结果**。下文 `✅` 凡涉及「通过」的，除非注明是你实跑的真实数据，否则视为「待你实跑确认」。

---

## 一、M4 做了什么（文件清单）

```
examples/cfd/vortex_shedding_mgn/
├── amr_m4gn/
│   ├── micro_gnn.py            ✅ 新建：MeshGraphNet wrapper（走到 processor，丢弃 decoder）
│   ├── macro_transformer.py    ✅ 新建：SegmentEncoder + MacroTransformer + dispatch
│   ├── model.py                ✅ 新建：AMRM4GN 顶层模型（串起全链路）
│   └── __init__.py             ✅ 改：导出 MicroGNN/SegmentEncoder/MacroTransformer/dispatch/AMRM4GN
├── preprocess_partitions.py    ✅ 新建：离线几何/分区/PE/area 缓存（与时间步无关）
├── data_amr.py                 ✅ 新建：VortexSheddingDatasetAMR（D5/P1：暴露 pos/gidx + get_cache）
├── train_amr_m4gn.py           ✅ 新建：单 case overfit 脚本（argparse，per-channel NMSE）
├── conf/config_amr_m4gn.yaml   ✅ 新建：参数清单（完整 hydra/DDP 训练留 M5/M7）
└── tests/
    ├── test_micro_gnn.py       ✅ 新建：4 例（已实跑通过）
    ├── test_macro_transformer.py ✅ 新建：5 例（已实跑通过）
    └── test_model.py           ✅ 新建：3 例集成测试（待实跑）
```
> `✅` 仅表示「文件已建/改」，**不代表已测试通过**（除 §五步骤 1 标注「实跑通过」者）。

M4 目标（Design Doc §八 M4）：**完整模型在单个 case 上 overfit 成功**——集成测试无 NaN、全参有梯度；overfit loss 单调降至接近 0、预测≈GT。

---

## 二、🚩 决策门 D4（micro_gnn 接口，U2）：已确认通过 → 路径 (a)

**问题**：`MeshGraphNet` 暴露的子模块属性名是否稳定？

**核对结果**（读 `physicsnemo/models/meshgraphnet/meshgraphnet.py` 源码）：`__init__` 建 `edge_encoder/node_encoder/processor/node_decoder`；`forward = edge_encoder→node_encoder→processor→node_decoder`（line 267–270）。→ 属性稳定，采用**路径 (a)**：`MicroGNN` 走到 `processor` 输出 `h_node[N,hidden]`，**丢弃 `node_decoder`**。

**关键实现**：`MicroGNN.__init__` 先构造临时 `MeshGraphNet`，**只把 `edge_encoder/node_encoder/processor` 注册为自身子模块**（局部 backbone 丢弃），故 `node_decoder` 不计入 `parameters()`——保证「每个参数都收到梯度」（§7.4.6）。

> **状态**：D4 已通过——源码核对 + `test_micro_gnn.py` **实跑 4/4 过**（含 `test_matches_meshgraphnet_processor`）。

## 三、🚩 决策门 D5（数据管线，U1/§7.2-A）：选 P1（已落地）

- **P1（采用）**：`VortexSheddingDatasetAMR(VortexSheddingDataset)` 子类，**所有 split** 都暴露 `graph.pos`、`graph.gidx`，不改原类，零风险。
- **P2（不用）**：前向里反归一化 `edge_attr` 凑 Δx，耦合差。
- **落地**：`data_amr.py`（mesh 静止，子类额外读一次 `mesh_pos`；`get_cache(gidx)` 懒加载分区缓存）。
- **D1（速度反归一化）在 overfit 里间接验证**：`model` 用 `node_stats` 的 `vel_mean/std` 反归一化后再算物理量；overfit 能正常收敛即说明该管线可用。

---

## 四、各模块速览（详细签名见代码 docstring）

| 模块 | 关键内容 |
| --- | --- |
| `micro_gnn.MicroGNN` | `forward(x[N,6], edge_attr[E,3], graph) → h_node[N,hidden]`，走 encoder→processor |
| `macro_transformer.SegmentEncoder` | 每 token mean-pool + `pe_proj([rwse16, depth1, centroid2])`，置换不变 |
| `macro_transformer.MacroTransformer` | Pre-LN 4层8头 d=128 FFN=512；预留 `key_padding_mask`（M5 批处理）|
| `macro_transformer.dispatch` | `h_cat=concat([h_node, h_seg'[token_of(i)]])`→`[N,2d]` |
| `model.AMRM4GN` | `forward(graph, cache, thresholds=None)`：micro→反归一化算 phys→route→per-token PE 组装→seg_enc→macro→dispatch→decoder→`[N,3]` |
| `preprocess_partitions` | 每 case 离线缓存：`levels/seg_adj/rwse{L0,L1}/node_pe/l1_to_l0/centroid{L0,L1}/area/pos/meta` |
| `data_amr` | P1 子类；`get_cache(gidx)` 取 `partition_cache_{split}_{gidx}.pt` |
| `train_amr_m4gn` | overfit：argparse + 固定 ω 阈值 + per-channel NMSE + 在线 `build_cache`（无 DDP/AMP/wandb）|

**per-token PE 组装**（`model._assemble_token_pe`）：细 token（depth=1）取其 **L1 段**的 rwse/质心，粗 token（depth=0）取 **L0 父段**的（用每 token 代表节点反查 L1/L0 标签，再 `torch.where`）。

---

## 五、运行步骤（每步：做什么 → 为什么 → 应该得到什么结果）

> 前提：M1/M2/M3 已跑通；`gnn` 环境（有 torch + physicsnemo + torch_geometric）。

### 步骤 0 — 同步本轮文件到运行机（`C:\GitHub`）

- **做什么**：确认下列文件都是 M4 最新版（注意三个脚本在 `vortex_shedding_mgn/` 根目录，不在 `amr_m4gn/`）：

  | 文件 | 操作 |
  | --- | --- |
  | `amr_m4gn/micro_gnn.py` | 新增 |
  | `amr_m4gn/macro_transformer.py` | 新增（已含 `enable_nested_tensor=False` 修复）|
  | `amr_m4gn/model.py` | 新增 |
  | `amr_m4gn/__init__.py` | 修改（导出 5 个新符号）|
  | `preprocess_partitions.py` | 新增（根目录）|
  | `data_amr.py` | 新增（根目录）|
  | `train_amr_m4gn.py` | 新增（根目录）|
  | `conf/config_amr_m4gn.yaml` | 新增 |
  | `tests/test_micro_gnn.py` / `test_macro_transformer.py` / `test_model.py` | 新增 |

- **为什么**：之前多次因文件没同步导致 import 报错；先核对再跑。
- **应该得到什么**：以上文件齐全且为最新。

### 步骤 1 — 跑模型组件单测（micro / macro）

- **做什么**：
  ```bash
  cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
  pytest tests/test_micro_gnn.py tests/test_macro_transformer.py -v
  ```
- **为什么**：先在合成数据上验证两个核心组件正确（形状、丢 decoder、与 MGN 一致、置换不变、padding mask、dispatch），不依赖真实数据。
- **应该得到什么**：**9 passed**（4+5）。
- **状态**：✅ **你已实跑：9 passed in 20.35s**。
  > 顺带修了一条本代码告警（`TransformerEncoder` 在 Pre-LN 下 `enable_nested_tensor` 失效），已置 `False`。其余告警（`torch.jit`/`torch_geometric.distributed`/physicsnemo `SyntaxWarning`）来自第三方，非本项目。

### 步骤 2 — 跑顶层模型集成测试（`test_model.py`）

- **做什么**：
  ```bash
  pytest tests/test_model.py -v
  ```
- **为什么**：用**合成图 + 合成缓存**（不依赖数据集/预处理）验证 `AMRM4GN` 整条链路串得对——前向出 `[N,3]` 无 NaN、反向**每个参数都有有限梯度**、全折回(T=K0)与全细(T=K1)两种极端都能跑（§7.4.6 集成测试）。
- **应该得到什么**：**3 passed**。
- **状态**：✅ **你已实跑：3 passed in 4.69s**。

### 步骤 3 — 离线预处理（生成某 case 的分区缓存）

- **做什么**：
  ```bash
  python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test --num_cases 1 --out_dir ./amr_cache
  ```
- **为什么**：把「模态分解→分区→RWSE→l1_to_l0→质心→面积」这套**只跟几何有关、与时间步无关**的重计算离线缓存一次。（overfit 脚本也能在线现算，此步是为后续多 case/训练准备，可选。）
- **应该得到什么**：生成 `./amr_cache/partition_cache_test_0.pt`，终端打印 `saved ... (K0=64, K1=256)`。
- **状态**：✅ **你已实跑：生成成功（K0=64, K1=256）**。

### 步骤 4 — overfit 单 case（M4 的核心验收）

- **做什么**（**用 `train` split**，见下「为什么」）：
  ```bash
  python train_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split train --case_idx 0 --num_steps 50 --epochs 300 --omega_thresh 30
  ```
  （`--split train` 是默认值，可省略。CPU 可跑，有 GPU 更快。）
- **为什么**：
  - M4 退出标准就是「单 case overfit 成功」——这是端到端管线（数据→反归一化→物理量→路由→段编码→Transformer→dispatch→decoder→NMSE→反向）**全链路正确且可学习**的最强证据。
  - **为什么是 `train` split 而不是 `test`**：`VortexSheddingDataset` 对 **非 train** split 会去 cwd 读 `edge_stats.json`/`node_stats.json`（这俩由 baseline 训练生成），你没跑过 baseline 所以会报 `FileNotFoundError: edge_stats.json`（你上次就是这个错）。而 **`train` split 会自算这两份 stats 并落盘**，无需先跑 baseline。脚本默认 `--noise_std 0` 关掉训练噪声，保证干净 overfit。
- **应该得到什么**：
  - 终端先打印 `Preparing the train dataset...`，并在 cwd 生成 `edge_stats.json`/`node_stats.json`（train split 的副产物，正常现象）；
  - 然后每 10 epoch 打印一行 `epoch xxxx  NMSE x.xxxe±xx`，NMSE 随 epoch 大体下降。
- **实测结果（你实跑，2026-06-11，device=cuda，49 步×300 epoch）**：

  | epoch | 0 | 40 | 90 | 160 | 250 | 299 |
  | --- | --- | --- | --- | --- | --- | --- |
  | NMSE | 0.915 | 0.133 | 0.038 | 0.022 | **0.011** | 0.013 |

  - **下降约 70×、趋势明确向下** → 端到端管线（数据→反归一化→物理量→路由→段编码→Transformer→dispatch→decoder→NMSE→反向）**正确且可学习**，M4 集成核心目标达成。
  - **但后期在 1e-2~6e-2 震荡、没收敛到 ≈0**：这是因为本脚本其实是对 **49 个时间帧**一起拟合（不是单一样本），且 `lr=1e-3` 偏大、epoch 偏少——属调参问题，**不影响「管线正确」的结论**，但严格的「overfit 到 loss≈0」**未完全达到**。
  - **若想压到更低**（可选，不阻塞 M4）：减帧（`--num_steps 2`，逼近真·单样本）、降 lr（`--lr 3e-4`）、增 epoch（`--epochs 1000`）。
- **状态**：🟢 **已实跑：管线可学（NMSE 0.92→0.013）**；严格 ≈0 收敛留调参/训练期。

> **仓库副产物**（运行后你机器上会多出）：cwd 的 `edge_stats.json`、`node_stats.json`（train split 自算 stats）；`./amr_cache/partition_cache_test_0.pt`（步骤 3 预处理）。这些是正常产物，可保留。

> 跑完把 ① `pytest tests/test_model.py` 输出 ② overfit 的 NMSE 那几行发我，我据实回填验收并最终确认 D1。

### 验收清单（M4 退出标准，Design Doc §八 M4）

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| A. 组件单测 | `pytest test_micro_gnn.py test_macro_transformer.py` | 9 passed | ✅ 已实跑通过 |
| B. 集成测试 | `pytest test_model.py` | 3 passed（无 NaN / 全参梯度 / 全折全细）| ✅ 已实跑通过（3 passed）|
| C. 预处理 | `preprocess_partitions.py` | 生成 `partition_cache_test_0.pt`（K0=64,K1=256）| ✅ 已实跑通过 |
| D. overfit | `train_amr_m4gn.py --split train` | NMSE 单调降至接近 0 | 🟢 **部分达成**：NMSE 0.92→0.013（降 70×、可学），但后期震荡、未到 ≈0（调参问题，见步骤 4）|
| D1 最终确认 | （随 D 一起）| 反归一化管线在真实训练里可用 | ✅ 管线可用：overfit 能正常下降，说明 `vel_mean/std` 反归一化→物理量→路由整条在真实数据上不崩、可学 |

---

## 六、跑不通时怎么排查（已知风险点）

| 现象 | 先查 | 说明 |
| --- | --- | --- |
| `FileNotFoundError: edge_stats.json`（或 node_stats）| `--split` | 非 train split 要读 baseline 生成的 stats json；用 `--split train`（自算 stats、默认值）即可，**已是默认** |
| `ImportError`/`cannot import name` | 步骤 0 文件是否同步齐全 | `__init__.py` 没同步是历史高频坑 |
| `RuntimeError: ... on the same device` | 已修 | `physics_ops.lstsq_gradient` 的 `A/B/eye` 之前没带 device（M2 只在 CPU 测过），GPU 上与输入不同 device → 报错。已让其跟随 `field.device`，并把 `edge_index` 对齐到同 device。CPU 路径数值完全等价（M2 的 8/8 不受影响）|
| overfit NMSE 不降/震荡 | `--omega_thresh`、`--lr` | 阈值让 routing 几乎全细/全粗时学习信号弱；先试 `--omega_thresh 30`、`--lr 1e-3` |
| NMSE 数值异常大 | NMSE 分母 | target 速度增量量级小，已加 `eps=1e-8`；仍异常则打印 `graph.y` 量级 |
| `loss` 变 NaN | 物理量/反归一化 | 查 `vel_mean/std` 是否正确加载、`compute_ns_quantities` 是否有除零 |

---

## 七、与 Design Doc 的对应关系

| Design Doc | M4 落地 |
| --- | --- |
| §7.4.4 micro_gnn（旁路 decoder）+ 单测 | `micro_gnn.py` + `test_micro_gnn.py`（9 例的一部分，已过）|
| §7.4.5 macro_transformer（seg enc/transformer/dispatch）+ 单测 | `macro_transformer.py` + `test_macro_transformer.py`（已过）|
| §7.4.6 model 顶层 + 集成测试 | `model.py` + `test_model.py`（待跑）|
| §7.5 预处理与缓存格式 | `preprocess_partitions.py` |
| §7.6 数据/训练入口（P1 子类、NMSE）| `data_amr.py` + `train_amr_m4gn.py` + `conf/config_amr_m4gn.yaml` |
| 决策门 D4 / D5 | §二 / §三（已定）；D1 随 overfit 最终确认 |

**M4 未覆盖（属后续）**：
- 批内段偏移 + padding（`graph.batch`、`key_padding_mask`）= M5。
- 完整 DDP/AMP/wandb/checkpoint 训练、全量数据 = M5/M7。
- 阈值采样区间 vs 物理量级的标定（D3 最终）= 训练期。

---

## 八、M5 注意事项（下一步）

0. **（可选先做）把 overfit 压到更低**：当前 NMSE 收敛在 1e-2 且震荡。M5 调参时验证 `--num_steps 2 --lr 3e-4 --epochs 1000` 能否压到 1e-3 以下，确认是纯调参问题而非管线缺陷。
1. **批处理**：`route`/`SegmentEncoder`/`dispatch` 按 `graph.batch` 做 token 偏移，`MacroTransformer` 用 `key_padding_mask`（变长 batch）。
2. **完整训练入口**：把 overfit 脚本扩成 hydra+DDP+AMP+checkpoint（复用 `train.py` 框架），多 case 训练。
3. **阈值采样**：训练期用 `sample_thresholds(training=True)`（绝对阈值），收集全训练集 T 分布 → 最终拍板 D3（K0/K1 与区间）。
4. **warm-up（U5）**：前若干 epoch 关 AMR（全 L1）的时长由训练曲线定。

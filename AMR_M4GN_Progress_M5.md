# AMR-M4GN 开发进度与操作手册（M5 阶段）

**更新日期**：2026年6月11日
**当前阶段**：M5 — 全量训练 + baseline 对比（批处理 `batch_size>1` → 完整训练入口 → `inference_amr_m4gn` rollout/指标/可视化 → D3 阈值标定）
**M5 状态**：🚧 **进行中**。M5 较重、分小步推进，**每小步都配可单独验证的单测 + 文档**。
- 小步 1（批处理注意力隔离）：**单测 3/3 实跑通过**（含跨图不泄漏）。
- 小步 2（model 批处理集成）：**单测 2/2 + 单图回归 3/3 实跑通过**（batch==逐图拼接）。
- 小步 3（完整训练入口）：**多 case batch 训练实跑通过**——NMSE 3.06→~0.08（降约 35×），checkpoint 已存。
- 小步 4（inference + 预测可视化）：**实跑通过**——预测场与真值场形态高度吻合（`09_prediction_case0_t25.png`），端到端「训练→推理→可视化」闭环打通。
- 小步 4b（rollout + 误差曲线）：**代码已实现，待你实跑**（自回归多步 + 速度 RMSE-vs-step 曲线）。
**前置**：M1/M2/M3 ✅、M4 🟢（端到端管线跑通，overfit NMSE 0.92→0.013）。
**配套设计文档**：`AMR_M4GN_Design_Doc.md`（§7.2-C 批处理 / §八 M5）
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`

> **里程碑总进度**（截至 2026-06-11）：M1 ✅ · M2 ✅ · M3 ✅ · M4 🟢 · **M5 🚧** · M6 ⬜ · M7 ⬜
> **文档索引**：M1〜M4 见各 `AMR_M4GN_Progress_M*.md`；设计 `AMR_M4GN_Design_Doc.md`
> 本文同样遵循「拿到代码后每步：做什么 → 为什么 → 应该得到什么结果」。

---

## M5 总体计划（分小步）

| 小步 | 内容 | 状态 |
| --- | --- | --- |
| **1. 批处理注意力隔离** | `pack_segments`/`run_macro_batched`：变长 token 打包成 `[B,Tmax,d]`+mask，Transformer 注意力按图隔离 | ✅ 单测 3/3 通过 |
| 2. model 批处理集成 | `model.forward` 支持 PyG batch：逐图 route → 全局偏移 token id + `token_batch` → 批处理 transformer | ✅ 单测 2/2 + 回归 3/3 |
| 3. 完整训练入口 | 把 overfit 脚本扩成正经训练（多 case、checkpoint、lr 调度），复用 `train.py` 框架 | ✅ 实跑通过（NMSE 3.06→~0.08）|
| 4. `inference_amr_m4gn.py` | 预测场可视化（pred vs GT + 误差）+ 指标；rollout 后续 | ✅ 实跑通过（预测≈真值）|
| 4b. rollout + 误差曲线 | 自回归多步 rollout（边界 mask）+ 速度 RMSE-vs-step 曲线 | ✅ 实现，待实跑 |
| 5. D3 最终标定 | 训练期绝对阈值采样，统计全训练集 T 分布，定 K0/K1 与区间 | ⬜ 待做 |

---

## 小步 1 — 批处理注意力隔离（本轮）

### 做了什么

```
amr_m4gn/macro_transformer.py   ✅ 改：加 pack_segments / unpack_segments / run_macro_batched
amr_m4gn/__init__.py            ✅ 改：导出上述 3 个函数
tests/test_batched_macro.py     ✅ 新建：3 个单测（pack/unpack 往返、mask 正确、跨图不泄漏）
```

### 为什么这么设计

- **`SegmentEncoder`/`dispatch` 不用改**：它们只认「全局连续 token id `kept_assign∈[0,T)`」+ `T`；只要把一个 batch 内各图的 token 统一编号（全局偏移），mean-pool（`index_add_`）和 dispatch（`index_select`）的逻辑与单图完全一致。
- **唯一必须 batch-aware 的是 Transformer 的注意力**：不同算例的 token **绝不能互相 attend**（否则 A 算例的涡街会"看到" B 算例，物理上错误、且破坏变长语义）。做法：把各图变长 token 序列 **pad 成 `[B, Tmax, d]`**，用 `key_padding_mask`（pad 位=True）让注意力只在**同图非 pad token**间进行；算完再**解包**回 `[T_total, d]`。
- **前提**：`token_batch` 是**分段连续**的（图 0 的 token 在前、图 1 紧随……），这正是 `model.py`（小步 2）拼接逐图路由结果的方式。

### 接口

| 函数 | 签名 | 作用 |
| --- | --- | --- |
| `pack_segments` | `(h_seg[T_total,d], token_batch[T_total], num_graphs=None) → (packed[B,Tmax,d], mask[B,Tmax], index)` | 变长 token 打包 + padding mask（True=pad）|
| `unpack_segments` | `(packed[B,Tmax,d], index) → [T_total,d]` | 解包回扁平 token |
| `run_macro_batched` | `(macro, h_seg[T_total,d], token_batch, num_graphs=None) → [T_total,d]` | 打包→按图隔离注意力→解包 |

### 运行步骤（你拿到代码后）

#### 步骤 0 — 同步文件
`amr_m4gn/macro_transformer.py`、`amr_m4gn/__init__.py`、`tests/test_batched_macro.py`。

#### 步骤 1 — 跑批处理单测
- **做什么**：
  ```bash
  cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
  pytest tests/test_batched_macro.py -v
  ```
- **为什么**：验证「打包/解包无损」「mask 标对了 padding」，尤其 **`test_per_graph_attention_isolation`**——批处理跑出的结果必须**逐字节等于**把每个图单独跑 transformer 再拼接，这是「跨图不泄漏」的硬证据。
- **应该得到什么**：**3 passed**：
  - `test_pack_unpack_roundtrip`：`unpack(pack(h))==h`，且 `packed.shape==(B,Tmax,d)`；
  - `test_padding_mask_correct`：每行非 pad 数 == 各图 token 数；
  - `test_per_graph_attention_isolation`：批处理结果 == 逐图单跑拼接（`allclose`）。
- **状态**：⏳ **待你实跑**（开发机无 torch，我未跑 pytest）。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| 批处理打包/隔离 | `pytest tests/test_batched_macro.py` | 3 passed（含跨图不泄漏）| ✅ **3 passed in 4.80s** |

> 小步 1 闭环。进入小步 2（`model.forward` 批处理集成）。

---

## 小步 2 — `model.forward` 批处理集成（本轮）

### 做了什么

```
amr_m4gn/model.py          ✅ 改：forward 支持 PyG batch（cache 传 list[dict]）
tests/test_model_batch.py  ✅ 新建：2 例（batch 前向/反向；batch==逐图拼接）
```

### 为什么这么设计（数据流）

一个 batch 是 `B` 个图用 `Batch.from_data_list` 拼成的大图（节点全局偏移、`graph.ptr` 标各图节点范围、边**不跨图**）。`forward(graph, caches, ...)` 里：
1. **micro 一次跑整个 batch**：`h_node[N_total,d]`（MeshGraphNet 消息传递不跨图，等价逐图）。
2. **物理量一次全局算**：把各图 cache 的 `pos`/`area` 拼接，配 batched `edge_index` 一次 `compute_ns_quantities`（边不跨图 → 等价逐图）。
3. **路由逐图做**：对每个图 `b`，用 `ptr` 切出它的 `phys_b`，调已验证的 `route(levels_b, phys_b, thr)`，得局部 `kept_assign_b∈[0,T_b)`；**token id 全局偏移** `+= tok_off`，并记 `token_batch=b`。
4. **段编码用全局 id**：`SegmentEncoder` 一次 mean-pool（全局连续 id，等价逐图）。
5. **Transformer 按图隔离**：`run_macro_batched`（小步 1）→ 各图 token 互不 attend。
6. **dispatch + decoder** 全局出 `[N_total,3]`。

> **单图路径保持与 M4 完全等价**：`cache` 传单 dict 时 `B=1`，走 `self.macro(h_seg)`（与 M4 同一行），`test_model.py` 不受影响。

### 运行步骤（你拿到代码后）

#### 步骤 0 — 同步文件
`amr_m4gn/model.py`、`tests/test_model_batch.py`。

#### 步骤 1 — 跑 batch 集成单测
- **做什么**：
  ```bash
  pytest tests/test_model_batch.py -v
  # 顺带回归单图：pytest tests/test_model.py -v
  ```
- **为什么**：验证 batch 路径正确，且**批处理结果逐字节等于把每个图单独前向再拼接**——这同时证明「跨图不泄漏」和「单图/批处理一致」。
- **应该得到什么**：`test_model_batch.py` **2 passed**：
  - `test_batch_forward_finite_and_grad`：`pred[N_total,3]` 全有限、反向全参有梯度；
  - `test_batch_equals_per_graph`：`batch 前向 == [图0前向; 图1前向]`（`allclose`）。
  - （回归）`test_model.py` 仍 **3 passed**（单图路径未变）。
- **状态**：⏳ **待你实跑**。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| batch 集成 | `pytest tests/test_model_batch.py` | 2 passed（含 batch==逐图拼接）| ✅ **2 passed in 5.02s** |
| 单图回归 | `pytest tests/test_model.py` | 仍 3 passed | ✅ **3 passed in 4.66s** |

> 小步 2 闭环（batch==逐图拼接、单图未回归）。进入小步 3（完整训练入口）。

---

## 小步 3 — 完整训练入口（本轮）

### 做了什么

```
train_amr_m4gn_full.py   ✅ 新建：多 case + PyG batch + checkpoint + lr 调度 + per-channel NMSE
（train_amr_m4gn.py 保留不动：M4 的单 case overfit 验证脚本）
```

### 为什么这么设计

- **新建而非改 overfit 脚本**：`train_amr_m4gn.py`（overfit）已验证通过，不动它；完整训练逻辑放新脚本 `train_amr_m4gn_full.py`，职责清晰。
- **单机单卡、不上 DDP/apex/wandb**：降低你跑通的门槛；模型/数据各部件本身是 DDP-ready 的，真正多卡时再包一层（M7/算力到位）。
- **batch 取 cache 的关键**：DataLoader 把 B 个图拼成一个 PyG Batch（带 `ptr` 与每图 `gidx`）；训练循环按 `batch.gidx` 取出**对应的 caches 列表**喂给 `AMRM4GN.forward(batch, caches)`（小步 2）。caches 预先全部搬到 device。
- **阈值**：默认用**固定 ω 阈值**（`--omega_thresh 30`，与 overfit/可视化一致，保证 AMR 真的在合并、可控）；`--sample_thresh` 可切到 Design Doc §4.7 的训练期随机采样（D3 标定时用，留小步 5）。

### 运行步骤（你拿到代码后）

#### 步骤 0 — 同步文件
`train_amr_m4gn_full.py`（其余 M5 文件此前已同步）。

#### 步骤 1 — 预处理若干 train case（生成缓存）
- **做什么**：
  ```bash
  python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split train --num_cases 4 --out_dir ./amr_cache
  ```
- **为什么**：完整训练按 `gidx` 读 `partition_cache_train_{gidx}.pt`；必须先离线生成（几何缓存，与时间步无关）。
- **应该得到什么**：`./amr_cache/` 下出现 `partition_cache_train_0.pt`〜`_3.pt`，每个打印 `(K0=64, K1=256)`。

#### 步骤 2 — 跑多 case batch 训练（smoke run）
- **做什么**：
  ```bash
  python train_amr_m4gn_full.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --batch_size 2 --epochs 50 --omega_thresh 30
  ```
- **为什么**：验证「多个算例 + batch_size>1」这条真实训练链路能跑通、loss 在下降、能存 checkpoint——这是 M5 从「单 case overfit」迈向「正经训练」的关键一跳。
- **应该得到什么**：
  - 终端打印 `[train] 4 cases x 49 steps, batch_size=2, ...`；
  - 每 5 epoch 打印 `epoch xxxx  NMSE x.xxxe±xx  lr x.xxe-xx`，**NMSE 总体随 epoch 下降**（多 case 比单 case 难，不会像 overfit 那么低，但应明显下行）；
  - `./checkpoints_amr/` 下生成 `amr_m4gn_epoch*.pt`。
- **状态**：⏳ **待你实跑**。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| 多 case 预处理 | `preprocess_partitions.py --split train --num_cases 4` | 生成 4 个 `partition_cache_train_*.pt` | ✅ **已实跑：train 0~3 全部生成（K0=64,K1=256）** |
| batch 训练跑通 | `train_amr_m4gn_full.py ... --batch_size 2` | 无报错、NMSE 总体下降、存 checkpoint | ✅ **实跑：NMSE 3.06→~0.08，checkpoint 已存** |

> 小步 3 闭环。实测训练日志：epoch0 NMSE 3.06 → epoch49 ~0.08（中段最低 ~0.07），总体下降约 35×（多 case 比单 case overfit 难，后期在 7e-2~1e-1 震荡属正常）。进入小步 4（`inference_amr_m4gn.py`：预测场可视化）。

---

## 小步 4 — `inference_amr_m4gn.py`：预测场可视化（本轮）

### 做了什么

```
inference_amr_m4gn.py   ✅ 新建：加载 checkpoint → 单步预测 → 反归一化 → 出 pred vs GT vs |err| 9 宫格图 + 指标
```

### 为什么这么设计

- **直接回答「能不能看到结果」**：M4 的 overfit 只给 loss 数字；这里把模型**预测的 (Δu, Δv, p) 场**反归一化回物理量，与**真值场**并排画在网格上（3 行 = du/dv/p，3 列 = 预测/真值/误差），一眼看出学得准不准。
- **单步预测先行，rollout 后续**：先做「给定真值输入预测下一帧」的单步可视化（最直接），多步 rollout（用预测反馈做长序列）+ §6.4 全套指标 + baseline 对比留作小步 4 的后续/M5 收尾。
- **指标双口径**：打印归一化空间 per-channel NMSE（与训练 loss 同尺度）+ 物理空间 RMSE（有量纲、可解释）。

### 运行步骤（你拿到代码后）

#### 步骤 0 — 同步文件
`inference_amr_m4gn.py`。

#### 步骤 1 — 用训练好的 checkpoint 出预测图
- **做什么**（用小步 3 存的 checkpoint）：
  ```bash
    python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch49.pt --case_idx 0 --timestep 25 --num_steps 50 --omega_thresh 30
  ```
- **为什么**：把训练结果**可视化验证**——这是你之前想看的「预测 vs 真值」图。
- **应该得到什么**：
  - 终端打印 `per-channel NMSE (norm space)` 与 `per-channel RMSE (physical)`；
  - `./inference_vis/09_prediction_case0_t25.png`：3×3 面板，**预测列与真值列形态应接近**（尾迹/壁面结构对得上），误差列整体偏小、主要残留在涡街等剧烈变化处。
  - 注意：这是 4 case、50 epoch 的 smoke 模型（NMSE~0.08），预测会**大致像**但不会完美；要更准需更多 case/epoch。
- **状态**：✅ **实跑通过**。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| 预测可视化 | `inference_amr_m4gn.py ...` | 出 `09_prediction_*.png`，预测列≈真值列；打印 NMSE/RMSE | ✅ 实跑通过 |

**实测结果（case0, t=25）**：
- per-channel NMSE（归一化空间）du/dv/p = **0.317 / 0.453 / 0.143**；
- per-channel RMSE（物理）du/dv/p = **1.0e-3 / 5.9e-4 / 1.7e-2**；
- `09_prediction_case0_t25.png`：**预测列与真值列形态高度吻合**——du/dv 的尾迹涡街红蓝结构、p 的圆柱前高压/后低压梯度都对得上；**误差集中在圆柱后方涡街/驻点等剧烈变化区**，大片来流误差≈0（物理上预期的难点分布）。
- **诚实说明**：这是 4 case×50 epoch 的 **smoke 模型**，形态已对、量级合理；单帧 NMSE（尤其 dv，量级小→分母小→NMSE 偏高）高于训练 batch 平均（~0.08），属正常，**非最终性能**。结论：端到端「训练→推理→可视化」**闭环成立、预测场形态正确**。

> 小步 4 闭环。M5 收尾还剩：小步 4b（多步 rollout + §6.4 全指标 + 与 MGN/X-MGN baseline 对比）、小步 5（D3 阈值标定）。

---

## 小步 4b — 自回归 rollout + 误差曲线（本轮）

### 做了什么

```
inference_amr_m4gn.py   ✅ 改：加 --rollout K 模式 + run_rollout（多步自回归 + 速度 RMSE-vs-step 曲线）
```

### 为什么这么设计

- **rollout 才是代理模型的真实用法**：单步预测是「给真值输入猜下一帧」；rollout 是「只给初始帧，之后用自己的预测一路往下推 K 步」，会暴露**误差累积/稳定性**——这是 §6.4 评估的核心。
- **边界条件用 GT（baseline 同款）**：只对 `rollout_mask`（内部节点）施加预测增量，**边界节点（inflow/wall/outflow）速度保持真值**，否则边界乱掉会让整场迅速发散。`rollout_mask` 只在 **test split** 提供，故 rollout 用 `--split test`。
- **物理空间积分**：每步把归一化预测增量反归一化为物理 `Δv`，`v_{t+1}=v_t+Δv`，再归一化喂下一步（与 `inference.py` 一致）。

### 运行步骤（你拿到代码后）

#### 步骤 0 — 同步文件 + 准备 test 缓存
- `inference_amr_m4gn.py`（已更新）。
- 确认有 test case0 的缓存（若没有先跑）：`python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test --num_cases 1 --out_dir ./amr_cache`

#### 步骤 1 — 跑 rollout
- **做什么**：`python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch49.pt --split test --case_idx 0 --num_steps 60 --rollout 50 --omega_thresh 30`
- **为什么**：验证 rollout 链路（自回归 + 边界 mask + 积分）能跑，并画出误差随步数的累积曲线。
- **应该得到什么**：
  - 终端打印 `rollout velocity RMSE per step (first/mid/last)`；
  - `./inference_vis/10_rollout_rmse_case0.png`：RMSE-vs-step 曲线。
  - **诚实预期**：当前是 **4 case×50 epoch 的 smoke 模型**，且 rollout 在**没训过的 test case** 上做泛化，**RMSE 大概率随步数明显上升甚至发散**——这正常，rollout 稳定需要充分训练 + 训练期加 noise（baseline 也靠这个）。本步先验证**机制正确**，稳定性是放大训练后的事。
- **状态**：⏳ **待你实跑**。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| rollout 跑通 | `inference_amr_m4gn.py ... --rollout 50` | 出 `10_rollout_rmse_case0.png` + 打印每步 RMSE，无报错 | ⏳ 待实跑 |

> 跑完把曲线 + RMSE 发我。注意：smoke 模型 rollout 发散是预期，重点是机制跑通。M5 收尾还剩小步 5（D3 阈值标定）+ 充分训练后的 baseline 对比。

---

## 与 Design Doc 的对应关系

| Design Doc | M5 落地 |
| --- | --- |
| §7.2-C 批内段偏移 + padding mask | 小步 1：`pack_segments`/`run_macro_batched`（本轮）；小步 2：`model` 集成 |
| §7.6 完整训练/推理入口 | 小步 3 `train_amr_m4gn_full.py`（✅）、小步 4 `inference_amr_m4gn.py`（✅ 待实跑）|
| §八 M5 退出标准（vs baseline 指标）| 小步 4b（rollout/误差曲线，✅ 待实跑）+ 充分训练后 baseline 对比（待做）|
| 决策门 D3 最终标定 | 小步 5（待做）|

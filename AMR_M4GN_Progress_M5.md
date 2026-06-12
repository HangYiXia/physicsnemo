# AMR-M4GN 开发进度与操作手册（M5 阶段）

**更新日期**：2026年6月11日
**当前阶段**：M5 — 全量训练 + baseline 对比（批处理 `batch_size>1` → 完整训练入口 → `inference_amr_m4gn` rollout/指标/可视化 → D3 阈值标定）
**M5 状态**：🟢 **小验证档全部实跑完成**（6 小步均闭环；全量/多 baseline 待算力）。M5 较重、分小步推进，**每小步都配可单独验证的单测 + 文档**。
- 小步 1（批处理注意力隔离）：**单测 3/3 实跑通过**（含跨图不泄漏）。
- 小步 2（model 批处理集成）：**单测 2/2 + 单图回归 3/3 实跑通过**（batch==逐图拼接）。
- 小步 3（完整训练入口）：**多 case batch 训练实跑通过**——NMSE 3.06→~0.08（降约 35×），checkpoint 已存。
- 小步 4（inference + 预测可视化）：**实跑通过**——预测场与真值场形态高度吻合（`09_prediction_case0_t25.png`），端到端「训练→推理→可视化」闭环打通。
- 小步 4b（rollout + 误差曲线）：**实跑通过**——50 步自回归 RMSE 0.107→~0.35 后**平台饱和、不发散**，rollout 机制正确。
- 小步 5（D3 阈值标定）：**实跑通过，D3 已拍板**——推荐绝对 ω 阈值≈8.9（T≈164），训练期采样区间 [2.83, 25.8]，K0=64/K1=256 确认合适。
- 小步 6（放大训练 + baseline 对比）：**全部实跑完成**——同预算下 **AMR-M4GN rollout RMSE 全程低于 MGN，长程优势明显**（见 `12_compare_rollout_case0.png`）。
**前置**：M1/M2/M3 ✅、M4 🟢（端到端管线跑通，overfit NMSE 0.92→0.013）。
**配套设计文档**：`AMR_M4GN_Design_Doc.md`（§7.2-C 批处理 / §八 M5）
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`

> **里程碑总进度**（截至 2026-06-12）：M1 ✅ · M2 ✅ · M3 ✅ · M4 🟢 · **M5 🟢 小验证档通过（AMR>MGN，全量待算力）** · M6 ⬜ · M7 ⬜
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
| 4b. rollout + 误差曲线 | 自回归多步 rollout（边界 mask）+ 速度 RMSE-vs-step 曲线 | ✅ 实跑通过（0.107→~0.35 饱和，不发散）|
| 5. D3 最终标定 | 训练期绝对阈值采样，统计全训练集 T 分布，定 K0/K1 与区间 | ✅ 实跑通过，D3 拍板（ω 阈值≈8.9，区间 [2.83,25.8]）|
| 6. 放大训练 + baseline 对比 | MGN baseline（同预算）+ rollout 对比脚本；小验证档 20case×100步×200ep | ✅ **实跑完成**（20case×100步×200ep；AMR rollout 全程优于 MGN）|

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
  - **诚实预期**：当前是 **4 case×50 epoch 的 smoke 模型**，且 rollout 在**没训过的 test case** 上做泛化，RMSE 会随步数上升。
- **实测结果（case0 test，rollout 50 步）**：每步速度 RMSE first/mid/last = **0.107 / 0.347 / 0.359**；`10_rollout_rmse_case0.png` 显示 **RMSE 前 ~20 步单调上升、之后平台饱和在 ~0.35，不发散**。
  - **解读**：rollout 机制正确（自回归 + 边界 mask + 积分均生效）；误差**有界饱和**（边界条件锚定 + 涡街周期性）说明模型没崩——这比"发散"更好。RMSE 0.35（相对来流 |U|~1.5 约 23%）是 smoke 模型水平，充分训练（更多 case/epoch + 训练期加 noise）应明显降低。
- **状态**：✅ **实跑通过**。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| rollout 跑通 | `inference_amr_m4gn.py ... --rollout 50` | 出 `10_rollout_rmse_case0.png` + 打印每步 RMSE，无报错 | ✅ 实跑通过（误差有界饱和）|

> 小步 4b 闭环。M5 收尾还剩小步 5（D3 阈值标定）+ 充分训练后的 baseline 对比。

---

## 小步 5 — D3 阈值标定（本轮）

### 做了什么

```
calibrate_thresholds.py   ✅ 新建：统计 L1 per-seg |ω| 分布 + 扫描绝对 ω 阈值看 T 分布，给 D3 建议
```

### 为什么这么设计

- **D3 的核心矛盾**：M3 已发现原文阈值区间（`omega:[0.2,4.0]`）是**归一化尺度**，本数据物理 `|ω|~1e2`，直接套用会让 T 恒等于 K1（全不合并）。要定**适合本数据的绝对阈值**，必须先看真实的 per-segment |ω| 分布。
- **纯物理统计、不需 checkpoint**：物理量只依赖速度场 + 几何，所以标定与模型训练无关，可独立、快速地跑。
- **方法**：多 case×多帧 → 每帧在 L1 上聚合每段 `max|ω|` → ① 汇总分布给"物理量级"；② 对一组候选绝对 ω 阈值，统计 route 产生的 T（mean/min/max）→ 画 `T-vs-阈值` 曲线，**挑使 T 落在 [K0,K1] 中段的阈值**作为 D3 建议，并据 p40~p85 给训练期采样区间。

### 运行步骤（你拿到代码后）

#### 步骤 0 — 同步文件
`calibrate_thresholds.py`（依赖已有的 train caches，无需 checkpoint）。

#### 步骤 1 — 跑标定
- **做什么**：`python calibrate_thresholds.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --stride 5`
- **为什么**：用真实数据定 D3——回答「绝对 ω 阈值取多少能让 token 数 T 落在合理区间」「训练期 ω 阈值该在什么范围采样」。
- **应该得到什么**：终端打印 |ω| 分位 + 推荐阈值 + 采样区间；`11_threshold_calibration.png`（左 |ω| 分布、右 T-vs-阈值）。
- **实测结果（4 case×t≤49，stride 5）**：
  - per-seg |ω| 分位：p10=5.2e-3, p30=0.64, **p50=7.6, p70=16.3, p90=33.0, p95=49.1, p99=107.6**；
  - T-vs-阈值：阈值 0→T≈240、**≈8.9→T≈164**、50→T≈77，覆盖 [77,240] 落在 [K0,K1] 内；min-max 带显示 T 随帧（物理状态）波动。
- **状态**：✅ **实跑通过**。

### 🚩 D3 结论（已拍板，据真实标定）

| 项 | 结论 | 依据 |
| --- | --- | --- |
| **K0 / K1** | **保持 64 / 256** | T 在阈值扫描下覆盖 [77,240]，落在 [64,256] 内，区间用满、不溢出 |
| **绝对 ω 阈值（测试/演示）** | **≈ 8.9** | 使 mean T≈164（[K0,K1] 中段），活跃区≈尾迹/壁面、合并区≈来流 |
| **训练期 ω 采样区间** | **[2.83, 25.8]**（≈ p40~p85）| 对应 T 跨度约 [90,175]，模型见过粗/细各种粒度，泛化稳 |
| **「T 随物理变化」** | **已证实** | 右图 min-max 带显示同一绝对阈值下 T 随帧波动（M3 遗留点闭环）|
| **物理量级 vs 原文** | 本数据 |ω| 比原文归一化区间 [0.2,4] 高 1~2 个数量级 | 必须用绝对物理阈值，不能照搬原文 |
| G / M / S 区间 | **待同法标定** | 本步只标定了主判据 ω；G/M/S 用同一工具扩展即可（当前训练/演示主用 ω）|

> **落地**：`amr_router.DEFAULT_RANGES["omega"]` 已据此更新为 **(2.83, 25.8)**（物理尺度）；G/M/S 暂留原文值并注明待标定。注意本标定基于 **4 case×t≤49 的 smoke 数据**，扩大训练规模后应复标确认。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| D3 标定 | `calibrate_thresholds.py ...` | 出 `11_threshold_calibration.png` + 打印分位/推荐阈值 | ✅ 实跑通过，D3 已拍板 |

> 小步 5 闭环、D3 拍板。**M5 五小步全部实跑通过。** 剩 M5 收尾：放大训练规模（更多 case/epoch + 训练期 noise）后做 rollout 稳定性 + 与 MGN/X-MGN baseline 的正式对比。

---

## 小步 6 — 放大训练 + baseline 对比（本轮：脚本就绪）

### 做了什么

```
train_mgn_baseline.py   ✅ 新建：MGN baseline，与 AMR 完全同训练条件（同 case/epoch/noise/NMSE/lr）
compare_baselines.py    ✅ 新建：两 checkpoint → test 同款 rollout → RMSE-vs-step 对比 + 参数量
（train_amr_m4gn_full.py 已支持 --noise_std / --sample_thresh）
```

### 为什么这么设计

- **公平对比的关键是「同预算同条件」**：MGN baseline 不用仓库 `train.py`（它是 MSE + hydra/DDP 框架，难严格对齐），而是新写 `train_mgn_baseline.py`，**与 `train_amr_m4gn_full.py` 同数据/同 case 数/同 epoch/同 noise/同 per-channel NMSE/同 lr**，唯一差别是模型（纯 MGN vs AMR-M4GN）。这样 rollout 对比才说明是「架构」带来的差异。
- **训练期加 noise（`--noise_std 0.02`）**：rollout 稳定性的关键——训练时给输入加噪声，模型学会纠正自身误差，否则自回归会快速发散（baseline 同款技巧）。
- **rollout 对比用同一函数**：`compare_baselines.rollout_eval(predict_fn, ...)` 对两模型用**完全相同**的自回归流程（边界 mask + 物理积分），只换 `predict_fn`，保证对比公平。

### 选定规模：小验证档（20 case × 100 步 × 各 200 epoch）

目的：先验证「放大后预测/rollout 确实变好」并跑通完整对比流程；确认有价值后再考虑上中/全量档。

### 运行步骤（你拿到代码后，命令均单行）

#### 步骤 0 — 同步文件
`train_mgn_baseline.py`、`compare_baselines.py`。

#### 步骤 1 — 预处理 20 个 train case
- **做什么**：`python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split train --num_cases 20 --out_dir ./amr_cache`
- **为什么**：AMR-M4GN 训练按 gidx 读这些几何缓存。
- **应该得到什么**：`./amr_cache/partition_cache_train_0.pt`〜`_19.pt`（约 1~2 分钟，每个打印 K0=64/K1=256）。

#### 步骤 2 — 训练 AMR-M4GN（20 case，200 epoch，加 noise）
- **做什么**：`python train_amr_m4gn_full.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02 --omega_thresh 8.9`
- **为什么**：用标定的 ω 阈值（8.9）+ 训练噪声做正经训练。
- **应该得到什么**：NMSE 逐 epoch 下降（比 4-case smoke 更稳）；`./checkpoints_amr/amr_m4gn_epoch199.pt`。

#### 步骤 3 — 训练 MGN baseline（完全同预算）
- **做什么**：`python train_mgn_baseline.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02`
- **为什么**：同条件对照组。
- **应该得到什么**：打印参数量（约 ?M）+ NMSE 下降；`./checkpoints_mgn/mgn_epoch199.pt`。

#### 步骤 4 — 准备 test 缓存（若没有）
- **做什么**：`python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test --num_cases 1 --out_dir ./amr_cache`
- **应该得到什么**：`partition_cache_test_0.pt`（已有可跳过）。

#### 步骤 5 — 对比 rollout
- **做什么**：`python compare_baselines.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --amr_ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --mgn_ckpt ./checkpoints_mgn/mgn_epoch199.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9`
- **为什么**：M5 退出标准——在相同预算下看 AMR-M4GN 的 rollout 误差与参数量相对 MGN 的表现。
- **应该得到什么**：终端打印两模型参数量 + step-1/final/mean 速度 RMSE；`./inference_vis/12_compare_rollout_case0.png`（两条 RMSE-vs-step 曲线）。
- **实测结果（20 case×100 步×200 epoch，case0 test，rollout 80 步）**：
  - **训练（in-sample）NMSE**：AMR-M4GN epoch199 = **6.22e-3**（最低 ~5.4e-3 @epoch180）；MGN epoch199 = **8.24e-3**（最低 ~8.1e-3 @epoch185）。**同预算下 AMR 训练误差也更低**。
  - 参数量：**AMR-M4GN 3.18M vs MGN 2.33M**（AMR 多 ~36%，多在 macro Transformer）；
  - 速度 RMSE（**test split case0，模型未训练过**）：step1 **2.021e-02 vs 2.300e-02**；final(step80) **1.143e-01 vs 1.331e-01**；**mean 7.804e-02 vs 9.423e-02**（AMR 平均低 ~17%）；
  - **AMR-M4GN 曲线全程在 MGN 之下，且 step≈20 之后长程差距明显拉大**——正是全局 Transformer 抑制长程误差累积的设计目标。
- **状态**：✅ **实跑通过，AMR-M4GN 全程优于 MGN（含 mean RMSE）**。

### 验收

| 验收点 | 命令 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| AMR-M4GN 放大训练 | 步骤 2 | NMSE 下降、出 checkpoint | ✅ 实跑完成（epoch199 NMSE 6.22e-3，checkpoint 已生成）|
| MGN baseline 训练 | 步骤 3 | NMSE 下降、出 checkpoint | ✅ 实跑完成（epoch199 NMSE 8.24e-3，checkpoint 已生成）|
| rollout 对比 | 步骤 5 | 出 `12_compare_rollout_case0.png` + 指标，AMR 不差于 MGN | ✅ **AMR 全程优于 MGN**（step1 0.020/0.023，final 0.114/0.133，mean 0.078/0.094）|

### 🚩 M5 退出结论（小验证档）

- **结论（正面）**：相同训练预算（20 case×100 步×200 epoch、同 noise/NMSE/lr）下，**AMR-M4GN 的 rollout 误差全程低于 MGN baseline（mean 0.078 vs 0.094，低 ~17%），且长程（step>20）优势明显**；训练 in-sample NMSE 也更低（6.22e-3 vs 8.24e-3）——初步验证了「局部 GNN + 全局段级 Transformer」对长程依赖/误差累积的价值。
- **如实标注的边界**：
  1. 这是 **3.18M vs 2.33M**，非等参对比；AMR 多 ~36% 参数。结论是「同训练预算下更优」，等参/等算力对比待补；
  2. 规模为 **小验证档（20 case，单 case 对比）**，非全量、非多 case 平均，**不构成论文级定论**；
  3. 对比对象只有 MGN，**X-MGN 等其他 baseline 未做**。
- **下一步（M5 完整退出 / M6-M7）**：放大到中/全量档、多 case 平均指标、加 X-MGN 对比、（可选）等参对照。

> **本次实跑精确数字（已补入上表）**：训练 NMSE AMR 6.22e-3 / MGN 8.24e-3；test case0 rollout 速度 RMSE step1 0.020/0.023、final 0.114/0.133、mean 0.078/0.094（AMR/MGN）。

---

## 与 Design Doc 的对应关系

| Design Doc | M5 落地 |
| --- | --- |
| §7.2-C 批内段偏移 + padding mask | 小步 1：`pack_segments`/`run_macro_batched`（本轮）；小步 2：`model` 集成 |
| §7.6 完整训练/推理入口 | 小步 3 `train_amr_m4gn_full.py`（✅）、小步 4 `inference_amr_m4gn.py`（✅ 实跑通过，含 rollout/GIF）|
| §八 M5 退出标准（vs baseline 指标）| 小步 6 `train_mgn_baseline.py` + `compare_baselines.py`（✅ 实跑完成，AMR 全程优于 MGN）|
| 决策门 D3 最终标定 | 小步 5 `calibrate_thresholds.py`（✅ 已拍板：ω∈[2.83,25.8]，K0/K1=64/256）|

---

## 附录 A：`vortex_shedding_mgn/` 全代码关系图

整个目录分四层：**① 原始 baseline（NVIDIA 自带）** → **② AMR-M4GN 模型库（`amr_m4gn/`）** → **③ 离线预处理 + 数据封装** → **④ 训练/推理/对比/标定入口**。数据流向：

```
                      raw_dataset/cylinder_flow/*.tfrecord
                                   │
          ┌────────────────────────┴───────────────────────────┐
          │ (原始 baseline 路线)                                  │ (AMR-M4GN 路线)
          ▼                                                      ▼
  VortexSheddingDataset                          preprocess_partitions.py ──写──► amr_cache/partition_cache_*.pt
   (physicsnemo 内置)                                   │  (几何/分区/PE/面积，与时间无关，只跑一次)
          │                                             │  用到: modal_decomp / segmentation / pe / amr_router(build_l1_to_l0)
          ▼                                             ▼
       train.py ──► checkpoints/                 data_amr.py: VortexSheddingDatasetAMR
   inference.py ──► animations/*.gif               (子类，额外暴露 graph.pos / gidx / get_cache)
   (stock MGN，hydra+DDP)                                 │
                                                          ▼
                                          amr_m4gn/model.py: AMRM4GN.forward(graph, cache, thresholds)
                                          ┌──────────────────┼─────────────────────────────┐
                                          ▼                  ▼                             ▼
                                  micro_gnn.MicroGNN   physics_ops.compute_ns_quantities   amr_router.route
                                  (MGN 旁路 decoder)    (G/ω/M/S，按物理速度)               (折叠/保留→token)
                                          │                  └──────────► thresholds ◄──────┘
                                          ▼
                                  macro_transformer: SegmentEncoder → MacroTransformer → dispatch → decoder → pred[N,3]
```

**训练/推理入口如何复用模型库**（实线=import 调用）：

| 入口脚本 | import 的核心件 | 产物 |
| --- | --- | --- |
| `train_amr_m4gn.py`（M4 单 case overfit）| `AMRM4GN`、`build_cache`、`VortexSheddingDatasetAMR`；定义 `per_channel_nmse`/`move_cache`（被其他脚本复用）| 终端 NMSE 日志 |
| `train_amr_m4gn_full.py`（M5 多 case 正式训练）| `AMRM4GN`、`VortexSheddingDatasetAMR`、`per_channel_nmse`/`move_cache`（来自 `train_amr_m4gn`）| `checkpoints_amr/amr_m4gn_epoch*.pt` |
| `train_mgn_baseline.py`（M5 同预算 MGN 基线）| `MeshGraphNet`、`VortexSheddingDataset`、`per_channel_nmse` | `checkpoints_mgn/mgn_epoch*.pt` |
| `inference_amr_m4gn.py`（M5 预测/rollout/GIF）| `AMRM4GN`、`VortexSheddingDatasetAMR`、`move_cache` | `inference_vis/09_*、10_*.png`、`animations/amr_m4gn_*.gif` |
| `compare_baselines.py`（M5 AMR vs MGN）| `AMRM4GN`、`MeshGraphNet`、`VortexSheddingDatasetAMR`、`move_cache` | `inference_vis/12_compare_rollout_*.png` |
| `calibrate_thresholds.py`（M5 D3 标定）| `compute_ns_quantities`/`denormalize_velocity`、`aggregate_per_segment`/`route` | `inference_vis/11_threshold_calibration.png` |
| `visualize_partition.py`（M1–M3 诊断图）| `modal_decomp`/`segmentation`/`physics_ops`/`amr_router`；定义 `load_single_case`（被 `preprocess_partitions` 复用）| `partition_vis/case*/*.png` |

> **关键复用关系（务必记牢）**：
> - `per_channel_nmse` 和 `move_cache` 的**单一定义在 `train_amr_m4gn.py`**，全量训练、推理、对比脚本都从它 import——改 loss/搬运逻辑只改这一处。
> - `build_cache`（建几何缓存）定义在 `preprocess_partitions.py`；`load_single_case`（读单 case TFRecord）定义在 `visualize_partition.py`。两者被 `train_amr_m4gn.py`（按需现建缓存）/`preprocess_partitions.py` 复用。
> - **rollout 逻辑只有一份**：`inference_amr_m4gn.run_rollout`（单模型 + GIF）与 `compare_baselines.rollout_eval`（双模型对比）是同一套自回归规则（内部节点用预测增量积分、边界保 GT），刻意保持一致以保证「09/10 图」与「12 对比图」口径相同。**新增 GIF 直接扩展 `run_rollout` 收集场，不另写脚本。**

---

## 附录 B：所有脚本作用 + 逐参数含义

> 约定：`required` = 必填；其余给默认值。`<...>` 为占位。

### B.1 `preprocess_partitions.py` — 离线几何/分区缓存（**第一步必跑**）

**作用**：每个 case 跑一次「模态分解 → 两级分区(L0/L1) → 段级&节点级 RWSE → l1→l0 父子映射 → 段质心 → 节点 Voronoi 面积」，存成 `partition_cache_{split}_{gidx}.pt`。因网格静止，缓存与时间步无关，只生成一次后被训练/推理反复加载。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 数据目录（含 `meta.json`、`{split}.tfrecord`）|
| `--split` | `test` | 数据划分：`train`/`valid`/`test`。**注意**：训练用 `train`、对比/推理用 `test`，要分别预处理 |
| `--num_cases` | 1 | 处理多少个 case（从 `case_start` 起）|
| `--case_start` | 0 | 起始 case 序号 |
| `--K0` | 64 | L0（粗）目标段数 |
| `--K1` | 256 | L1（细）目标段数；L1 嵌套细分 L0 |
| `--num_modes` | 6 | Laplacian 特征模态数（分区引导特征 `f_md`）|
| `--tau` | 1.0 | SLIC 紧致度（段形状规整 vs 贴合特征的权衡）|
| `--steps` | 16 | RWSE 随机游走步数（PE 维度）|
| `--out_dir` | `./amr_cache` | 缓存输出目录 |

### B.2 `train_amr_m4gn_full.py` — AMR-M4GN 正式训练（M5）

**作用**：多 case、PyG 批处理（`batch_size>1`）、逐通道 NMSE、指数 LR 衰减、定期 checkpoint 的训练入口。**前置**：先跑 `preprocess_partitions.py --split train`。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | `partition_cache_train_*.pt` 所在目录 |
| `--split` | `train` | 训练划分 |
| `--num_cases` | 4 | 训练 case 数（须 ≤ 已预处理的缓存数）|
| `--num_steps` | 50 | 每 case 取前 N 帧；实际训练样本 = `num_steps-1`（相邻帧增量）|
| `--batch_size` | 2 | 每批图数；批内逐图路由 + 全局 token 偏移 + 注意力按图隔离 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` | 1e-3 | Adam 初始学习率 |
| `--lr_decay` | 0.9999991 | 指数 LR 衰减 γ（每 epoch `lr*=γ`）|
| `--noise_std` | 0.0 | 训练噪声标准差（加在输入速度上，提升 rollout 稳定性）|
| `--hidden` | 128 | 隐藏维度（encoder/processor/decoder/transformer 通用）|
| `--processor_size` | 15 | MicroGNN 消息传递层数（局部感受野跳数）|
| `--omega_thresh` | 30.0 | 固定涡量路由阈值（绝对物理量级）。**本数据集标定值≈8.9** |
| `--sample_thresh` | False | 开启后改为「每图按区间采样阈值」（Design Doc 4.7），忽略 `--omega_thresh` |
| `--ckpt_dir` | `./checkpoints_amr` | checkpoint 输出目录 |
| `--ckpt_every` | 10 | 每多少 epoch 存一次（末轮必存）|
| `--device` | 自动 | `cuda`/`cpu` |

### B.3 `train_mgn_baseline.py` — MGN 同预算基线（M5）

**作用**：用**完全相同**的数据/损失/噪声/lr/epoch 训练原版 `MeshGraphNet`，唯一区别是模型本身（无 AMR 路由），以保证对比公平。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--split` | `train` | 训练划分 |
| `--num_cases` | 20 | 训练 case 数 |
| `--num_steps` | 100 | 每 case 帧数（样本数 = `num_steps-1`）|
| `--batch_size` | 2 | 批大小 |
| `--epochs` | 200 | 训练轮数 |
| `--lr` / `--lr_decay` | 1e-3 / 0.9999991 | 同 AMR，保证公平 |
| `--noise_std` | 0.02 | 训练噪声 |
| `--hidden` / `--processor_size` | 128 / 15 | 同 AMR 的 MicroGNN 配置 |
| `--ckpt_dir` | `./checkpoints_mgn` | checkpoint 目录 |
| `--ckpt_every` | 20 | 存档间隔 |
| `--device` | 自动 | 设备 |

> MGN baseline **不需要** `cache_dir`/`omega_thresh`（无分区路由）。

### B.4 `inference_amr_m4gn.py` — 预测 / rollout / **GIF** 可视化（M5）

**作用**：加载 AMR-M4GN checkpoint，三种模式：
- **默认（单帧）**：预测某一帧，画 (du,dv,p) 的 pred/GT/|误差| 3×3 面板 → `09_prediction_*.png`，并打印逐通道 NMSE/RMSE。
- **`--rollout R`**：自回归滚动 R 步，画速度 RMSE-步数曲线（误差累积）→ `10_rollout_rmse_*.png`。
- **`--gif`**：在 rollout 基础上，为每个 `--gif_fields` 变量存「上=预测 / 下=真值」的场动画 GIF（仿原 `inference.py` 风格）→ `animations/amr_m4gn_case{idx}_{field}.gif`。

**rollout/gif 需内部节点掩码 `rollout_mask`，故必须 `--split test`。**

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | 分区缓存目录（须含对应 split 的缓存）|
| `--ckpt` | required | AMR-M4GN checkpoint 路径（如 `amr_m4gn_epoch199.pt`）|
| `--split` | `train` | 划分；**用 rollout/gif 时改 `test`** |
| `--case_idx` | 0 | 用哪个 case |
| `--timestep` | 25 | 单帧模式画第几帧 |
| `--num_steps` | 50 | 数据集每 case 取帧数 |
| `--hidden` / `--processor_size` | 128 / 15 | 须与训练时一致，否则权重 load 失败 |
| `--omega_thresh` | 30.0 | 路由阈值，须与训练一致（本数据集 8.9）|
| `--rollout` | 0 | >0 时滚动该步数；0 且无 `--gif` 则单帧模式 |
| `--gif` | False | 开启则在 rollout 后生成 GIF |
| `--gif_fields` | `u v p` | 要动画化的场（任意子集）|
| `--frame_skip` | 1 | GIF 抽帧（每 N 帧取一帧，加速/减小体积）|
| `--gif_dir` | `./animations` | GIF 输出目录 |
| `--out_dir` | `./inference_vis` | png 输出目录 |
| `--device` | 自动 | 设备 |

### B.5 `compare_baselines.py` — AMR-M4GN vs MGN rollout 对比（M5）

**作用**：同一 test case 上对两模型跑**同一套** rollout，输出 step-1/最终/平均速度 RMSE、参数量，以及两条误差曲线 → `12_compare_rollout_case{idx}.png`。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | 分区缓存目录 |
| `--amr_ckpt` | required | AMR-M4GN checkpoint |
| `--mgn_ckpt` | required | MGN baseline checkpoint |
| `--split` | `test` | 划分（需 `rollout_mask`）|
| `--case_idx` | 0 | 对比哪个 case |
| `--num_steps` | 90 | 数据集帧数 |
| `--rollout` | 80 | 滚动步数 |
| `--hidden` / `--processor_size` | 128 / 15 | 须与两模型训练一致 |
| `--omega_thresh` | 8.9 | AMR 路由阈值（标定值）|
| `--out_dir` | `./inference_vis` | 输出目录 |
| `--device` | 自动 | 设备 |

### B.6 `calibrate_thresholds.py` — D3 阈值标定（M5，**无需 checkpoint**）

**作用**：统计真实数据上每个 L1 段的 |ω| 分布，扫描候选绝对阈值看 token 数 T 落点，推荐让 T 落在 [K0,K1] 中段的阈值 + 训练期采样区间 → `11_threshold_calibration.png`。物理量只依赖速度+几何，**不需要训练好的模型**。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | 分区缓存目录 |
| `--split` | `train` | 划分 |
| `--num_cases` | 4 | 统计用 case 数 |
| `--num_steps` | 50 | 每 case 帧数 |
| `--stride` | 5 | 每隔几帧采一帧（降算量）|
| `--n_thresh` | 12 | 候选阈值个数（扫描分辨率）|
| `--out_dir` | `./inference_vis` | 输出目录 |

### B.7 `visualize_partition.py` — 预处理诊断图（M1–M3）

**作用**：单 case 跑模态分解+分区+（可选）物理量/路由，输出网格/特征模态/障碍距离/L0/L1 分区/段邻接/物理指标/路由结果等诊断 png 到 `partition_vis/case{idx}/`。也提供被 `preprocess_partitions` 复用的 `load_single_case`。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | `./raw_dataset/.../cylinder_flow` | TFRecord 目录 |
| `--split` / `--case_idx` | `test` / 0 | 划分与 case |
| `--timestep` | 0 | 取哪帧速度（0=初始未发展；建议 300 看成熟卡门街）|
| `--num_modes` | 6 | 特征模态数 |
| `--K0` / `--K1` | 64 / 256 | 两级段数 |
| `--tau` | 1.0 | SLIC 紧致度 |
| `--output_dir` | `./partition_vis` | 输出根目录（自动加 `case{idx}` 子目录）|
| `--use_cotangent` | True | 用余切(FEM) Laplacian（更贴近连续算子）|
| `--boundary_type` | `neumann` | 边界条件：`neumann`/`dirichlet` |
| `--plot_physics` | False | 额外画 G/ω/M/S 四物理指标 |
| `--plot_routing` | False | 额外跑 M3 路由并画折叠/保留结果 + 打印 T |
| `--route_pct` | 70.0 | 路由 DEMO 阈值（L1 段聚合 |phys| 的百分位）|
| `--route_channels` | `omega` | DEMO 用哪些通道判活跃（默认仅 ω）|
| `--log_file` / `--no_log` | None / False | 控制台输出另存文本日志 / 关闭日志 |

### B.8 `train_amr_m4gn.py` — 单 case overfit（M4，集成自检）

**作用**：在单 case 上过拟合验证端到端管线（NMSE 应单调下降至 ~0）。同时是 `per_channel_nmse`/`move_cache` 的定义处。参数与 B.2 类似，特有：`--cache_dir` 不填时用 `build_cache` 现场建缓存；`--K0/--K1` 现建缓存时的段数。

### B.9 原始 baseline（NVIDIA 自带，非 AMR）

- `train.py`：hydra + DDP 的 stock MGN 训练，配置在 `conf/config.yaml`。
- `inference.py`：stock MGN rollout，出 `animations/animation_{u,v,p}.gif`（**本次新增的 AMR GIF 即仿此风格**）。
- `data_amr.py`：`VortexSheddingDataset` 的子类，额外暴露 `graph.pos`/`gidx` 及 `get_cache`，是 AMR 路线的数据入口。

---

## 附录 C：标准复现流程（端到端命令序列）

```bash
cd examples/cfd/vortex_shedding_mgn

# 0) 预处理：train 与 test 各建一次几何缓存
python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split train --num_cases 20 --out_dir ./amr_cache
python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test  --num_cases 1  --out_dir ./amr_cache

# 1) （可选）标定 D3 阈值 → 11_threshold_calibration.png（推荐 ω≈8.9）
python calibrate_thresholds.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --stride 5

# 2) 训练 AMR-M4GN 与 MGN（同预算）
python train_amr_m4gn_full.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02 --omega_thresh 8.9
python train_mgn_baseline.py  --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02

# 3) 单帧预测面板 + rollout 曲线 + 场 GIF
python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9 --gif --gif_fields u v p

# 4) AMR vs MGN 对比曲线
python compare_baselines.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --amr_ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --mgn_ckpt ./checkpoints_mgn/mgn_epoch199.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9
```

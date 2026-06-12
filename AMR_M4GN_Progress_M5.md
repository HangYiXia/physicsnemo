# AMR-M4GN 开发文档 — M5：批处理 → 全量训练 → 评估对比

**更新日期**：2026-06-12
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`
**配套文档**：设计 `AMR_M4GN_Design_Doc.md`；阶段 `AMR_M4GN_Progress_M1.md`〜`_M4.md`

> **里程碑总进度**：M1 ✅ · M2 ✅ · M3 ✅ · M4 ✅ · **M5 🟢（小验证档全部完成，仅"更大训练集"待算力）** · M6 ⬜ · M7 ⬜

---

## 0. M5 是什么 / 现在到哪了

M5 把 M4 跑通的「单 case overfit 管线」升级为**正经可训练、可评估、可对比**的系统，分 6 个小步：

| # | 小步 | 交付物 | 状态 |
| --- | --- | --- | --- |
| 1 | 批处理注意力隔离 | `macro_transformer.py`: `pack_segments`/`run_macro_batched` | ✅ 单测 3/3 |
| 2 | model 批处理集成 | `model.py`: `forward` 支持 PyG batch | ✅ 单测 2/2 + 单图回归 3/3 |
| 3 | 完整训练入口 | `train_amr_m4gn_full.py` | ✅ 实跑（NMSE 6.22e-3）|
| 4 | 推理 + 可视化 | `inference_amr_m4gn.py`（单帧 / rollout / **GIF**）| ✅ 实跑 |
| 5 | D3 阈值标定 | `calibrate_thresholds.py` | ✅ 实跑，D3 拍板（ω≈8.9）|
| 6 | 放大训练 + baseline 对比 | `train_mgn_baseline.py` + `compare_baselines.py` | ✅ 实跑（AMR 全程优于 MGN）|

**一句话结论**：相同训练预算（20 case×100 步×200 epoch）下，AMR-M4GN 的 test rollout 误差**全程低于** MGN baseline（平均低 ~17%，长程优势更明显），训练 in-sample NMSE 也更低。**唯一未做的是更大规模训练集**（待目标机算力）。

---

## 1. 脚本总览（一句话职责 + 输入→输出）

| 脚本 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `preprocess_partitions.py` | **第一步必跑**：建几何/分区/PE/面积缓存（与时间无关，每 case 一次）| TFRecord | `amr_cache/partition_cache_{split}_{idx}.pt` |
| `calibrate_thresholds.py` | 标定 D3 路由阈值（无需 checkpoint）| 缓存 + 速度场 | `inference_vis/11_threshold_calibration.png` |
| `train_amr_m4gn_full.py` | 训练 AMR-M4GN（多 case + batch + 调度 + checkpoint）| TFRecord + 缓存 | `checkpoints_amr/amr_m4gn_epoch*.pt` |
| `train_mgn_baseline.py` | 训练 MGN 基线（与上**同预算**，仅换模型）| TFRecord | `checkpoints_mgn/mgn_epoch*.pt` |
| `inference_amr_m4gn.py` | 单帧预测面板 / rollout 误差曲线 / 场 GIF | checkpoint + test 缓存 | `inference_vis/09_*,10_*.png`、`animations/*.gif` |
| `compare_baselines.py` | AMR vs MGN 同款 rollout 对比（+ **AMR/MGN/GT 三行场 GIF**）| 两 checkpoint | `inference_vis/12_compare_rollout_*.png`、`animations/compare_*.gif` |
| `train_amr_m4gn.py` | (M4) 单 case overfit 自检；`per_channel_nmse`/`move_cache` 定义处 | TFRecord | 终端日志 |
| `visualize_partition.py` | (M1–M3) 预处理诊断图；`load_single_case` 定义处 | TFRecord | `partition_vis/case*/*.png` |
| `train.py` / `inference.py` | NVIDIA 原版 stock MGN（hydra+DDP），非 AMR | — | `animations/animation_*.gif` |

逐参数含义见 **§7 脚本参数大全**。

---

## 2. 一键复现（直接复制）

> 全部命令在 `examples/cfd/vortex_shedding_mgn/` 下执行；均为单行。阈值 `omega_thresh 8.9` 为本数据集标定值（见 §4.4）。

```bash
cd examples/cfd/vortex_shedding_mgn

# ── 0) 预处理：train 20 case + test 1 case（几何缓存，只跑一次）─────────────
python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split train --num_cases 20 --out_dir ./amr_cache
python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test  --num_cases 1  --out_dir ./amr_cache

# ── 1) (可选) D3 阈值标定 → 11_threshold_calibration.png（推荐 ω≈8.9）────────
python calibrate_thresholds.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --stride 5

# ── 2) 训练 AMR-M4GN 与 MGN（完全同预算）────────────────────────────────────
python train_amr_m4gn_full.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02 --omega_thresh 8.9
python train_mgn_baseline.py  --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02

# ── 3) 可视化：单帧面板 / rollout 曲线 / 场 GIF ────────────────────────────
# 3a 单帧 9 宫格 (pred vs GT vs |err|) → 09_prediction_*.png
python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --split test --case_idx 0 --timestep 25 --num_steps 90 --omega_thresh 8.9
# 3b rollout 误差曲线 + 速度/压强场 GIF → 10_rollout_*.png + animations/amr_m4gn_case0_{u,v,p}.gif
python inference_amr_m4gn.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9 --gif --gif_fields u v p

# ── 4) AMR vs MGN 对比曲线 → 12_compare_rollout_case0.png ──────────────────
python compare_baselines.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --amr_ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --mgn_ckpt ./checkpoints_mgn/mgn_epoch199.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9
# 4b 加 --gif：AMR/MGN/GT 三行场动画 → animations/compare_case0_{u,v,p}.gif
python compare_baselines.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --amr_ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --mgn_ckpt ./checkpoints_mgn/mgn_epoch199.pt --split test --case_idx 0 --num_steps 90 --rollout 80 --omega_thresh 8.9 --gif --gif_fields u v p

# ── (可选) 单元测试 ────────────────────────────────────────────────────────
pytest tests/ -v
```

> **依赖提示**：GIF 用 matplotlib 的 `pillow` writer，目标机需 `pip install pillow`。`pymetis` 缺失时分区自动退化为谱聚类（慢、质量略低）。

---

## 3. 关键结果记录（实跑数据，完整保存）

> 规则：本节是**结果档案**。每次回传的实跑数字按「配置 + 数字 + 文件名」三段记录，重要数据**全量保留**，不做删改。

### 3.1 训练收敛

| 配置 | 模型 | 最终 NMSE | 最低 NMSE | 备注 |
| --- | --- | --- | --- | --- |
| 20 case×100 步×200 ep, noise 0.02, ω=8.9 | AMR-M4GN | **6.22e-3** (epoch199) | ~5.4e-3 (epoch180) | 参数 3.18M |
| 20 case×100 步×200 ep, noise 0.02 | MGN baseline | **8.24e-3** (epoch199) | ~8.1e-3 (epoch185) | 参数 2.33M |
| (smoke) 4 case×50 步×50 ep, ω=30 | AMR-M4GN | ~0.08 (epoch49) | ~0.07 | 从 3.06 降约 35× |

**要点**：同预算下 AMR-M4GN 训练误差比 MGN 更低（6.22e-3 < 8.24e-3）。

### 3.2 单步预测（4 case×50 ep smoke 模型, test case0, t=25）

- per-channel NMSE（归一化空间）du/dv/p = **0.317 / 0.453 / 0.143**
- per-channel RMSE（物理空间）du/dv/p = **1.0e-3 / 5.9e-4 / 1.7e-2**
- `09_prediction_case0_t25.png`：预测列与真值列形态高度吻合（尾迹涡街红蓝结构、圆柱前高压/后低压梯度对得上），误差集中在涡街/驻点剧烈变化区，来流区误差≈0。
- 说明：smoke 模型，形态正确、量级合理；dv 量级小→NMSE 偏高属正常。**非最终性能**。

### 3.3 自回归 rollout（smoke 模型 epoch49, test case0, 50 步）

- 每步速度 RMSE first/mid/last = **0.107 / 0.347 / 0.359**
- `10_rollout_rmse_case0.png`：前 ~20 步上升，之后**平台饱和在 ~0.35、不发散**。
- 解读：rollout 机制正确（自回归 + 边界 mask + 物理积分）；误差有界饱和说明模型未崩。

### 3.4 D3 阈值标定（4 case×t≤49, stride 5）

- per-seg |ω| 分位：p10=5.2e-3, p30=0.64, **p50=7.6, p70=16.3, p90=33.0, p95=49.1, p99=107.6**
- T-vs-阈值：阈值 0→T≈240、**≈8.9→T≈164**、50→T≈77（均落在 [K0,K1]=[64,256] 内）

**D3 结论（已拍板）**：

| 项 | 结论 | 依据 |
| --- | --- | --- |
| K0 / K1 | **64 / 256** | T 覆盖 [77,240]，区间用满不溢出 |
| 绝对 ω 阈值（测试/演示）| **≈ 8.9** | mean T≈164（中段）；活跃≈尾迹/壁面，合并≈来流 |
| 训练期 ω 采样区间 | **[2.83, 25.8]**（≈p40~p85）| T 跨度 ~[90,175]，见过粗/细各粒度 |
| 「T 随物理变化」| **已证实** | min-max 带显示同阈值下 T 随帧波动 |
| G/M/S 区间 | **待同法标定** | 本步只标主判据 ω |

> 已落地：`amr_router.DEFAULT_RANGES["omega"]` = (2.83, 25.8)。基于 smoke 数据，扩大规模后应复标。

### 3.5 AMR-M4GN vs MGN 对比（20 case×100×200 ep, test case0, rollout 80 步）

| 指标 | AMR-M4GN | MGN baseline |
| --- | --- | --- |
| 参数量 | 3.18M | 2.33M |
| 速度 RMSE @step1 | **2.021e-02** | 2.300e-02 |
| 速度 RMSE @final(step80) | **1.143e-01** | 1.331e-01 |
| 速度 RMSE @mean | **7.804e-02** | 9.423e-02 |

- `12_compare_rollout_case0.png`：AMR-M4GN 曲线**全程在 MGN 之下**，step≈20 后长程差距明显拉大。
- **test split 为模型未训练过的数据**，故这是泛化结果。
- 边界：3.18M vs 2.33M（非等参，AMR 多 ~36%）；单 test case；仅对比 MGN。

---

## 4. 实现要点（为什么这么写）

### 4.1 批处理注意力隔离（小步 1）
- `SegmentEncoder`/`dispatch` 不改：只认全局连续 token id（`kept_assign∈[0,T)`）+ `T`；批内各图 token 全局偏移后，mean-pool/dispatch 逻辑与单图一致。
- 唯一须 batch-aware 的是 **Transformer 注意力**：不同算例 token 绝不能互相 attend。做法：变长 token pad 成 `[B,Tmax,d]` + `key_padding_mask`（pad=True），注意力只在同图非 pad token 间进行，算完解包回 `[T_total,d]`。
- 前提：`token_batch` 分段连续（图 0 token 在前），正是 `model.py` 拼接逐图路由的方式。

### 4.2 model 批处理集成（小步 2）— 数据流
一个 batch 是 B 个图拼成的大图（节点全局偏移、`ptr` 标范围、边不跨图）：
1. micro 一次跑整 batch → `h_node[N_total,d]`（消息传递不跨图，等价逐图）；
2. 物理量一次全局算（拼接各 cache 的 pos/area + batched edge_index）；
3. 路由**逐图**做，token id 全局偏移、记 `token_batch`；
4. `SegmentEncoder` 用全局 id 一次 mean-pool；
5. `run_macro_batched` 按图隔离注意力；
6. dispatch + decoder 出 `[N_total,3]`。
- 单图路径（`cache` 传单 dict, B=1）与 M4 逐字节等价。

### 4.3 训练入口（小步 3）
- 新建 `train_amr_m4gn_full.py`，不动已验证的 overfit 脚本；单机单卡（DDP/wandb 留 M7）。
- DataLoader 把 B 个图拼成 PyG Batch（带 `ptr`、每图 `gidx`）；按 `gidx` 取对应 caches 列表喂 `forward`。
- 损失 `per_channel_nmse`、搬运 `move_cache` **单一定义在 `train_amr_m4gn.py`**，全脚本复用。

### 4.4 rollout / GIF（小步 4 + 4b）
- rollout 是代理模型真实用法：只给初始帧，之后用自己预测往下推，暴露误差累积/稳定性。
- 边界用 GT：只对 `rollout_mask`（内部节点）施加预测增量，边界保真值，否则迅速发散；`rollout_mask` 仅 test split 提供 → rollout/gif 必须 `--split test`。
- 物理积分：每步反归一化增量 `v_{t+1}=v_t+Δv` 再归一化喂下一步（与 stock `inference.py` 一致）。
- **GIF 复用 `run_rollout`**：rollout 时顺带收集 (u,v,p) 物理场，`make_gif` 画「上=预测/下=真值」动画（仿 `inference.py` 风格，色标固定 GT 范围逐帧可比）。**不另写脚本**，与 `compare_baselines` 的 rollout 同口径。

### 4.5 公平对比（小步 6）
- MGN baseline 不用 stock `train.py`（MSE+hydra/DDP，难对齐），而是 `train_mgn_baseline.py`：与 AMR **同数据/case/epoch/noise/NMSE/lr**，唯一差别是模型。
- 训练期加 noise（0.02）：模型学会纠正自身误差，rollout 才不发散。
- `compare_baselines.rollout_eval(predict_fn,...)` 两模型用同一自回归流程，只换 `predict_fn`。

---

## 5. 代码关系图

四层结构：**① stock baseline（NVIDIA 自带）** → **② AMR-M4GN 模型库 `amr_m4gn/`** → **③ 预处理 + 数据封装** → **④ 训练/推理/对比/标定入口**。

```
                      raw_dataset/cylinder_flow/*.tfrecord
                                   │
          ┌────────────────────────┴───────────────────────────┐
          │ (stock baseline)                                     │ (AMR-M4GN)
          ▼                                                      ▼
  VortexSheddingDataset                          preprocess_partitions.py ─写─► amr_cache/partition_cache_*.pt
   (physicsnemo 内置)                              用到 modal_decomp/segmentation/pe/amr_router(build_l1_to_l0)
          │                                             │
          ▼                                             ▼
   train.py / inference.py                       data_amr.VortexSheddingDatasetAMR
   (stock MGN, hydra+DDP)                          (子类: 暴露 graph.pos/gidx + get_cache)
                                                          │
                                                          ▼
                            amr_m4gn/model.py: AMRM4GN.forward(graph, cache, thresholds)
                            ┌──────────────────┼─────────────────────────────┐
                            ▼                  ▼                             ▼
                    micro_gnn.MicroGNN  physics_ops.compute_ns_quantities  amr_router.route
                    (MGN 旁路 decoder)   (G/ω/M/S, 按物理速度)              (折叠/保留→token)
                            │                  └──────► thresholds ◄────────┘
                            ▼
            macro_transformer: SegmentEncoder→MacroTransformer→dispatch→decoder→pred[N,3]
```

**关键复用点（改动只改一处）**：
- `per_channel_nmse` / `move_cache` → 定义在 `train_amr_m4gn.py`，全训练/推理/对比脚本 import。
- `build_cache`（建几何缓存）→ `preprocess_partitions.py`。
- `load_single_case`（读单 case TFRecord）→ `visualize_partition.py`。
- rollout 自回归逻辑 → `inference_amr_m4gn.run_rollout`（单模型+GIF）与 `compare_baselines.rollout_eval`（双模型）同一套规则。

---

## 6. 单元测试一览

| 测试文件 | 数量 | 覆盖 |
| --- | --- | --- |
| `tests/test_pe.py` | 5 | RWSE 段级/节点级 |
| `tests/test_physics_ops.py` | 8 | G/ω/M/S 物理量 |
| `tests/test_amr_router.py` | 11 | 折叠/保留路由、聚合 |
| `tests/test_micro_gnn.py` | 4 | MGN 旁路 decoder |
| `tests/test_macro_transformer.py` | 5 | 段编码/transformer/dispatch |
| `tests/test_batched_macro.py` | 3 | 打包/解包/跨图不泄漏 |
| `tests/test_model.py` | 3 | 单图端到端 |
| `tests/test_model_batch.py` | 2 | batch 前反向 + batch==逐图拼接 |

一次跑全：`pytest tests/ -v`。

---

## 7. 脚本参数大全

> `required` = 必填；其余为默认值。

### 7.1 `preprocess_partitions.py`
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录（含 `meta.json`、`{split}.tfrecord`）|
| `--split` | `test` | 划分：`train`/`valid`/`test`（训练用 train、对比用 test，各跑一次）|
| `--num_cases` | 1 | 处理 case 数（从 `case_start` 起）|
| `--case_start` | 0 | 起始 case 序号 |
| `--K0` / `--K1` | 64 / 256 | L0（粗）/ L1（细嵌套）目标段数 |
| `--num_modes` | 6 | Laplacian 特征模态数（分区引导特征 f_md）|
| `--tau` | 1.0 | SLIC 紧致度（形状规整 vs 贴合特征）|
| `--steps` | 16 | RWSE 随机游走步数（PE 维度）|
| `--out_dir` | `./amr_cache` | 缓存输出目录 |

### 7.2 `train_amr_m4gn_full.py`
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | `partition_cache_train_*.pt` 目录 |
| `--split` | `train` | 训练划分 |
| `--num_cases` | 4 | 训练 case 数（≤ 已预处理缓存数）|
| `--num_steps` | 50 | 每 case 取帧数；样本数 = `num_steps-1` |
| `--batch_size` | 2 | 每批图数 |
| `--epochs` | 50 | 训练轮数 |
| `--lr` / `--lr_decay` | 1e-3 / 0.9999991 | Adam 初始 lr / 指数衰减 γ |
| `--noise_std` | 0.0 | 输入速度训练噪声（提升 rollout 稳定）|
| `--hidden` | 128 | 隐藏维度 |
| `--processor_size` | 15 | MicroGNN 消息传递层数（局部感受野跳数）|
| `--omega_thresh` | 30.0 | 固定涡量路由阈值；**本数据集标定值 8.9** |
| `--sample_thresh` | False | 改为每图按区间采样阈值（忽略 `--omega_thresh`）|
| `--ckpt_dir` | `./checkpoints_amr` | checkpoint 目录 |
| `--ckpt_every` | 10 | 每多少 epoch 存档（末轮必存）|
| `--device` | 自动 | `cuda`/`cpu` |

### 7.3 `train_mgn_baseline.py`
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--split` | `train` | 训练划分 |
| `--num_cases` | 20 | 训练 case 数 |
| `--num_steps` | 100 | 每 case 帧数 |
| `--batch_size` | 2 | 批大小 |
| `--epochs` | 200 | 训练轮数 |
| `--lr` / `--lr_decay` | 1e-3 / 0.9999991 | 同 AMR（保证公平）|
| `--noise_std` | 0.02 | 训练噪声 |
| `--hidden` / `--processor_size` | 128 / 15 | 同 AMR 的 MicroGNN 配置 |
| `--ckpt_dir` | `./checkpoints_mgn` | checkpoint 目录 |
| `--ckpt_every` | 20 | 存档间隔 |
| `--device` | 自动 | 设备 |

> MGN baseline **不需要** `cache_dir`/`omega_thresh`。

### 7.4 `inference_amr_m4gn.py`
三模式：默认**单帧**面板；`--rollout R` 出**误差曲线**；`--gif` 出**场动画**。rollout/gif 需 `rollout_mask` → 必须 `--split test`。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | 分区缓存目录（对应 split）|
| `--ckpt` | required | AMR-M4GN checkpoint 路径 |
| `--split` | `train` | 划分；**rollout/gif 时改 `test`** |
| `--case_idx` | 0 | 用哪个 case |
| `--timestep` | 25 | 单帧模式画第几帧 |
| `--num_steps` | 50 | 每 case 取帧数 |
| `--hidden` / `--processor_size` | 128 / 15 | 须与训练一致，否则 load 失败 |
| `--omega_thresh` | 30.0 | 路由阈值，须与训练一致（本数据集 8.9）|
| `--rollout` | 0 | >0 时滚动该步数；0 且无 `--gif` 则单帧模式 |
| `--gif` | False | rollout 后生成 GIF |
| `--gif_fields` | `u v p` | 要动画化的场（任意子集）|
| `--frame_skip` | 1 | GIF 抽帧（每 N 帧取一帧）|
| `--gif_dir` | `./animations` | GIF 输出目录 |
| `--out_dir` | `./inference_vis` | png 输出目录 |
| `--device` | 自动 | 设备 |

### 7.5 `compare_baselines.py`
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | 分区缓存目录 |
| `--amr_ckpt` / `--mgn_ckpt` | required | 两模型 checkpoint |
| `--split` | `test` | 划分（需 `rollout_mask`）|
| `--case_idx` | 0 | 对比哪个 case |
| `--num_steps` | 90 | 数据集帧数 |
| `--rollout` | 80 | 滚动步数 |
| `--hidden` / `--processor_size` | 128 / 15 | 须与两模型训练一致 |
| `--omega_thresh` | 8.9 | AMR 路由阈值（标定值）|
| `--gif` | False | 额外出 **AMR/MGN/GT 三行**场动画 GIF |
| `--gif_fields` | `u v p` | 三行 GIF 要动画化的场（任意子集）|
| `--frame_skip` | 1 | GIF 抽帧（每 N 帧取一帧）|
| `--gif_dir` | `./animations` | 对比 GIF 输出目录 |
| `--out_dir` | `./inference_vis` | 输出目录 |
| `--device` | 自动 | 设备 |

> 三行 GIF 文件名 `compare_case{idx}_{field}.gif`，上→下 = AMR-M4GN / MGN / Ground-Truth，色标统一按 GT 范围，逐帧/逐行可比。

### 7.6 `calibrate_thresholds.py`（无需 checkpoint）
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | 分区缓存目录 |
| `--split` | `train` | 划分 |
| `--num_cases` | 4 | 统计用 case 数 |
| `--num_steps` | 50 | 每 case 帧数 |
| `--stride` | 5 | 每隔几帧采一帧 |
| `--n_thresh` | 12 | 候选阈值个数（扫描分辨率）|
| `--out_dir` | `./inference_vis` | 输出目录 |

### 7.7 `visualize_partition.py`（诊断图）
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | `./raw_dataset/.../cylinder_flow` | TFRecord 目录 |
| `--split` / `--case_idx` | `test` / 0 | 划分与 case |
| `--timestep` | 0 | 取哪帧速度（0=初始；建议 300 看成熟卡门街）|
| `--num_modes` | 6 | 特征模态数 |
| `--K0` / `--K1` | 64 / 256 | 两级段数 |
| `--tau` | 1.0 | SLIC 紧致度 |
| `--output_dir` | `./partition_vis` | 输出根目录（自动加 `case{idx}` 子目录）|
| `--use_cotangent` | True | 用余切(FEM) Laplacian |
| `--boundary_type` | `neumann` | 边界条件 `neumann`/`dirichlet` |
| `--plot_physics` | False | 额外画 G/ω/M/S |
| `--plot_routing` | False | 额外跑 M3 路由并画 + 打印 T |
| `--route_pct` | 70.0 | 路由 DEMO 阈值（百分位）|
| `--route_channels` | `omega` | DEMO 判活跃的通道 |
| `--log_file` / `--no_log` | None / False | 另存日志 / 关闭日志 |

### 7.8 `train_amr_m4gn.py`（M4 overfit 自检）
与 7.2 类似；特有：`--case_idx` 选单 case；`--cache_dir` 不填时用 `build_cache` 现场建缓存；`--K0/--K1` 为现建缓存段数。

---

## 8. M5 退出结论

**正面**：同训练预算下，AMR-M4GN rollout 误差全程低于 MGN（mean 0.078 vs 0.094，低 ~17%），长程优势明显；训练 in-sample NMSE 也更低（6.22e-3 vs 8.24e-3）。初步验证「局部 GNN + 全局段级 Transformer」对长程依赖/误差累积的价值。

**如实标注的边界**：
1. 3.18M vs 2.33M，**非等参**（AMR 多 ~36%）；结论是「同预算下更优」，等参对照待补。
2. **小验证档**（20 case、单 test case 对比），非全量、非多 case 平均，不构成论文级定论。
3. 对比对象仅 MGN，X-MGN 等未做。

**下一步（M5 完整退出 / M6–M7）**：
- 放大到中/全量训练集（**唯一待算力项**）；
- 多 test case 平均 rollout 指标（建议加批量评估脚本）；
- 加 X-MGN 等 baseline、（可选）等参对照；
- §6.4 全套指标（FLOPs/显存/GPU 时间/Strouhal 数）。

---

## 9. 结果回传记录区（待目标机跑完填入）

> 目标机跑出新结果后，按下表回传，本人据此**完整记录**进 §3，并更新 §8 结论。

| 待办 | 命令（见 §2）| 回传内容 |
| --- | --- | --- |
| 更大训练集（中/全量档）| §2 步骤 2（调大 `--num_cases`/`--epochs`）| 训练 NMSE 曲线末值 + checkpoint 名 |
| 多 test case 平均 | 多次 §2 步骤 4（改 `--case_idx`）| 各 case step1/final/mean RMSE |
| 场 GIF 验证 | §2 步骤 3b | 确认 `animations/amr_m4gn_case0_{u,v,p}.gif` 生成 |
| 单帧/rollout 新数 | §2 步骤 3a/3b | 终端 NMSE/RMSE 打印 + png 文件名 |

**回传数据请贴终端原始打印**（如 `compare_baselines.py` 的 `params/step-1/final/mean` 那几行），本人原样录入、不改数字。

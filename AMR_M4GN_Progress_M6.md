# AMR-M4GN 开发文档 — M6：模块消融实验

**更新日期**：2026-06-12
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`
**配套文档**：设计 `AMR_M4GN_Design_Doc.md`（§6.5 消融计划 / §八 M6）；阶段 `AMR_M4GN_Progress_M1.md`〜`_M5.md`

> **里程碑总进度**：M1 ✅ · M2 ✅ · M3 ✅ · M4 ✅ · M5 🟢 · **M6 🟡（代码+脚本+单测就绪，消融实跑待目标机算力）** · M7 ⬜
>
> **状态图例**：✅ 完成并实跑验证 · 🟢 主体完成（仅更大规模待算力）· 🟡 **代码就绪、等待目标机实跑/测试** · ⬜ 未开始

---

## 0. M6 是什么 / 现在到哪了

M6 = **模块消融**：逐个关掉 AMR-M4GN 的某个模块，在**同数据/同预算/同评估**下看 rollout 误差变化，从而量化每个模块的净贡献（Design Doc §6.5 八组表 + 决策门 D8「裁剪净贡献为负的模块」）。

| # | 工作 | 交付物 | 状态 |
| --- | --- | --- | --- |
| 1 | 模型加消融开关 | `model.py`: `use_amr` / `use_transformer` / `use_rwse` | 🟡 已实现，单测待跑 |
| 2 | 预处理加模态开关 | `preprocess_partitions.py`: `--no_modal` | 🟡 已实现 |
| 3 | 训练入口透传开关 | `train_amr_m4gn_full.py`: `--no_amr/--no_transformer/--no_rwse/--tag` | 🟡 已实现 |
| 4 | 消融编排脚本 | `run_ablation.py`（训练 N 组 → 同 rollout 评估 → 表 + 柱状图）| 🟡 已实现，实跑待算力 |
| 5 | 单元测试 | `tests/test_ablation.py` | 🟡 已写，待目标机 `pytest` |

**一句话现状**：M6 的**全部代码、脚本、单测已写完**，本机无环境/算力**未实跑**；待目标机跑 `run_ablation.py` 产出 §6.5 消融表后，本人据回传数据填入 §3 并给出 D8 裁剪结论。

### 可消融模块清单（如实标注是否已接线）

| Design Doc 6.5 行 | 开关 | 是否可跑 | 说明 |
| --- | --- | --- | --- |
| Full model | — | ✅ | 完整模型 |
| w/o AMR | `use_amr=False` | ✅ | 所有 L1 段保细（固定 K=K1，不折叠）|
| w/o Transformer | `use_transformer=False` | ✅ | 仅 15 步 GNN，全局分支置零（≈MGN）|
| w/o Modal Decomp | 预处理 `--no_modal` | ✅ | 几何-only 分区缓存（SLIC 不用模态特征）|
| w/o RWSE PE | `use_rwse=False` | ✅ | 段级位置编码置零 |
| 7 步 vs 15 步 GNN | `processor_size=7/15` | ✅ | GNN 深度收益 vs 过平滑 |
| w/o δ=1 重叠 | — | ❌ **未实现** | `segmentation.py` 无段重叠逻辑，缺该消融对象 |
| w/o 虚拟步 | — | ❌ **未接线** | `physics_ops.virtual_step()` 已实现但 `model.forward` 未喂 `u_prev`；当前模型本就「无虚拟步」，缺「有虚拟步」对照臂 |

> 后两项要做需先补实现（见 §6 待办）。本阶段消融表覆盖 **6/8 行**，并明确标注未覆盖的 2 行及原因。

---

## 1. 消融开关怎么接线（实现要点）

均为**零侵入**改动（不改张量形状、不影响已验证的全量路径）：

- **`use_amr=False`**：`model.forward` 路由时把 4 个阈值强制设为 `-inf` → 每个 L1 段都「活跃」→ 全部保留为细 token（`T=K1`），等价 M4GN 固定 K。
- **`use_transformer=False`**：跑完 `micro` 后**直接 `decoder([h_node ; 0])` 返回**，跳过路由/段编码/Transformer/dispatch。全局分支置零 → 退化为「局部 GNN + 解码器」≈ MGN。段编码/Transformer 子模块仍构造但不参与前向，故其参数**不收梯度**（单测校验）。
- **`use_rwse=False`**：段编码前把 `rwse_t` 置零（depth/centroid 保留），消融段级随机游走结构编码。
- **`--no_modal`（预处理）**：`build_cache` 把模态特征 `f_md` 置零再做 SLIC，得到「几何-only」分区，存为 `*_nomodal.pt`（**不覆盖**模态缓存）。
- **`processor_size`**：本就是构造参数，传 7 / 15 即可。

> 设计取舍：消融用「置零/强制」而非删模块，保证各配置**结构等价、参数可比**，且全量路径一字不改。

---

## 2. 一键复现（直接复制）

> 在 `examples/cfd/vortex_shedding_mgn/` 下执行。前置：M5 的 modal 缓存已建；w/o Modal 行额外需 `_nomodal` 缓存。

```bash
cd examples/cfd/vortex_shedding_mgn

# ── 0) 额外预处理：几何-only 缓存（供 w/o Modal 行）──────────────────────────
python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split train --num_cases 20 --out_dir ./amr_cache --no_modal
python preprocess_partitions.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test  --num_cases 1  --out_dir ./amr_cache --no_modal

# ── 1) 跑全部 6 组消融（训练 + 同 rollout 评估 + 出表/图）────────────────────
python run_ablation.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02 --omega_thresh 8.9 --test_case 0 --rollout 80
#   产物：inference_vis/13_ablation.csv + inference_vis/13_ablation.png + 终端表

# ── (可选) 只跑某几组 ──────────────────────────────────────────────────────
python run_ablation.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 20 --num_steps 100 --epochs 200 --noise_std 0.02 --omega_thresh 8.9 --only full "w/o Transformer" "w/o AMR"

# ── (可选) 单独训练某个消融配置（产 checkpoint，便于后续可视化）─────────────
python train_amr_m4gn_full.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 20 --num_steps 100 --batch_size 2 --epochs 200 --noise_std 0.02 --omega_thresh 8.9 --no_transformer --tag amr_no_tf

# ── (可选) 单元测试 ────────────────────────────────────────────────────────
pytest tests/test_ablation.py -v
```

> **算量提示**：`run_ablation.py` 默认串行训练 6 个模型（每个 = 一次完整 M5 训练）。目标机算力有限时，先用 `--only`/小 `--epochs` 跑子集冒烟，再放全量。GIF/pillow 不涉及本阶段。

---

## 3. 结果记录（**待目标机实跑后填入**）

> 规则同 M5 §3：回传按「配置 + 数字 + 文件名」三段记录，重要数据全量保留、不改。

### 3.1 消融总表（占位，待回传 `13_ablation.csv` / 终端表）

| 配置 | params(M) | 训练 NMSE | rollout mean RMSE(test) | 相对 full 变化 | 结论 |
| --- | --- | --- | --- | --- | --- |
| full | _待填_ | _待填_ | _待填_ | — | 基准 |
| w/o AMR | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| w/o Transformer | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| w/o Modal | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| w/o RWSE | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| proc7 | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| w/o δ重叠 | — | — | — | — | 未实现，未跑 |
| w/o 虚拟步 | — | — | — | — | 未接线，未跑 |

读法：`mean RMSE` 比 full **高**得越多 → 该模块净贡献越大、应保留；**持平或更低** → 候选裁剪（D8）。

### 3.2 D8 裁剪结论（**待数据后给出**）

待 §3.1 填满后，对净贡献≈0 或为负的模块给出保留/裁剪建议（重点观察 RWSE、proc 深度）。

---

## 4. 代码改动一览

| 文件 | 改动 | 说明 |
| --- | --- | --- |
| `amr_m4gn/model.py` | + `use_amr/use_transformer/use_rwse` 构造参数 + forward 接线 | 见 §1 |
| `preprocess_partitions.py` | + `--no_modal`，`build_cache(use_modal=)`，文件名 `_nomodal` 后缀 | 几何-only 缓存 |
| `train_amr_m4gn_full.py` | + `--no_amr/--no_transformer/--no_rwse/--tag` | 单配置可独立训练 |
| `run_ablation.py` | **新建**：编排 6 组训练+评估，出 `13_ablation.{csv,png}` | M6 主入口 |
| `tests/test_ablation.py` | **新建**：开关前反向 + 分支不收梯度 + token 数校验 | 3 测试 |

---

## 5. 参数大全（M6 新增/相关）

### 5.1 `run_ablation.py`
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--data_dir` | required | TFRecord 目录 |
| `--cache_dir` | required | 分区缓存目录（含 modal；w/o Modal 行需同目录有 `_nomodal`）|
| `--num_cases` | 20 | 训练 case 数 |
| `--num_steps` | 100 | 训练每 case 帧数 |
| `--batch_size` | 2 | 批大小 |
| `--epochs` | 200 | 每个配置训练轮数 |
| `--lr` / `--lr_decay` | 1e-3 / 0.9999991 | 学习率 / 衰减 |
| `--noise_std` | 0.02 | 训练噪声 |
| `--hidden` | 128 | 隐藏维度 |
| `--omega_thresh` | 8.9 | AMR 路由阈值（标定值；w/o AMR 行内部忽略）|
| `--test_split` / `--test_case` | `test` / 0 | 评估用划分与 case |
| `--test_num_steps` | 90 | 评估数据集帧数 |
| `--rollout` | 80 | 评估 rollout 步数 |
| `--only` | None | 只跑指定配置名（默认全部可跑项）|
| `--out_dir` | `./inference_vis` | 表/图输出目录 |
| `--device` | 自动 | 设备 |

### 5.2 `preprocess_partitions.py` 新增
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--no_modal` | False | 几何-only 分区（f_md 置零），写 `*_nomodal.pt`（不覆盖 modal 缓存）|

### 5.3 `train_amr_m4gn_full.py` 新增
| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--no_amr` | False | 关 AMR 路由（固定 K=K1）|
| `--no_transformer` | False | 关 macro Transformer（仅 GNN）|
| `--no_rwse` | False | 段级 RWSE PE 置零 |
| `--tag` | `amr_m4gn` | checkpoint 文件名前缀（区分不同消融）|

---

## 6. M6 退出标准 与 待办

**退出标准（Design Doc §八 M6）**：§6.5 消融表 + 分析 + D8 裁剪结论。

**当前状态**：
- ✅ 代码/脚本/单测**全部就绪**（6/8 行可跑）。
- 🟡 **未实跑**（本机无算力）→ 消融表、D8 结论待目标机数据。
- ❌ 两行未覆盖，需补实现才能纳入：
  1. **δ=1 重叠**：在 `segmentation.py` 加段重叠（节点可属多段一圈邻居），并让 token 池化/缓存支持软归属；
  2. **虚拟步**：把上一帧速度 `u_prev` 喂进 `model.forward`（数据侧需提供前帧），用 `virtual_step()` 算预refine物理量。

**下一步**：目标机跑 §2 → 回传 `13_ablation.csv` + 终端表 → 本人填 §3 + 给 D8 结论；如需补 δ/虚拟步两行，单列小任务实现后再跑。

---

## 7. 结果回传记录区（待目标机跑完填入）

| 待办 | 命令（见 §2）| 回传内容 |
| --- | --- | --- |
| 6 组消融主结果 | §2 步骤 1 | 终端表全文 + `13_ablation.csv` + `13_ablation.png` |
| 单测验证 | `pytest tests/test_ablation.py -v` | 通过数（应 3/3）|
| (可选) 子集冒烟 | §2 `--only` | 对应几行数字 |

**回传请贴 `run_ablation.py` 终端表原文**（含 params/train NMSE/mean RMSE 各列），本人原样录入 §3.1、不改数字，并据此完成 §3.2 的 D8 裁剪结论。

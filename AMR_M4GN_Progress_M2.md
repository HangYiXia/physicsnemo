# AMR-M4GN 开发进度与操作手册（M2 阶段）

**更新日期**：2026年6月10日
**当前阶段**：M2 — N-S 物理量算子（G/ω/M/S + 1-ring 最小二乘梯度 + 虚拟步）+ 四标量场可视化
**M2 状态**：⚠️ **代码已实现，但尚未经你实跑测试通过**（pytest 与发展态可视化都还没跑）。下文 `✅` 仅表示「文件已新建/修改」，**不代表已验证**。M2 能否关闭取决于 §七 验收由你实跑达成。
**前置**：M1 已由你实跑 4 个 case 验证关闭（环境、数据集下载、`visualize_partition.py` 基础用法见 `AMR_M4GN_Progress_M1.md`，本文不重复）
**配套设计文档**：`AMR_M4GN_Design_Doc.md`（§4.6 / §7.4.2 / §八 M2）
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`

> 本文只讲 M2 新增的东西：做了什么、为什么、怎么跑怎么验、决策门 D1 怎么拍板、M3 注意事项。M1 已覆盖的（装环境、下数据集）只引用不重写。

---

## 一、M2 做了什么

```
examples/cfd/vortex_shedding_mgn/
├── amr_m4gn/
│   ├── __init__.py             ✅ 改：导出 physics_ops 的 4 个函数
│   └── physics_ops.py          ✅ 新建：G/ω/M/S + 最小二乘梯度 + 虚拟步 + 反归一化
├── visualize_partition.py      ✅ 改：加 --plot_physics，多出 07_physics_fields.png
└── tests/
    └── test_physics_ops.py     ✅ 新建：8 个 pytest 单测（解析场验证）
```

M2 目标（Design Doc §八 M2）：**算对 4 个物理判据，能区分活跃区（尾迹/壁面）与平静区（来流）**，为 M3 的 AMR Router 提供输入。**不涉及训练。**

（M1 已建的 `modal_decomp.py`/`segmentation.py` 本阶段未改。）

---

## 二、`amr_m4gn/physics_ops.py` 详解

对应 Design Doc **§4.6**。

| 函数 | 签名 | 作用 |
| --- | --- | --- |
| `lstsq_gradient` | `(field[N,C], pos[N,2], edge_index[2,E], eps=1e-8) -> grad[N,C,2]` | 1-ring 加权最小二乘梯度（`[...,0]=d/dx, [...,1]=d/dy`）|
| `compute_ns_quantities` | `(u[N], v[N], pos[N,2], edge_index, area=None, rho=1.0, eps=1e-8, vel_mean=None, vel_std=None) -> {"G","omega","M","S"}` | 一次性算 4 个判据 |
| `virtual_step` | `(uv_t[N,2], uv_prev[N,2]或None) -> uv'[N,2]` | 前向欧拉 `uv'=uv_t+(uv_t−uv_prev)`（AMR-Transformer Eq.11）|
| `denormalize_velocity` | `(u, v, vel_mean[2], vel_std[2]) -> (u_phys, v_phys)` | 训练期把归一化速度还原为物理量（D1）|

**四个判据公式（与 Design Doc §4.6 一致）**：
- `G = √(du_dx² + du_dy² + dv_dx² + dv_dy²)` —— 速度梯度幅值（间断/急变）
- `omega = dv/dx − du/dy` —— 涡量（速度梯度的**反对称**部分，旋转）
- `S = √(2·du_dx² + 2·dv_dy² + (du_dy+dv_dx)²)` —— 应变率幅值 √(2·S_ij·S_ij)（**对称**部分，剪切/拉伸）
- `M = ρ·|U|·area` —— 动量指示量（`area` 来自 M1 的 `compute_node_area`）

> **⚠️ S 定义已修正（vs AMR-Transformer 原文）**：原文 `S=∂u/∂y−∂v/∂x` **恒等于 −ω**，Router 取 |·| 后与涡量完全冗余，4 判据塌成 3。改用应变率幅值 `√(2·S_ij·S_ij)`（速度梯度对称部分），与 ω 严格独立且恒 ≥ 0。验证：纯剪切 `u=y` → ω=−1、S=1（区分开）；刚体旋转 `u=−y,v=x` → ω=2、S=0（应变为零）。

**关键设计决策（为什么这么做）**：

1. **为什么用 1-ring 最小二乘而非差分**：非结构三角网格没有规则 stencil，AMR-Transformer 原文的四叉树差分用不了。最小二乘对每个节点解 `(ΣΔxΔxᵀ)∇f=(ΣΔxΔf)`，是任意非结构网格上的标准一致梯度估计，且**对线性场精确**（见 §四 验证误差 1e-6）。
2. **为什么加 `eps·I` 正则**：边界/角点邻居少或共线时，2×2 法矩阵奇异；正则保证 `torch.linalg.solve` 不出 NaN（单测 `test_no_nan_tiny_graph` 覆盖）。
3. **为什么用 `index_add_` 而非 torch_scatter**：`index_add_` 是 torch 核心算子，零额外依赖，按节点聚合每条边的外积贡献，等价 scatter-add。
4. **为什么留 `vel_mean/vel_std` 参数（D1 核心）**：可视化用的是物理速度（直读 TFRecord），但**训练时 `graph.x[:,:2]` 是归一化的**；归一化空间算出的 ω 无物理意义、阈值不可解释。训练期须传 `node_stats.json` 的均值/方差先反归一化。详见 §五。

---

## 三、`visualize_partition.py` 的改动

- 新增 `--plot_physics` 开关（**默认关，不影响 M1 既有 6 图流程**）。
- 新增 `--timestep` 参数（默认 0）：选轨迹的哪一帧取速度场。**t=0 是未发展的初始流，没有涡街**；要看 Kármán 脱涡须用后期帧（如 `--timestep 300`）。网格几何是静态的，pos/cells/node_type 与时间无关。
- `--plot_physics` 开启后：`compute_node_area` 算面积 → `compute_ns_quantities`（物理速度，无需反归一化）→ 出 `07_physics_fields.png`（2×2：G/ω/M/S）；终端打印每个量的 `min/max/|·|p99`。
- 配色：ω 用对称发散色标 `RdBu_r`（有正负号）；G/M/S 恒 ≥ 0 用 `viridis`；均按 p99 截断防离群值压扁色阶。
- **已实现 2 个 M1 遗留告警的修复（尚未实跑确认）**：① `modal_decomp` 把 L、M 统一转 float64，意图消除 ARPACK `M does not have the same type precision` 告警；② 分区配色改用 `matplotlib.colormaps[name].resampled(K)`，意图消除 `get_cmap` 弃用告警。**这两处改动你还没重跑过，是否真的消除告警需你重跑确认。**
- **新增「终端输出保存为纯文本」功能**（方便你在另一台机器跑完后直接把日志文件发我，免去复制粘贴）：
  - 默认开启：所有 `print` 内容在显示到终端的同时，写入 `<output_dir>/run_log_case<case_idx>_t<timestep>.txt`（如 `partition_vis/run_log_case0_t300.txt`）。
  - 实现方式：一个 `_Tee` 类把 `sys.stdout` 同时指向真实终端和日志文件，每次写入即 flush（**脚本中途崩溃也能留下已产生的日志**），并用 `atexit` 保证退出时关闭文件。
  - 新参数：`--log_file <路径>` 自定义日志文件名/位置；`--no_log` 关闭该功能。

---

## 四、单元测试 `tests/test_physics_ops.py`

对应 Design Doc **§7.4.2**。在**规则三角网格 + 解析速度场**上验证，**不依赖数据集**。

| 测试 | 场 | 期望 | 容差 |
| --- | --- | --- | --- |
| `test_linear_gradient_exact` | f=3x+2y | ∇f=(3,2) | <1e-3 |
| `test_shear_flow_vorticity` | u=y,v=0 | ω=−1, **S=1**, G=1（ω≠S，验证独立）| <5e-2 |
| `test_rotation_vorticity` | u=−y,v=x | ω=2, **S=0**（纯旋转无应变）| <5e-2 |
| `test_uniform_flow_zero_gradient` | u=1,v=0 | G≈ω≈S≈0 | <1e-3 |
| `test_momentum_indicator` | u=3,v=4,area=1 | M=5 | <1e-4 |
| `test_denormalize_velocity` | — | phys=norm·std+mean | 精确 |
| `test_virtual_step` | — | uv+(uv−prev) | 精确 |
| `test_no_nan_tiny_graph` | 3 节点退化图 | 全有限 | — |

**关于验证程度（请注意区分）**：
- 我只在开发机用**纯 numpy 复刻了梯度/应变公式**做离线数学校验（**不是仓库的 torch 代码，也不是 pytest**），结果：linear 1.08e-5、shear ω=−1/S=1/G=1（误差 1e-6）、rotation ω=2/S=0（误差<2e-6）。这只能说明**公式推导对**。
- 仓库里 `physics_ops.py`（torch 版）与更新后的 `test_physics_ops.py` **尚未实跑 pytest**（开发机 `base` Python 无 torch）。
- 因此现状是「**已实现，未经实跑测试**」。需你在装好 torch 的环境跑 `pytest` 后才能确认是否真的全过。
- 你上一轮跑过的 8/8 PASS 是**改 S 公式之前**的版本；本轮改了 S 公式与 2 个测试，**必须重跑**。

---

## 五、🚩 决策门 D1：物理量是否反归一化（暂定，未最终确认）

**问题**：训练时 `u,v` 归一化，算物理量前要不要用 `node_stats.json` 反归一化？

**已有的真实数据（你上一轮 4 个 case 的实测，t=0）**：`|ω|p99` = case0 359、case1 253、case50 360、case99 142 /s；与边界层涡量估计 `ω~U/δ≈1.5/0.004≈375 /s` 同数量级，跨 case 同量级。**这些是你实跑出来的真实数字。**

**暂定结论（尚未最终确认）**：倾向**训练期反归一化**（`compute_ns_quantities` 传 `vel_mean/vel_std`）。理由：① D2 倾向用绝对阈值，前提是物理量纲一致、跨 case 可比；② 归一化把 u、v 各自缩放到不同尺度，会扭曲 ω 的相对大小。

**为什么还不能算最终确认**：上一轮量级数据取自 **t=0 未发展流**，尚未看到涡街已发展时（`--timestep 300`）的 ω 量级与空间结构是否仍然合理、是否真能区分活跃/平静区。**请你跑完 §六 步骤 3（带 `--timestep`）后再回填最终结论。** 在那之前 U4 不算闭环。

---

## 六、运行步骤（每步：做什么 → 得到什么 → 为什么）

> 前提：M1 的环境与数据集已就绪（见 M1 文档）。M2 唯一可能新增的依赖是 `pytest`（没有就 `pip install pytest`，或用纯 Python 跑法）。

### 步骤 0 — 同步 M2 改动的文件（你在 `C:\GitHub`，我改在另一路径）

- **做什么**：确认下列 5 个文件是 M2 最新版（新建的存在、修改的已更新）：

  | 文件 | 操作 |
  | --- | --- |
  | `amr_m4gn/physics_ops.py` | 新增 |
  | `amr_m4gn/__init__.py` | 修改（加 `from .physics_ops import ...`）|
  | `amr_m4gn/modal_decomp.py` | 修改（L、M 转 float64）|
  | `visualize_partition.py` | 修改（`--plot_physics`、`--timestep`、配色 API、终端日志保存 `--log_file`/`--no_log`）|
  | `tests/test_physics_ops.py` | 新增（注意 S 测试本轮改过）|

- **得到什么**：5 个文件齐全且为最新。
- **为什么**：上次 `cannot import name compute_ns_quantities` 就是 `__init__.py` 没同步导致的。**先核对再跑，避免重复踩这个坑。**

### 步骤 1 — 包导入自检

- **做什么**：
  ```bash
  cd <你的仓库>\examples\cfd\vortex_shedding_mgn
  python -c "from amr_m4gn import compute_ns_quantities, lstsq_gradient, virtual_step; print('physics_ops OK')"
  ```
- **得到什么**：打印 `physics_ops OK`。
- **为什么**：先确认新模块能被包正确导出、没有 import 链错误。**若报 `cannot import name ... from 'amr_m4gn'`**：说明你的 `amr_m4gn/__init__.py` 没更新（缺 `from .physics_ops import ...`），补上即可——这是文件没同步导致，不是代码错。

### 步骤 2 — 跑单元测试（不依赖数据集，先跑这个）

- **做什么**：
  ```bash
  pytest tests/test_physics_ops.py -v
  # 没装 pytest 时等价跑法：
  python tests/test_physics_ops.py
  ```
- **得到什么**：8 个测试全部 `PASS`（pytest 显示 `8 passed`）。
- **为什么**：物理量算子的正确性**不需要真实数据**，先用解析场（已知答案的 shear/rotation/uniform 流）确认梯度、涡量、应变、动量、反归一化、虚拟步全对。先测算子、再上真实网格看图，出问题能立刻分清是「算子错」还是「数据/可视化错」。

### 步骤 3 — 在真实 case 上出物理量图（**务必带 `--timestep`**）

- **做什么**：
  ```bash
  # t=0 是未发展的初始流，看不到涡街；用后期帧看 Karman 脱涡：
  python visualize_partition.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test --case_idx 0 --timestep 300 --plot_physics
  ```
- **得到什么**：
  - 照常产出 `01`~`06` 图 + **新增 `07_physics_fields.png`**（2×2：G/ω/M/S）；
  - 终端 `[1/6]` 行显示 `(timestep 300)`，并打印 4 个量的 `min/max/|·|p99`；
  - **新增**：终端全部输出自动存到 `partition_vis/run_log_case0_t300.txt`（脚本第一行会打印该日志路径）。**你在另一台机器跑完后，把这个 txt 发我即可，不用复制粘贴终端。**
  - ARPACK / get_cmap 告警预期已消失（M2 已实现修复，**以这次重跑的终端/日志为准**）。
- **为什么**：上次用默认 t=0 看不到尾迹（验收点 B/C/D 看似不过）——根因是初始流未发展，不是算子错。带 `--timestep 300` 后涡街已发展，ω 子图应出现圆柱后方红蓝相间的脱涡。

> **关于日志文件**：默认开启，文件名按 `case_idx`/`timestep` 自动区分（多次跑不同参数不会互相覆盖日志）。可用 `--log_file <路径>` 自定义，或 `--no_log` 关闭。

### 步骤 4 — 多看几个 case / 多个时间步（可选但建议）

- **做什么**：换 `--case_idx 1/50/99`、或同一 case 换 `--timestep 100/200/300` 各跑一次（图会覆盖，但日志按参数分文件不覆盖）。
- **得到什么**：不同 case/时刻的 ω 量级与尾迹形态，以及对应的多份 `run_log_*.txt`。
- **为什么**：确认物理量量级跨 case 稳定（呼应 D2 的「绝对阈值」前提）；若某 case 量级差一个数量级，说明阈值不能用全局绝对值，要回到 Top-r 分位方案。

---

## 七、验收清单（M2 退出标准）

| 验收点 | 看哪里 | 合格判据 | 为什么 |
| --- | --- | --- | --- |
| **A. 单测** | `pytest` 输出 | 8/8 PASS（**本轮改了 S，须重跑**）| 算子在解析场上正确 |
| **B. 涡量结构** | `07` 的 ω 子图（**须 `--timestep 300`**）| 圆柱后方涡街清晰、上下两侧异号（红蓝相间）、来流区≈0 | ω 能区分活跃/平静（M3 核心输入）|
| **C. 梯度 G** | `07` 的 G 子图 | 壁面/圆柱表面、剪切层处高，来流低 | G 捕捉边界层与间断 |
| **D. 应变 S** | `07` 的 S 子图 | 尾迹剪切层/涡街两侧显著、纯旋转涡核处相对低 | S 捕捉剪切/拉伸，且**与 ω 不同**（独立判据）|
| **E. 量级合理（D1）** | 终端 `omega |·|p99` | 物理速度下量级合理、跨 case 同量级（t=0 已实测，发展态待确认）| 用于最终拍板 D1 |
| **F. 无残留告警** | 终端 | 不再出现 ARPACK / get_cmap 告警 | 确认 M2 的告警修复真的生效（你尚未重跑过）|

> **重要**：B/C/D 必须用 **`--timestep 300`**（或其它涡街已发展的帧）评估。默认 t=0 是未发展初始流，看不到尾迹，**不代表算子错**。

**M2 通过标准（待你实跑达成）**：A~F 全部满足 + D1 在发展态下确认 → 才能关闭 M2、进入 M3。**目前代码均为「已实现、未经你实跑测试」状态。**

---

## 八、与 Design Doc 的对应关系

| Design Doc | M2 落地 |
| --- | --- |
| §4.6 N-S 物理量算子（含 S 改用应变率幅值的修正说明）| `physics_ops.py`（已实现，未测试通过）|
| §7.4.2 physics_ops 规格 + 单元测试 | `physics_ops.py` + `tests/test_physics_ops.py`（待重跑）|
| §八 M2（四标量场可视化、解析场单测 <5%）| `--plot_physics` + 测试 |
| 决策门 D1 / 不确定点 U4 | §五（暂定反归一化，未最终确认）|

**M2 未覆盖（属后续）**：
- 虚拟步在可视化里未启用（无 t−1 历史；M3/M5 rollout 时接入）。`virtual_step` 函数已写好，但未在真实流程中跑过。
- 物理量 → 段聚合 → 保留/折回决策 = **M3 的 AMR Router**。

---

## 九、M3 注意事项（下一步，待 M2 实跑通过后再开始）

1. **新建 `amr_m4gn/amr_router.py` + `amr_m4gn/pe.py`**（Design Doc §7.4.1/§7.4.3）。
2. **D2 暂定用绝对阈值**（M1 实测 4 case 坐标全等）；**D1 暂定阈值作用在反归一化后的物理量上**（均待 M2 发展态确认）。
3. **复用 M2 的 `compute_ns_quantities`**：M3 在 L1（256 段）上对每段做 `max|ω|`、`max(G)`、`max(S)`、`max(M)` 聚合，再按各自阈值判活跃。注意 ω 取 `max|·|`（有正负），G/M/S 恒 ≥ 0 直接 `max`。
4. **盯住 M1 的「流向带状」隐患**（Progress_M1 §7.3.1.1）：M3 出 token 分布统计（决策门 D3）时，重点看尾迹段是否「一段跨高涡量+平静区」；若浪费明显，调 num_modes/K/τ。
5. **M2 已实现 M1 两个告警的修复，但需你重跑确认生效**（ARPACK dtype + matplotlib colormap）。

**M3 退出标准**（Design Doc §八 M3）：四例路由单测全过（全平静→T=64；全活跃→T=256；半场→中间值；阈值可复现）；可视化中尾迹保细、来流被合并；产出 T 分布统计（喂 D3）。

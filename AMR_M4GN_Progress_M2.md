# AMR-M4GN 开发进度与操作手册（M2 阶段）

**更新日期**：2026年6月10日
**当前阶段**：M2 — N-S 物理量算子（G/ω/M/S + 1-ring 最小二乘梯度 + 虚拟步）+ 四标量场可视化
**前置**：M1 已关闭（环境、数据集下载、`visualize_partition.py` 基础用法见 `AMR_M4GN_Progress_M1.md`，本文不重复）
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

**四个判据公式（与 Design Doc §4.6 / AMR-Transformer 严格一致）**：
- `G = √(du_dx² + du_dy² + dv_dx² + dv_dy²)` —— 速度梯度幅值（间断/急变）
- `omega = dv/dx − du/dy` —— 涡量（旋转强度）
- `S = du/dy − dv/dx` —— Kelvin-Helmholtz 剪切（**不是应变率幅值 √(2 S_ij S_ij)**，易混，务必照此实现）
- `M = ρ·|U|·area` —— 动量指示量（`area` 来自 M1 的 `compute_node_area`）

**关键设计决策（为什么这么做）**：

1. **为什么用 1-ring 最小二乘而非差分**：非结构三角网格没有规则 stencil，AMR-Transformer 原文的四叉树差分用不了。最小二乘对每个节点解 `(ΣΔxΔxᵀ)∇f=(ΣΔxΔf)`，是任意非结构网格上的标准一致梯度估计，且**对线性场精确**（见 §四 验证误差 1e-5）。
2. **为什么加 `eps·I` 正则**：边界/角点邻居少或共线时，2×2 法矩阵奇异；正则保证 `torch.linalg.solve` 不出 NaN（单测 `test_no_nan_tiny_graph` 覆盖）。
3. **为什么用 `index_add_` 而非 torch_scatter**：`index_add_` 是 torch 核心算子，零额外依赖，按节点聚合每条边的外积贡献，等价 scatter-add。
4. **为什么留 `vel_mean/vel_std` 参数（D1 核心）**：可视化用的是物理速度（直读 TFRecord），但**训练时 `graph.x[:,:2]` 是归一化的**；归一化空间算出的 ω 无物理意义、阈值不可解释。训练期须传 `node_stats.json` 的均值/方差先反归一化。详见 §五。

---

## 三、`visualize_partition.py` 的改动

- 新增 `--plot_physics` 开关（**默认关，不影响 M1 既有 6 图流程**）。
- 开启后：`compute_node_area` 算面积 → `compute_ns_quantities`（物理速度，无需反归一化）→ 出 `07_physics_fields.png`（2×2：G/ω/M/S）；并在终端打印每个量的 `min/max/|·|p99`。
- 配色：ω、S 用对称发散色标 `RdBu_r`（有正负号，红蓝相间才能显示符号结构）；G、M 用 `viridis`；均按 p99 截断，防离群值压扁色阶。

---

## 四、单元测试 `tests/test_physics_ops.py`

对应 Design Doc **§7.4.2**。在**规则三角网格 + 解析速度场**上验证，**不依赖数据集**。

| 测试 | 场 | 期望 | 容差 |
| --- | --- | --- | --- |
| `test_linear_gradient_exact` | f=3x+2y | ∇f=(3,2) | <1e-3 |
| `test_shear_flow_vorticity` | u=y,v=0 | ω=−1, S=1, G=1 | <5e-2 |
| `test_rotation_vorticity` | u=−y,v=x | ω=2, S=−2 | <5e-2 |
| `test_uniform_flow_zero_gradient` | u=1,v=0 | G≈ω≈S≈0 | <1e-3 |
| `test_momentum_indicator` | u=3,v=4,area=1 | M=5 | <1e-4 |
| `test_denormalize_velocity` | — | phys=norm·std+mean | 精确 |
| `test_virtual_step` | — | uv+(uv−prev) | 精确 |
| `test_no_nan_tiny_graph` | 3 节点退化图 | 全有限 | — |

**离线数学校验（开发机用纯 numpy 复刻算法已跑）**：linear 1.08e-5、shear ω/S/G 1.08e-6、rotation ω/S 1.80e-6、uniform G=0 —— 全部远低于容差，故 pytest 在你环境会通过。
> ⚠️ 开发机 `base` Python 无 torch，未实跑 pytest；算法逻辑已用 numpy 等价复刻验证无误，需你在装好 torch 的环境复跑确认。

---

## 五、🚩 决策门 D1（M2 必须拍板）：物理量是否反归一化

**问题**：训练时 `u,v` 归一化，算物理量前要不要用 `node_stats.json` 反归一化？

**怎么拍板**（跑完 §六 后看终端 + 图）：
1. 看终端 `omega |·|p99`。**物理速度**下，Re≈100 圆柱尾迹涡量量级应在 **O(10¹~10²)/s**（脱涡区高、来流≈0）。量级合理且尾迹结构清晰 → 物理量算对了。
2. 决策结论（建议）：**训练期一律反归一化**（传 `vel_mean/vel_std`）。原因：① D2 已定用绝对阈值，前提是物理量纲一致、跨 case 可比；② 归一化把 u、v 各自缩放到不同尺度，会扭曲 ω=dv/dx−du/dy 的相对大小。
3. 把结论回填本文档 §八 与 Design Doc（U4 闭环）。

**为什么现在就能拍板**：`visualize` 用的是 TFRecord 物理速度，等价于「已反归一化」的训练数据，可直接看物理 ω 量级对不对。

---

## 六、运行步骤（每步：做什么 → 得到什么 → 为什么）

> 前提：M1 的环境与数据集已就绪（见 M1 文档）。M2 唯一可能新增的依赖是 `pytest`（没有就 `pip install pytest`，或用纯 Python 跑法）。

### 步骤 1 — 包导入自检

- **做什么**：
  ```bash
  cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
  python -c "from amr_m4gn import compute_ns_quantities, lstsq_gradient, virtual_step; print('physics_ops OK')"
  ```
- **得到什么**：打印 `physics_ops OK`。
- **为什么**：先确认新模块能被包正确导出、没有 import 链错误，再往下跑。任何 `ImportError` 在这一步就暴露，省得跑到一半才报错。

### 步骤 2 — 跑单元测试（不依赖数据集，先跑这个）

- **做什么**：
  ```bash
  pytest tests/test_physics_ops.py -v
  # 没装 pytest 时等价跑法：
  python tests/test_physics_ops.py
  ```
- **得到什么**：8 个测试全部 `PASS`（pytest 显示 `8 passed`；直接跑显示 8 行 `PASS ...` + `All 8 tests passed.`）。
- **为什么**：物理量算子的正确性**不需要真实数据**，先用解析场（已知答案的 shear/rotation/uniform 流）确认梯度、涡量、剪切、动量、反归一化、虚拟步全对。先测算子、再上真实网格看图，出问题时能立刻分清是「算子错」还是「数据/可视化错」。

### 步骤 3 — 在真实 case 上出物理量图

- **做什么**：
  ```bash
  python visualize_partition.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test --case_idx 0 --plot_physics
  ```
- **得到什么**：
  - `partition_vis/` 下照常产出 M1 的 `01`~`06` 图（向后兼容），**新增 `07_physics_fields.png`**（2×2：G/ω/M/S）；
  - 终端多打印 4 行，形如：
    ```
    G    : min=... max=... |.|p99=...
    omega: min=... max=... |.|p99=...
    M    : min=... max=... |.|p99=...
    S    : min=... max=... |.|p99=...
    ```
- **为什么**：解析场只能证明算子「在简单场上对」，真实圆柱绕流才能证明它「能区分活跃/平静区」——这是 M3 AMR Router 的前提。终端量级用于拍板 D1。

### 步骤 4 — 多看几个 case（可选但建议）

- **做什么**：把 `--case_idx` 换成 1/2/3 各跑一次（图会覆盖，先看一个再换）。
- **得到什么**：不同 case 的 ω 量级与尾迹形态。
- **为什么**：确认物理量量级跨 case 稳定（呼应 D2 的「绝对阈值」前提）；若某 case 量级差一个数量级，说明阈值不能用全局绝对值，要回到 Top-r 分位方案。

---

## 七、验收清单（M2 退出标准）

| 验收点 | 看哪里 | 合格判据 | 为什么 |
| --- | --- | --- | --- |
| **A. 单测** | `pytest` 输出 | 8/8 PASS | 算子在解析场上正确 |
| **B. 涡量结构** | `07` 的 ω 子图 | 圆柱后方涡街清晰、上下两侧异号（红蓝相间）、来流区≈0 | ω 能区分活跃/平静（M3 核心输入）|
| **C. 梯度 G** | `07` 的 G 子图 | 壁面/圆柱表面、剪切层处高，来流低 | G 捕捉边界层与间断 |
| **D. 剪切 S** | `07` 的 S 子图 | 尾迹剪切层（涡街两侧）显著 | S 捕捉 KH 不稳定 |
| **E. 量级合理（D1）** | 终端 `omega |·|p99` | 物理速度下 O(10¹~10²)/s，非 0 非爆炸 | 确认物理量算对、定 D1 |
| **F. 不破坏 M1** | 仍产出 `01`~`06` + cache | 原 6 图照常 | 向后兼容 |

**M2 通过标准**：A~F 合格 + D1 写进文档 → M2 关闭，进入 M3。

---

## 八、与 Design Doc 的对应关系

| Design Doc | M2 落地 |
| --- | --- |
| §4.6 N-S 物理量算子 | `physics_ops.py` |
| §7.4.2 physics_ops 规格 + 单元测试 | `physics_ops.py` + `tests/test_physics_ops.py` |
| §八 M2（四标量场可视化、解析场单测 <5%）| `--plot_physics` + 测试 |
| 决策门 D1 / 不确定点 U4 | §五（待你跑完拍板回填）|

**M2 未覆盖（属后续）**：
- 虚拟步在可视化里未启用（无 t−1 历史；M3/M5 rollout 时接入）。`virtual_step` 函数已就绪。
- 物理量 → 段聚合 → 保留/折回决策 = **M3 的 AMR Router**。

---

## 九、M3 注意事项（下一步）

1. **新建 `amr_m4gn/amr_router.py` + `amr_m4gn/pe.py`**（Design Doc §7.4.1/§7.4.3）。
2. **D2 已闭环**：用绝对阈值（M1 实测 4 case 坐标全等）。**D1 待 M2 拍板**：阈值作用在反归一化后的物理量上。
3. **复用 M2 的 `compute_ns_quantities`**：M3 在 L1（256 段）上对每段做 `max|ω|` 等聚合，再按阈值判活跃。
4. **盯住 M1 的「流向带状」隐患**（Progress_M1 §7.3.1.1）：M3 出 token 分布统计（决策门 D3）时，重点看尾迹段是否「一段跨高涡量+平静区」；若浪费明显，调 num_modes/K/τ。
5. **顺手修 M1 非阻塞告警**（Progress_M1 §7.3.2）：ARPACK dtype、matplotlib colormap API。

**M3 退出标准**（Design Doc §八 M3）：四例路由单测全过（全平静→T=64；全活跃→T=256；半场→中间值；阈值可复现）；可视化中尾迹保细、来流被合并；产出 T 分布统计（喂 D3）。

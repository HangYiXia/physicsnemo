# AMR-M4GN：结合空间划分与多尺度 Transformer 的大规模流体仿真代理模型

## 研究设计文档（开题对应版）

**作者**：储润东（124037910068）
**日期**：2026年6月10日（修订）
**仓库**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`
**主线 Baseline**：PhysicsNeMo MeshGraphNet (MGN) 圆柱绕流训练脚本
**对比 Baseline**：MGN、X-MeshGraphNet (X-MGN)、M4GN、节点级 Graph-Transformer

> **里程碑总进度**（截至 2026-06-11）：M1 ✅ · M2 ✅ · M3 ✅ · M4 🟢 管线跑通（overfit 收敛度待调参）· M5 ⬜ · M6 ⬜ · M7 ⬜
> **文档索引**：本设计文档 `AMR_M4GN_Design_Doc.md`；阶段手册 `AMR_M4GN_Progress_M1.md`〜`_M4.md`（各含「拿到代码后每步做什么/为什么/应得什么」的操作说明）。

> **一句话定位**：在非结构三角网格上，用「局部 GNN（短程、高频物理）+ 段级 Transformer（全局、长程压力耦合）+ 物理驱动的自适应 Token 化（AMR）」三者融合的混合架构，在几乎不增加计算量的前提下解决 MeshGraphNet 的长程依赖丢失与过平滑问题，并为大规模湍流（EAGLE）提供可扩展路径。

---

## 一、研究背景与问题

### 1.1 从 CNN 到 GNN 再到 Transformer 的演进

现代图形学与物理引擎对流体仿真的视觉保真度要求极高，逼真的水花、激波、湍流细节往往需要**数百万甚至上亿**级别的网格或粒子。传统数值方法在面对**大规模、高雷诺数**湍流时陷入「精度-成本」的零和博弈：细节要求提升时，计算复杂度指数级爆炸，单帧离线解算可能耗时数小时甚至数天。

深度学习代理模型的演进路线：

| 阶段 | 代表方法 | 能力 | 根本局限 |
| --- | --- | --- | --- |
| CNN | UNet 风格 | 规则笛卡尔网格上加速 | 只能处理规则网格；复杂边界需「体素化/插值」，引入巨大数值误差，丢失几何边缘与碰撞细节 |
| GNN | MeshGraphNet (DeepMind 2020) | 直接处理非结构网格与粒子，完美贴合任意几何边界 | 消息传递逐跳扩展感受野，长程依赖需极深网络；过平滑 |
| 多尺度 GNN | X-MeshGraphNet (NVIDIA 2024) | METIS 分区 + Halo + 梯度聚合，突破显存墙；多尺度图扩大感受野 | **只优化了长程交互的「开销」，并未真正解决长程消息传递问题** |
| Transformer | EAGLE (ICLR 2023)、AMR-Transformer (CVPR 2025) | 自注意力全局感受野，一跳建立长程联系 | 节点级注意力 O(N²) 显存/算力爆炸 |

### 1.2 MeshGraphNet 的核心机制与瓶颈

MGN 采用 **Encoder–Processor–Decoder (EPD)** 结构：

- **Encoder**：物理量编码为高维特征（如 128 维）。节点特征 = 速度 (u, v) + 节点类型 one-hot；边特征 = 相对坐标 (dx, dy) + 距离 ‖d‖。
- **Processor**：L 步消息传递。L 层意味着每个节点只能聚合其 **L 跳（L-hops）** 邻居信息。
- **Decoder**：将状态解码回物理量增量。

**三大瓶颈**（与开题 PPT「研究意义及目标」一致）：

1. **长程特征丢失**：`processor_size=15` 时最远只覆盖 15 跳邻居。圆柱绕流中压力波的全局传播（上游压力变化对下游尾迹的影响）、湍流中的涡-涡远程相互作用，需要远超 15 步的传递。**对压力场这类「全局瞬时更新」的物理量尤其致命**——在水体仿真中某处压力突然增大时，远距离水体若毫无响应即为物理错误。
2. **过平滑 (over-smoothing)**：每步消息传递通常做聚合（如 sum/mean），等价于一次低通滤波。远程物理量互相影响需极深网络，不仅计算昂贵（O(L·E·d²)），还会使深层节点嵌入趋于同质化，丧失局部高频细节。
3. **缺乏自适应性**：MGN 对所有区域一视同仁——来流均匀区与尾迹涡街区投入相同计算量，造成资源浪费。

### 1.3 X-MeshGraphNet 的贡献与未竟之处（定位为对比 Baseline）

X-MGN 针对 MGN 的**可扩展性**做了三点优化：

1. **图分区 + Halo + 梯度聚合**：用 METIS 把大图切成含 Halo（光环）重叠区的子图，Halo 厚度 = 消息传递层数，配合梯度聚合，使「分块训练」在数学上**严格等价于**整图一次性处理，从而突破单卡显存墙（解决网格尺寸增大带来的二次方显存开销）。
2. **免网格推理**：直接从 STL 几何生成点云图（k-NN 连边），消除推理时高质量网格生成的速度瓶颈。
3. **多尺度图生成**：迭代组合粗/细分辨率点云，每层是上一层超集，无需复杂上/下采样层即可高效长程传播。

**但 X-MGN 的本质仍是 GNN 消息传递**：它优化了长程交互的「显存开销」，却**没有改变长程信息逐跳传递、必然产生「延迟」的事实**。对压力场这种全局耦合量依旧会产生误差。本工作因此把 X-MGN 列为**对比 Baseline**，而非架构基座——我们用 Transformer 的全局自注意力**从机理上**解决长程交互。

### 1.4 本工作要解决的核心问题

> 如何在**非结构三角网格**上，既保留 GNN 对局部高频物理（黏性、局部涡、对流）的精确建模能力，又获得 Transformer 的全局感受野来建模长程压力耦合，同时把 Transformer 的 O(N²) 复杂度压到可接受范围？

---

## 二、关键参考文献与启发

### 2.1 M4GN (TMLR 2025)：层次化分区骨架

- **论文**：Lei et al., "M4GN: Mesh-based Multi-segment Hierarchical Graph Network for Dynamic Simulations."
- **三层架构**：Micro-level（局部 GNN 消息传递）+ Intermediate-level（混合网格分区，离线）+ Macro-level（Segment Transformer 段级全局交互）。
- **关键贡献**：
  - **混合分区**：METIS 粗分（保证连通性、最小化边切割、速度快）+ SLIC 超像素精修（基于模态分解特征，保证几何保真与物理一致）。Table 1 论证其在 Heuristic / Contiguity / Geometric Fidelity / Physics-Aware 四项全面优于 learnable pooling、spatial proximity pooling、Bi-Stride、same-size k-means。
  - **置换不变聚合器**：用 average-pool + MLP 替代 EAGLE 的 GRU，复杂度从 O(Nd²) 降到 O(Nd)，且无顺序敏感、无长序列信息稀释、无梯度消失。
  - **段级 Transformer**：在 K 个段（K≪N）上做全连接自注意力，O(K²)≪O(N²)。
  - **位置编码**：段级 RWSE（随机游走结构编码）+ 节点级绝对 PE（注入 Encoder 输入）。
  - **重叠分区 δ=1**：允许段间重叠一圈邻居，平滑段边界、减少不连续（消融见其 Table 9）。
- **默认配置**：Micro-level 7 步消息传递；Segment Transformer 4 层 8 头。精度提升最高 56%，推理加速最高 22%。
- **局限**：Segment 数量 K **固定**，不能根据物理状态动态调整——平静区与活跃区分到同等数量的段，仍有冗余。

### 2.2 AMR-Transformer (CVPR 2025)：物理驱动的动态 Token 化

- **论文**：Xu et al., "AMR-Transformer: Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation."
- **核心思想**：用**自适应网格细化 (AMR)** 作为 Tokenizer——基于多叉树（2D 四叉树/3D 八叉树），自顶向下逐层 (depth) 细分；用 **Navier-Stokes 约束感知的快速剪枝模块**决定哪些区域细分保留、哪些合并粗化；合并后用 Encoder-only Transformer 做全局自注意力。
- **四个物理判据**（逐 cell 计算）：速度梯度 G、涡量 ω、动量 M、Kelvin-Helmholtz 剪切 S。
- **阈值采样**：训练时阈值因子 `t = {tG, tω, tM, tS}` 在预定义区间均匀随机采样；测试时手动固定以平衡精度/效率。速度梯度额外用 **Top-r 百分位**机制（取当前 depth 分布的前 r 分位触发细分）。
- **虚拟步外推**：用前向欧拉 `u'_{t+Δt} = u_t + (u_t − u_{t−Δt})` 估计虚拟速度场，与当前场**并集**决定细分区域，提前细化「即将变活跃」的区域。
- **损失**：NMSE（归一化均方误差），保证不同物理量尺度一致；标签也经 AMR Tokenizer 处理为多尺度表示。
- **效果**：Token 数减少 2~10 倍，因 self-attention O(N²)，FLOPs 最高减少 60 倍；CFDBench 上精度提升一个量级。
- **局限**：仅适用于**结构化网格** (H×W×c) 的四叉树，不能直接用于非结构三角网格。

### 2.3 EAGLE (ICLR 2023)：大规模湍流数据集与 Mesh Transformer

- **论文**：Janny et al., "EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers."
- **数据集**：~110 万个 2D 非结构网格快照，600 个不同场景，3 种类型；由移动气流源（2D 无人机）与非线性场景结构交互产生高度湍流、非周期涡。**作为本工作大规模扩展验证的目标数据集**。
- **模型**：节点聚类 + 图池化 + 全局注意力，用 GRU 聚合段特征（M4GN 正是用置换不变 pooling 改进了它）。

### 2.4 三者关系与本工作的融合点

```
   MGN (局部精确, 长程弱)        X-MGN (可扩展, 长程仍弱)  ← 对比 Baseline
            │                              │
            └──────────── 局部建模骨架 ─────┘
                          │
   M4GN (层次化分区 + 段级Transformer, 段数固定)  ← 分区骨架 + 段级全局交互
                          │
   AMR-Transformer (物理驱动动态Token, 仅结构化网格)  ← 动态自适应 Token 化
                          │
                          ▼
            AMR-M4GN（本工作）：非结构网格 + 动态层次 Token + GNN/Transformer 互补
```

---

## 三、本工作创新点

**核心思路**：把 M4GN 的「层次化分区骨架」+ AMR-Transformer 的「物理感知动态 Token 化」+ MGN 的「15 步深度局部 GNN」融合为统一框架，并将 AMR-Transformer 受限于结构化四叉树的思想**迁移到非结构三角网格的层次 METIS 分区树**上。

| 对比维度 | MGN | X-MGN | M4GN | AMR-Transformer | **本工作 (AMR-M4GN)** |
| --- | --- | --- | --- | --- | --- |
| 数据类型 | 非结构 | 非结构/点云 | 非结构 | **结构化** | **非结构** |
| 长程机制 | 逐跳消息传递 | 逐跳+多尺度图 | 段级 Transformer | 全局 Transformer | 段级 Transformer |
| Token/Segment 数 | — | — | **固定 K** | 动态（四叉树） | **动态（层次 AMR 决策）** |
| 物理先验 | 无 | 无 | 模态分解（静态） | N-S 物理量（动态） | **模态分解 + N-S 量（静+动）** |
| 局部建模深度 | 15 步 GNN | 15 步 GNN | 7 步 GNN | 无 | **15 步 GNN** |
| 自适应性 | 无 | 无 | 无 | 有 | **有** |

**创新点提炼**（对齐 PPT「创新点」页）：

1. **方法**：提出一种「物理驱动」的 Token 剪枝机制，利用流体力学先验（涡量阈值等）决定计算资源分配，提升效率与可解释性——首次将 AMR-Transformer 的动态剪枝**搬到非结构网格的层次分区树**上。
2. **架构**：融合 GNN 的局部灵活性与 Transformer 的全局感受野，用动态 Token 聚合（而非固定 K）解决大规模扩展性难题。
3. **应用**：无需重新生成网格，而是通过神经网络内部的 Token **动态聚合模拟网格自适应过程**，为实时流体仿真提供新思路。

---

## 四、方法设计

### 4.1 总体架构

整个流程分为**离线预处理**（每个 case 几何固定，只算一次）与**在线前向**（每个时间步执行）两段：

```
                          OFFLINE（每个 case 一次性预处理，写盘缓存）
  ┌────────────────────────────────────────────────────────────────────┐
  │  ① Laplacian Eigenfunctions (m=6 modes, cotangent FEM, Neumann BC)  │
  │  ② 障碍物有符号距离 f_obs（到圆柱壁面）                              │
  │  ③ METIS 粗分 (K0=64) + SLIC 精修 → L0                              │
  │  ④ 递归 METIS：每个 L0 段内切 4 块 → L1 (K1≈256)                    │
  │  ⑤ 段重叠 δ=1：每段并入一圈邻居（平滑段边界）                       │
  │  ⑥ RWSE 段级位置编码（L0/L1 各算一份，16 步随机游走）               │
  │  ⑦ 节点级绝对位置编码（注入 Encoder 输入）                          │
  │  ⑧ 写盘缓存 partition_cache.pt                                      │
  └────────────────────────────────────────────────────────────────────┘

                          ONLINE（每个时间步的前向传播）
  ┌────────────────────────────────────────────────────────────────────┐
  │  graph_t ─► ① Micro GNN (15步 MGN, EPD去掉decoder) ─► h_node [N,d]  │
  │                                                                      │
  │          ─► ② 物理量算子 (G, ω, M, S, 含虚拟步外推) ─► phys [N,4]   │
  │                                                                      │
  │          ─► ③ AMR Router：在 L1 上聚合 phys，                       │
  │                 活跃段保持 L1 细粒度；平静段折回 L0 父段；           │
  │                 → 变长 token 数 T (64 ≤ T ≤ 256)                    │
  │                                                                      │
  │          ─► ④ Segment Encoding：                                    │
  │                 mean-pool h_node per token + RWSE PE + (depth,x̄,ȳ)  │
  │                 → h_seg [T, d]                                       │
  │                                                                      │
  │          ─► ⑤ Macro Transformer (4层×8头, Pre-LN):                  │
  │                 段级全局自注意力 → h_seg' [T, d]                     │
  │                                                                      │
  │          ─► ⑥ Feature Dispatch：                                    │
  │                 h_seg' scatter 回节点 → h_global [N, d]              │
  │                 拼接 h_cat = [h_node, h_global] [N, 2d]              │
  │                                                                      │
  │          ─► ⑦ Decoder MLP → (Δu, Δv, p̂) [N, 3]                    │
  └────────────────────────────────────────────────────────────────────┘
```

**数学符号约定**：图 G=(V,E)，N=|V| 节点，E=|E| 边；隐藏维 d=128。分区策略 π(G)=fs(G,I) 输出段集合 {S₁,…,S_K}，I 为先验物理信息（边界条件、障碍）。重叠量 δ∈ℕ，δ=0 无重叠，δ=1 时 V^δ_{S_k}=V^{δ−1}_{S_k}∪{Adj(i)|i∈V^{δ−1}_{S_k}}。

### 4.2 模态分解 (Modal Decomposition)

**目的**：提取网格的几何-物理结构特征，指导分区算法产出「物理一致」的段。

**方法**（M4GN 流体路径，Laplacian Eigenfunctions）：求解

```
−∇²φ = λφ,  subject to boundary constraints
取最小的 m=6 个非零特征值对应特征向量
节点 i 的特征 f_md(i) = (φ₁(i), …, φ₆(i))
```

**离散实现要点**（已在 `modal_decomp.py` 落地）：
- 两种 Laplacian：`graph_laplacian`（L=D−A，任意图）与 `cotangent_laplacian`（FEM 余切权重，对 −∇² 的一致离散，物理意义更好，**默认**）。
- 边界条件：`neumann`（默认，全域求解，跳过常数零模）或 `dirichlet`（固定边界为 0，仅内部求解后 scatter 回全域）。
- 鲁棒性：`scipy.sparse.linalg.eigsh` 用 shift-invert (sigma=0) 加速小特征值收敛；异常 fallback 宽松容差；每模归一化到单位范数。

**物理意义**：低频模捕捉大尺度流动结构（来流方向）；中频模反映圆柱附近分离与回流；模态相近的节点动力学行为相似。圆柱驻点区/加速区/尾迹区会因模态值差异被分到不同段，边界层节点因近壁约束自动与远场分离。

### 4.3 混合网格分区 (Hybrid Segmentation)

**Stage 1 — METIS 粗分**：输入网格图 G，输出 K0=64 个大致等尺寸的**连通**子图；保证连通性、最小化跨段边切割；优先 `pymetis`，无则 fallback 到谱聚类（eigsh+KMeans，会告警建议装 pymetis）。

**Stage 2 — SLIC 精修**（M4GN Algorithm 2）：基于物理感知特征的 K-means 变体，距离度量

```
d(i, C_k) = ‖f_obs_i − f_obs_Ck‖ + ‖f_md_i − f_md_Ck‖ + τ·‖x_i − x_Ck‖
```

- `f_obs`：节点到固体壁面的距离（1 维，自动检测 `node_type==6` WALL_BOUNDARY 节点）。
  > **不确定点 U1-b**：DeepMind 约定下圆柱壁与上下通道壁**同为 type 6**，仅凭 node_type 无法区分圆柱与外墙；当前用「到最近壁面距离」（含外墙）作为 SLIC 特征，对边界层细分仍合理。若需「仅圆柱」距离，后续用几何过滤（排除位于域包围盒上的 type-6 簇）。`compute_obstacle_distance` 默认 `obstacle_type_id=6`（5 是 OUTFLOW 出口，曾误设为 5，M1 已修正）。
- `f_md`：模态特征（6 维）；`x`：空间坐标（2 维）；`τ=1.0` 控制紧凑性。
- 特征与坐标均做 [0,1] 归一化，保证 τ 物理含义一致。
- **连通性约束**：节点只能重分配到「邻居所属」的段，避免跨空隙连接（修正 same-size k-means 的缺陷）。

**递归二层树 + 重叠**：

```
Level 0 (粗): K0=64  segments  ← METIS+SLIC（全图）
Level 1 (细): K1≈256 segments  ← 每个 L0 段内部再 METIS 切 4 块
重叠 δ=1: 每段并入一圈邻居 → 平滑段边界、减少 Feature Dispatch 处的不连续
```

每节点携带两个分配 ID：`L0_assign[i]`、`L1_assign[i]`；并返回段级邻接矩阵供 RWSE 与可视化使用。过小的段（< 4 节点）不再细分。

### 4.4 位置编码（段级 RWSE + 节点级绝对 PE）

**段级 RWSE（Random Walk Structural Encoding）**：
1. 对每层构建段级邻接 A_K：`A_K[Si,Sj] = Σ_{m∈Si}Σ_{n∈Sj} A_{mn}`（两段间有跨段边即相邻）。
2. 归一化转移矩阵 P = D⁻¹A_K，计算 diag(P), diag(P²), …, diag(P¹⁶) → 16 维向量。
3. 第 j 维 = 「从段 k 出发走 j 步回到自身的概率」，编码段在拓扑图中的局部连通结构。

**节点级绝对 PE**（M4GN §3.3.1）：用 MLP `fnp` 处理每节点 PE，**注入 Encoder 输入**：`h⁰_i = fn(x_i + fnp(p_i))`，参与微观消息传递，提升所提段特征的连续性。

### 4.5 Micro-level GNN（15 步消息传递）

**架构**：直接复用 PhysicsNeMo 的 `MeshGraphNet`，但**去掉其 decoder 头**——decoder 推迟到 Macro 层之后（M4GN EPD 拆分思想）。即只取 `node_encoder + edge_encoder + processor` 的输出 `h_node [N, d]`。

**分工**：负责局部物理——边界层黏性扩散、局部涡旋转、对流项。15 步消息传递使每节点「看到」15 跳邻居（约覆盖圆柱直径 2~3 倍范围）。与 Transformer 互补：GNN 管高频/短程，Transformer 管低频/长程。

**噪声注入训练**（Godwin et al. 2021，M4GN §3.2 沿用）：训练时对输入图加噪声并加入「噪声纠正」的节点级损失，提升长时 rollout 稳定性。

**实现要点**：

```python
self.micro = MeshGraphNet(
    input_dim_nodes=6,    # u, v + node_type one-hot(4)
    input_dim_edges=3,    # relative_x, relative_y, distance
    output_dim=128,       # 隐藏维 d；注意：实际取 processor 输出，旁路 decoder
    processor_size=15,    # 15 步消息传递
    aggregation="sum",
    recompute_activation=True,   # 省显存
)
# forward 内部：edge_enc → node_enc → processor → (本工作旁路 node_decoder)
```

> **落地注意**：`MeshGraphNet.forward` 默认返回经 decoder 的结果。本工作需直接调用其内部 `processor` 输出 `h_node`，或将 `output_dim` 设为 d 后把 decoder 当作首个投影层、把最终物理量预测交给 §4.9 的统一 decoder。两种方式在 §九 实现计划中确定。

### 4.6 N-S 约束感知物理量算子

**目的**：为 AMR Router 提供「每个区域是否需细分」的依据。四个判据（AMR-Transformer §3.2）：

| 物理量 | 公式 | 物理含义 | 圆柱绕流表现 |
| --- | --- | --- | --- |
| 速度梯度 G | √(‖∇u‖² + ‖∇v‖²) | 急剧速度变化/间断 | 壁面附近极高 |
| 涡量 ω | ∂v/∂x − ∂u/∂y | 旋转强度（反对称部分）| 脱落涡处极高 |
| 动量 M | ρ·√(u²+v²)·area | 局部动量 | 加速区高 |
| 应变率 S | √(2·S_ij·S_ij) = √(2u_x² + 2v_y² + (u_y+v_x)²) | 剪切/拉伸强度（对称部分）| 尾迹剪切层高 |

> **关于 S 的设计修正（M2 实现，已实跑测试通过）**：AMR-Transformer 原文把 KH 剪切定义为 `∂u/∂y − ∂v/∂x`，但它**恒等于 −ω**，Router 取绝对值后与涡量完全冗余，4 判据退化为 3。本工作改用**应变率幅值** `S=√(2·S_ij·S_ij)`（速度梯度的对称部分），与 ω（反对称部分）严格独立且恒 ≥ 0，使四个物理先验真正互补。已实跑验证：`pytest tests/test_physics_ops.py` → 8/8 通过，其中纯剪切流 `u=y` → ω=−1、S=1（区分开）；刚体旋转 `u=−y,v=x` → ω=2、S=0（应变为零）。真实数据（cylinder_flow，t=300，4 case）`S min` 全 > 0 且 `S p99 ≠ |ω|p99`，证明 S 与 ω 在实际流场上确实独立。

**非结构网格离散**（1-ring 最小二乘梯度，避免结构网格四叉树的限制）：

```
对节点 i，收集邻居 j：
  Δx = pos_j − pos_i      [deg(i), 2]
  Δu = u_j − u_i          [deg(i), 1]
最小二乘：∇u_i = (ΔxᵀΔx)⁻¹ Δxᵀ Δu     （∇v_i 同理）
```

**虚拟步外推**（AMR-Transformer Eq.11）：用前向欧拉 `u'_{t+Δt}=u_t+(u_t−u_{t−Δt})` 估计虚拟速度场，对 `u_t` 与 `u'_{t+Δt}` 各算一遍物理量，**取并集**触发细分，提前细化「即将变活跃」的区域，提升 rollout 时的前瞻性。

### 4.7 AMR Router（自适应 Token 路由）

**核心逻辑**：把 AMR-Transformer 的「四叉树逐层细分」迁移为「层次 METIS 树（L0=64↔L1=256）的保留 vs 折回」决策。L1≈对应四叉树更深一层（细），L0≈更浅一层（粗）。

```
输入：partition_levels=[L0_assign, L1_assign]，phys_per_node={G,ω,M,S}，阈值 T（采样）

Step 1  在 L1（256 段）聚合物理量：每段取 max|phys| → agg_G[k], agg_ω[k], …
Step 2  活跃判定（满足任一即活跃）：
        is_active[k] = (agg_G[k] > T_G) ∨ (agg_ω[k] > T_ω)
                       ∨ (agg_M[k] > T_M) ∨ (agg_S[k] > T_S)
        速度梯度 G 额外支持 Top-r 百分位触发（取当前层分布前 r 分位）
Step 3  分配最终 token：
        活跃 L1 段 → 保持独立 token（细粒度）
        平静 L1 段 → 折回 L0 父段（兄弟合并为 1 token）
输出：变长 token 数 T（64 ≤ T ≤ 256），kept_assign[N]（每节点 token id），kept_depth[T]（每 token 深度，1=细 L1 / 0=折回 L0；每节点深度 = kept_depth[kept_assign]）
```

**阈值采样机制**（AMR-Transformer 的巧妙设计）：
- **训练时**：阈值在预定义区间均匀随机采样——`G:[0.1,2.0], ω:[0.2,4.0], M:[0.5,10.0], S:[0.2,4.0]`。模型见过各种粒度的 Token 化，泛化性强。
- **测试时**：固定阈值，手动调节平衡精度/效率。

**直觉**：来流均匀区（ω≈0, G≈0）→ 大块合并省算；尾迹涡街区（ω/S 极高）→ 保持 256 粒度精确建模；壁面区（G 极高）→ 保持细分保边界层精度。

### 4.8 Segment Encoding + Macro Transformer + Feature Dispatch

**Segment Encoding**（置换不变，M4GN §3.3.1）：

```python
h_seg_k = MLP( mean_{i∈S_k} h_node_i )           # average pooling，O(Nd)
h_seg_k += PE_proj([RWSE_k, depth_k, x̄_k, ȳ_k]) # 段级 PE
```

相比 EAGLE 的 GRU：置换不变、O(Nd) vs O(Nd²)、无梯度消失、无长序列信息稀释。

**Macro Transformer**（M4GN §3.3.2，Pre-LN）：在 T 个段构成的**全连接段图**上做多头自注意力。

```
配置：4 层 TransformerEncoder, 8 头, d_model=128, FFN=512, Pre-LN
输入：h_seg [T,128] + padding mask（变长 batch）
输出：h_seg' [T,128]
复杂度：O(L_S · T² · d)，因 T≪N 故 O(T²)≪O(N²)
```

T=150 时 4·150²·128≈11.5 MFLOPs；T=256 时≈33.6 MFLOPs，均极低。

**Feature Dispatch**（M4GN §3.3.2）：

```python
h_global_i = h_seg'[token_of(i)]              # 节点取所属 token 输出
h_cat_i    = Concat([h_node_i, h_global_i])   # [N, 2d=256]
```

**设计哲学**：`h_node`（15 步 GNN）保留高频局部细节；`h_global`（Transformer）注入全局上下文（压力传播、远程涡-涡相互作用）；**拼接而非相加**，让 decoder 自学如何融合两种尺度。δ=1 重叠进一步平滑段边界处的 dispatch 不连续。

### 4.9 Decoder + 损失函数

**Decoder**：`MLP([256, 128, 3])` → (Δu, Δv, p̂)。速度预测增量（残差），压力预测绝对值。

**损失**：Per-channel NMSE（与 AMR-Transformer 一致）：

```
NMSE = mean((pred − target)²) / mean(target²).clamp_min(eps)
```

优势：自动适应尺度差异——速度增量 ~O(10⁻³) 与压力 ~O(1) 不会互相淹没。可叠加噪声纠正项（§4.5）。

---

## 五、与 Baseline 的对比分析

### 5.1 计算复杂度对比

圆柱绕流主线参数：N≈1900 节点，E≈5500 边，d=128。

| 模型 | 主要计算项 | 估算 FLOPs | 长程机制 | 备注 |
| --- | --- | --- | --- | --- |
| MGN (15 步) | O(L·E·d²) | ~1.35 GFLOPs | 15 跳 | 纯消息传递，长程弱 |
| X-MGN | O(L·E·d²)+多尺度图 | ≈MGN 量级 | 逐跳+多尺度 | 省显存但长程仍弱 |
| 节点级 Graph-Transformer | O(N²·d) | ~462 MFLOPs | 全局 | 显存 ~14.4 GB，不可行 |
| M4GN (K=64 固定) | MGN + O(K²·d) | +1.3 MFLOPs | 段级全局 | K 固定，活跃区粒度不足 |
| **AMR-M4GN (ours)** | MGN + O(T²·d) | +11.5 MFLOPs (T=150) | 段级全局+自适应 | **自适应粒度** |

**核心观察**：Transformer 部分开销远小于 GNN 部分（T≪N）；总开销约为原 MGN 的 **1.01 倍**——几乎「免费」获得全局建模能力。相比节点级 Transformer，AMR-M4GN 把 O(N²) 降到 O(T²)，显存与算力均可控。

### 5.2 精度预期

| 场景 | MGN (baseline) | X-MGN | AMR-M4GN (预期) | 提升原因 |
| --- | --- | --- | --- | --- |
| 涡脱落频率 (Strouhal) | 中 | 中 | 高 | Transformer 捕捉全局周期 |
| 尾迹速度衰减 | 中偏低 | 中 | 高 | 远程压力-速度耦合 |
| 回流区长度 | 中 | 中 | 高 | AMR 在分离点保留细 token |
| 表面压力分布 | 高 | 高 | 高 | GNN 15 步已覆盖 |
| 全局压力场瞬时响应 | 低 | 低 | 高 | **Transformer 一跳全局** |
| 长时 rollout 稳定性 | 低 | 中 | 中偏高 | 全局约束+噪声注入减缓误差累积 |
| 千万级网格显存峰值 | 高（线性/二次）| 低（Halo 分块）| 中 | AMR 减 token；扩展见 §十二 |

---

## 六、数据集与实验设置

### 6.1 主线数据集：CylinderFlow（圆柱绕流）

**VortexSheddingDataset**（PhysicsNeMo 内置，与现有 `train.py`/`config.yaml` 兼容）：
- 来源：圆柱绕流直接数值仿真 (DNS)；网格 ~1900 节点的非结构三角网格（stationary，拓扑不随时间变化）。
- 节点特征：速度 (u,v) + node_type one-hot（4 维）；边特征：相对坐标 (dx,dy) + 距离 |d|。
- 预测目标：速度增量 (Δu,Δv) + 压力 p；Reynolds 数通过不同圆柱直径/来流速度覆盖。
- 训练规模（对齐现有 config）：`num_training_samples` 个 case × `num_training_time_steps` 步。

**选它做主线的理由**：现有 `amr_m4gn/` 代码（modal_decomp + segmentation）已基于此构建，网格小、可在单卡快速迭代，便于把全链路跑通、做完整消融，与 MGN baseline 公平对比。

### 6.2 扩展数据集：EAGLE（大规模非定常湍流）

- ~110 万个 2D 非结构网格快照、600 场景、3 类型；高度湍流、非周期涡，每个 mesh 节点数（数千级）远大于圆柱绕流，graph diameter 更大、长程耦合更强。
- **作为「大规模能力验证」目标**：在主线跑通后迁移，验证混合架构在强湍流/长程场景下相对 MGN/X-MGN 的优势，并验证 AMR Token 剪枝在大网格上的算力节省。
- 数据加载需新增 EAGLE 的 reader（区别于 TFRecord 的 cylinder_flow）。

### 6.3 训练设置

```yaml
optimizer: Adam（或 Apex FusedAdam）
lr: 1e-4
lr_decay: exponential, rate≈0.9999991 per step   # 对齐现有 config
epochs: 25~200（先小后大）
batch_size: 1~4（图级别）
amp: True（FP16 混合精度，先关后开以排查数值问题）
ddp: 支持多卡
noise_injection: True（训练时输入加噪 + 噪声纠正损失）
# AMR 阈值（训练随机采样）
G:U[0.1,2.0]  ω:U[0.2,4.0]  M:U[0.5,10.0]  S:U[0.2,4.0]
# AMR warm-up：前 5 epoch 关闭 AMR，全用 L1 细粒度（见风险表）
```

### 6.4 评价指标（对齐 PPT「实验方案与评估指标」两页）

**A. 物理精度与保真度**
1. **单步 / 多步 RMSE**：评估速度场 (u,v) 与压力场 p 的均方根误差；并报 NMSE/MAE（对标 AMR-Transformer Table 1）。
2. **长程误差累积**：50/100/200 帧自回归 rollout 误差曲线；测算连续推演数百帧后的**能量守恒与发散**情况。

**B. 计算效能与资源消耗**
3. **显存峰值**：验证在更大网格（EAGLE/千万级）下是否能突破单卡显存墙（AMR 减 token 的贡献）。
4. **单帧推理延迟**：对比传统物理求解器（如 OpenFOAM），测算加速比（期望 1~2 个数量级加速）；并报单步 FLOPs / GPU 时间。

**C. 视觉效果评估**
5. 观察湍流核心与涡旋边缘是否出现「过度平滑」，评估视觉保真度（直接对应 §1.2 的过平滑瓶颈）。

**D. AMR 统计**
6. 平均 token 数 T、T 分布、T 与物理量（涡量/梯度）的相关性，验证「物理驱动」剪枝的可解释性。

### 6.5 消融实验计划

| 实验 | 配置 | 验证什么 |
| --- | --- | --- |
| Full model | AMR-M4GN | 完整性能 |
| w/o AMR | 固定 K=256 | AMR 的效率贡献（对照 M4GN 固定 K）|
| w/o Transformer | 只有 15 步 GNN | Transformer 的精度贡献（对照 MGN）|
| w/o Modal Decomp | METIS-only 分区 | 模态分解的分区质量贡献 |
| w/o RWSE PE | 去掉段级 RWSE | 位置编码必要性 |
| w/o δ=1 重叠 | δ=0 | 段重叠对段边界连续性的贡献 |
| w/o 虚拟步 | 仅当前帧 phys | 虚拟步外推对 rollout 的贡献 |
| 7 步 vs 15 步 GNN | processor_size=7/15 | 深度 GNN 的收益 vs 过平滑 |

---

## 七、代码实现计划

### 7.1 文件结构

```
physicsnemo/examples/cfd/vortex_shedding_mgn/
├── train.py                       # [不动] 原 MGN baseline
├── inference.py                   # [不动] 原推理脚本
├── conf/
│   ├── config.yaml                # [不动] 原 MGN 配置
│   └── config_amr_m4gn.yaml       # [新建] AMR-M4GN 配置
├── train_amr_m4gn.py              # [新建] AMR-M4GN 训练入口
├── inference_amr_m4gn.py          # [新建] 推理+评估脚本
├── preprocess_partitions.py       # [新建] 离线预处理 CLI（封装下列离线步骤）
├── visualize_partition.py         # [已有] 预处理诊断可视化
└── amr_m4gn/                      # 模型包
    ├── __init__.py                # [已有]
    ├── modal_decomp.py            # [已有] Laplacian eigenfunctions
    ├── segmentation.py            # [已有] METIS+SLIC 混合分区 + 递归 + δ重叠
    ├── pe.py                      # [新建] RWSE + 节点级绝对 PE
    ├── physics_ops.py             # [新建] G/ω/M/S + 1-ring 最小二乘 + 虚拟步
    ├── amr_router.py              # [新建] AMR 二层 fold/keep 路由 + 阈值采样
    ├── micro_gnn.py               # [新建] MeshGraphNet wrapper（旁路 decoder）
    ├── macro_transformer.py       # [新建] Segment Encode + Transformer + Dispatch
    └── model.py                   # [新建] AMRM4GN 顶层模型
```

### 7.2 数据管线现状与关键改造点（务必先读）

> 以下源自对 `VortexSheddingDataset`、`train.py`、`MeshGraphNet` 真实代码的核对，是后续所有实现的前提。**若忽略，整条链路会在拿不到坐标/分区时崩溃。**

**A. 坐标 `pos` 与 `cells` 在训练 split 不可见（关键障碍）**

`VortexSheddingDataset.__getitem__` 仅在 `split != "train"` 时把 `mesh_pos`、`cells`、`rollout_mask` 附到样本上；训练 split 的 `graph` 只有 `graph.x [N,6]`、`graph.edge_attr [E,3]`、`graph.y [N,3]`、`graph.edge_index`。而 `edge_attr` 是**归一化后**的 `[dx, dy, |d|]`（减均值除标准差），**无法直接当作真实相对坐标**。

- **影响**：`physics_ops`（梯度需真实 Δx）、`segmentation`/`modal_decomp`（需 pos）、AMR 段质心（需 pos）全都依赖 `pos`。
- **应对（推荐方案，二选一并在 M4 前定夺）**：
  - **方案 P1（推荐）**：新增轻量子类 `VortexSheddingDatasetAMR(VortexSheddingDataset)`，在 `__init__` 缓存每个 case 的 `mesh_pos[i]`、`cells[i]`，并在 `__getitem__`（train 分支）把 `graph.pos`（原始坐标）、`graph.gidx`（case 索引）一并返回。**不改动原 `VortexSheddingDataset`**，零风险，与 baseline 共存。
  - **方案 P2**：在 `physics_ops` 内用 `edge_stats`（`edge_mean/edge_std`，已存 `edge_stats.json`）**反归一化** `edge_attr[:, :2]` 得到真实 Δx。可行但把统计量耦合进模型前向，可读性差。**仅当 P1 受阻时用。**
- **不确定点 U1**：原始 `mesh_pos` 是否已做坐标归一化、不同 case 是否同坐标系——需在 M2 用 `visualize_partition.py` 打印 `pos.min/max` 确认。若各 case 坐标尺度不一，`physics_ops` 的绝对阈值需改为「每 case 自适应分位数」（见 §4.7 Top-r）。

**B. 分区是「逐 case 几何」的，需按 `gidx` 缓存**

每个 case 有自己的网格（`data_np["cells"][0]`，stationary）。离线预处理须**对每个训练/测试 case 各算一份** `partition_cache_{split}_{gidx}.pt`，在线按 `graph.gidx` 取用。CylinderFlow 同一 case 的拓扑不随时间变化，故每 case 只算一次。

**C. PyG 批处理时段 ID 需偏移**

`PyGDataLoader(batch_size>1)` 会把多张图拼成一张大图并提供 `batch [N_total]` 向量。AMR 的 `kept_assign`、Transformer 的 token 必须**按图偏移**（图 b 的段 ID 加上前面所有图的段数），否则跨图节点会被错误聚到同一 token。`macro_transformer` 内需用 `batch` 做分组与 padding-mask。**M4 起 `batch_size=1` 跑通，再在 M5 支持 `batch_size>1`。**

**D. `MeshGraphNet` 旁路 decoder 的落地决策**

`MeshGraphNet.forward` = `edge_encoder → node_encoder → processor → node_decoder`。本工作要的是 `processor` 输出 `h_node [N,d]`。两条路径：
- **(a) 推荐**：在 `micro_gnn.py` 内分别持有 `node_encoder/edge_encoder/processor`（可直接构造 `MeshGraphNet` 后引用其子模块 `.node_encoder/.edge_encoder/.processor`），forward 只走到 processor，跳过 `node_decoder`。语义清晰、与 M4GN「detach decoder」一致。
- **(b) 备选**：构造时设 `output_dim=d`，把内置 `node_decoder` 当作首个投影层用，最终物理量预测交给 §4.9 统一 decoder。
- **不确定点 U2**：需在 M4 写单测确认 `MeshGraphNet` 暴露的子模块属性名（`processor` 等）在当前版本稳定；若内部命名变动，退化为路径 (b)。

### 7.3 各文件职责与接口（概览）

| 文件 | 状态 | 输入 | 输出 | 关键依赖 |
| --- | --- | --- | --- | --- |
| `amr_m4gn/modal_decomp.py` | ✅ | edge_index, pos, node_type, cells, m=6 | f_md [N,6], eigvals | scipy.eigsh |
| `amr_m4gn/segmentation.py` | ✅ | edge_index, pos, f_md, f_obs, K=[64,256], τ | levels=[L0,L1], seg_adj | pymetis/谱聚类 |
| `amr_m4gn/pe.py` | ✅ 实现（单测 5/5）| seg_adj, steps=16 | rwse[K,16]（每层）| torch（稠密）|
| `amr_m4gn/physics_ops.py` | ✅ 实现（M2 pytest 8/8）| u,v,pos,edge_index,(u_prev) | {G,ω,M,S}[N] | torch（index_add_）|
| `amr_m4gn/amr_router.py` | ✅ 实现（单测 11/11、4 case 实跑）| levels, phys, thresholds | kept_assign[N], depth[T], T | torch（scatter_reduce_）|
| `amr_m4gn/micro_gnn.py` | ✅ 实现（单测 4/4）| x, edge_attr, graph | h_node [N,d] | MeshGraphNet（旁路 decoder）|
| `amr_m4gn/macro_transformer.py` | ✅ 实现（单测 5/5）| h_node, kept_assign, rwse, depth, centroid | h_cat [N,2d] | nn.Transformer |
| `amr_m4gn/model.py` | ✅ 实现（集成测试 3/3）| PyG Data + 预处理缓存 | pred [N,3] | 上述全部 |
| `preprocess_partitions.py` | ✅ 实现（实跑生成缓存）| data_dir, split, K_list, m | 写 partition_cache_*.pt | modal/seg/pe |
| `data_amr.py`（子类）| ✅ 实现（D5/P1，overfit 实跑）| 同 VortexSheddingDataset | graph(+pos,+gidx)+get_cache | 见 §7.2-A |
| `train_amr_m4gn.py` | ✅ 实现（overfit 实跑，NMSE 0.92→0.013）| argparse | overfit 日志 | model, data_amr |
| `inference_amr_m4gn.py` | ⬜ 新建（M5）| checkpoint, test split | rollout + 评估指标 | model, data_amr |

下面给出**逐文件详细规格**（函数签名、输入输出张量形状、前置数据、单元测试、不确定点）。

### 7.4 逐文件详细规格

#### 7.4.1 `amr_m4gn/pe.py`（新建）— 位置编码

**职责**：段级 RWSE + 节点级绝对 PE。离线计算，写入缓存。

```python
def rwse_segment(seg_adj: Tensor[2, E_seg], num_segments: int,
                 steps: int = 16) -> Tensor:   # → [num_segments, steps]
    """段级随机游走结构编码：P=D⁻¹A_seg，返回 diag(P^1..P^steps)。"""

def rwse_node(edge_index: Tensor[2, E], num_nodes: int,
              steps: int = 16) -> Tensor:       # → [num_nodes, steps]
    """节点级 RWSE（绝对 PE），注入 Encoder 输入（§4.4）。"""
```

- **输入来源**：`seg_adj` 来自 `segmentation.build_partition_tree` 已返回的 `segment_adjacency`；`edge_index` 来自 graph。
- **输出**：写入缓存 `rwse_L0 [K0,16]`、`rwse_L1 [K1,16]`、`node_pe [N,16]`。
- **单元测试**：(1) 链状 5 段图，验证端点段的回归概率 < 中间段；(2) 全连接 3 段，RWSE 应近似相等；(3) 数值检查 `P` 行和为 1。
- **不确定点 U3**：节点级绝对 PE 是否真正提升精度，M4GN 消融显示「视情况」。**先实现接口、默认开启，在 M6 用 `w/o RWSE PE` 消融决定保留与否。**

#### 7.4.2 `amr_m4gn/physics_ops.py`（新建）— N-S 物理量算子

```python
def lstsq_gradient(field: Tensor[N, C], pos: Tensor[N, 2],
                   edge_index: Tensor[2, E]) -> Tensor:  # → [N, C, 2]
    """1-ring 最小二乘梯度：每节点对邻居解 (ΔxᵀΔx)⁻¹ΔxᵀΔf。"""

def compute_ns_quantities(u: Tensor[N], v: Tensor[N], pos: Tensor[N,2],
                          edge_index, area: Tensor[N]=None, rho: float=1.0,
                          eps: float=1e-8,
                          vel_mean=None, vel_std=None
                          ) -> dict:   # {"G":[N],"omega":[N],"M":[N],"S":[N]}
    """G=√(‖∇u‖²+‖∇v‖²); ω=∂v/∂x−∂u/∂y; M=ρ·|U|·area;
       S=√(2·S_ij·S_ij)=√(2u_x²+2v_y²+(u_y+v_x)²)（应变率幅值，见 §4.6 修正）。
       传 vel_mean/vel_std 则先反归一化（D1/U4）。"""

def virtual_step(u_t, u_prev) -> Tensor:        # u' = u_t + (u_t − u_prev)
    """前向欧拉虚拟速度场（§4.6），与当前场取并集触发细分。"""
```

- **输入来源**：`u,v` = `graph.x[:, :2]`（**注意：已归一化**）；`pos` = `graph.pos`（需 §7.2-A 改造）；`area` 可用 `modal_decomp.compute_node_area`。
- **关键不确定点 U4（速度归一化对物理量的影响）**：`graph.x` 的速度是减均值除标准差后的值，直接算的 ω/G 是「归一化空间」的量，**不是真实涡量**。两种处理：
  - (i) 用 `node_stats.json` 的 `velocity_mean/std` **反归一化**后再算物理量（物理正确，阈值有物理意义）；
  - (ii) 直接在归一化空间算，把阈值也理解为归一化量（实现简单，但失去物理可解释性，与 PPT「物理驱动/可解释」诉求冲突）。
  - **建议**：采用 (i)。在 `physics_ops` 入口接收 `vel_mean/vel_std` 反归一化。**最终方案需 M2 可视化对比两种 ω 场后定夺**——若 (ii) 的活跃区与 (i) 高度一致，可为简洁选 (ii)。
- **单元测试**：(1) 解析场 `u=y, v=0` → ω=−1、S=1、G=1 全场常数，误差 < 5%（规则网格上）；(2) 旋转场 `u=−y,v=x` → ω=2、S=0（纯旋转无应变）；(3) 均匀流 `u=1,v=0` → G≈0；(4) 退化邻居（共线）时最小二乘加 ε·I 正则不崩。**M2 已实跑 8/8 通过。**
- **测试数据**：构造 20×20 规则三角网格 + 解析速度场（脚本内生成，不依赖数据集），对比解析梯度。

#### 7.4.3 `amr_m4gn/amr_router.py`（新建）— 自适应 Token 路由

```python
def aggregate_per_segment(phys: dict, assign: Tensor[N],
                          num_seg: int, reduce="max") -> dict:  # 每段 max|·|

def sample_thresholds(ranges: dict, training: bool, fixed: dict=None,
                      generator=None) -> dict:
    """训练随机采样（可传 seeded generator 复现）；测试固定/取区间中点。见 §4.7。"""

def route(levels: list, phys: dict, thresholds: dict, reduce="max"
          ) -> tuple:   # → (kept_assign[N], kept_depth[T], T:int, token_batch[T])
    """L1 活跃段保留；平静段折回 L0 父段；返回连续 token id、每 token 深度
       （kept_depth 为 per-token [T]；per-node 深度 = kept_depth[kept_assign]）。"""
```

- **实现状态（M3）**：已实现并实跑——单测 11/11 通过；4 case×t=300 真实路由 T=129–135、reduction ~48%。`kept_depth` 实现为 per-token `[T]`（与 `token_batch[T]` 配套、供 SegmentEncoder 取 depth）。

- **输入来源**：`levels=[L0_assign,L1_assign]`（缓存）；`phys` 来自 `physics_ops`；阈值来自 `sample_thresholds`。
- **关键设计点**：需建立 `L1→L0` 父子映射（离线缓存 `l1_to_l0 [K1]`）。`route` 输出 `kept_assign[i]`∈[0,T) 为节点最终 token id，连续编号。
- **单元测试**：(1) 全场 phys=0 → 全部折回 → T=K0=64；(2) 全场 phys 极大 → 全保留 → T=K1；(3) 半场活跃 → 64<T<256 且活跃区 token 来自 L1；(4) 阈值采样落在区间内、可复现（固定种子）。
- **不确定点 U5（warm-up 时长）**：前若干 epoch 关闭 AMR（全用 L1）的时长需实验定。**M5 训练曲线决定**：若关 AMR 期间 loss 已平稳，则可早开 AMR。

#### 7.4.4 `amr_m4gn/micro_gnn.py`（新建）— 局部 GNN（旁路 decoder）

```python
class MicroGNN(nn.Module):
    def __init__(self, in_nodes=6, in_edges=3, hidden=128, processor_size=15,
                 activation="silu", recompute_activation=True): ...
    def forward(self, x, edge_attr, graph) -> Tensor:  # → h_node [N, hidden]
        # 走 node_encoder/edge_encoder/processor，跳过 node_decoder（见 §7.2-D）
```

- **单元测试**：(1) 形状测试，N×6 + E×3 → N×128；(2) 与原 `MeshGraphNet` 前 processor 输出逐元素一致（同权重）；(3) 反向传播梯度非 NaN。
- **不确定点 U2**：见 §7.2-D（子模块属性名稳定性）。

#### 7.4.5 `amr_m4gn/macro_transformer.py`（新建）— 段编码 + Transformer + Dispatch

```python
class SegmentEncoder(nn.Module):      # mean-pool per token + PE_proj
    def forward(self, h_node, kept_assign, T, rwse, depth, centroid
                ) -> Tensor: ...      # → h_seg [T, d]

class MacroTransformer(nn.Module):    # 4层×8头, Pre-LN, d=128, FFN=512
    def forward(self, h_seg, key_padding_mask=None) -> Tensor: ...  # [T, d]

def dispatch(h_seg_out, kept_assign, h_node) -> Tensor:  # → h_cat [N, 2d]
```

- **批处理**：`kept_assign` 与 `T` 需按 `graph.batch` 偏移；变长 batch 用 `key_padding_mask`（见 §7.2-C）。
- **单元测试**：(1) mean-pool 置换不变性（打乱节点顺序输出不变）；(2) T=1 时退化为全局平均；(3) padding mask 正确屏蔽（被 mask 段不影响他段输出）；(4) dispatch 后 `h_cat[i, :d]==h_node[i]`。

#### 7.4.6 `amr_m4gn/model.py`（新建）— 顶层模型

```python
class AMRM4GN(nn.Module):
    def forward(self, graph, cache) -> Tensor:   # → pred [N, 3]
        h_node = self.micro(graph.x, graph.edge_attr, graph)
        phys   = compute_ns_quantities(...)             # 含虚拟步
        kept_assign, depth, T = route(cache.levels, phys, thr)
        h_seg  = self.seg_enc(h_node, kept_assign, T, cache.rwse, depth, ...)
        h_seg  = self.macro(h_seg, mask)
        h_cat  = dispatch(h_seg, kept_assign, h_node)
        return self.decoder(h_cat)                      # MLP [2d→128→3]
```

- `cache` 为该 case 的预处理缓存（levels、rwse、l1_to_l0、centroid 等）。
- **集成测试**：单 case 单步前向，输出 `[N,3]` 无 NaN；`loss.backward()` 全参数有梯度。

### 7.5 离线预处理与缓存格式

**`preprocess_partitions.py`（新建）**：对指定 split 的每个 case 跑「modal_decomp → segmentation(δ=1) → pe → l1_to_l0 映射 → 段质心」，写盘。

```
缓存文件：partition_cache_{split}_{gidx}.pt（dict）
  ├── levels:      [L0_assign[N], L1_assign[N]]   (long)
  ├── seg_adj:     [L0_adj, L1_adj]                (用于可视化/调试)
  ├── rwse:        {"L0":[K0,16], "L1":[K1,16]}
  ├── node_pe:     [N,16]
  ├── l1_to_l0:    [K1]    (L1 段 → L0 父段)
  ├── centroid:    {"L0":[K0,2], "L1":[K1,2]}     (段质心，供 PE)
  ├── area:        [N]     (Voronoi 面积，供动量 M)
  └── meta:        {K0,K1,m,tau,boundary_type,pos_min,pos_max}
```

- **CLI**：`python preprocess_partitions.py --data_dir ... --split train --K0 64 --K1 256 --num_cases 400`。
- **耗时估算**：单 case 模态分解 ~1-3s + 分区 ~0.5s + PE ~0.2s ≈ 4s；400 case ≈ 27 min（一次性）。
- **不确定点 U6（SLIC 速度）**：`slic_refinement` 当前是 Python 循环，~1900 节点可接受；扩到 EAGLE（数千节点）或 case 数大时需并行/Numba。**M7 前评估，必要时多进程预处理。**

### 7.6 训练与推理入口改造

- **`data_amr.py`**：`VortexSheddingDatasetAMR(VortexSheddingDataset)`，train 分支额外返回 `graph.pos`、`graph.gidx`；并在 `__init__` 按需加载/触发 `preprocess_partitions`（缺缓存则报错提示先跑预处理）。
- **`train_amr_m4gn.py`**：复制 `train.py` 框架，替换 `MeshGraphNet` 为 `AMRM4GN`，`MSELoss` 替换为 per-channel NMSE（§4.9），前向时按 `graph.gidx` 取缓存。保留 DDP/AMP/checkpoint 逻辑。
- **`inference_amr_m4gn.py`**：复制 `inference.py`，加入 rollout、§6.4 全部指标计算与可视化（速度/压力场、token 着色、误差累积曲线）。
- **`conf/config_amr_m4gn.yaml`**：在原 `config.yaml` 基础上增 `K0/K1/num_modes/tau/processor_size/transformer{layers,heads,ffn}/amr_thresholds/warmup_epochs/loss=nmse` 等字段。

### 7.7 测试策略总览

| 层级 | 对象 | 方法 | 通过标准 |
| --- | --- | --- | --- |
| 单元测试 | 每个新文件核心函数 | `pytest`，解析场/合成图（见各 §7.4）| 数值误差 < 阈值、形状正确、可复现 |
| 可视化校验 | 模态/分区/物理量/token | `visualize_partition.py` 扩展出图 | 肉眼符合物理直觉（见 §八验收）|
| 集成测试 | `model.py` 前向+反向 | 单 case 单步 | 输出 [N,3] 无 NaN、全参有梯度 |
| Overfit | 完整模型 | 单 case 多步训练 | loss 单调下降至接近 0，预测≈GT |
| 回归对比 | vs MGN/X-MGN | 全量训练 | §6.4 指标，AMR-M4GN ≥ baseline |

### 7.8 关键第三方依赖

| 库 | 用途 | 安装 |
| --- | --- | --- |
| `pymetis` | METIS 图分区 | `pip install pymetis`（Windows 困难则 fallback 谱聚类）|
| `torch_geometric` / `torch_scatter` | 图结构、scatter 聚合 | 已有 |
| `scipy` | 稀疏特征值 | 已有 |
| `tfrecord` | 读 cylinder_flow TFRecord | `pip install tfrecord` |
| `pytest` | 单元测试 | `pip install pytest` |


---

## 八、研发里程碑（含进入/退出条件与决策门）

> 每个里程碑标注：**目标 / 新增改动文件 / 配合数据 / 验收（退出）标准 / 决策门**。「决策门」= 必须看到某类结果（训练曲线、可视化）才能决定下一步的点。

### M1 — 离线预处理（✅ 已完成）

- **改动**：`modal_decomp.py`、`segmentation.py`、`visualize_partition.py`。
- **数据**：1 个 test case（`raw_dataset/cylinder_flow`）。
- **退出标准**：6 张诊断图合理（模态非死模、分区连通、尾迹/来流分到不同段）。

### M2 — 物理量算子（2 天）✅ 已完成

- **目标**：`physics_ops.py` 正确计算 G/ω/M/S，能区分活跃区与平静区。
- **新增**：`amr_m4gn/physics_ops.py`；扩展 `visualize_partition.py` 出四标量场图（+ `--timestep`、按 case 分目录、终端日志落盘）。
- **配合数据**：(a) 合成 20×20 规则网格 + 解析速度场（单元测试）；(b) cylinder case 4 个（0/1/50/99）的 t=300 真实速度场（可视化）。
- **退出标准**：解析场单测误差 < 5%；可视化中圆柱后方涡街 ω 显著高于来流区。**实测结果**：pytest 8/8 通过；t=300 四 case `|ω|p99=68–208` 同数量级，ω 图脱涡涡街清晰、来流≈0。详见 `AMR_M4GN_Progress_M2.md` §七/§十。
- **🚩 决策门 D1（速度归一化处理，U4）**：暂定反归一化（物理速度量级与 BL 涡量估计同阶，跨 case 同数量级）；归一化空间下的最终取舍待 M4 真实数据管线确认。
- **🚩 决策门 D2（坐标系，U1）**：M1 实测 4 case `pos` 范围全等（x∈[0,1.6]、y∈[0,0.41]），暂定用绝对阈值；M3 出 token 分布后复核。
- **设计修正**：S 由原文 KH 剪切 `∂u/∂y−∂v/∂x`（≡−ω，冗余）改为应变率幅值 `√(2·S_ij·S_ij)`，与 ω 独立（见 §4.6 note）。

### M3 — AMR Router（2 天）✅ 已完成

- **目标**：二层 fold/keep 决策正确，T 随物理状态变化。
- **新增**：`amr_m4gn/amr_router.py`、`amr_m4gn/pe.py`；`visualize_partition.py` 加 `--plot_routing`/`--route_pct`/`--route_channels`（出 `08_routing.png`）。
- **配合数据**：M2 的真实涡街场（4 case×t=300）+ 缓存的 `levels`、`l1_to_l0`。
- **退出标准**：单测四例全过 + 可视化尾迹保细/来流合并。**实测结果**：`amr_router` 单测 6/6、`pe` 单测 5/5（共 11/11）通过；4 case×t=300（ω 单通道 p70）T=129–135、reduction ~48%，`08_routing.png` 红（保细）集中于尾迹+底壁、蓝（合并）覆盖来流。详见 `AMR_M4GN_Progress_M3.md`。
- **🚩 决策门 D3（K0/K1 取值）**：观察 T 分布。**M3 现状**：演示用分位阈值（p70），T 恒≈130 是分位的数学必然，**「T 随物理变化」需绝对阈值才能体现**，留训练期（M5）用 `sample_thresholds` 采样后看全训练集 T 分布最终拍板。

### M4 — 端到端跑通（3 天）🟢 管线跑通（overfit 收敛度待调参）

- **目标**：完整模型在**单个 case** 上 overfit 成功。
- **新增**：`micro_gnn.py`、`macro_transformer.py`、`model.py`、`data_amr.py`、`preprocess_partitions.py`、`train_amr_m4gn.py`、`conf/config_amr_m4gn.yaml`。
- **配合数据**：1 个 case 全时间步 + 在线/离线 `partition_cache`。
- **退出标准**：(1) 集成测试无 NaN、全参有梯度；(2) overfit 单 case：loss 单调降至接近 0。**实测结果**：组件单测 9/9 + 集成测试 3/3 通过；overfit 单 case NMSE **0.92→0.013（降约 70×、趋势向下）→ 端到端管线正确且可学习**；但后期在 1e-2~6e-2 震荡、未压到 ≈0（49 帧同拟合 + lr 未精调，属调参问题，非管线缺陷）。详见 `AMR_M4GN_Progress_M4.md`。
- **🚩 决策门 D4（micro_gnn 接口，U2）**：✅ **通过**——`MeshGraphNet` 子模块属性稳定，用路径 (a)（走 processor、丢 decoder）。
- **🚩 决策门 D5（数据管线方案，U1/§7.2-A）**：✅ **选 P1**（`VortexSheddingDatasetAMR` 子类暴露 pos/gidx）。
- **🚩 决策门 D1（速度反归一化，U4）**：✅ 管线可用——overfit 用 `node_stats` 的 `vel_mean/std` 反归一化算物理量，loss 正常下降，证明该管线在真实数据上不崩、可学。
- **前置约束**：本里程碑 `batch_size=1`，批内段偏移留 M5。

### M5 — 全量训练 + 对比（5 天）

- **目标**：与 MGN、X-MGN baseline 公平对比。
- **新增改动**：`macro_transformer` 支持 `batch_size>1`（批内段偏移 + padding mask，§7.2-C）；`inference_amr_m4gn.py`（rollout + §6.4 指标）。
- **配合数据**：全量 train（`num_training_samples` cases）+ test split + 全部缓存。
- **退出标准**：训练收敛；对比表（NMSE/MAE/单步&多步 RMSE/FLOPs/GPU 时间/显存峰值）；rollout 误差曲线；token 数统计。
- **🚩 决策门 D6（AMR warm-up 时长，U5）**：根据「关 AMR 阶段」loss 曲线决定何时开 AMR。
- **🚩 决策门 D7（是否达预期）**：若 AMR-M4GN 长程指标未超 MGN → 排查 Transformer 是否真起作用（查注意力权重是否非平凡），据此决定是否调整段粒度/PE/损失权重。**此为整个课题的关键验证点。**

### M6 — 消融实验（5 天）

- **目标**：验证各模块贡献。
- **改动**：在 `config_amr_m4gn.yaml` 加开关（`use_amr/use_transformer/use_modal/use_rwse/use_overlap/use_virtual_step/processor_size`），无需大改代码。
- **配合数据**：全量数据，复用 M5 流程。
- **退出标准**：§6.5 八组消融表 + 分析；明确各模块净贡献。
- **🚩 决策门 D8（裁剪）**：根据消融结果，对净贡献为负/可忽略的模块（如 RWSE、虚拟步）在最终模型中关闭，简化架构。

### M7 — EAGLE 大规模扩展（可选，5 天）

- **目标**：大规模湍流能力验证。
- **新增**：EAGLE 数据 reader（区别于 TFRecord）；预处理并行化（U6）。
- **配合数据**：EAGLE 子集（先少量场景验证管线，再扩量）。
- **退出标准**：在 EAGLE 上跑通并与 MGN/X-MGN 对比；展示 AMR token 节省与长程优势。
- **🚩 决策门 D9（扩展性瓶颈）**：若单图过大致显存不足 → 评估是否引入 X-MGN 式 Halo 分块（§十一-2，与本架构正交）。

**主线总工期**：约 20 工作日（M1–M6，4 周）；含 EAGLE 扩展约 5 周。

### 8.1 可视化验收清单（肉眼判断，沿用并扩展 M1）

| 图 | 内容 | 合格判据 |
| --- | --- | --- |
| 模态图 | 前 6 个 Laplacian 模 | 无死模；中频在圆柱/尾迹有结构 |
| 分区图 | L0(64)/L1(256) 着色 + 段邻接 | 同色连通；圆柱附近段更密 |
| 物理量图 | G/ω/M/S 四标量场 | 尾迹 ω 高、壁面 G 高 |
| Token 图 | AMR 保留/合并着色 | 尾迹保细、来流合并 |
| 预测对比图 | pred vs GT 的 u/v/p | overfit 后近乎一致 |
| 误差曲线 | rollout RMSE vs 步数 | AMR-M4GN 增长慢于 MGN |
| 注意力图（可选）| 段级注意力权重 | 关注气流/尾迹（呼应 EAGLE）|


---

## 九、技术风险与应对

| 风险 | 可能性 | 影响 | 应对 |
| --- | --- | --- | --- |
| Laplacian 边界条件不当致特征值退化 | 中 | 模态无意义 | Neumann 替代 Dirichlet；shift-invert 稳定；可视化校验 |
| METIS 在窄长区域产生不均匀段 | 低 | 某些段仅几节点 | SLIC 精修 + 最小尺寸检查（<4 不细分）|
| 训练初期物理量不稳，AMR 抖动 | 高 | 收敛慢 | 前 5 epoch 关 AMR 全用 L1（warm-up）|
| 变长 token batch padding 比例高 | 中 | 浪费显存 | 按 T 排序组 batch / nested tensor / mask |
| 15 步 GNN 过平滑 | 低 | 节点嵌入趋同 | residual（MGN 已有）+ recompute_activation；消融对比 7/15 步 |
| `MeshGraphNet` decoder 旁路接口不稳 | 中 | 集成报错 | 采用 §7.2 路径 (a)，加单测确认 processor 输出形状 |
| pymetis Windows 安装困难 | 中 | 无法分区 | fallback 谱聚类；或 WSL/容器内装 |
| EAGLE 数据规模大，单卡训不动 | 中 | 扩展受阻 | DDP 多卡 + AMR 减 token；必要时引入 X-MGN 式 Halo 分块（见 §十二）|

---

## 十、与开题 PPT 的对应关系

| PPT 章节 | 本文档对应 |
| --- | --- |
| 研究背景（CNN→MGN→X-MGN→Transformer）| §一 |
| 研究意义及目标（长程丢失、过平滑、混合架构+AMR）| §1.2 / §三 / §四 |
| 创新点（物理驱动 Token 剪枝、混合架构、动态聚合）| §三 |
| 实验方案（EAGLE/METIS/Halo、局部 GNN、全局 Token 剪枝、自回归+噪声、对比 MGN/X-MGN）| §六 |
| 评估指标（RMSE、长程累积、显存峰值、推理延迟、过平滑）| §6.4 |
| 计划进度 | §八 |

---

## 十一、后续扩展方向

1. **EAGLE / 大规模湍流**：迁移到数千~百万节点网格，验证 AMR 剪枝的算力节省与 Transformer 长程优势。
2. **X-MGN 式可扩展性（正交叠加）**：当单图大到塞不进单卡时，引入 METIS 分区 + Halo 重叠区（厚度=消息传递层数）+ 梯度聚合，使分块训练严格等价于整图——与本工作 Transformer 全局层**正交**，可叠加。小网格主线不启用。
3. **3D 扩展**：四叉树→八叉树，2D METIS→3D METIS，Laplacian 3D 版本。
4. **多物理场**：加密度、温度通道，AMR 判据扩展为「密度梯度」等。
5. **可学习阈值**：把 AMR 阈值从采样改为可学习参数（Gumbel-Softmax 或强化学习，参考 Swarm RL for AMR）。

---

## 十二、参考文献

1. Xu et al., "AMR-Transformer: Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation," CVPR 2025.
2. Lei et al., "M4GN: Mesh-based Multi-segment Hierarchical Graph Network for Dynamic Simulations," TMLR 2025.
3. Nabian et al., "X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation," NVIDIA 2024.
4. Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks," ICLR 2021.
5. Janny et al., "EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers," ICLR 2023.
6. Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs," SIAM J. Sci. Comput., 1998.
7. Achanta et al., "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods," IEEE TPAMI, 2012.
8. Godwin et al., "Simple GNN Regularisation for 3D Molecular Property Prediction & Beyond," 2021（噪声注入）.
9. Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer (GPS)," NeurIPS 2022（RWSE PE）.

---

## 十三、总结

本工作提出 **AMR-M4GN** 混合架构，创新性融合三个思想：

1. **M4GN 的层次化分区**：METIS+SLIC+模态分解，保证分区连通性、几何保真与物理一致。
2. **AMR-Transformer 的动态 Token 化**：把结构化四叉树的物理驱动剪枝迁移到非结构层次 METIS 树，根据实时物理状态自适应调节计算粒度。
3. **深层 GNN + 段级 Transformer 的互补分工**：GNN 管局部精细物理（高频/短程），Transformer 管全局长程交互（低频/全局压力耦合）。

该方案在理论上几乎不增加计算成本（Transformer 部分仅占总 FLOPs 的 ~1%），同时从机理上解决 MGN 的长程依赖丢失与过平滑问题，有望显著提升 rollout 稳定性。实现完全基于现有 PhysicsNeMo 生态，与原 MGN baseline 兼容共存，便于公平对比；并预留向 EAGLE 大规模湍流与 X-MGN 式 Halo 扩展的清晰路径。

# AMR-M4GN：面向大规模流体仿真的自适应 Token 化多尺度图-Transformer 代理模型

> 毕业论文初稿 · 中文 · 大白话版
> 作者：储润东（124037910068）
> 日期：2026-06-12
> 仓库：`E:\phys\physicsnemo\`
> 配套设计文档：`docs/design.md`；阶段开发手册：`docs/progress/M1.md`〜`M7.md`

---

## 摘要

把流体的运动算清楚一直很费机器。传统方法（CFD 数值解算）精度高，但跑一个高雷诺数湍流可能要几个小时甚至几天；近几年深度学习代理模型试图用神经网络一次前向就把下一帧"预测"出来，已经能做到 10–100 倍加速。但目前这条路上有几个绕不开的痛点：

1. **CNN 系**只能处理规则网格，遇到不规则几何（圆柱、机翼、复杂边界）就要做体素化插值，把几何细节糊掉；
2. **图神经网络系（以 MeshGraphNet 为代表）**完美支持任意网格，但消息传递要"一跳一跳"地传，**长程依赖天然弱**——压力是瞬时全局耦合的物理量，靠 15 步消息传递根本传不过去；
3. **Transformer 系**全局自注意力一步到位，但节点级注意力是 O(N²) 的，几十万节点根本上不去；
4. 最近的 **AMR-Transformer**（CVPR 2025）想出"按物理状态动态决定 Token 多少"的好主意，可是它只在结构化笛卡尔网格的四叉树上做，**用不到非结构三角网格上**。

本文提出 **AMR-M4GN**：把"局部 GNN（管短程）+ 段级 Transformer（管长程）+ 物理驱动的自适应 Token 化（管效率）"这三个东西**焊到同一套非结构网格的层次分区树上**。具体做法是先离线把网格切成两层段树（粗 64 段、细 256 段），每帧根据涡量、速度梯度等物理量动态决定哪些段用细粒度、哪些折回粗粒度，得到一个**变长 Token 序列**送进 Transformer。这样 Transformer 看到的 Token 数远少于节点数（K≪N），既保留了全局视野，又把算力压回可承受范围。

在 DeepMind cylinder_flow 数据集（圆柱绕流，~5000 节点）上的小规模验证（20 个训练算例 × 100 步 × 200 轮）表明：相同训练预算下，AMR-M4GN 在 test 集上的 rollout 误差**全程低于** MGN baseline（平均 RMSE 0.078 vs 0.094，低约 17%），训练集 NMSE 也更低（6.22e-3 vs 8.24e-3）。本文还实现了完整的模块消融脚本与 EAGLE 大规模数据集的扩展接口，供后续在大算力上验证。

**关键词**：流体仿真代理模型；图神经网络；Transformer；自适应网格细化；MeshGraphNet

---

## 第一章 引言

### 1.1 选题背景：为什么要做"流体仿真代理模型"

现代图形学、工程仿真（汽车空气动力、飞行器气动、流体力学教学演示）对流体运动的视觉与物理保真度要求越来越高：水花、激波、湍流尾迹这些细节往往需要几百万甚至上亿网格才能解算清楚。传统数值方法（有限元 / 有限体积 / SPH）在面对**大规模、高雷诺数**湍流时陷入"精度 vs 成本"的硬约束——网格越细、时间步越短、精度越高，但**单帧计算时间可能从几秒涨到几小时甚至几天**。

近年深度学习代理模型（neural surrogate）的思路是：**用一个神经网络去学"上一帧→下一帧"的映射**，训练时离线消耗算力，推理时一次前向就给出下一帧，从而实现 10× ~ 1000× 的加速。这个方向已经从 CNN（UNet 风格）、GNN（MeshGraphNet 系列）一路演进到 Transformer，但**没有一个方法能同时满足以下四点**：

1. 支持任意非结构网格（不必体素化）；
2. 长程物理量（如压力）能瞬时全局耦合；
3. 计算量不爆炸到 O(N²)；
4. 能根据流场状态自适应分配算力（活跃区算细、平静区算粗）。

[图 1-1：传统 CFD vs 神经代理模型的精度-速度权衡示意图（占位）]

### 1.2 现有方法存在的问题

把现有方法按时间和能力梳理一下：

| 阶段 | 代表方法 | 能干什么 | 干不了什么 |
| --- | --- | --- | --- |
| CNN | UNet | 规则网格、加速明显 | 不规则几何要体素化、丢边界细节 |
| GNN | MeshGraphNet（DeepMind 2020）| 任意非结构网格 | 长程依赖弱（要靠多跳传递）、过平滑 |
| 多尺度 GNN | X-MeshGraphNet（NVIDIA 2024）| 大规模图能训（METIS 分区+Halo）| **本质还是消息传递，长程仍要逐跳** |
| Mesh Transformer | EAGLE（ICLR 2023）| 全局自注意力 | 节点级 O(N²)，规模上不去 |
| 自适应 Token | AMR-Transformer（CVPR 2025）| 物理驱动动态 Token 化 | **只支持结构化网格的四叉树** |
| 层次化分区 | M4GN（TMLR 2025）| 段级 Transformer、复杂度 O(K²)≪O(N²)| **段数固定 K**，活跃/平静一视同仁 |

**MeshGraphNet 的瓶颈细看**：它是 Encoder–Processor–Decoder 三段，Processor 跑 L=15 步消息传递。每步把节点和它的图邻居做一次特征聚合（类似一次"低通滤波"），所以一个节点最远能"看到"15 跳邻居。问题是：

- **长程信号丢失**：圆柱绕流里，上游来流的压力变化要瞬间影响下游尾迹，物理上是整场耦合。15 跳根本传不到——传到的时候已经迟了，相位错了。
- **过平滑（over-smoothing）**：消息传递做多了，深层节点嵌入越来越像，丢局部高频细节。
- **没有自适应**：来流均匀区和尾迹涡街区投入同样算力，浪费大头。

### 1.3 本文工作

我们提出 **AMR-M4GN**，思路其实很朴素：

> "**让局部短程的事归 GNN 管，长程全局的事归 Transformer 管；至于 Transformer 的 Token 数太多算不动——那就让流体自己告诉我们哪儿要细哪儿不要细。**"

具体三步落地：

1. **离线把网格切成层次段树**（M4GN 的混合分区思路）：用 METIS 粗分成 64 段、再 SLIC 精修，每段内部递归切成 4 块得到 256 个细段。这样网格上每个节点都有"所属粗段 / 所属细段"两个 ID。
2. **在线每帧动态决定 Token 粒度**（AMR-Transformer 的物理驱动思路，迁移到非结构层次树）：在细段上算速度梯度 G、涡量 ω、动量 M、应变率 S，超过阈值就保持细粒度，否则折回粗段。这样 Transformer 看到的 Token 数 T 在 64~256 之间动态变化。
3. **GNN+Transformer 合作**：节点先过 15 步 MGN 拿到局部特征 h_node，h_node 按 token 分组做均值得到段特征 h_seg，h_seg 进 4 层 Transformer 做全局自注意力得到 h_seg'，再 scatter 回节点拼接成 [h_node, h_global]，最后一个解码 MLP 输出 (Δu, Δv, p)。

[图 1-2：AMR-M4GN 总体架构示意图（占位，对应 docs/figures 里的架构图）]

**主要贡献**：

1. **方法**：首次把 AMR-Transformer 的"物理驱动动态 Token 剪枝"从结构化四叉树**搬到非结构网格的层次 METIS 分区树**上，得到一个真正能在任意三角网格上跑的自适应 Token 化模块；
2. **架构**：把 M4GN 的层次分区骨架 + AMR-Transformer 的动态 Token + MGN 的 15 步深度局部 GNN 三件事**焊接成一个端到端管线**，并保证基线路径（不开任何模块）等价于纯 MGN，便于公平消融；
3. **工程**：在 PhysicsNeMo 框架上实现完整代码（10+ 模块、12 个入口脚本、10 个 pytest 单测）+ 阶段开发文档 + EAGLE 数据集扩展接口；
4. **实验**：在 cylinder_flow 上小规模验证 AMR-M4GN 全程优于 MGN（17% 平均 RMSE 改善）；M6 八组消融、M7 EAGLE 大规模扩展的代码、命令、文档全部就绪，待算力。

### 1.4 论文结构

- 第二章：相关工作。从 CFD 数值方法到神经代理模型的演进，重点拆 MGN / X-MGN / M4GN / AMR-Transformer / EAGLE 这几篇关键论文。
- 第三章：AMR-M4GN 方法。先讲离线预处理（模态分解、混合分区、位置编码），再讲在线前向（局部 GNN、物理量算子、AMR 路由、段级 Transformer、特征分发与解码），最后讲训练目标。
- 第四章：实验。数据集、训练设置、AMR vs MGN 主对比、阈值标定、模块消融、EAGLE 扩展。
- 第五章：结论与展望。

---

## 第二章 相关工作

> 本章只引用本仓库 `docs/papers/Paper/` 已收集的 18 篇论文，按"做什么 / 启发了我们什么 / 我们和它的差别"三段式过一遍。

### 2.1 流体仿真发展脉络（传统 → 神经）

[图 2-1：流体仿真方法演进时间轴（占位）]

**传统 CFD**：有限元、有限体积、谱方法等。算得准，但贵。论文`An unstructured adaptive mesh refinement for steady flows based on physics-informed neural networks` 是 PINN 方向的代表，把物理方程作为软约束嵌进神经网络。它和我们的差别：PINN 一般针对单个算例做隐式求解，没法做大规模数据驱动训练；我们走的是数据驱动监督学习路线。

**Position-based / XPBD / FLIP**（论文 `PBD.pdf`、`XPBD.pdf`、`Adaptive_Phase_Field_FLIP_preprint.pdf`、`Neural Monte Carlo Fluid Simulation.pdf`、`FastSubspaceFluidSimulation.pdf`）：这些是图形学社区追求"实时、视觉真实感"的传统方法。算法很巧妙、视觉效果好，但精度和可解释性偏弱，主要做交互式仿真。我们的目标是物理精度可量化的 CFD 代理，不直接竞争这一系。

### 2.2 MeshGraphNet 系列：图神经网络做流体的开山之作

**Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021**（仓库：`Learning Mesh-Based Simulation with Graph Networks.pdf` / `MeshGraphNets.pdf`）

- **做了什么**：把网格当成图，每个节点是顶点，每条边是网格边。模型 Encoder–Processor–Decoder：先把节点和边的物理量（速度、相对坐标）编码到 128 维，然后做 15 层消息传递（每步聚合邻居），最后解码出每个节点下一帧的物理量增量。
- **启发**：证明了图神经网络可以直接吃任意非结构网格，**不需要体素化**。这是我们的局部建模骨架。
- **不足**：长程依赖弱（15 跳的局部视野）、过平滑、不自适应。

**X-MeshGraphNet**（论文：`X-MeshGraphNet Scalable Multi-Scale Graph Neural Networks for Physicis Simulation.pdf`）

- **做了什么**：用 METIS 把大图切成子图带 Halo（光环重叠区），分块训练；多尺度图（粗细分辨率叠加）扩感受野。
- **启发**：分区思路 + Halo 机制对显存爆炸很有效。
- **不足**：本质还是消息传递，**没解决长程依赖延迟问题**——只是让大图能塞进 GPU。我们把它当对比 baseline。

**Mesh-based Super-Resolution of Fluid Flows with Multiscale Graph Neural Networks**（仓库同名 PDF）

- **做了什么**：多尺度 GNN 做流场超分辨率。
- **启发**：多尺度是对的，但他们做的是"分辨率提升"，我们做的是"自适应 Token 化"。

### 2.3 M4GN：层次化分区 + 段级 Transformer

**Lei et al., "M4GN: Mesh-based Multi-segment Hierarchical Graph Network for Dynamic Simulations", TMLR 2025**（仓库：`M4GN.pdf`）

- **做了什么**：三层架构。Micro（局部 GNN 7 步消息传递）+ Intermediate（混合分区，离线）+ Macro（段级 Transformer 全局自注意力）。
- **混合分区**：METIS 粗分（连通、最小边切割、快） + SLIC 超像素精修（基于 Laplacian 模态分解特征，物理一致）。这一步是重点：他们论文 Table 1 论证 METIS+SLIC 在连通性 / 几何保真 / 物理感知四项全面优于 learnable pooling、k-means 等替代方案。
- **段级 Transformer**：在 K 个段上做全连接自注意力，复杂度从 O(N²) 降到 O(K²)，**K≪N**。
- **位置编码**：段级 RWSE（随机游走结构编码） + 节点级绝对 PE。
- **重叠分区 δ=1**：段间允许并入一圈邻居，平滑边界。
- **启发**：层次分区 + 段级 Transformer 是把 Transformer 用在非结构网格上的最佳实践之一。我们的"层次 METIS 树"骨架直接借自 M4GN。
- **不足**：**Segment 数 K 固定**，活跃和平静区域分到一样多的段，仍有冗余。我们正是要把这个"固定 K"换成"动态 T"。

**Eagle dataset & Mesh Transformer**（论文：`Eagle_Large-Scale_Learning_of_Turbulent_Fluid_Dyna.pdf`）

- **做了什么**：节点聚类 + 图池化 + 全局注意力 + GRU 段聚合。给社区贡献了 110 万快照的大规模湍流数据集。
- **启发**：M4GN 用 average-pool+MLP 替代了它的 GRU，复杂度从 O(Nd²) 降到 O(Nd)。我们沿用 M4GN 的做法。EAGLE 数据集本身是我们大规模扩展（M7）的目标数据。

### 2.4 自适应网格细化与 AMR-Transformer

**Xu et al., "AMR-Transformer: Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation", CVPR 2025**（仓库：`AMR-Transformer Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation.pdf`）

- **做了什么**：用自适应网格细化（AMR）作为 Tokenizer——基于多叉树（2D 四叉树），自顶向下逐层细分；用 Navier-Stokes 物理约束决定哪些区域细分保留、哪些合并粗化。四个物理判据：速度梯度 G、涡量 ω、动量 M、Kelvin-Helmholtz 剪切 S。再加一个虚拟步外推：用前向欧拉 `u' = u_t + (u_t − u_{t-1})` 估计虚拟速度场，提前细化"即将变活跃"区域。
- **效果**：Token 数减少 2–10 倍；因 self-attention 是 O(N²)，FLOPs 最高减 60×。
- **启发**：物理驱动的动态剪枝是个好东西。
- **不足**：**只支持结构化网格的四叉树**。我们的核心创新就是把这套思想搬到非结构层次 METIS 树上。

**Adaptive_Physics_Transformer with Fused Global-Local Attention for Subsurface Energy Systems**（仓库同名 PDF）

- **做了什么**：地下能源系统的物理 Transformer，融合全局-局部注意力。
- **启发**：全局+局部融合是趋势。我们也是同样思路（GNN 局部 + Transformer 全局）。

**Fast Fluid Simulation via Dynamic Multi-Scale Gridding**（仓库同名 PDF）、**Swarm Reinforcement Learning for Adaptive Mesh Refinement**

- **做了什么**：动态多尺度网格 / 用 RL 学 AMR 细分策略。
- **启发**：动态网格化是个共识。但 RL 做 AMR 训练复杂、采样昂贵，我们走的是"物理量阈值 + 可微剪枝"的轻量路线。

### 2.5 网格生成相关

**Graph Neural Networks for Mesh Generation and Adaptation in Structural and Fluid Mechanics**（仓库同名 PDF）

- **做了什么**：用 GNN 做网格生成 / 自适应。
- **启发**：网格本身可以学；但我们的工作不动网格，只在固定网格上动态聚合 Token。

### 2.6 与本文的差别小结

| 维度 | MGN | X-MGN | M4GN | AMR-Transformer | **AMR-M4GN（本文）** |
| --- | --- | --- | --- | --- | --- |
| 数据类型 | 非结构 | 非结构 | 非结构 | 结构化 | **非结构** |
| 长程机制 | 逐跳消息 | 逐跳+多尺度 | 段级 Transformer | 全局 Transformer | 段级 Transformer |
| Token 数 | — | — | **固定 K** | 动态四叉树 | **动态层次 AMR** |
| 物理先验 | 无 | 无 | 模态（静态） | N-S（动态） | **模态+N-S（静+动）** |
| 局部建模 | 15 步 GNN | 15 步 GNN | 7 步 GNN | 无 | **15 步 GNN** |
| 自适应性 | 无 | 无 | 无 | 有 | **有** |

一句话：**M4GN 的层次骨架 × AMR-Transformer 的物理驱动动态 Token，迁移到非结构网格**。

---

## 第三章 方法：AMR-M4GN

### 3.1 总体框架

整个流程分两段：**离线预处理**（每个仿真算例的几何固定，只算一次）和**在线前向**（每个时间步执行）。这样设计是因为分区、位置编码这些"几何相关"的东西不依赖时间步，提前算好写盘缓存即可，训练/推理时直接读，省一大半时间。

```
                          OFFLINE（每个 case 一次性，写盘缓存）
  ┌────────────────────────────────────────────────────────────────────┐
  │  ① Laplacian 模态分解 (m=6 模, 余切 FEM, Neumann 边界)              │
  │  ② 障碍物有符号距离 f_obs                                            │
  │  ③ METIS 粗分 (K0=64) + SLIC 精修 → L0                              │
  │  ④ 递归 METIS：每个 L0 段内切 4 块 → L1 (K1≈256)                    │
  │  ⑤ RWSE 段级位置编码（L0/L1 各算一份，16 步随机游走）               │
  │  ⑥ 节点级绝对位置编码                                               │
  │  ⑦ 写盘缓存 partition_cache.pt                                      │
  └────────────────────────────────────────────────────────────────────┘

                          ONLINE（每个时间步前向）
  ┌────────────────────────────────────────────────────────────────────┐
  │  graph_t ─► ① Micro GNN (15步 MGN, 去掉 decoder) ─► h_node [N,d]    │
  │          ─► ② 物理量算子 (G, ω, M, S)         ─► phys [N,4]         │
  │          ─► ③ AMR Router：活跃段保 L1、平静段折回 L0                │
  │                 → 变长 token 数 T (64 ≤ T ≤ 256)                    │
  │          ─► ④ Segment Encoding：mean-pool h_node + RWSE PE          │
  │                                              ─► h_seg [T, d]        │
  │          ─► ⑤ Macro Transformer (4层×8头)   ─► h_seg' [T, d]        │
  │          ─► ⑥ Feature Dispatch：scatter 回节点                       │
  │                                              ─► h_global [N, d]     │
  │          ─► ⑦ Decoder MLP → (Δu, Δv, p̂) [N, 3]                    │
  └────────────────────────────────────────────────────────────────────┘
```

[图 3-1：AMR-M4GN 总体架构示意（占位）]

### 3.2 离线预处理

#### 3.2.1 模态分解（Laplacian Eigenfunctions）

**目的**：给每个网格节点算一组"几何-物理结构特征"，让后面分区算法分得物理一致。

**做法**：解一个广义特征值问题：

```
−∇²φ = λφ,   带边界条件
取最小的 m=6 个非零特征值对应的特征向量
节点 i 的特征 f_md(i) = (φ₁(i), …, φ₆(i)) ∈ ℝ⁶
```

实现要点（代码 `amr_m4gn/modal_decomp.py`）：

- 用余切 FEM Laplacian（对 -∇² 一致离散，物理意义好）；
- Neumann 边界（默认）跳过常数零模；
- `scipy.sparse.linalg.eigsh` 加 shift-invert（sigma=0）加速小特征值收敛。

**直观理解**：低频模捕捉大尺度流动结构（比如来流方向），中频模反映圆柱附近的分离与回流。模态值相近的节点动力学行为相似，会被自动分到同一段。

[图 3-2：圆柱绕流网格上前 6 个 Laplacian 模态可视化（占位，对应 docs/figures/图片1.png 之类）]

#### 3.2.2 混合分区：METIS + SLIC

**两步走**（代码 `amr_m4gn/segmentation.py`）：

**Stage 1 — METIS 粗分**：把整张网格图切成 K0=64 个大致等大、连通、跨段边最少的子图。优先用 `pymetis`，没装就退化到谱聚类（更慢、质量稍差）。

**Stage 2 — SLIC 精修**：基于物理感知特征做 K-means 变体，距离度量是：

```
d(i, C_k) = ‖f_obs_i − f_obs_Ck‖     # 障碍物距离差
          + ‖f_md_i − f_md_Ck‖       # 模态特征差
          + τ · ‖x_i − x_Ck‖         # 空间位置差
```

其中 `f_obs` 是节点到固体壁面的距离（自动从 node_type==6 算起），`f_md` 是 §3.2.1 的 6 维模态特征，`x` 是 2D 坐标。所有特征都归一化到 [0,1]，τ=1.0 控制段的紧凑性。**关键约束**：节点只能重分配到"邻居所属"的段，避免跨空隙连接（这是 same-size k-means 的经典缺陷）。

**递归两层段树**：

- L0（粗）：K0=64 段，整图做一次 METIS+SLIC；
- L1（细）：对每个 L0 段内部再 METIS 切 4 块，K1≈256 段；
- 过小的段（<4 节点）不再细分。

每个节点最终带两个 ID：`L0_assign[i]` 和 `L1_assign[i]`。

[图 3-3：圆柱绕流网格的 L0 / L1 分区可视化（占位）]

#### 3.2.3 位置编码：段级 RWSE + 节点级绝对 PE

**段级 RWSE（Random Walk Structural Encoding）**（代码 `amr_m4gn/pe.py`）：

1. 对每层算段间邻接：A_K[Si,Sj] = 两段间跨段边数；
2. 转移矩阵 P = D⁻¹A_K，算 diag(P), diag(P²), …, diag(P¹⁶) → 16 维向量；
3. 第 j 维 = "从段 k 出发走 j 步回到自身的概率"，编码段在拓扑图中的局部连通结构。

**节点级绝对 PE**：MLP 处理每节点的 (x,y) → 加到 Encoder 输入。

#### 3.2.4 缓存内容

预处理完写盘 `partition_cache.pt`，里面包含：levels（L0/L1 分配）、seg_adj（段邻接）、rwse（L0/L1 各一份位置编码）、l1_to_l0 映射、segment centroids、node Voronoi area、节点位置等。训练时按 case 索引懒加载。

支持**并行预处理**（M7 引入）：`preprocess_partitions.py --workers 8` 用 `ProcessPoolExecutor` 并发跑多个 case，对 EAGLE 这种几百到几千 case 的大数据集很关键。

### 3.3 在线前向

#### 3.3.1 局部 Micro-GNN（15 步消息传递）

直接复用 PhysicsNeMo 的 `MeshGraphNet`，**但去掉它的 decoder 头**——decoder 推迟到 Macro 之后。即只取 `node_encoder + edge_encoder + processor` 的输出 `h_node ∈ ℝ^{N×128}`。

**为什么 15 步**：M4GN 自己用的是 7 步，我们沿用 MGN 默认的 15 步。15 步意味着每节点能聚合 15 跳邻居的信息，覆盖圆柱直径的 2~3 倍范围。**与 Transformer 互补**：GNN 管高频/短程（黏性扩散、局部涡），Transformer 管低频/长程（压力波）。

#### 3.3.2 物理量算子（G, ω, M, S）

参考 AMR-Transformer 的四个判据，在节点上算（代码 `amr_m4gn/physics_ops.py`）：

- **速度梯度幅值 G** = ‖∇u‖ + ‖∇v‖（用三角网格梯度算子）
- **涡量 ω** = ∂v/∂x − ∂u/∂y
- **动量 M** = √(u² + v²)
- **应变率幅值 S** = √(2 e_ij e_ij)（替代了 AMR-Transformer 论文的 KH 剪切——KH 剪切公式在 t=0 速度为 0 时全场为 0、不可用，开发期 D1 决策门拍板用应变率幅值代替）

这些物理量在归一化空间还是物理空间算？开发期 D1 决策门拍板**用反归一化的物理空间**，物理可解释性更强。

#### 3.3.3 自适应 Token 路由（AMR Router）

**核心 idea**（代码 `amr_m4gn/amr_router.py`）：在 L1 细段上聚合物理量；超过阈值的段 → 保 L1 细粒度；否则折回 L0 粗段。这样**变长 Token 数 T 在 64~256 之间动态变化**。

```
对每个 L1 段 s:
    ω̄_s = mean(|ω_i|, i ∈ s)
    Ḡ_s = mean(G_i, i ∈ s)
    ...
若 ω̄_s > thr_ω 或 Ḡ_s > thr_G 或 ... :
    保留 s 作为 L1 token
否则:
    把 s 折回它的 L0 父段（同父段的多个子段合并成一个 token）
```

阈值 `thr_ω` 等怎么定？开发期 M5 引入 `calibrate_thresholds.py`：扫一段时间窗内 ω 分布，把"该激活的尾迹区"和"该平静的来流区"的分位数对齐，**圆柱绕流上拍板 ω≈8.9**。

[图 3-4：AMR Router 路由结果可视化——尾迹保细、来流折粗（占位）]

#### 3.3.4 Segment Encoding + Macro Transformer

**Segment Encoding**（代码 `amr_m4gn/macro_transformer.py`）：

1. 对每个 token，取它包含的所有节点的 h_node 做 **mean-pool**（M4GN 用的 average-pool+MLP，不是 EAGLE 的 GRU——更快、置换不变）；
2. 加上 RWSE 段级位置编码（depth 信息：来自 L0 还是 L1）；
3. 加上段中心坐标 (x̄, ȳ) 编码；
4. 得到 h_seg ∈ ℝ^{T×128}。

**Macro Transformer**：4 层 × 8 头自注意力，Pre-LN。复杂度 O(T²) ≈ O(K²) ≪ O(N²)。

#### 3.3.5 特征分发与解码

**Feature Dispatch**：把 h_seg' scatter 回每个节点（节点取它所属 token 的特征）→ h_global ∈ ℝ^{N×128}。

**Decoder**：拼接 [h_node, h_global] ∈ ℝ^{N×256}，过一个 MLP → (Δu, Δv, p̂) ∈ ℝ^{N×3}。

预测增量 Δu/Δv（不是绝对速度）+ 绝对压强 p。增量预测对 rollout 长程稳定性更好。

#### 3.3.6 几个工程细节

- **批处理（PyG batch）**：M5 引入 `pack_segments` / `run_macro_batched`，把同一 batch 内多个 case 的段拼成一个大序列做注意力，**用 attention mask 隔离不同 case**。这一步对训练吞吐很关键。
- **段重叠 δ=1**（M6 消融）：段间允许并入一圈邻居（halo），平滑边界。运行期用 `edge_index + kept_assign` 直接计算，不改缓存。
- **虚拟步外推**（M6 消融）：路由时用前向欧拉虚拟场 `u' = u_t + (u_t − u_{t-1})` 代替当前场，提前细化"即将变活跃"区域。`u_{t-1}` 由 `data/vortex.py` 暴露的 `graph.x_prev` 提供。
- **零开销消融开关**：模型构造函数加 5 个布尔开关 `use_amr / use_transformer / use_rwse / use_overlap / use_virtual_step`，默认配置等价于纯 MGN（关掉 use_transformer 时 Macro 路径短路），保证消融时**结构等价、参数可比**。

### 3.4 训练目标

- **损失**：per-channel NMSE（归一化均方误差），即 `mean(MSE_c / mean(target_c²))`。NMSE 比 MSE 好的地方：**自动归一化不同物理量的尺度**（u/v/p 单位、量级都不一样）。
- **噪声注入**（Godwin et al. 2021）：训练时给输入图加 σ=0.02 的高斯噪声，rollout 长程稳定性显著提升。
- **优化器**：Adam，lr=1e-3 → 余弦退火到 1e-5；epoch=200。
- **batch_size=2**（M5），AMR 和 MGN baseline 用**完全相同的训练预算**（同 case 数 / 步数 / epoch / noise / lr / NMSE）保证公平。

**训练数值稳定性**（实战发现并修复）：

batched 训练初版无梯度防护时，在 20 case × 100 步 × 200 epoch 设定下会**在 epoch 5~10 突发发散**——某 batch 的极端梯度（多由物理量算子 G/ω/M/S 在随机权重下的高方差导致）把权重推进死区，后续所有 batch 的输出塌缩到 0，NMSE 卡在 ~1.0。修复后默认开启三件套：

1. **梯度范数裁剪** `clip_grad_norm_(max_norm=1.0)`：最关键的一项，把单步参数更新限制在合理范围；
2. **线性 lr 暖启动**：前 5 epoch lr 从 base_lr/10 线性升到 base_lr，跨过早期梯度尖峰；
3. **NaN/inf batch 守护**：单个非有限 loss 直接跳过该 batch，杜绝污染权重。

附带每 5 epoch 打印 `grad_norm` 与 `nan_skipped` 字段，便于事后诊断。AMR 与 MGN baseline 同步加同样默认值，保证对比公平。详见 `docs/progress/M5.md` §9 失败案例。

---

## 第四章 实验

### 4.1 数据集

**主数据集：DeepMind cylinder_flow**
- 2D 圆柱绕流，~5000 节点的非结构三角网格；
- 600 个仿真算例，每个 600 时间步；
- 来源：DeepMind MeshGraphNet 公开数据集（TFRecord 格式）；
- 训练用 20 case × 100 步，测试用 1~10 case。

**目标数据集：EAGLE（M7 扩展）**
- ~110 万个 2D 非结构网格快照，600 个不同场景，3 种几何（spline / triangular / step）；
- 高度湍流、非周期涡（移动气流源 + 非线性场景结构）；
- 论文：Janny et al., ICLR 2023；
- 状态：reader + 并行预处理代码已实现并通过合成数据单测；真实数据待目标机算力。

### 4.2 评价指标

- **训练 NMSE**：训练 in-sample 收敛指标；
- **rollout RMSE**：从测试集某 case 的 t=0 出发，模型自回归 80 步，每步算速度 RMSE（物理空间），看误差累积；
- **mean / first / final RMSE**：rollout 80 步的均值 / 第 1 步 / 最末步 RMSE；
- **多 case 平均**（M5+M6 加）：N 个 test case 的 mean ± std。

[图 4-1：rollout 评估示意（自回归过程）（占位）]

### 4.3 主结果：AMR-M4GN vs MGN（M5）

**实验设置**：20 case × 100 步 × 200 epoch；noise_std=0.02；ω 阈值=8.9；test case 0；rollout 80 步。AMR 和 MGN baseline **完全同预算**，唯一差别是模型。

**训练 NMSE（in-sample）**：

| 模型 | 最终 NMSE (epoch199) | 最低 NMSE | 参数量 |
| --- | --- | --- | --- |
| **AMR-M4GN** | **6.22 × 10⁻³** | ~5.4 × 10⁻³ (epoch180) | 3.18M |
| MGN baseline | 8.24 × 10⁻³ | — | 2.33M |

**rollout RMSE（test case 0，80 步）**：

| 指标 | AMR-M4GN | MGN baseline | 改善 |
| --- | --- | --- | --- |
| step 1 | **2.021 × 10⁻²** | 2.300 × 10⁻² | -12% |
| step 80 (final) | **1.143 × 10⁻¹** | 1.331 × 10⁻¹ | -14% |
| **mean** | **7.804 × 10⁻²** | 9.423 × 10⁻² | **-17%** |

**结论**：

1. AMR-M4GN 在训练 NMSE 上更低（6.22e-3 < 8.24e-3）；
2. 在 test rollout 上**全程低于** MGN，平均 RMSE 低 17%；
3. **长程优势更明显**——final step 改善 14%，远端误差累积慢；
4. test split 是模型未训练过的数据，所以这是泛化结果。

**注意点**：参数量 AMR 比 MGN 多 36%（多了 Transformer），不算严格等参；test case 单一，需要 §4.5 的多 case 平均补强；对比对象只有 MGN，X-MGN 等未做。

[图 4-2：AMR vs MGN rollout RMSE 曲线对比（占位，对应 12_compare_rollout_case0.png）]

[图 4-3：AMR / MGN / GT 三行场动画截屏（占位，对应 animations/compare_case0_*.gif）]

### 4.4 阈值标定（D3 决策门）

ω 阈值是 AMR Router 的关键超参。`calibrate_thresholds.py` 在 4 case × 50 步上扫 |ω| 分布，把"应保细的尾迹区"和"应折粗的来流区"的分位数对齐，得出 **ω ≈ 8.9** 是好的拍板值（M5 §4.4）。M6 实验也用同一阈值。

[图 4-4：阈值标定结果（占位，对应 11_threshold_calibration.png）]

### 4.5 模块消融实验（M6，待实跑）

**8 组配置**（代码 `scripts/run_ablation.py`，每组同预算 200 epoch 训练 + 同 rollout 80 步评估）：

| # | 配置 | 模型差异 | 期望解读 |
| --- | --- | --- | --- |
| 1 | `full` | 默认完整模型 | 基准 |
| 2 | `w/o AMR` | 路由阈值置 -∞，L1 全保细 | 验证 AMR 减算量是否伤精度 |
| 3 | `w/o Transformer` | 段编码器关闭，仅 micro-GNN | 验证 Transformer 长程贡献 |
| 4 | `w/o Modal` | 几何-only 分区（模态特征置零）| 验证模态分解引导价值 |
| 5 | `w/o RWSE` | RWSE 位置编码置零 | 验证位置编码贡献 |
| 6 | `proc7` | processor_size=7（默认 15）| 算力 vs 精度权衡 |
| 7 | `w/ overlap` | δ=1 halo 池化与分发 | 段边界平滑是否有益 |
| 8 | `w/ virtual` | 路由用前向欧拉虚拟场 | 提前细化是否有益 |

**读法**：mean RMSE 比 `full` **高**得越多 → 该模块净贡献越大、应保留；**持平或更低** → 候选裁剪。

**状态**：8 组代码、单测、阈值缓存、运行命令、出表/出图脚本全部就绪；本机无算力，待目标机实跑。M6 文档（`docs/progress/M6.md`）有完整复现命令与结果表占位。

[图 4-5：8 组消融柱状图（占位，对应 13_ablation.png）]

### 4.6 多 case 平均评估（补 M5 单 case 短板）

`eval_rollout.py` 在 N 个 test case 上对 AMR / MGN 各跑 rollout，输出 mean ± std 曲线。M5 单 test case 不构成定论，多 case 平均才严谨。**状态**：代码就绪，待算力。

[图 4-6：多 test case 平均 ±std 曲线（占位，对应 14_eval_multicase.png）]

### 4.7 EAGLE 大规模扩展（M7，待数据）

**目标**：把 cylinder_flow 上验证过的 AMR-M4GN 在 EAGLE（~110 万快照、湍流强度大得多的非周期涡）上跑通，与 MGN/X-MGN 对比，展示 AMR Token 节省与长程优势。

**状态**：

- ✅ EAGLE reader（`data/eagle.py`）：按官方文档 `mesh_pos / VX / VY / PS / node_type` 实现，三角形分离文件支持，合成数据单测通过；
- ✅ 并行预处理（`preprocess_partitions.py --source eagle --workers 8`）：EAGLE 几百到几千 case 必需；
- ✅ 训练/推理/评估脚本全 `--dataset eagle` 切换；
- 🟡 真实数据未验证：本机无 EAGLE 数据；用户已下载 `eagle_clusters.tar.gz`（仅聚类预算结果，无仿真数据）；正在下载 `triangular.tar.gz`（29G，sim 数据）。
- ⚠️ EAGLE 网格是**动态的**（每帧位置/cell 可能变），开发期文档化假设 A1：使用 t=0 mesh 的连接做整 sim 的固定分区，与 cylinder_flow 的"静止网格"假设一致。真正的逐帧重新分区是后续工作。

---

## 第五章 结论与展望

### 5.1 工作小结

本文针对**非结构网格上流体仿真代理模型的长程依赖弱、计算量爆炸、不能自适应**三个核心痛点，提出 AMR-M4GN 混合架构：把 M4GN 的层次分区骨架、AMR-Transformer 的物理驱动动态 Token 化（迁移到非结构层次树）、MGN 的 15 步深度局部 GNN **焊接到一个端到端管线**。

主要贡献：

1. **方法**：首次把 AMR-Transformer 的物理驱动动态剪枝从结构化四叉树搬到非结构网格的层次 METIS 分区树上；
2. **架构**：以"GNN 局部 + Transformer 段级全局 + AMR 动态 Token"三元组为骨架，用零开销消融开关保证基线路径等价于纯 MGN，便于公平对比；
3. **工程**：在 PhysicsNeMo 上实现完整代码（10 模块、12 入口、10 单测）+ 完整阶段开发文档（M1~M7） + EAGLE 数据集扩展接口；
4. **实验**：cylinder_flow 上小规模验证 AMR-M4GN 全程优于 MGN（rollout mean RMSE 低 17%），训练 NMSE 也更低；M6 消融、多 case 评估、EAGLE 扩展的代码、命令、文档全部就绪。

### 5.2 局限

诚实列一下：

1. **算力受限**：本机训练慢，主结果只是"小验证档"（20 case × 100 步 × 200 epoch、单 test case），不构成论文级定论；M6 八组消融、多 test case 平均、EAGLE 实跑都待目标机算力。
2. **对比对象单一**：只对比了 MGN baseline，未对比 X-MGN、M4GN 原版、AMR-Transformer。
3. **参数量不严格等参**：AMR 比 MGN 多 36%，公平性有待补充等参实验。
4. **EAGLE 真实数据未验证**：reader 按官方文档写、合成数据单测通过，但真实文件的字段名、节点类型 id、动态网格行为需在目标机上核对。
5. **无逐帧重分区**：当前对 EAGLE 这种动态网格用 t=0 固定分区（A1 假设），真正的逐帧 AMR 是后续工作。

### 5.3 未来工作

短期（1~2 个月内可补）：

- **算力到位后**：跑全量 cylinder_flow（200 case × 600 步），跑 M6 八组消融，跑 N 个 test case 平均评估，把 §4.5 / §4.6 的占位表/图填满；
- **下载 EAGLE triangular 子集**：跑通 M7 的 reader → 预处理 → 训练 → 评估全链条，与 MGN/X-MGN 对比，写进论文；
- **等参对比**：把 MGN 的 hidden 维度调到 156 让参数量也 ≈3.18M，做严格等参 baseline。

中期：

- **逐帧重分区**：对 EAGLE 这种动态网格，去掉 A1 假设，让分区缓存随时间自适应；
- **更多 AMR 物理判据**：现在用 G/ω/M/S，可以加入压力梯度、对流 CFL 等；
- **自学习的阈值**：当前 ω≈8.9 是离线标定的硬阈值，可以让网络端到端学一个软阈值（可微剪枝）；
- **3D 扩展**：当前算子按 2D 写，3D 需要改物理量与分区算法。

长期：

- **与 X-MGN 的合作**：X-MGN 解决"图大塞不进显存"，AMR-M4GN 解决"长程依赖弱+不自适应"，两者正交，理论上可以合并成 X-AMR-M4GN，同时支持百万节点 + 全局动态 Token；
- **从纯监督到物理一致性约束**：在 NMSE 损失外加上 N-S 残差软约束（PINN 思路），降低数据需求。

---

## 参考文献

> 以下论文均收录于本仓库 `docs/papers/Paper/` 目录。

1. Pfaff T, Fortunato M, Sanchez-Gonzalez A, Battaglia P W. *Learning Mesh-Based Simulation with Graph Networks*. ICLR 2021.（MeshGraphNet 主要参考，本文 GNN 局部建模骨架）
2. Lei et al. *M4GN: Mesh-based Multi-segment Hierarchical Graph Network for Dynamic Simulations*. TMLR 2025.（层次分区 + 段级 Transformer 思路来源）
3. Xu et al. *AMR-Transformer: Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation*. CVPR 2025.（物理驱动动态 Token 化思路来源）
4. Janny et al. *EAGLE: Large-scale Learning of Turbulent Fluid Dynamics with Mesh Transformers*. ICLR 2023.（M7 大规模扩展目标数据集）
5. NVIDIA. *X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation*. 2024.（对比 baseline，多尺度图 + Halo 思路）
6. Mesh-based Super-Resolution of Fluid Flows with Multiscale Graph Neural Networks.（多尺度 GNN 流场超分辨率）
7. Adaptive Physics Transformer with Fused Global-Local Attention for Subsurface Energy Systems.（全局-局部注意力融合）
8. Fast Fluid Simulation via Dynamic Multi-Scale Gridding.（动态多尺度网格化）
9. Swarm Reinforcement Learning for Adaptive Mesh Refinement.（RL 学 AMR 策略）
10. Graph Neural Networks for Mesh Generation and Adaptation in Structural and Fluid Mechanics.（GNN 做网格生成）
11. An unstructured adaptive mesh refinement for steady flows based on physics-informed neural networks.（PINN+AMR）
12. Adaptive Phase Field FLIP（FLIP 方法）
13. Position-Based Dynamics（PBD）/ XPBD：图形学传统物理引擎
14. FastSubspaceFluidSimulation.（子空间流体仿真）
15. Neural Monte Carlo Fluid Simulation.（神经蒙特卡洛流体）
16. Godwin et al. *Simple GNN Regularization for 3D Molecular Property Prediction & Beyond*. ICLR 2022.（噪声注入训练，M4GN §3.2 沿用）

---

## 附录 A：仓库结构与代码索引

```
physicsnemo/                    # 仓库根
├── amr_m4gn/                   # AMR-M4GN 模型包
│   ├── modal_decomp.py         #   §3.2.1 Laplacian 模态分解
│   ├── segmentation.py         #   §3.2.2 METIS+SLIC 两级分区树
│   ├── pe.py                   #   §3.2.3 RWSE 位置编码
│   ├── physics_ops.py          #   §3.3.2 物理量算子 + 虚拟步
│   ├── amr_router.py           #   §3.3.3 自适应 Token 路由
│   ├── micro_gnn.py            #   §3.3.1 MGN 旁路
│   ├── macro_transformer.py    #   §3.3.4 段编码 + Transformer
│   └── model.py                #   AMRM4GN 顶层（含 5 个消融开关）
├── data/                       # 数据集适配
│   ├── vortex.py               #   cylinder_flow（M4–M6 主用）
│   └── eagle.py                #   EAGLE（M7）
├── scripts/                    # 入口脚本
│   ├── preprocess_partitions.py
│   ├── train_amr_m4gn_full.py / train_mgn_baseline.py
│   ├── inference_amr_m4gn.py / compare_baselines.py
│   ├── eval_rollout.py / run_ablation.py
│   └── calibrate_thresholds.py / visualize_partition.py
├── tests/                      # 10 个 pytest 单测
├── conf/                       # Hydra 配置
├── docs/                       # 设计 + 阶段手册 + 论文 + 图
└── physicsnemo/                # NVIDIA PhysicsNeMo 框架（依赖）
```

## 附录 B：开发里程碑

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M1 | 预处理（模态分解 + 混合分区）| ✅ |
| M2 | N-S 物理量算子（G/ω/M/S）| ✅ |
| M3 | AMR Token 路由 | ✅ |
| M4 | 端到端管线 + overfit 自检（NMSE 0.92→0.013）| ✅ |
| M5 | 批处理 + 全量训练 + baseline 对比 | 🟢 小验证档通过（17% 改善），仅"更大训练集"待算力 |
| M6 | 八组模块消融 | 🟡 代码/脚本/单测就绪，实跑待算力 |
| M7 | EAGLE 大规模扩展 | 🟡 reader+并行预处理就绪，真实数据待验证 |

详见 `docs/progress/M1.md` 〜 `M7.md`，每篇按"做什么 → 为什么 → 应得什么 → 实测结果"四段写。





# AMR-M4GN：面向圆柱绕流的混合 GNN-Transformer 自适应网格架构

## 研究工作汇报文档

**作者**：储润东  
**日期**：2026年5月14日  
**仓库**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`  
**Baseline**：PhysicsNeMo MeshGraphNet (MGN) 圆柱绕流训练脚本

---

## 一、研究背景与问题

### 1.1 现有方法的局限性

当前使用 MeshGraphNet (MGN) 进行圆柱绕流 (Vortex Shedding) 仿真代理建模时，存在以下核心问题：

1. **长程依赖捕捉不足**：MGN 依赖消息传递 (Message Passing) 逐步扩展感受野，15步消息传递最远只能覆盖15跳邻居。对于圆柱绕流中压力波的全局传播（尤其是上游压力变化对下游尾迹的影响），需要大量消息传递步才能建立远距离联系。

2. **计算效率与精度的矛盾**：增加消息传递步数虽然能扩大感受野，但会带来：
   - 过平滑 (over-smoothing)：深层节点嵌入趋于一致，丧失局部细节
   - 计算代价线性增长：O(L * E * d^2)
   - 梯度消失/爆炸风险

3. **缺乏自适应性**：MGN 对所有区域一视同仁——来流均匀区域和尾迹涡街区域投入相同计算量，造成资源浪费。

### 1.2 两篇关键参考文献

#### 参考文献 1：AMR-Transformer (CVPR 2025)

- **论文**：Xu et al., "AMR-Transformer: Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation"
- **核心思想**：将自适应网格细化 (AMR) 作为 Tokenizer，基于 Navier-Stokes 物理量（速度梯度、涡量、动量、KH不稳定性）决定哪些区域需要细分保留、哪些区域可以合并粗化。合并后用 Transformer 的全局自注意力高效建模长程依赖。
- **关键贡献**：
  - Token 数量减少 2~10 倍，FLOPs 减少最高 60 倍（因为 self-attention 是 O(N^2)）
  - 在 CFDBench 上精度提升一个量级
- **局限**：仅适用于结构化网格 (H x W x c)，不能直接用于非结构三角网格

#### 参考文献 2：M4GN (TMLR 2025)

- **论文**：Lei et al., "M4GN: Mesh-based Multi-segment Hierarchical Graph Network for Dynamic Simulations"
- **核心思想**：三层层次化架构——Micro-level (局部 GNN 消息传递) + Intermediate-level (混合网格分区) + Macro-level (Segment Transformer 全局交互)
- **关键贡献**：
  - 混合分区策略：METIS 粗分 + SLIC 精修（基于模态分解特征），保证分区连通性、几何保真度、物理一致性
  - 置换不变聚合器替代 EAGLE 的 GRU，O(Nd) vs O(Nd^2)
  - 精度提升 56%，推理加速 22%
- **局限**：Segment 数量固定，不能根据物理状态动态调整

### 1.3 本工作的创新点

**核心思路**：将 M4GN 的层次化架构骨架 + AMR-Transformer 的物理感知自适应 token 化 + 15步深度 GNN 融合为一个统一框架。

**创新点总结**：

| 对比维度 | M4GN | AMR-Transformer | 本工作 (AMR-M4GN) |
| --- | --- | --- | --- |
| 数据类型 | 非结构网格 | 结构化网格 | 非结构网格 |
| Segment/Token 数 | 固定 K | 动态（四叉树） | 动态（AMR 决策） |
| 物理先验 | 模态分解（静态） | N-S 物理量（动态） | 模态分解 + N-S 量 |
| 局部建模 | 7步 GNN | 无 | 15步 GNN |
| 全局建模 | 固定 K 个 token Transformer | 变长 token Transformer | 变长 token Transformer |
| 自适应性 | 无 | 有 | 有 |

---

## 二、方法设计

### 2.1 总体架构

```
                          OFFLINE（每个 case 一次性预处理）
  ┌─────────────────────────────────────────────────────────────────┐
  │  ① Laplacian Eigenfunctions (m=6 modes)                         │
  │  ② METIS 粗分 (64 segments) + SLIC 精修 → L0                    │
  │  ③ 递归 METIS 在每个 L0 段内切4块 → L1 (256 segments)           │
  │  ④ RWSE 位置编码（两层各自计算）                                  │
  │  ⑤ 写盘缓存 partitions.pt                                       │
  └─────────────────────────────────────────────────────────────────┘

                          ONLINE（每个时间步的前向传播）
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  graph_t  ─► ① Micro-level GNN (15步 MGN)  ─► h_node [N, d]    │
  │                                                                  │
  │            ─► ② 物理量计算 (G, omega, M, S) ─► phys [N, 4]      │
  │                                                                  │
  │            ─► ③ AMR Router: L1 聚合 phys →                       │
  │                   活跃段保持细分 (L1)                             │
  │                   平静段折回粗层 (L0)                             │
  │                   → 变长 token 数 T (64 <= T <= 256)             │
  │                                                                  │
  │            ─► ④ Segment Encoding:                                │
  │                   mean-pool h_node per token                     │
  │                   + RWSE PE + (depth, x_mean, y_mean) PE         │
  │                   → h_seg [T, d]                                 │
  │                                                                  │
  │            ─► ⑤ Macro Transformer (4层 x 8头):                   │
  │                   全局自注意力 → h_seg' [T, d]                    │
  │                                                                  │
  │            ─► ⑥ Feature Dispatch:                                │
  │                   h_seg' scatter back → h_global [N, d]          │
  │                   拼接 h_cat = [h_node, h_global]                │
  │                                                                  │
  │            ─► ⑦ Decoder MLP → (du, dv, p_hat) [N, 3]            │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
```

### 2.2 各模块详细说明

#### 2.2.1 模态分解 (Modal Decomposition)

**目的**：提取网格上的几何-物理结构特征，用于指导分区算法产出"物理一致"的 segment。

**方法**：对流体问题，采用 Laplacian Eigenfunctions（M4GN 的流体路径）：

```
解 -nabla^2 phi = lambda phi
取最小的 m=6 个特征值对应的特征向量
每个节点 i 获得特征 f_md(i) = (phi_1(i), phi_2(i), ..., phi_6(i))
```

**物理意义**：
- 低频模态捕捉大尺度流动结构（如来流方向）
- 中频模态反映圆柱附近的流动分离和回流区
- 模态特征相近的节点倾向于有相似的动力学行为

**在圆柱绕流中的预期效果**：
- 圆柱正前方（驻点区）和两侧（加速区）模态值差异明显 → 被分到不同 segment
- 尾迹区域内部模态相似 → 被分为一组
- 边界层节点因靠近壁面，模态受 Dirichlet 约束 → 自动与远场分离

#### 2.2.2 混合网格分区 (Hybrid Segmentation)

**两阶段流程**：

**Stage 1：METIS 粗分**
- 输入：网格图 G = (V, E)
- 输出：K=64 个大致等尺寸的连通子图
- 优势：保证连通性、最小化跨 segment 的边切割

**Stage 2：SLIC 精修**
- 基于物理感知特征的 K-means 变体
- 距离度量：

```
d(i, C_k) = ||f_md_i - f_md_Ck|| + ||f_obs_i - f_obs_Ck|| + tau * ||x_i - x_Ck||
```

其中：
- `f_md`：模态分解特征（6维）
- `f_obs`：节点到圆柱表面的有符号距离（1维）
- `x`：节点空间坐标（2维）
- `tau=1.0`：紧凑性参数（控制空间接近度的权重）

**递归构建二层树**：
```
Level 0 (粗): 64 segments  ← METIS+SLIC
Level 1 (细): 256 segments ← 对每个 L0 segment 内部再 METIS 切 4 块
```

每个节点最终携带两个分配 ID：`L0_assign[i]` 和 `L1_assign[i]`。

#### 2.2.3 RWSE 位置编码

**全称**：Random Walk Structural Encoding

**目的**：让 Transformer 知道 segment 之间的拓扑相邻关系，而不仅仅是空间坐标。

**计算方法**：
1. 对每一层 (L0, L1) 构建 segment 级别的邻接矩阵 A_K
   - 如果两个 segment 之间有跨 segment 的边 → 相邻
2. 归一化得到随机游走转移矩阵 P = D^{-1} A_K
3. 计算 P, P^2, P^3, ..., P^16 的对角线元素 → 16维向量

**物理意义**：RWSE(k) 的第 j 维代表"从 segment k 出发、走 j 步后回到自身的概率"。这编码了 segment 在拓扑图中的局部连通结构。

#### 2.2.4 Micro-level GNN（15步消息传递）

**架构**：直接复用 PhysicsNeMo 的 `MeshGraphNet`，设置 `processor_size=15`。

**分工**：
- 负责捕捉局部物理：边界层内的黏性扩散、局部涡结构的旋转、对流项
- 15 步消息传递让每个节点能"看到" 15 跳邻居（约覆盖圆柱直径的 2-3 倍范围）

**与后续 Transformer 的互补关系**：
- GNN：精确的局部物理（高频、短程）
- Transformer：全局压力场（低频、长程）

**实现细节**：
```python
self.micro = MeshGraphNet(
    input_dim_nodes=6,    # u, v + 4-dim node_type one-hot
    input_dim_edges=3,    # relative_x, relative_y, distance
    output_dim=128,       # hidden dim d
    processor_size=15,    # 15步消息传递
    mlp_activation_fn='silu',
    recompute_activation=True,  # 节省显存
)
```

#### 2.2.5 N-S 约束感知物理量计算

**目的**：为 AMR Router 提供"每个区域是否需要细分"的依据。

**四个判据**（来自 AMR-Transformer 论文）：

| 物理量 | 公式 | 物理含义 | 在圆柱绕流中的表现 |
| --- | --- | --- | --- |
| 速度梯度 G | sqrt(\|\|nabla u\|\|^2 + \|\|nabla v\|\|^2) | 检测急剧速度变化/间断 | 圆柱壁面附近极高 |
| 涡量 omega | dv/dx - du/dy | 流体旋转强度 | 脱落涡处极高 |
| 动量 M | rho * sqrt(u^2+v^2) * area | 局部动量大小 | 加速区域高 |
| KH 剪切 S | du/dy + dv/dx | 剪切层不稳定性 | 尾迹两侧剪切层高 |

**图上离散实现**：在非结构三角网格上，用 1-ring 邻居做最小二乘梯度估计：
```
对节点 i，收集所有邻居 j：
  delta_x = pos_j - pos_i    形状 [deg(i), 2]
  delta_u = u_j - u_i        形状 [deg(i), 1]
求解最小二乘：grad_u_i = (delta_x^T delta_x)^{-1} delta_x^T delta_u
```

#### 2.2.6 AMR Router（自适应 Token 路由）

**核心逻辑**：二层（L0=64, L1=256）的"保留 vs 合并"决策。

```
输入：partition_levels = [L0_assign, L1_assign]
      phys_per_node = {G, omega, M, S}

Step 1: 在 L1 (细层 256 个 segment) 上聚合物理量
        对每个 segment 取 max(abs(phys)) → agg_G[k], agg_omega[k], ...

Step 2: 判断每个 L1 segment 是否"活跃"
        is_active[k] = (agg_G[k] > T_G) OR (agg_omega[k] > T_omega)
                        OR (agg_M[k] > T_M) OR (agg_S[k] > T_S)

Step 3: 分配最终 token
        活跃的 L1 segment → 保持为独立 token（细粒度）
        平静的 L1 segment → 折回 L0 parent（几个兄弟合并为 1 个 token）

输出：变长 token 数 T (64 <= T <= 256)
```

**阈值采样机制**（AMR-Transformer 的巧妙设计）：
- **训练时**：阈值从预定义区间均匀随机采样
  - G: [0.1, 2.0], omega: [0.2, 4.0], M: [0.5, 10.0], S: [0.2, 4.0]
- **测试时**：使用固定阈值（手动调节，平衡精度与效率）
- **好处**：模型对不同粒度的 token 化都见过，泛化性强

**直觉解释**：
- 来流均匀区（涡量~0，梯度~0）→ 大块合并成一个 token → 节省计算
- 尾迹涡街区（涡量极高、剪切极高）→ 保持 256 粒度 → 精确建模
- 圆柱壁面区（梯度极高）→ 保持细分 → 边界层精度

#### 2.2.7 Segment Encoding + Macro Transformer

**Segment Encoding（M4GN §3.3.1 风格）**：
```python
# 平均池化：segment k 的所有节点的 h 取均值
h_seg_k = MLP( mean_{i in S_k} h_node_i )

# 加位置编码
h_seg_k += PE_proj([RWSE_k, depth_k, x_mean_k, y_mean_k])
```

优势（相比 EAGLE 的 GRU）：
- 置换不变：不依赖节点顺序
- O(Nd) vs O(Nd^2)：计算更轻
- 无梯度消失：避免长序列信息衰减

**Macro Transformer**：
```
配置：4 层 TransformerEncoder, 8 注意力头, d_model=128, FFN=512
输入：h_seg [T, 128] + padding mask
输出：h_seg' [T, 128]
复杂度：O(4 * T^2 * 128)
```

当 T=150 时：4 * 150^2 * 128 = 11.5M FLOPs（极低）
当 T=256 时：4 * 256^2 * 128 = 33.6M FLOPs（仍很低）

**Feature Dispatch（M4GN §3.3.2）**：
```python
# 每个节点 i 找到自己所属 token 的输出
h_global_i = h_seg'[token_of(i)]

# 拼接局部+全局
h_cat_i = Concat([h_node_i, h_global_i])   # [N, 2d=256]
```

**设计哲学**：
- `h_node_i`（来自 15步 GNN）：保留高频局部物理细节
- `h_global_i`（来自 Transformer）：注入全局上下文（压力传播、远程涡-涡相互作用）
- 拼接而非相加：让 decoder 自己学习如何融合两种尺度的信息

#### 2.2.8 Decoder + Loss

**Decoder**：
```python
self.decoder = MLP([256, 128, 3])  # 输出 (delta_u, delta_v, p_hat)
```

**Loss Function**：Per-channel NMSE（Normalized Mean Squared Error）
```
NMSE = mean((pred - target)^2) / mean(target^2).clamp_min(eps)
```

优势（vs 普通 MSE）：
- 自动适应不同物理量的尺度差异
- 速度增量 ~O(0.001) 和压力 ~O(1) 不会互相淹没

---

## 三、与 Baseline 的对比分析

### 3.1 计算复杂度对比

圆柱绕流数据集参数：N ≈ 1900 节点，E ≈ 5500 边，d = 128

| 模型 | 主要计算项 | 估算 FLOPs | 备注 |
| --- | --- | --- | --- |
| MGN (15步) | O(L*E*d^2) | 1.35 GFLOPs | 纯消息传递 |
| 节点级 Transformer | O(N^2*d) | 462 MFLOPs | 但显存 14.4 GB |
| M4GN (K=36固定) | MGN + O(K^2*d) | +0.66 MFLOPs | K 太小 |
| **AMR-M4GN (ours)** | MGN + O(T^2*d) | +11.5 MFLOPs (T=150) | 自适应 |

**核心观察**：
- Transformer 部分的开销远小于 GNN 部分（因为 T << N）
- 总开销约为原 MGN 的 1.01 倍——几乎"免费"获得了全局建模能力

### 3.2 精度预期

| 场景 | MGN (baseline) | AMR-M4GN (预期) | 提升原因 |
| --- | --- | --- | --- |
| 涡脱落频率 | 中 | 高 | Transformer 捕捉全局 Strouhal 数 |
| 尾迹速度衰减 | 中偏低 | 高 | 远程压力-速度耦合 |
| 回流区长度 | 中 | 高 | AMR 在分离点处保留细 token |
| 表面压力分布 | 高 | 高 | GNN 15步已覆盖 |
| 长时 rollout 稳定性 | 低 | 中偏高 | 全局约束减缓误差累积 |

---

## 四、数据集与实验设置

### 4.1 数据集

**VortexSheddingDataset**（PhysicsNeMo 内置）：
- 来源：圆柱绕流直接数值仿真 (DNS)
- 训练集：1000 个 case × 600 时间步 = 599,000 样本
- 网格：~1900 节点的非结构三角网格（stationary，拓扑不随时间变化）
- 节点特征：速度 (u,v) + node_type one-hot (4维)
- 边特征：相对坐标 (dx, dy) + 距离 |d|
- 预测目标：速度增量 (delta_u, delta_v) + 压力 p
- Reynolds 数变化范围：通过不同圆柱直径和来流速度覆盖

### 4.2 训练设置

```yaml
# 优化器
optimizer: Adam (或 Apex FusedAdam)
lr: 1e-4
lr_decay: exponential, rate=0.999985 per step

# 训练
epochs: 200
batch_size: 4 (图级别，每张图 ~1900 节点)
AMP: True (FP16 混合精度)
DDP: 支持多卡

# AMR 阈值（训练时随机采样）
G: U[0.1, 2.0]
omega: U[0.2, 4.0]
M: U[0.5, 10.0]
S: U[0.2, 4.0]
```

### 4.3 评价指标

1. **单步预测精度**：NMSE, MAE, MSE（对标 AMR-Transformer 论文 Table 1）
2. **Rollout 精度**：50/100/200 步自回归的误差累积曲线
3. **推理效率**：单步 FLOPs, GPU 时间, 峰值显存
4. **AMR 统计**：平均 token 数 T、token 数分布、与物理量的相关性

### 4.4 消融实验计划

| 实验 | 配置 | 验证什么 |
| --- | --- | --- |
| Full model | AMR-M4GN | 完整性能 |
| w/o AMR | 固定 K=256 | AMR 的效率贡献 |
| w/o Transformer | 只有 15步 GNN | Transformer 的精度贡献 |
| w/o Modal Decomp | METIS-only 分区 | 模态分解的分区质量贡献 |
| w/o RWSE PE | 去掉 RWSE | 位置编码的必要性 |
| 7步 vs 15步 GNN | processor_size=7 | 深度 GNN 的收益 |

---

## 五、代码实现计划

### 5.1 文件结构

```
physicsnemo/examples/cfd/vortex_shedding_mgn/
├── train.py                       # [不动] 原 MGN baseline
├── inference.py                   # [不动] 原推理脚本
├── conf/
│   ├── config.yaml                # [不动] 原 MGN 配置
│   └── config_amr_m4gn.yaml       # [新建] AMR-M4GN 配置
├── train_amr_m4gn.py              # [新建] AMR-M4GN 训练入口
├── inference_amr_m4gn.py          # [新建] 推理+评估脚本
├── preprocess_partitions.py       # [新建] 离线预处理（模态分解+分区+RWSE）
└── amr_m4gn/                      # [新建] 模型包
    ├── __init__.py
    ├── modal_decomp.py            # Laplacian eigenfunctions
    ├── segmentation.py            # METIS + SLIC 混合分区 + 递归
    ├── pe.py                      # RWSE 位置编码
    ├── physics_ops.py             # 速度梯度/涡量/动量/KH剪切
    ├── amr_router.py              # AMR 二层路由决策
    ├── micro_gnn.py               # MeshGraphNet wrapper (15步)
    ├── macro_transformer.py       # Segment Transformer + dispatch
    └── model.py                   # AMRM4GN 顶层模型
```

### 5.2 各文件职责与接口

#### `modal_decomp.py`
```
输入: edge_index [2,E], pos [N,2], node_type [N], num_modes=6
输出: f_md [N, 6]   (每个节点的模态特征)
依赖: scipy.sparse.linalg.eigsh (LOBPCG)
耗时: ~1-3 秒/case
```

#### `segmentation.py`
```
输入: edge_index, pos, f_md, f_obs, K_list=[64,256], tau=1.0
输出: partition_levels = [L0_assign[N], L1_assign[N]]
依赖: pymetis (METIS), torch_scatter
耗时: ~0.5 秒/case
```

#### `pe.py`
```
输入: partition_levels, edge_index, num_steps=16
输出: rwse_pe = [rwse_L0[64, 16], rwse_L1[256, 16]]
依赖: torch.sparse
耗时: ~0.2 秒/case
```

#### `physics_ops.py`
```
输入: u[N], v[N], pos[N,2], edge_index[2,E]
输出: dict {G[N], omega[N], M[N], S[N]}
依赖: torch_scatter (1-ring least-square)
耗时: ~1ms/step (在线)
```

#### `amr_router.py`
```
输入: partition_levels, phys_per_node, thresholds (sampled)
输出: kept_assign[N], kept_depth[N], num_tokens T
依赖: torch_scatter
耗时: ~0.5ms/step (在线)
```

#### `model.py`
```
输入: PyG Data (graph_t)
输出: predictions [N, 3]
组合: micro_gnn → physics_ops → amr_router → segment_encode → transformer → dispatch → decoder
```

### 5.3 关键第三方依赖

| 库 | 用途 | 安装命令 |
| --- | --- | --- |
| `pymetis` | METIS 图分区 | `pip install pymetis` |
| `torch_geometric` | 图数据结构、scatter 操作 | 已有 |
| `scipy` | 稀疏特征值求解 | 已有 |
| `torch_scatter` | 高效聚合 | 随 PyG 安装 |

---

## 六、研发里程碑

### M1：离线预处理验证（预计 3 天）

**目标**：确认模态分解和分区在圆柱绕流数据上表现合理

**交付物**：
- `modal_decomp.py` + `segmentation.py` + `pe.py`
- 可视化图：前 6 个 Laplacian 模画在 mesh 上
- 可视化图：64/256 两层分区着色画在 mesh 上
- 验证：尾迹处和来流处分到不同 segment

### M2：物理量算子验证（预计 2 天）

**目标**：确认物理量计算正确，能区分活跃区和平静区

**交付物**：
- `physics_ops.py`
- 可视化图：G, omega, M, S 四个标量场画在 mesh 上
- 验证：圆柱后方（涡街）omega 显著高于来流区

### M3：AMR Router 单测（预计 2 天）

**目标**：确认 AMR 决策逻辑正确

**交付物**：
- `amr_router.py`
- 测试：给定固定阈值，画出"哪些 segment 被保留为细 token，哪些被合并"
- 统计：不同时间步的 T 值分布

### M4：端到端训练跑通（预计 3 天）

**目标**：完整模型在单个 case 上 overfit 成功

**交付物**：
- `model.py` + `train_amr_m4gn.py` + `config_amr_m4gn.yaml`
- 训练曲线：loss 稳定下降
- 验证：overfit case 的预测 vs ground truth 可视化

### M5：全数据集训练 + 对比（预计 5 天）

**目标**：与 MGN baseline 进行公平对比

**交付物**：
- 全量训练完成的 checkpoint
- 对比表：NMSE/MAE/MSE/FLOPs/时间
- Rollout 曲线对比
- token 数统计分析

### M6：消融实验（预计 5 天）

**目标**：验证每个模块的贡献

**交付物**：
- 消融实验表（§4.4 中列出的 6 组实验）
- 分析报告

**总预计工期**：约 20 个工作日（4 周）

---

## 七、技术风险与应对方案

| 风险 | 可能性 | 影响 | 应对措施 |
| --- | --- | --- | --- |
| Laplacian 求解在边界条件不正确时特征值退化 | 中 | 模态无意义 | 用 Neumann BC 替代 Dirichlet；加 shift-invert 稳定 |
| METIS 在窄长区域产生不均匀 segment | 低 | 某些 segment 只有几个节点 | SLIC 精修 + 最小尺寸检查 |
| 训练初期物理量分布不稳定，AMR 随机抖动 | 高 | 收敛慢 | 前 5 epoch 关闭 AMR，全部使用 L1 (warm-up) |
| 变长 token batch padding 比例过高 | 中 | 浪费显存 | 按 T 排序组 batch / nested tensor |
| 15步 GNN 过平滑 | 低 | 节点嵌入趋同 | recompute_activation + residual connection (MGN 已有) |
| pymetis 安装困难 (Windows) | 中 | 无法分区 | 备选：networkx 的 metis binding 或 torch_geometric 自带 |

---

## 八、后续扩展方向

1. **3D 扩展**：quadtree → octree，2D METIS → 3D METIS，Laplacian 3D 版本
2. **更大网格**：当 N > 10万时，micro GNN 可改为 segment 内并行
3. **多物理场**：加密度、温度通道，AMR 判据可扩展为"密度梯度"等
4. **时序 AMR**：当前 AMR 只看 t 时刻，可加入 t-1 时刻外推（AMR-Transformer 的 virtual step）
5. **可学习阈值**：把阈值从采样改为可学习参数（Gumbel-Softmax 或强化学习）

---

## 九、参考文献

1. Xu et al., "AMR-Transformer: Enabling Efficient Long-range Interaction for Complex Neural Fluid Simulation," CVPR 2025.
2. Lei et al., "M4GN: Mesh-based Multi-segment Hierarchical Graph Network for Dynamic Simulations," TMLR 2025.
3. Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks," ICLR 2021.
4. Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs," SIAM J. Sci. Comput., 1998.
5. Achanta et al., "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods," IEEE TPAMI, 2012.
6. Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations," ICLR 2021.

---

## 十、总结

本工作提出 AMR-M4GN 混合架构，创新性地将三个思想融合：

1. **M4GN 的层次化分区**：确保分区连通性和物理一致性
2. **AMR-Transformer 的动态 token 化**：根据实时物理状态自适应调节计算粒度
3. **深层 GNN + Transformer 的互补分工**：GNN 负责局部精细物理，Transformer 负责全局长程交互

该方案在理论上几乎不增加计算成本（Transformer 部分仅占总 FLOPs 的 ~1%），同时有望显著提升长程依赖建模能力和 rollout 稳定性。代码实现完全基于现有 PhysicsNeMo 生态，与原 MGN baseline 兼容共存，便于公平对比。

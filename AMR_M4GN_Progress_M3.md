# AMR-M4GN 开发进度与操作手册（M3 阶段）

**更新日期**：2026年6月11日
**当前阶段**：M3 — AMR Router（二层 fold/keep token 路由）+ 段级/节点级 RWSE 位置编码 + 路由可视化
**M3 状态**：⚠️ **代码已实现，尚未由你实跑测试通过**。下文凡涉及「测试通过/闭环」的，除非注明是你实跑的真实数据，否则一律视为「待你重跑确认」。
**前置**：M1、M2 已关闭（M2 验收见 `AMR_M4GN_Progress_M2.md`）。
**配套设计文档**：`AMR_M4GN_Design_Doc.md`（§4.7 Router / §7.4.1 pe / §7.4.3 amr_router / §八 M3）
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`

> 本文只讲 M3 新增的东西：做了什么、为什么、怎么跑怎么验、决策门 D3 怎么看、M4 注意事项。

---

## 一、M3 做了什么

```
examples/cfd/vortex_shedding_mgn/
├── amr_m4gn/
│   ├── __init__.py             ✅ 改：导出 amr_router 与 pe 的接口
│   ├── amr_router.py           ✅ 新建：aggregate_per_segment / sample_thresholds
│   │                                    / build_l1_to_l0 / route
│   └── pe.py                   ✅ 新建：rwse_segment / rwse_node（RWSE 位置编码）
├── visualize_partition.py      ✅ 改：加 --plot_routing + --route_pct，多出 08_routing.png
└── tests/
    ├── test_amr_router.py      ✅ 新建：6 个 pytest 单测（含规格 4 例）
    └── test_pe.py              ✅ 新建：5 个 pytest 单测（含规格 3 例）
```

> 上面的 `✅` 仅表示「文件已新建/修改」，**不代表已测试通过**。能否关闭 M3 取决于 §五 验收由你实跑达成。

M3 目标（Design Doc §八 M3）：**二层 fold/keep 决策正确、token 数 T 随物理状态变化**，并能可视化「尾迹保细、来流被合并」。**不涉及训练。**

---

## 二、`amr_m4gn/amr_router.py` 详解

对应 Design Doc **§4.7 / §7.4.3**。

| 函数 | 签名 | 作用 |
| --- | --- | --- |
| `aggregate_per_segment` | `(phys: dict, assign[N], num_seg, reduce="max") -> dict[num_seg]` | 每段聚合 `max\|phys\|`（ω 先取绝对值）|
| `sample_thresholds` | `(ranges=None, training=True, fixed=None, generator=None) -> dict` | 训练随机采样；测试固定/取区间中点 |
| `build_l1_to_l0` | `(levels) -> l1_to_l0[K1]` | L1→L0 父子映射（靠嵌套性质单次 scatter）|
| `route` | `(levels, phys, thresholds, reduce="max") -> (kept_assign[N], kept_depth[T], T, token_batch[T])` | 活跃 L1 段保留、平静段折回 L0 父段 |

**路由逻辑（§4.7 三步）**：
1. 在 L1（256 段）聚合：`agg_X[k] = max_{i∈段k} |X_i|`，X∈{G,ω,M,S}。
2. 活跃判定（满足任一即活跃）：`active[k] = (agg_G>T_G) ∨ (agg_ω>T_ω) ∨ (agg_M>T_M) ∨ (agg_S>T_S)`。
3. 分配 token：活跃 L1 段 → 自己成 1 个细 token（`depth=1`）；平静 L1 段 → 折回其 L0 父段（`depth=0`），**同一 L0 父的所有平静子段合并成 1 个 token**。

输出 `kept_assign[i]∈[0,T)` 为节点最终 token id（连续编号），`kept_depth[T]∈{0,1}`，`T`（K0≤T≤K1）。

**关键设计决策**：
1. **父子映射靠「嵌套性质」而非内部 offset**：`segmentation.build_partition_tree` 的 L1 是在每个 L0 段内部再 METIS 细分，所以同一 L1 段所有节点的 L0 标签相同。`l1_to_l0[L1_assign] = L0_assign` 单次赋值即得，稳健、不依赖 segmentation 的实现细节。
2. **取 `max|·|` 而非 mean**：一个段只要局部有强信号（尾迹涡核穿过）就该细分，max 比 mean 更敏感、更符合「宁可多留不可漏」。ω 取绝对值（有正负），G/M/S 恒 ≥ 0。
3. **纯 torch（`scatter_reduce_`），不引入 `torch_scatter`**：与 `physics_ops.py` 风格一致，零额外依赖。
4. **`route` 里 token 编号用一个 K1 次的循环**：K1≤256 极小，循环换可读性；连续编号逻辑清晰、无歧义。
5. **`token_batch` 单图版先返回全 0**：M3 不处理批内段偏移（Design Doc M4 `batch_size=1`、批处理留 M5）。

**阈值区间（默认 `DEFAULT_RANGES`，来自 AMR-Transformer 原文）**：`G:[0.1,2.0]`、`ω:[0.2,4.0]`、`M:[0.5,10.0]`、`S:[0.2,4.0]`。
> ⚠️ **这些是原文针对其归一化数据的区间，与本数据集物理速度的量级（如 |ω|p99≈188）不在一个尺度**。直接套用会让真实数据「全部活跃」。真实阈值标定 = **决策门 D3**（见 §四），要等看过 T 分布 / 进入训练期采样后才定。M3 的单测用合成数据（全 0 / 全大 / 半场），**不依赖真实量级**，只验证路由逻辑。

---

## 三、`amr_m4gn/pe.py` 详解

对应 Design Doc **§7.4.1**。RWSE（随机游走结构编码）：`P=D⁻¹A`（行随机矩阵），`rwse[i]=[ (P¹)_ii, …, (P^steps)_ii ]`，即长度 1..steps 的随机游走回到出发点的概率。

| 函数 | 签名 | 作用 |
| --- | --- | --- |
| `rwse_segment` | `(seg_adj[2,E_seg], num_segments, steps=16) -> [num_segments, steps]` | 段级结构编码（喂 SegmentEncoder，§4.8）|
| `rwse_node` | `(edge_index[2,E], num_nodes, steps=16) -> [num_nodes, steps]` | 节点级绝对 PE（Encoder 输入，§4.4）|

**关键设计决策**：
1. **稠密实现**：段数 K1≤256、节点数 N~2000，稠密 `P` 矩阵幂完全够用，避免 `torch.sparse` 的复杂度；用 float64 算幂、最后转 float32（防数值误差累积）。
2. **孤立节点（度=0）行全 0**：无回归概率，安全（不会出现在连通网格里，但单测覆盖）。
3. **M3 只产出编码、不接模型**：RWSE 在 M4 的 SegmentEncoder 才真正用到。**不确定点 U3**（节点级绝对 PE 是否真提精度）留到 M6 消融。

---

## 四、🚩 决策门 D3：K0/K1 与阈值区间（M3 产出统计，最终待训练定）

**问题**：T（token 数）的分布合不合理？K0=64/K1=256 与阈值区间要不要调？

**M3 怎么产出输入**：`--plot_routing` 会打印每次路由的 `T`、活跃/折回段数、相对 256 的 reduction%，并出 `08_routing.png`（尾迹保细=红、来流合并=蓝）。

**怎么判读**：
- T 长期贴近 256（几乎不合并）→ 阈值过松或 K1 过大 → 调阈值区间或 K1；
- T 长期贴近 64（几乎全合并）→ 阈值过严；
- 健康区间：T 落在 64~256 中段，且红色（保细）集中在尾迹/壁面、蓝色（合并）覆盖来流。

> **注意**：`--plot_routing` 用的是**数据相对阈值**（`--route_pct`，默认每通道取该层 per-segment |phys| 的 70 分位），只是为了让 M3 在**未标定**时也能出一张有意义的演示图。**真正的 D3 拍板要到 M5 训练期阈值采样后**，看整个训练集的 T 分布再定 K0/K1 与区间。

---

## 五、运行步骤与验收（每步：做什么 → 得到什么 → 为什么）

> 前提：M2 已通过、`visualize_partition.py` 等已同步到运行机（`C:\GitHub`）。

### 步骤 0 — 同步 M3 改动的文件

| 文件 | 操作 |
| --- | --- |
| `amr_m4gn/amr_router.py` | 新增 |
| `amr_m4gn/pe.py` | 新增 |
| `amr_m4gn/__init__.py` | 修改（导出 router/pe）|
| `visualize_partition.py` | 修改（`--plot_routing`、`--route_pct`、`08_routing.png`）|
| `tests/test_amr_router.py` | 新增 |
| `tests/test_pe.py` | 新增 |

### 步骤 1 — 跑单元测试

```bash
cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
pytest tests/test_amr_router.py tests/test_pe.py -v
```

- **得到什么**：router 6 个 + pe 5 个单测结果。
- **为什么**：先在合成数据上验证路由逻辑与 RWSE 的正确性，不依赖真实网格。
- **状态**：⏳ **待你实跑确认**（开发机无 torch，我未实跑 pytest）。

### 步骤 2 — 在真实涡街场上跑路由可视化

```bash
python visualize_partition.py --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test --case_idx 0 --timestep 300 --plot_physics --plot_routing
```

- **得到什么**：除 M2 的 7 张图外，多出 **`08_routing.png`**（上：保细/合并红蓝图；下：token id 图）；终端打印 `Tokens T = ...`、活跃/折回段数、reduction%。
- **为什么**：肉眼确认「尾迹保细、来流被合并」（M3 退出标准），并产出 D3 统计。
- **状态**：⏳ **待你实跑确认**。

### 验收清单（M3 退出标准，Design Doc §八 M3）

| 验收点 | 看哪里 | 合格判据 | 状态 |
| --- | --- | --- | --- |
| **A. router 单测** | `pytest test_amr_router.py` | 全平静→T=K0=64；全活跃→T=K1=256；半场→64<T<256；阈值采样在区间内且可复现 | ⏳ 待实跑 |
| **B. pe 单测** | `pytest test_pe.py` | 链状端点回归<中间；全连接近似相等；P 行和=1 | ⏳ 待实跑 |
| **C. 路由可视化** | `08_routing.png` | 尾迹/壁面保细（红）、来流被合并（蓝）| ⏳ 待实跑 |
| **D. T 统计（D3）** | 终端 `Tokens T=` | T 落在 [64,256] 中段、随物理状态变化 | ⏳ 待实跑 |

---

## 六、与 Design Doc 的对应关系

| Design Doc | M3 落地 |
| --- | --- |
| §4.7 AMR Router（聚合/活跃判定/fold-keep）| `amr_router.route` |
| §7.4.3 amr_router 规格 + 4 例单测 | `amr_router.py` + `tests/test_amr_router.py` |
| §7.4.1 pe（rwse_segment / rwse_node）+ 3 例单测 | `pe.py` + `tests/test_pe.py` |
| §八 M3（可视化保留/合并、T 分布 D3）| `--plot_routing` + `08_routing.png` |

**M3 未覆盖（属后续）**：
- 批内段偏移（`graph.batch`、`key_padding_mask`）= M5。
- RWSE 真正接入模型（SegmentEncoder）= M4 的 `macro_transformer.py`。
- 真实阈值标定（D3 最终拍板）= 训练期阈值采样后。

---

## 七、M4 注意事项（下一步）

1. **新建 `micro_gnn.py` / `macro_transformer.py` / `model.py` / `data_amr.py` / `preprocess_partitions.py` / `train_amr_m4gn.py` / `conf/config_amr_m4gn.yaml`**（Design Doc §7.4.4~§7.4.6、§9）。
2. **🚩 决策门 D4（micro_gnn 接口，U2）**：M4 第一步先写 `micro_gnn` 单测，确认 `MeshGraphNet` 子模块属性（node_encoder/edge_encoder/processor）稳定 → 通过用路径 (a)，否则退 (b)（§7.2-D）。
3. **🚩 决策门 D5（数据管线，U1/§7.2-A）**：确认 P1（子类暴露 pos）还是 P2（反归一化）。推荐 P1，M4 开工即定。
4. **前置约束**：M4 `batch_size=1`，暂不处理批内段偏移（留 M5）。
5. **SegmentEncoder 接 RWSE**：M4 用 `pe.rwse_segment` 的输出做段级 PE；记得离线缓存进 `partition_cache`。
6. **D1 仍暂定**：训练期反归一化在 M4 真实数据管线里最终确认。

# AMR-M4GN 开发进度与操作手册（M4 阶段）

**更新日期**：2026年6月11日
**当前阶段**：M4 — 端到端模型（micro GNN + 段编码 + Macro Transformer + dispatch + 统一 decoder）+ 数据管线 + overfit 单 case
**M4 状态**：🚧 **进行中**。本轮交付**纯模型组件**（`micro_gnn.py`、`macro_transformer.py`）+ 单测，并确认两个决策门 D4/D5。**单测 9/9 已实跑通过**。`model.py` / `data_amr.py` / `preprocess_partitions.py` / `train_amr_m4gn.py` / `conf` 下一轮。
**前置**：M1/M2/M3 已关闭。
**配套设计文档**：`AMR_M4GN_Design_Doc.md`（§4.3/4.8/4.9、§7.2、§7.4.4~7.4.6、§八 M4）

---

## 一、本轮做了什么

```
amr_m4gn/
├── micro_gnn.py            ✅ 新建：MeshGraphNet wrapper（走到 processor，丢弃 decoder）
├── macro_transformer.py    ✅ 新建：SegmentEncoder + MacroTransformer + dispatch
└── __init__.py             ✅ 改：导出 MicroGNN / SegmentEncoder / MacroTransformer / dispatch
tests/
├── test_micro_gnn.py       ✅ 新建：4 个单测（形状/无 decoder 参数/与 MGN 一致/全参有梯度）
└── test_macro_transformer.py ✅ 新建：5 个单测（置换不变/T=1/padding mask/dispatch/形状）
```
> `✅` 仅表示「文件已建/改」，**不代表已测试通过**。

**本轮未做（下一轮）**：`model.py`（顶层 AMRM4GN）、`data_amr.py`（P1 子类）、`preprocess_partitions.py`、`train_amr_m4gn.py`、`conf/config_amr_m4gn.yaml`、overfit 验证。

---

## 二、🚩 决策门 D4（micro_gnn 接口，U2）：已确认通过 → 用路径 (a)

**问题**：`MeshGraphNet` 暴露的子模块属性名在当前版本是否稳定？

**核对结果**（读 `physicsnemo/models/meshgraphnet/meshgraphnet.py` 真实源码）：
- `MeshGraphNet.__init__` 确实建 `self.edge_encoder` / `self.node_encoder` / `self.processor` / `self.node_decoder`；
- `forward` 即 `edge_encoder → node_encoder → processor → node_decoder`（line 267–270）。
- **结论**：属性名稳定 → 采用 **路径 (a)**：`MicroGNN` 走到 `processor` 输出 `h_node[N,hidden]`，**丢弃 `node_decoder`**。

**关键实现细节**：`MicroGNN.__init__` 先构造一个临时 `MeshGraphNet`，然后**只把 `edge_encoder/node_encoder/processor` 注册为自身子模块**（局部 backbone 变量随后丢弃），所以 `node_decoder` 既不存储也**不计入 `parameters()`**——这样 §7.4.6 的「每个参数都收到梯度」才成立（否则 decoder 参数永远无梯度）。

> **D4 通过**基于源码核对 + `test_micro_gnn.py` **已实跑 4/4 通过**（含 `test_matches_meshgraphnet_processor` 验证 forward==MGN 的 encoder→processor）。

## 三、🚩 决策门 D5（数据管线，U1/§7.2-A）：选 P1

- **P1（采用）**：新增 `VortexSheddingDatasetAMR(VortexSheddingDataset)`，train 分支额外返回 `graph.pos`、`graph.gidx`，不动原 `VortexSheddingDataset`，零风险。
- **P2（不用）**：在前向里用 `edge_stats` 反归一化 `edge_attr` 得 Δx，耦合差。
- **落地**：在下一轮的 `data_amr.py` 实现 P1。

---

## 四、`micro_gnn.py` 详解

| 类/方法 | 签名 | 作用 |
| --- | --- | --- |
| `MicroGNN.__init__` | `(in_nodes=6, in_edges=3, hidden=128, processor_size=15, ...)` | 建 MeshGraphNet、只留 encoder+processor |
| `MicroGNN.from_backbone` | `(backbone: MeshGraphNet) -> MicroGNN` | 复用已有 MGN 的子模块（对齐/对比权重用）|
| `MicroGNN.forward` | `(x[N,in_nodes], edge_attr[E,in_edges], graph) -> h_node[N,hidden]` | `edge_encoder→node_encoder→processor` |

---

## 五、`macro_transformer.py` 详解（Design Doc §4.8）

| 类/方法 | 作用 |
| --- | --- |
| `SegmentEncoder` | 每 token 对所属节点做 mean-pool → `node_mlp`，再加 `pe_proj([rwse(16), depth(1), centroid(2)])`。置换不变、O(Nd)。|
| `MacroTransformer` | Pre-LN `TransformerEncoder`，默认 4 层 8 头 d=128 FFN=512；支持 `key_padding_mask`（变长 batch，M5 用）|
| `dispatch` | `h_global_i = h_seg'[token_of(i)]`；`h_cat=concat([h_node, h_global])` → `[N,2d]` |

**关键设计决策**：
1. **mean-pool 而非 GRU**（M4GN §3.3.1）：置换不变、无长序列稀释、O(Nd)。
2. **per-token PE 输入 = rwse(16)+depth(1)+centroid(2)=19 维**：由 `model.py` 在 M4 下一轮按 token 组装（细 token 用 L1 段的 rwse/质心，粗 token 用 L0 段的）后传入。
3. **concat 而非 add**（§4.8）：让 decoder 自学融合 local/global 两种尺度。
4. **batch_size=1**：M4 单图；批内段偏移 + padding 留 M5（§7.2-C）。`MacroTransformer` 已预留 `key_padding_mask` 接口。

---

## 六、运行步骤与验收（待你实跑）

### 步骤 0 — 同步本轮文件
`amr_m4gn/micro_gnn.py`、`amr_m4gn/macro_transformer.py`、`amr_m4gn/__init__.py`、`tests/test_micro_gnn.py`、`tests/test_macro_transformer.py`。

### 步骤 1 — 跑单测
```bash
cd E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn
pytest tests/test_micro_gnn.py tests/test_macro_transformer.py -v
```

| 验收点 | 合格判据 | 状态 |
| --- | --- | --- |
| micro_gnn 形状 | x[N,6]→h_node[N,hidden] | ✅ 通过 |
| micro_gnn 无 decoder 参数 | named_parameters 无 `node_decoder` | ✅ 通过 |
| micro_gnn 与 MGN 一致（D4）| forward==MGN 的 encoder→processor | ✅ 通过 |
| micro_gnn 全参有梯度 | 反向后每参数 grad 有限非 NaN | ✅ 通过 |
| SegmentEncoder 置换不变 | 打乱节点序输出不变 | ✅ 通过 |
| SegmentEncoder T=1 | 退化为全局平均 | ✅ 通过 |
| MacroTransformer padding mask | 被 mask token 不影响他 token | ✅ 通过 |
| dispatch | `h_cat[:, :d]==h_node` | ✅ 通过 |

> **实跑结果：9 passed in 20.35s**（gnn 环境）。另修了一条本代码的告警：`TransformerEncoder` 在 Pre-LN 下 `enable_nested_tensor` 失效的 UserWarning，已显式置 `False` 消除（无功能影响）。其余告警（torch.jit deprecated、torch_geometric.distributed、physicsnemo 的 SyntaxWarning）来自第三方/框架，非本项目代码。

---

## 七、下一轮 M4 计划（model + 数据 + 训练 + overfit）

1. **`preprocess_partitions.py`**：对每 case 跑 modal→seg→pe→l1_to_l0→质心，写 `partition_cache_{split}_{gidx}.pt`（§7.5 格式）。
2. **`data_amr.py`**：P1 子类，train 分支暴露 `pos`/`gidx`，按 gidx 取缓存。
3. **`model.py`**：顶层 `AMRM4GN`，串起 micro→physics→route→seg_enc→macro→dispatch→decoder（§7.4.6）；集成测试单 case 单步无 NaN、全参有梯度。
4. **`train_amr_m4gn.py` + `conf/config_amr_m4gn.yaml`**：复制 train.py 框架，换模型 + per-channel NMSE（§4.9）。
5. **overfit 单 case**：loss 单调降至接近 0、预测≈GT（§八 M4 退出标准）。
6. **决策门**：D1（训练期反归一化）在此最终确认；阈值采样区间 vs 物理量级的标定（D3 延续）。

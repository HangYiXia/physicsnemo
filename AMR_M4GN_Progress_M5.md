# AMR-M4GN 开发进度与操作手册（M5 阶段）

**更新日期**：2026年6月11日
**当前阶段**：M5 — 全量训练 + baseline 对比（批处理 `batch_size>1` → 完整训练入口 → `inference_amr_m4gn` rollout/指标/可视化 → D3 阈值标定）
**M5 状态**：🚧 **进行中**。M5 较重、分小步推进，**每小步都配可单独验证的单测 + 文档**。
- 小步 1（批处理注意力隔离）：**代码已实现，单测待你实跑**。
**前置**：M1/M2/M3 ✅、M4 🟢（端到端管线跑通，overfit NMSE 0.92→0.013）。
**配套设计文档**：`AMR_M4GN_Design_Doc.md`（§7.2-C 批处理 / §八 M5）
**工作目录**：`E:\phys\physicsnemo\examples\cfd\vortex_shedding_mgn\`

> **里程碑总进度**（截至 2026-06-11）：M1 ✅ · M2 ✅ · M3 ✅ · M4 🟢 · **M5 🚧** · M6 ⬜ · M7 ⬜
> **文档索引**：M1〜M4 见各 `AMR_M4GN_Progress_M*.md`；设计 `AMR_M4GN_Design_Doc.md`
> 本文同样遵循「拿到代码后每步：做什么 → 为什么 → 应该得到什么结果」。

---

## M5 总体计划（分小步）

| 小步 | 内容 | 状态 |
| --- | --- | --- |
| **1. 批处理注意力隔离** | `pack_segments`/`run_macro_batched`：变长 token 打包成 `[B,Tmax,d]`+mask，Transformer 注意力按图隔离 | ✅ 实现，单测待实跑 |
| 2. model 批处理集成 | `model.forward` 支持 PyG batch：逐图 route → 全局偏移 token id + `token_batch` → 批处理 transformer | ⬜ 待做 |
| 3. 完整训练入口 | 把 overfit 脚本扩成正经训练（多 case、checkpoint、lr 调度），复用 `train.py` 框架 | ⬜ 待做 |
| 4. `inference_amr_m4gn.py` | rollout + §6.4 指标 + **预测场可视化**（你想看的图在这里）| ⬜ 待做 |
| 5. D3 最终标定 | 训练期绝对阈值采样，统计全训练集 T 分布，定 K0/K1 与区间 | ⬜ 待做 |

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
| 批处理打包/隔离 | `pytest tests/test_batched_macro.py` | 3 passed（含跨图不泄漏）| ⏳ 待实跑 |

> 跑完把输出发我；通过后进入小步 2（`model.forward` 批处理集成）。

---

## 与 Design Doc 的对应关系

| Design Doc | M5 落地 |
| --- | --- |
| §7.2-C 批内段偏移 + padding mask | 小步 1：`pack_segments`/`run_macro_batched`（本轮）；小步 2：`model` 集成 |
| §7.6 完整训练/推理入口 | 小步 3/4（待做）|
| §八 M5 退出标准（vs baseline 指标）| 小步 3/4/5（待做）|
| 决策门 D3 最终标定 | 小步 5（待做）|

# AMR-M4GN：自适应 Token 化的多尺度流体仿真代理模型

> 在非结构三角网格上，把 **局部 GNN（MeshGraphNet）** + **段级全局 Transformer** + **物理驱动的自适应 Token 化（AMR）** 融合成一个混合架构，几乎不增加算力就缓解 MeshGraphNet 的「长程依赖丢失 / rollout 误差累积」问题。
>
> **当前最好结果**（圆柱绕流，test case0，rollout 80 步，同训练预算）：
> **AMR-M4GN 的滚动误差全程低于 MGN baseline（平均 RMSE 0.078 vs 0.094，低 ~17%）**，长程优势更明显。

本仓库 fork 自 NVIDIA PhysicsNeMo，仅保留 `physicsnemo/` 框架包作为依赖；其余 NVIDIA 示例与项目元文件已清理。AMR-M4GN 的所有代码、脚本、文档都在根目录下重新组织。

---

## 1. 一眼看懂：我要做什么

```
下载数据 ──► 预处理建缓存 ──► 训练 ──► 可视化看效果
            preprocess        train       inference / compare
```

三条主线脚本（统一从仓库根目录用 `python -m scripts.X` 运行）：

| 你想干嘛 | 跑哪个脚本 | 看什么 |
| --- | --- | --- |
| 训练 AMR-M4GN | `scripts.train_amr_m4gn_full` | 终端 NMSE 下降 + `checkpoints_amr/*.pt` |
| 训练对照基线 MGN | `scripts.train_mgn_baseline` | 同上，`checkpoints_mgn/*.pt` |
| **看预测场动画（GIF）** | `scripts.inference_amr_m4gn --gif` | `animations/amr_m4gn_case0_{u,v,p}.gif` |
| **AMR vs MGN vs 真值** | `scripts.compare_baselines --gif` | `animations/compare_case0_*.gif` + 误差曲线 |

---

## 2. 环境准备

```bash
# 1) 克隆后进入仓库根目录
cd /path/to/physicsnemo

# 2) 安装 PhysicsNeMo 框架（physicsnemo/ 子包）+ 项目依赖
pip install -e .
pip install -r requirements.txt
pip install pymetis    # 分区加速；缺失会自动退化为谱聚类（慢、质量略低）
pip install pillow     # 保存 GIF 所需

# 3) 下载数据（DeepMind cylinder_flow，约几 GB）
mkdir -p raw_dataset && cd raw_dataset
sh download_dataset.sh cylinder_flow
cd ..
```

> 需要 NVIDIA GPU（CUDA）。无 GPU 也能跑，但训练很慢。
> EAGLE 数据请见 `docs/progress/M7.md`。

---

## 3. 五分钟基础示例（小规模冒烟，能直接看到效果）

目标：**最快产出一个预测场 GIF**。用少量算例、短训练，验证「训练→推理→可视化」闭环。

```bash
# 全程在仓库根目录执行

# (1) 预处理：训练用 4 个算例 + 测试用 1 个算例（只需跑一次）
python -m scripts.preprocess_partitions --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split train --num_cases 4 --out_dir ./amr_cache
python -m scripts.preprocess_partitions --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --split test  --num_cases 1 --out_dir ./amr_cache

# (2) 训练（小档：4 算例 × 50 步 × 50 轮）
python -m scripts.train_amr_m4gn_full --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --num_cases 4 --num_steps 50 --batch_size 2 --epochs 50 --noise_std 0.02 --omega_thresh 8.9

# (3) 看效果：rollout 误差曲线 + 速度/压强场 GIF
python -m scripts.inference_amr_m4gn --data_dir ./raw_dataset/cylinder_flow/cylinder_flow --cache_dir ./amr_cache --ckpt ./checkpoints_amr/amr_m4gn_epoch49.pt --split test --case_idx 0 --num_steps 50 --rollout 40 --omega_thresh 8.9 --gif --gif_fields u v p
```

**应该看到**：
- 训练终端 NMSE 从 ~3 降到 ~0.08；
- `inference_vis/10_rollout_rmse_case0.png`：误差随步数上升后**平台饱和、不发散**；
- `animations/amr_m4gn_case0_u.gif`（及 `_v`/`_p`）：上=预测、下=真值，圆柱后方涡街红蓝交替结构对得上。

> 这是 smoke 模型，形态对、量级合理但非最终精度。要正经结果见 §4。

---

## 4. 完整复现 + AMR vs MGN 对比

正式档（20 算例 × 100 步 × 200 轮）+ 公平对比。完整命令序列见 **`docs/progress/M5.md` §2「一键复现」**。核心四步：

```bash
# 预处理 train 20 + test 1（同 §3 但 --num_cases 20）
# 训练 AMR-M4GN 与 MGN（同预算）
python -m scripts.train_amr_m4gn_full ... --num_cases 20 --num_steps 100 --epochs 200 --omega_thresh 8.9
python -m scripts.train_mgn_baseline  ... --num_cases 20 --num_steps 100 --epochs 200
# AMR vs MGN vs 真值：误差曲线 + 三行场动画
python -m scripts.compare_baselines ... --amr_ckpt ./checkpoints_amr/amr_m4gn_epoch199.pt --mgn_ckpt ./checkpoints_mgn/mgn_epoch199.pt --rollout 80 --omega_thresh 8.9 --gif
# 多 test case 平均（更严谨）
python -m scripts.eval_rollout ... --amr_ckpt ... --mgn_ckpt ... --num_cases 10 --rollout 80
```

---

## 5. 你能看到什么效果（产物清单）

| 文件 | 内容 |
| --- | --- |
| `inference_vis/09_prediction_*.png` | 单帧 9 宫格：预测 / 真值 / 误差（du,dv,p）|
| `inference_vis/10_rollout_rmse_*.png` | 单模型 rollout 误差累积曲线 |
| `inference_vis/11_threshold_calibration.png` | AMR 路由阈值标定（D3）|
| `inference_vis/12_compare_rollout_*.png` | **AMR vs MGN 误差曲线**（核心结果图）|
| `inference_vis/14_eval_multicase.png` | 多 test case 平均 ±std 曲线 |
| `animations/amr_m4gn_case0_{u,v,p}.gif` | AMR 预测场动画（上预测/下真值）|
| `animations/compare_case0_{u,v,p}.gif` | **AMR / MGN / 真值 三行对比动画** |
| `partition_vis/case*/*.png` | 预处理诊断：模态、分区、物理量、路由 |

---

## 6. 项目结构

```
physicsnemo/                    # 仓库根
├── amr_m4gn/                   # AMR-M4GN 模型包
│   ├── modal_decomp.py         #   Laplacian 模态分解（分区引导特征）
│   ├── segmentation.py         #   METIS+SLIC 两级分区树
│   ├── pe.py                   #   RWSE 位置编码
│   ├── physics_ops.py          #   N-S 物理量 G/ω/M/S + 虚拟步
│   ├── amr_router.py           #   自适应 Token 路由（折叠/保留）
│   ├── micro_gnn.py            #   局部 GNN（MGN 旁路 decoder）
│   ├── macro_transformer.py    #   段编码 + 全局 Transformer + dispatch
│   └── model.py                #   AMRM4GN 顶层（含 M6 消融开关）
├── data/                       # 数据集适配层
│   ├── vortex.py               #   cylinder_flow TFRecord（VortexSheddingDatasetAMR）
│   └── eagle.py                #   EAGLE .npz reader（M7）
├── scripts/                    # 入口脚本（python -m scripts.X 运行）
│   ├── preprocess_partitions.py    # 离线建几何/分区/PE 缓存（第一步必跑）
│   ├── train_amr_m4gn.py           # AMR-M4GN 训练（M4 单算例自检版）
│   ├── train_amr_m4gn_full.py      # AMR-M4GN 训练（M5 batched 多算例）
│   ├── train_mgn_baseline.py       # MGN baseline 训练（同预算公平对比）
│   ├── inference_amr_m4gn.py       # 单帧/rollout/GIF 可视化
│   ├── compare_baselines.py        # AMR vs MGN(+GT) 对比 + 三行 GIF
│   ├── eval_rollout.py             # 多 test case 平均评估
│   ├── calibrate_thresholds.py     # AMR 路由阈值标定
│   ├── run_ablation.py             # M6 模块消融编排
│   ├── visualize_partition.py      # 预处理诊断可视化
│   └── train.py / inference.py     # [原版 NVIDIA MGN，未改动]
├── conf/                       # Hydra 配置
│   ├── config.yaml             #   原版 MGN
│   └── config_amr_m4gn.yaml    #   AMR-M4GN
├── tests/                      # pytest 单元测试
├── docs/                       # 设计 + 阶段手册 + 论文资料
│   ├── design.md               #   AMR-M4GN 设计文档
│   ├── progress/M1..M7.md      #   每阶段「做什么 / 为什么 / 应得什么」+ 命令 + 结果
│   ├── papers/                 #   参考文献 PDF
│   └── figures/                #   图示
├── physicsnemo/                # NVIDIA PhysicsNeMo 框架（依赖，不改）
├── pyproject.toml              # PhysicsNeMo 安装定义
├── requirements.txt            # 项目额外依赖
└── README.md                   # 本文
```

每个脚本的**逐参数含义**见 `docs/progress/M5.md` §7。

---

## 7. 进度与文档导航

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M1 | 预处理：模态分解 + 混合分区 | ✅ |
| M2 | N-S 物理量算子（G/ω/M/S）| ✅ |
| M3 | AMR Token 路由 | ✅ |
| M4 | 端到端管线 + overfit 自检 | ✅ |
| M5 | 批处理 + 全量训练 + baseline 对比 | 🟢 小验证档通过，仅"更大训练集"待算力 |
| M6 | 八组模块消融 | 🟡 代码/脚本/单测就绪，实跑待算力 |
| M7 | EAGLE 大规模扩展（可选）| 🟡 reader+并行预处理就绪（按官方格式），真实数据待验证 |

文档：

- **设计文档** `docs/design.md`：动机、架构、决策门、消融计划。
- **阶段手册** `docs/progress/M1.md`〜`M7.md`：每阶段「做什么 / 为什么 / 应得什么」+ 命令 + 结果。
  - 想直接跑全套命令 → 看 **M5 §2**；想了解消融 → 看 **M6**；想跑 EAGLE → 看 **M7**。

---

## 8. 引用

- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, 2020（MeshGraphNet）
- Lei et al., *M4GN*, TMLR 2025（混合网格分区）
- Xu et al., *AMR-Transformer*, CVPR 2025（自适应 Token 化 / 物理量）
- NVIDIA PhysicsNeMo（baseline 框架与数据管线）

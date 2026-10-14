# 6-token curriculum100：最终评估（2026-10-12）

模型：`checkpoints/flowmatch_label_6tok_curriculum100/best.pt`（validation 选型=epoch 65；6-token
transformer、bs1024、lr 7e-4、TF30+ramp10+rollout60 课程、unroll 4、反馈 Euler2）。
协议：60fps / 240 帧 autoregressive rollout、Euler8、noise std1、clip→requantize→decode，
classifier（无镜像 240f）排除 GT 首帧，CI 按 source_entry_id 聚类 bootstrap。

## 封存 test（1000 sources × 5 targets，比对上一代正式结果）

| 指标 | **6tok（本轮）** | label_nomirror_150ep（上一代） | 变化 |
|---|---:|---:|---:|
| matched style acc | **96.40%** [94.48, 98.01] | 90.30% [85.98, 93.84] | **+6.1pp** |
| cross style acc | **91.42%** [90.32, 92.47] | 86.48% ±0.92 | **+4.9pp** |
| matched RootADE | **7.96cm** [6.71, 9.44] | 8.34cm [7.42, 9.42] | −0.4cm |
| cross RootADE | **9.93cm** [8.73, 11.28] | 10.78cm [9.82, 11.74] | −0.9cm |
| real motion 上界 | 99.80% | 99.80% | — |

**目标（matched ≥85、cross ≥80）在封存 test 上达成且大幅超过。** 原始输出：
`eval_results/6tok_rollout/seen/`。

## MPJPE（封存 source 权威口径，2026-10-12 补充）

正式评估协议只报风格准确率与 RootADE；这里补上 MPJPE（同一批 1000 个封存 matched source、
相同 noise plan、240 帧 rollout、std=1），全局坐标（root 积分后）与 local 两种口径，CI 按
source_entry_id 聚类 bootstrap：

| 模型 | **MPJPE（global）** | local MPJPE | RootADE | style acc（封存） |
|---|---:|---:|---:|---:|
| **VQ 表示层下限**（编码→解码，无控制器） | **2.20cm** [1.99, 2.45] | 1.02cm | 1.67cm | — |
| **6tok（本轮 best）** | **11.23cm** [9.97, 12.75] | 5.79cm | 7.96cm | **96.40%** |
| tr6（旧 label150） | 11.32cm [10.32, 12.48] | 5.61cm | 8.34cm | 90.30% |
| ep86（5-token curriculum，对照） | 9.91cm [9.06, 10.90] | 5.31cm | 7.00cm | ~88.5%（validation） |

读法：

- **6tok 与 tr6 的 MPJPE 基本持平**（CIs 大幅重叠），但 6tok 的风格准确率高 6.1pp、cross 高
  4.9pp——本轮训练（课程 + bs1024 + 配置）在**不牺牲 MPJPE** 的前提下拿下了风格指标。
- **误差分解**：VQ 下限 2.20 → 单步 TF 上界 ~4.2（90-clip 探针）→ rollout 11.2cm。即约 7cm
  来自自回归反馈环的误差累积，是当前主要瓶颈；表示层（2.2cm）与单步动力学（+2cm）只占小头。
- **5-token ep86 的 MPJPE 略好**（9.91，RootADE 7.00）但风格差 ~8pp 且低噪声闭环滑行严重
  （见闭环验收节），综合仍推荐 6tok。
- 方法自校验：本表 RootADE（6tok 7.96 / tr6 8.34）与正式封存评估逐值一致，口径可信。

原始数据：`analysis/motion_quality_diagnosis/probes/mpjpe_{6tok,tr6,ep86_5tok}.json`（含逐 source
数值）；复现脚本 `analysis/controller_compare/eval_rollout_mpjpe.py`（带 `--noise-std`）。

### 采样噪声档位阶梯（同一 1000 个封存 source，含 jitter/F.S.）

| 模型 | std | MPJPE | local | RootADE | **jitter** | **F.S.** |
|---|---:|---:|---:|---:|---:|---:|
| 6tok | 1.0（协议） | 11.23 | 5.79 | 7.96 | **1.791** | **1.676** |
| **6tok** | **0.5（显示推荐）** | **10.10** | **5.09** | **7.27** | **0.828** | **0.996** |
| 6tok | 0.25 | 11.48 | 5.16 | 8.69 | 0.561 | 0.770 |
| tr6 | 1.0 | 11.32 | 5.61 | 8.34 | 0.809 | 1.000 |
| tr6 | 0.5 | 11.03 | 5.40 | 8.18 | 0.422 | 0.662 |

读法：

- **std=0.5 是 6tok 的最佳档位**：ADE 7.27cm（比 std=1 的 7.96 更好）、MPJPE 10.10cm（
  全模型全档位最优）、jitter 从 1.79 降到 0.83、F.S. 1.68→1.00；闭环滑行 1.27%（见验收节）。
- **6tok 的 std=1.0 jitter（1.79）显著差于 tr6（0.81）**：这是本轮训练的新代价——课程/rollout
  训练让模型对采样噪声更敏感（与 5-token ep86 的 5.1× 放大同源，6tok 更高），但同时它的风格
  准确率、cross、ADE@0.5 都更强。**正式协议（std=1）下 6tok 的逐帧平滑度是明显弱点**，
  建议在论文/报告中标注 0.5 档位，或后续专门做噪声鲁棒性训练。
- std=0.25 的 MPJPE/ADE 反而回升（11.48/8.69）——过度确定性的运动质量下降（低噪声滑行
  同源），0.5 是折中最优。

## 90-clip 探针（seed42，与历史对照同口径）

| 采样 std | ADE | MPJPE | localMPJPE | jitter | F.S. | hits |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0（协议） | 9.14 | 11.85 | 4.92 | 1.624 | 1.507 | 86/90 (95.6%) |
| **0.5（推荐显示档）** | **6.84** | **9.24** | 4.21 | 0.663 | 0.875 | 82/90 (91.1%) |
| 0.25 | 8.11 | 10.48 | 4.24 | 0.456 | 0.618 | 85/90 (94.4%) |

对照 tr6（旧 label150）：std=1.0 → ADE 7.89 / hits 85/90；std=0.25 → ADE 8.20 / jitter 0.283 / F.S. 0.448。

## 闭环控制验收（脚本化行走 300 帧，多 seed）

| 设置 | 滑行占比（行走帧） | 备注 |
|---|---:|---|
| std=1.0 | **0.42%** | 正式口径，优于上一代全部模型 |
| std=0.5 | **1.27%** | 推荐显示档：低抖动（0.66）+ 无滑行 |
| std=0.25 | 6.3 / 10.5 / 13.9%（3 seeds） | 未达 ≤2% 验收线；好于 ep86（13.1%）但差于 tr6（~0.5%） |
| std=0.05 | 55.3% | 近确定性下仍会坍缩（tr5 62%、ep86 33%、tr6 1.7%） |

结论：**motion token 恢复 + 课程 + bs1024 把 std≥0.5 的闭环全部修好**（这也是正式评估与推荐的
显示档位）；std≤0.25 的低噪声吸引子仍在（程度减轻），是"训练噪声分布为 std=1"的固有外推缺口。
若必须用 std≤0.25 交互，剩余方案：① 训练 z0 噪声随机化（每 batch U[0.1,1]，未实施）；
② 旧 tr6 仍可用作低噪声显示模型（滑行 ~0.5%、jitter 0.28）。

## 决定与产物

- **交互/展示默认推荐**：`NOISE_STD=0.5 FLOW_CKPT=checkpoints/flowmatch_label_6tok_curriculum100/best.pt`；
  正式指标仍用 std=1.0。
- 控制视频（6tok@0.5，双面板）：`analysis/motion_quality_diagnosis/videos/6tok_final_std05/6tok_final_std05_control.mp4`。
- 训练日志：`analysis/controller_compare/runs/6tok_curriculum100.log`；
  history：`checkpoints/flowmatch_label_6tok_curriculum100/history.jsonl`。
- 验收标准与依据：[RETRAIN_6TOK_PLAN.md](../../analysis/controller_compare/RETRAIN_6TOK_PLAN.md)。

## 未竟事项（备选，非必需）

1. 低噪声（std≤0.25）滑行吸引子：可做 z0 噪声随机化重训验证（一轮 100ep ≈ 3.5h）。
2. 官方口径的 ADE 门限（matched ≤5cm / cross ≤10cm）：本轮 matched RootADE 7.96 仍 >5cm（选型
   约束未过，与上一代相同的口子）；若要压 ADE，需专门的轨迹损失/更长课程实验。

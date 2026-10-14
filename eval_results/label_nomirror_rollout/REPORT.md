# 无镜像链路 label controller：最终评估（2026-10-10）

模型：`checkpoints/flowmatch_label_nomirror_150ep/best.pt`（validation 选型 = epoch 150）
协议：60fps / **240 帧 autoregressive rollout**、Euler8、noise std1、clip→requantize→decode、
只在初始帧读 GT，classifier（无镜像 240f，best=ep33）排除 GT 首帧；CI 按 source_entry_id
聚类 bootstrap 1000 次。评估范围仅 seen（in-distribution）：label 条件模型没有 OOD 风格的
条件通道，unseen/OOD 不属于本模型评估范围。

## Seen（test split，1000 sources × 5 targets）

| 设置 | n | Style acc | 95% CI | RootADE | ADE 95% CI |
|---|---:|---:|---|---:|---|
| real motion 上界 | 1000 | 99.80% | [99.32, 100] | — | — |
| **matched** | 1000 | **90.30%** | [85.98, 93.84] | 8.34 cm | [7.42, 9.42] |
| **cross** | 5000 | **86.48%** | ±0.92 | 10.78 cm | [9.82, 11.74] |

- cross 的 **source retention 仅 0.38%**：模型几乎从不照抄 source 风格，86.48% 命中请求目标。
- 逐 target style：matched 70/90 个 style ≥80%（8 个 <50%）；cross 68/90 ≥80%（同 8 个 <50%），
  最差的是 style 70(6%)、75(25%)、29(31%)、2(31%)、25(36%)。

## 达成情况

- 目标 **matched ≥85 → 90.30%**、**cross ≥80 → 86.48%**：均在封存 test + 正式 rollout 协议下达成。
- validation 曲线（ep150）：matched 85.93 / cross 84.81（ADE 8.30/10.70），test 略优。
- 对照旧镜像链路最好 label 模型（rollout 81.90/74.02，ADE 12.53/16.27）：matched +8.4、
  cross +12.5 个点，ADE 同步下降。
- 选型说明：`constraints_passed=False`（ADE 门限 5/10cm 未过），best.pt 为诊断最优候选；
  若严格按门限则无合规 checkpoint，本轮按"取最高 S_TF 的诊断候选"规则锁定。

原始输出：`seen/`；曲线：`../checkpoints/flowmatch_label_nomirror_150ep/history.jsonl`。

# 6 层 field 的 10ep 纯 TF + 观测噪声短训（bs512）与分析

2026-10-13。目的：在「100ep 纯 TF + 观测噪声」主线配方上，把 field encoder 从 2 层加深到
6 层，用 10ep 短训验证加深是否值得进入正式长训；batch 由用户指定 512，其余参数按仓库既有
缩放规则适配（sqrt 规则：5e-4@512 → 7e-4@1024 → 1e-3@2048；10ep 试点令 cosine 调度随
EPOCHS 收尾）。

## 一、配置（今日两跑唯一变量 = FLOW_LAYERS）

| 项目 | 值（两跑相同） |
|---|---|
| 入口 | `scripts/run_train_flowmatch_history_obsnoise_tf100.sh`（本次新增 `EPOCHS` 旋钮；`FLOW_LAYERS` 覆盖层数） |
| 数据 | `database_100sty_375_memmap_nomirror`，VQ/cache/prior/latent_stats 全部冻结资产 |
| 结构 | transformer field、motion token（6 token）、history 双帧、`--train-feedback-context` |
| 课程 | 全程纯 TF（p=0，无 rollout、无 Euler/decode 训练路径） |
| 观测噪声 | A=U[0,0.5]·transport_std（`--observation-noise-std 0.5`），训练/验证门控同 100ep 主线 |
| batch / lr | 512 / 5e-4 cosine→1e-5（`lr-schedule-epochs=10`，随 10ep 收尾） |
| 其他 | seq-len 6、unroll 4、wd 1e-4、grad-clip 1、dropout 0、seed 42、从零训练 |
| 验证/选型 | 每 5ep 240f rollout 验证（90 clips×3 windows）、motion-quality selection、绝对门限 |
| 评估 | 封存 test：1000 matched / 5000 cross、240f@60fps、std1、Euler4、clip→quantize→decode |
| 产物 A | `checkpoints/flowmatch_label_history_obsnoise_tf10_l6/`（6 层，best=ep10） |
| 产物 B | `checkpoints/flowmatch_label_history_obsnoise_tf10_l2_bs512/`（2 层同参对照） |
| 评估 A | `eval_results/tf10_l6_obsnoise_rollout/seen/`；探针 `analysis/motion_quality_diagnosis/probes/mpjpe_tf10_l6_obsnoise_ns1.0.json` |
| 评估 B | `eval_results/tf10_l2_bs512_obsnoise_rollout/seen/`；探针 `analysis/motion_quality_diagnosis/probes/mpjpe_tf10_l2_bs512_obsnoise_ns1.0.json` |

## 二、6 层 10ep bs512 结果

### 训练内滚动验证（270 matched + 540 cross，每 5ep）

| epoch | matched | cross | mADE / cADE (cm) | jitter | world FS |
|---|---:|---:|---:|---:|---:|
| 5 | 71.85 | 21.85 | 8.55 / 8.92 | 0.696 | 0.783 |
| 10 | **75.56** | **26.67** | 4.87 / 6.57 | 0.571 | 0.632 |

选型：ep5/ep10 均过约束，联合分 ep10 更优（7.113 vs 9.104）→ best.pt = ep10。

### 封存 test（与全部历史 run 同协议）

| 指标 | **6 层 10ep bs512** | CI95 |
|---|---:|---|
| matched style acc | **83.60%** | ±5.24 [78.12, 88.60] |
| cross style acc | **31.72%** | ±2.67 [28.93, 34.27] |
| matched RootADE | 5.30cm | ±0.57 |
| cross RootADE | 6.74cm | ±0.51 |
| MPJPE std1（global） | 7.57cm | [6.97, 8.24] |
| local MPJPE | 4.14cm | [3.74, 4.58] |
| jitter | 0.560cm | [0.522, 0.605] |
| world F.S. | 0.625cm | [0.570, 0.683] |

### 训练成本

10 epoch 总计墙钟 ~30min（ep1–3 ~185s，ep4–10 ~171s/epoch，含 2 次滚动验证）+
封存评估 ~3min + MPJPE 探针 ~20s。峰值显存 8.65GB（bs512；同配方 bs2048 为 9.2GB@2 层 /
12.5GB@6 层）。

## 三、对照与归因

### 3.1 直接对照（同配方唯一变量=层数）

| 指标（封存 test） | 6 层 10ep | 2 层 10ep bs512 同参 | Δ |
|---|---:|---:|---:|
| matched acc | 83.60% [78.12, 88.60] | 82.30% [76.83, 87.66] | +1.30pp（CI 重叠） |
| cross acc | **31.72% [28.93, 34.27]** | **21.18% [18.60, 23.46]** | **+10.54pp（CI 不重叠）** |
| matched ADE | 5.30cm | 6.71cm | −1.40cm |
| cross ADE | 6.74cm | 8.14cm | −1.40cm |
| MPJPE | 7.57cm [6.97, 8.24] | 8.88cm [8.25, 9.62] | −1.30cm（CI 几乎不重叠） |
| local MPJPE | 4.14cm | 4.27cm | −0.13cm |
| jitter | 0.560cm | 0.585cm | −0.025cm |
| world F.S. | 0.625cm | 0.662cm | −0.037cm |

**净效应（6 层 vs 2 层，同配方同参、从零、同 seed）：cross +10.5pp 显著，matched 持平
（+1.3pp 在噪声内），全部质量指标同向更好。加深 field 的收益集中在跨风格表达力。**

训练内滚动验证（270 matched + 540 cross）同向：

| epoch | 6 层 matched / cross | 2 层 matched / cross |
|---|---:|---:|
| 5 | 71.85 / 21.85 | 72.59 / 16.30 |
| 10 | 75.56 / 26.67 | 76.67 / 16.30 |

（内部验证 matched 小样本上 2 层 ep10 略高，封存翻转，属噪声范围；cross 两处一致偏向 6 层。）

成本：2 层对照 ~88s/epoch（总 ~15min），6 层 ~171s/epoch（总 ~30min）；封存评估各 ~3min。
两跑均从零、同数据/协议、`best.pt` 均为质量选型 ep10。

### 3.2 历史参照（非同参，注意混淆）

| 指标（封存 test） | 6 层 10ep bs512（本轮） | 2 层 10ep bs1024 试点 | 2 层 100ep bs2048（主线） | ODE4 100ep（无噪声） |
|---|---:|---:|---:|---:|
| matched acc | 83.60% | 72.90% | 84.90% | 77.90% |
| cross acc | **31.72%** | 15.88% | 20.90% | 8.18% |
| matched / cross ADE | 5.30 / 6.74cm | 8.86 / 10.05cm | 4.52 / 7.54cm | 3.18 / 4.87cm |
| MPJPE | 7.57cm | 11.36cm | 6.89cm | 6.43cm |
| jitter | 0.560cm | 0.638cm | 0.474cm | 0.791cm |
| F.S. | 0.625cm | 0.736cm | 0.610cm | ~0.65cm |

读法：

- **10ep 的 6 层模型已接近 2 层 100ep 主线**：matched 83.6 vs 84.9（差 1.3pp，CI 重叠），
  质量指标略逊（MPJPE +0.7cm、jitter +0.09、FS +0.015），**但 cross 高出 +10.8pp
  （31.7 vs 20.9，CI 不重叠）**——用 1/10 训练量拿到更高的跨风格表达力。
- 对 2 层 10ep 试点（不同 batch/lr/调度）全指标大幅领先，但含 batch/lr/schedule 混淆，
  严格归因看 3.1 节同参对照。

## 四、结论与建议

1. **层数效应已隔离确认：6 层在同配方同参下 cross +10.5pp、matched 持平、质量全面略优。
   加深 field 是当前唯一被干净证据支持、能推动 cross 向 80% 目标靠近的结构性改动**；
   且成本可控（bs512 峰值 8.65GB，16GB 卡无压力）。
2. **10ep 的 6 层模型已接近 2 层 100ep 主线**：matched 83.6 vs 84.9（CI 重叠），cross 31.7 vs
   20.9（+10.8pp，CI 不重叠），也就是说 1/10 预算就超过了主线 100ep 的跨风格水平。
3. 下一步建议：**6 层 100ep 正式长训**。batch/lr 有两种已验证可选项：本轮的 bs512/5e-4
   （~4.7h 墙钟）或 sqrt 规则的 bs2048/1e-3（~1.6h，峰值 ~12.5GB，冒烟已过）；batch 轴
   在 6 层上尚未单独隔离，若关心该轴可先跑 bs2048 短对照再定。观察 100ep 的 cross 爬升
   幅度（10ep 已达 31.7%，若保持同比例爬升 cross 有望继续显著上行）。
4. 混淆与边界：本轮 l6 与其历史对照（bs1024 试点）除层数外还有 batch/lr/调度差异（已用
   同参对照隔离）；两跑均从零、同 seed、同数据、同验证/评估协议。

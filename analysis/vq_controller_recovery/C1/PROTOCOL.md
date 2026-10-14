# C1 预注册协议：不训练的推理旁路扫描（默认关闭，仅诊断）

依据 [docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md) §4 C1。
本文件在 C1 运行前冻结；结果写入 `per_source.json` / `manifest.json` / `REPORT.md`。

## 1. 目标

在 C0 确认"自反馈分布偏移是主因、量化不是"之后，扫描**推理期**可调项，判断是否存在无需训练即可
缩小缺口的旁路。所有候选权重全冻结、单次只改一项，基准为 C0-C（GT 轨迹、官方量化反馈、
Euler 4、noise 1.0、latent_clip=True、requantize=True）。

## 2. 扫描矩阵（唯一变量，其它严格同 C0-C）

| ID | 变量 | 取值 | 备注 |
|---|---|---|---|
| REF | 基准（=C0-C） | — | Euler 4 / noise 1.0 / requantize on |
| R0 | `requantize` | **off** | 仅诊断（部署需显式关闭正式反馈语义） |
| S2 | Euler 步数 | 2 | 速度/质量权衡下界 |
| S8 | Euler 步数 | 8 | 官方 smoke 默认（历史 100ep 训练即时 4） |
| N0 | inference noise `std` | **0** | `z0=0`；已知风险：冻结/滑行 |
| N05 | inference noise `std` | 0.5 | — |
| H0 | history ablation | `history_latent` 置零（等价于无历史增量） | `history_delta` 用 zeros；分布外诊断 |

- N0 必须同时报告静止/冻结比例（moving frame fraction、pose step）与 style 准确率；
  若冻结比例上升明显，判 `NO_USEFUL_INFERENCE_FIX`（不得以"更平滑"换取静止）。
- 不做有界 latent 修正投影（需先写设计与单测，属未来工作）。
- 同源、同风格、同 z0 序列（`case_noise_plan seed=42`），12 源 × 241 帧，与 C0 完全一致。

## 3. 停止条件（来自规约）

任一候选满足以下任一条件即从"可用方向"中剔除，并记录反对证据：
1. style 准确率相对 REF 下降；
2. 速度响应变差（速度方向误差或 planner 跟随）；
3. 冻结比例上升（moving/strong 帧占比下降、帧均位移下降）；
4. local/world FS 恶化超过 0.5%。

**不得只凭 Jitter 改善推广。** 最终给出：最佳诊断方向或 `NO_USEFUL_INFERENCE_FIX`。

## 4. 指标与判定

与 C0 完全同口径（joint-frame L2 cm、官方 motion_eval、jitter 恒等、activity 代理、
code match 仅诊断）。逐源记录 + 12 源汇总；成对比较（每源 REF vs 候选），报告 wins/losses
与均值差。C1 不打 PASS/FAIL 门槛，输出方向性结论与部署开销（每帧延迟）。

# V0 报告：ep90 VQ 基线诊断复核（只读，不训练）

依据 [docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md) §5 V0。
原始数据：[baseline.json](baseline.json)（本次复跑）对
[`analysis/vq_temporal_source/S1/S1_baseline_ep90.json`](../../vq_temporal_source/S1/S1_baseline_ep90.json)（归档）。

## 0. 阶段报告必备字段

- **stage**: V0
- **hypothesis**: 归档 S1 数字可由当前代码在固定 val12 + 90 val 风格代理上逐位复现；
  单位/归一化口径无回归。
- **source_commit**: `58b8870`（C0 提交；probe 脚本自 S1 以来未改）
- **tracked_dirty**: false
- **model_ckpt_hashes**: VQ `…150ep_nomirror/best.pt` file sha `825be6785b3e7fc2…`，
  state sha `918b5e07046266f7…`（运行前后一致）
- **normalization_hash**: 同目录 `norm_stats.npz`（显式加载）
- **db_split_fingerprints**: fixed_val12 与归档 audit starts 一致（probe 内置断言）
- **protocol_and_seed**: 12 源 ×121 帧、seed 42、FP32/highest/TF32 off
- **actual_update_counts**: 0（只读；EMA 无更新）
- **verdict**: **PASS**（逐位复现）
- **next_stage_approval_needed**: V1 Decoder-only 物理位置二阶损失 A/B（5ep/臂）训练授权

## 1. 复现表（本次 vs 归档 S1，全部 0.00e+00 差）

| 指标 | 本次 | 归档 S1 | 差 |
|---|---:|---:|---:|
| 官方 Jitter (quant) | 0.411635 | 0.411635 | 0 |
| MPJPE (quant) | 1.579437 | 1.579437 | 0 |
| local MPJPE (quant) | 1.069192 | 1.069192 | 0 |
| root ADE | 0.851379 | 0.851379 | 0 |
| FS local / world | 0.248099 / 0.488175 | 0.248099 / 0.488175 | 0 |
| 连续 AE Jitter / FS world | 0.120469 / 0.206672 | 0.120469 / 0.206672 | 0 |
| GT Jitter | 0.091734 | 0.091734 | 0 |
| `‖Δ²r‖` per-source mean / p95 | 0.389659 / 1.245851 | 0.389659 / 1.245851 | 0 |
| `‖Δ²e_q‖` per-source mean | 0.397473 | 0.397473 | 0 |
| codec 往返翻码率 | 0.625968 | 0.625968 | 0 |
| 风格代理（90 val source, 240f VQ 重建） | 0.9444 | 0.9444 | 0 |
| 死码 | 0 never-assigned，initted=true | 同 | — |

逐源 `d2r_local` 最大差亦为 0；checkpoint file sha 相同。

## 2. 结论

- 单位/索引/归一化口径无回归；V0 基线可信，可作为 V1 的对照参照。
- 已知局限（沿用 S1 归档，不重复上报）：val12 是分析用固定集；部署价值对照需 90 val
  风格代理 + 后续 240f controller 协议；`code_match`/码身份仅诊断。
- V1 的假设与此基线直接相关：`‖Δ²r‖`=0.390cm 与 quant/cont Jitter 之比 0.412/0.120=3.4×
  表明量化附加的二阶误差集中在 decode 侧；V1 用**物理位置空间**的二阶监督替换失败的
  motion291 混合空间二阶段，检验"能否降低该附加误差而不牺牲位置精度"。

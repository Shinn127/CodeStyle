# C0 报告：闭环因果分解与单次扰动恢复（只读诊断，无训练）

依据 [PROTOCOL.md](PROTOCOL.md)（运行前冻结）与
[docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md) §4 C0。
原始数据：[per_source.json](per_source.json)、[manifest.json](manifest.json)。

## 0. 阶段报告必备字段

- **stage**: C0
- **hypothesis**: 闭环质量衰退可由五组同源对照分解为 (a) 单步 FM/solver 误差、(b) 观测流形
  （raw vs 解码）、(c) 自反馈分布偏移、(d) planner 条件、(e) 量化反馈；且单次有界扰动可恢复。
- **source_commit**: `3ab090d`（G0 后；本阶段无 tracked 代码改动，见 manifest）
- **tracked_dirty**: false
- **model_ckpt_hashes**: Controller `flowmatch_label_history_obsnoise_tf100/best.pt`
  file sha256=`adc4deb2d7da200c…`、state sha=`26dc40eb30bb9370…`（运行前后一致）；VQ
  `…150ep_nomirror/best.pt` state sha=`918b5e07046266f7…`
- **normalization_hash**: VQ run `norm_stats.npz`（显式加载，未重算）
- **db_split_fingerprints**: `database_100sty_375_memmap_nomirror`（只用 val；fixed_val12 与 S1 同源）
- **model_version_compatibility**: VQ ep90 ↔ Controller v12（obs-noise 0.5、pure TF）为现役配套；
  两者 fingerprint 在 G0 交叉核验通过
- **protocol_and_seed**: 241 帧/源、seed=42、`case_noise_plan` 全组共享同一 z0、Euler 4、
  noise_std=1.0、latent_clip=True、requantize=True（除 E 组）
- **architecture_and_optimizer**: 无训练（无 optimizer）；2L/256dim/heads4、history 双帧
- **control_and_candidate**: 五组（A/B/C/D/E）+ 扰动恢复（t0=120，幅度 0.5×transport_std）
- **actual_update_counts**: 0（只读）
- **unit_tests_and_regressions**: G0 套件 CPU 425 / GPU 22 通过（本阶段仅新增分析脚本）
- **per_source_metrics_and_CI**: per_source.json（12 源逐源全部字段；C0 为诊断，未做门槛判定）
- **metric_definitions_and_units**: 与 S1 归档同口径（joint-frame L2 cm；官方 `motion_eval`；
  jitter 恒等已断言；world 使用零 traj 占位）
- **contact_count_and_activity**: 每组含预测接触覆盖与活动代理；逐源记录
- **unexpected_behaviors**: ① 连续反馈（E）姿态误差**不优于**量化反馈（C）；② teacher-anchored
  组（A/B）jitter 高于闭环组；③ D 组单一源（Aeroplane）规划跟随误差 70°（其余 0.7–2.9°）；
  ④ 单次扰动注入改变 15.6/32 槽，但闭环 32 帧内仅残留 0.31cm。
- **verdict**: PASS（诊断完成；结论见 §4）
- **reason**: 五组不变量 0 违规、状态指纹前后一致、结论指向 C2（纯 TF 观测鲁棒性），不支持
  "量化反馈是闭环质量主要来源"。
- **next_stage_approval_needed**: 请求 **C2a** 纯 TF 观测噪声 3 臂 ×10ep 的限额训练授权；
  V1 代码/单测/标定同步准备，但训练需另行批准（本轮只选一个训练 A/B）。

## 1. 五组汇总（12 源均值；A/B/C/E 与 GT 对照；D 为 planner 条件）

| 指标 | A (GT raw) | B (GT decoded) | C (闭环量化) | D (planner) | E (闭环连续) |
|---|---:|---:|---:|---:|---:|
| local MPJPE (cm) | 1.164 | 1.171 | 4.431 | n/a | 5.184 |
| world MPJPE (cm) | 5.205 | 5.160 | 8.145 | n/a | 9.243 |
| root ADE (cm) | 4.769 | 4.712 | 5.927 | n/a | 6.528 |
| 官方 Jitter (cm) | 0.723 | 0.733 | 0.460 | 0.799 | 0.425 |
| local FS (cm) | 0.946 | 0.905 | 0.803 | n/a | 0.727 |
| world FS (cm) | 0.906 | 0.915 | 0.888 | n/a | 0.748 |
| 速度方向误差 (deg) | 4.73 | 4.62 | 6.70 | n/a | 7.78 |
| code match（诊断） | 0.331 | 0.329 | 0.057 | n/a | 0.047 |
| moving 帧占比 | 0.629 | 0.632 | 0.508 | 0.687 | 0.549 |
| strong 帧占比 | 0.186 | 0.191 | 0.107 | 0.167 | 0.108 |
| 预测接触覆盖 | 0.094 | 0.095 | 0.122 | 0.025 | 0.112 |
| style 准确率（matched 240f 分类器） | 12/12 | 12/12 | 10/12 | 7/12 | 9/12 |
| root 末端漂移 (cm) | 11.09 | 10.97 | 12.29 | n/a | 14.94 |

时序分桶（local MPJPE / Jitter，帧区间均值）：

| 组 | 指标 | 1–16 | 17–32 | 33–60 | 61–120 | 121–240 |
|---|---|---:|---:|---:|---:|---:|
| A | local MPJPE | 1.155 | 1.225 | 1.231 | 1.161 | 1.143 |
| C | local MPJPE | 2.007 | 3.440 | 4.650 | 4.715 | 4.692 |
| E | local MPJPE | 1.982 | 3.368 | 5.047 | 5.602 | 5.676 |
| A | Jitter | 0.764 | 0.753 | 0.716 | 0.696 | 0.730 |
| C | Jitter | 0.499 | 0.463 | 0.433 | 0.454 | 0.464 |
| E | Jitter | 0.431 | 0.365 | 0.361 | 0.415 | 0.451 |
| A | code match | 0.329 | 0.323 | 0.340 | 0.332 | 0.330 |
| C | code match | 0.117 | 0.053 | 0.049 | 0.053 | 0.053 |

## 2. 扰动恢复（t0=120，注入 0.568 raw L2，15.6/32 槽真的换码）

| 窗口 | root 偏差 (cm) | 速度方向误差 vs clean (deg) | Jitter (clean) | 帧均位移 (clean) |
|---|---:|---:|---:|---:|
| +8 | 0.036 | 1.97 | 0.449 (0.430) | 0.558 (0.560) |
| +16 | 0.112 | 1.57 | 0.472 (0.449) | 0.590 (0.586) |
| +32 | 0.306 | 1.53 | 0.464 (0.442) | 0.619 (0.618) |

结论：闭环对单步码级扰动**有界恢复**（32 帧残留 0.31cm、方向误差回落、活动不降），
未见扰动放大或失稳；被扰动的码身份（49% 槽）被系统吸收为小幅、平滑的轨迹偏差。

## 3. 因果归因（对照 PROTOCOL §7 规则）

1. **A vs B（观测流形 raw vs 解码）：无实质差异**（local MPJPE 差 ≤0.02cm，全部指标同量级）。
   GT 观测的 raw/codec 流形差不是缺口来源。
2. **B vs C（teacher-anchored → 自反馈闭环）：主导缺口**。local MPJPE 1.17→4.43cm（+3.26cm，
   3.8×），strong 活动 0.191→0.107（−44%），style 12/12→10/12；误差在前 60 帧内累积后于
   ≈4.7cm 平台稳定（自洽吸引子）。此为**暴露偏差/自反馈分布偏移**的直接证据。
3. **C vs E（移除量化）：连续反馈不改善姿态**。E 的 local MPJPE 5.18 > C 的 4.43（E 仅 5/12
   源更优），style 9/12 < 10/12，root 漂移 14.94 > 12.29；只有 Jitter 略低（0.425 vs 0.460）。
   **量化反馈不是闭环质量的主要约束**；换码率/码身份（C 的 code match 0.057）不能用来解释
   姿态缺口——同代码路径的 E 组（无量化）误差相当甚至更大。
4. **C vs D（planner 条件）：另一条件域**。D 的 style 7/12、Jitter 0.799（最差），但规划跟随
   尚可（11/12 源方向误差 0.7–2.9°，Aeroplane 源 70.5°——该 source 的命令速度高于 planner
   上限 1.2，属条件超限）。D 不与 A/B/C/E 混比绝对分数。
5. **单步扰动恢复**：见 §2，闭环局部稳定。

## 4. 结论与指向

- 闭环质量缺口 = **自反馈分布偏移**（训练永远是 GT 观测，推理接自生成观测），而不是量化、
  也不是观测流形（decode 与否）。
- 该证据支持计划 §9 决策树的「仍有分布偏移证据 → 请求 C2 纯 TF 限额训练授权」；**不支持**把
  VQ 侧（V1/V2）当作闭环姿态缺口的主要杠杆。V 线按计划继续准备（对照 A/B 需单独授权）。
- 附带事实：现役 controller 已用 `OBS_NOISE_STD=0.5`（v12）训练，闭环误差仍 4.4cm → C2a 的
  意义在于扫描 {0.25, 0.5, 0.75}（含当前配方对照）而非仅证明噪声有效。
- 与 S1/S4 的衔接：C0-C 的 4.7cm 平台是 controller 侧缺口；S4 的 VQ 侧收益（jitter −9.8%）在
  该缺口下不足以改变闭环质量排序——两线独立，符合规约设计。

## 5. 效率备忘

12 源 × 5 组 × 241 帧 + 12 次扰动重放 = 30.3s 墙钟（单卡，FP32，含分类器推理）。
C1 复用同一库可零成本扩展扫描矩阵。

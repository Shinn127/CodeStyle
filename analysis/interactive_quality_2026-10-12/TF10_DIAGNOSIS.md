# TF-only + history10 短训诊断（2026-10-12）

> 归因复核：下文保留历史数值，但TF10继承了donor-era模型权重，且比较同时涉及history、donor、rollout等变化，不能分别确证各因素因果作用，也不能证明从零matched-only TF有同样的cross能力。文中“必须保留donor/增加rollout”的建议已撤回；当前方案见 [无donor纯TF风格计划](../../docs/DONOR_FREE_TF_STYLE_PLAN.md)。

对照实验：从 `flowmatch_label_6tok_curriculum100/best.pt` 初始化，只加 history 双帧条件，
**纯 teacher forcing（epoch 全程 feedback_probability=0，训练不做 Euler/decode）**，
**完全不用 donor（decouple_prob=0，无 cross-style 合成目标）**，10 epoch。
入口脚本 [run_train_flowmatch_history_tf10.sh](../../scripts/run_train_flowmatch_history_tf10.sh)，
输出 `checkpoints/flowmatch_label_history_tf10`（best=ep10，key [1, 6.5936]）。

目的：把"history 条件"这一条单独摘出来，看它本身对运动质量和风格准确率的影响，
不与 donor 相位配对、rollout 反馈耦合。

## 结果

| 指标 | **tf10（本实验）** | phase10（history+donor+rollout） | 6tok baseline | 说明 |
|---|---:|---:|---:|---|
| 封存 matched style acc | **79.40%** | 95.90% | 96.40% | tf10 掉了约 16pp |
| 封存 cross style acc | **65.68%** | 90.84% | 91.42% | tf10 掉了约 25pp |
| 封存 matched RootADE | **4.78cm** | 7.32cm | 7.96cm | tf10 最好 |
| 封存 cross RootADE | **6.10cm** | 9.28cm | 9.93cm | tf10 最好 |
| MPJPE std1.0（封存 1000 src） | **7.97cm** | 10.28cm | 11.23cm | tf10 最好 |
| MPJPE std0.5 | **7.80cm** | 8.32cm | 10.10cm | tf10 最好 |
| jitter（封存 std1.0） | **0.568** | 2.140 | 1.791 | tf10 最好 |
| F.S.（封存 std1.0） | **0.717** | 2.100 | 1.676 | tf10 最好 |
| 闭环控制视频滑行（std1，Neutral） | 0.42% | 0.42% | 0.42% | 都低 |
| 闭环控制视频滑行（std0.25） | 4.64% | 5.06% | 10.55% | tf10 略好 |
| 闭环控制视频步数 L/R（std0.25） | 5/6 | 3/5 | 7/5 | tf10 更接近 GT（GT 5/6） |

原始数据：`probes/mpjpe_tf10_ns{1.0,0.5}.json`、`videos/tf10_ns{1.0,0.25}/metrics.json`、
封存评估 `eval_results/tf10_shorttrain/`。

## 读法

- **history 条件本身极大地改善了运动质量与轨迹误差**：MPJPE 从 6tok 的 11.23 降到 7.97cm
  （−3.3cm），jitter 1.79→0.57，F.S. 1.68→0.72，ADE 7.96→4.78cm。这是目前所有配置里静态
  质量最好的一档，说明"给模型真实最近两帧 latent 变化方向"确实是有效的观测补充。
- **但没有 donor 的 cross-style 监督后，闭环风格准确率塌了**：matched 96.4→79.4、
  cross 91.4→65.7。train 日志里 cross_samples 全程为 0，即模型从未见过"在别人的内容上执行
  目标风格"的样本；rollout 验证/封存里 cross 场景要求把源动作改成另一风格，泛化就明显不足。
  **这正是 decouple/donor 路径存在的意义**——它不是为了质量，而是为了 cross 泛化。
- 因此 tf10 **不满足**论文目标（matched≥85、cross≥80），不能作为交付模型；但它是干净的
  归因点：**history 条件 → 质量与轨迹提升；donor → cross 泛化**。两者需要同时保留。

## 结论与下一步

- 想同时拿到"tf10 的质量 + phase10 的准确率"，正确组合是 **history 条件 + donor + 少量
  rollout**（即 phase10 的配置），而不是去掉 donor 或去掉 rollout。
- phase10 的准确率已保住（95.9/90.8），质量相对 6tok 有改善但不彻底（jitter 2.14 仍高于
  tf10 的 0.57）。**差距来自 rollout 反馈闭环**：纯 TF 的观测与监督完全一致，质量最好；
  一旦闭环用自采样的 D(q) 反馈，噪声放大重新引入抖动。
- 若要继续压 phase10 的抖动，可考虑：增大 rollout 占比、提高 feedback_solver_steps 与推理
  对齐（8 步）、或在 rollout 批上加入 tf10 那种更强的一致观测。这条路线需另跑验证。

## 后续（2026-10-12 晚）：donor 路径全面移除

按上述归因，用户决定全面清理 donor 训练内容。已删除：主训练脚本的 `matched_root_targets`、
`apply_style_decoupling(_windows)`、`--decouple-prob/--decouple-latent`、`--cross-donor-source`、
`--donor-*`、全部 donor 指标与 resume 字段；clip_recipe v2 的 paired-cross donor 路径；所有 wrapper
中的 donor/decouple 参数；相关测试。`Utils/style_donor_bank.py` 及 donor-only 分析脚本归档于
`docs/archive/donor_sources_2026-10-12/`。训练 objective 统一为 **version 11**，donor 时代
（<=10）的 checkpoint 不能 resume，防止静默续训。

保留的是：history 双帧条件、TF→rollout 课程、FP32 GPU 缓存、八步反馈与质量选型。

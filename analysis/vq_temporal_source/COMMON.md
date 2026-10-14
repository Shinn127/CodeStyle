# VQ 源头时序连续性实验：预注册协议（写于 B0/B1 开跑之前）

> 依据 [docs/VQ_SOURCE_AGENT_EXECUTION_PLAN.md](../../docs/VQ_SOURCE_AGENT_EXECUTION_PLAN.md) 执行。
> 本文在 S3 训练启动前冻结；S1/S2 只做只读诊断与梯度标定，不更新任何权重。
> 本实验是独立实验分支，不是已验证修复，不改现役默认配置。

## 1. 身份与版本（S0 记录实际值）

| 项目 | 值 |
|---|---|
| 代码基线 | git commit `5cdc6e3`（运行前 `s0_contract.py` 复核并记录 dirty 状态） |
| 初始化 | `checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt`，checkpoint 内 `epoch=90` |
| norm | 同目录 `norm_stats.npz`（显式传入，不重算） |
| 数据 | `database_100sty_375_memmap_nomirror`，只用 train 训练、val 调参；OOD 库不参与 |
| 精度 | FP32、matmul highest、TF32 off、cudnn TF32 off；记录 torch/CUDA/GPU/vector-quantize-pytorch 版本 |

## 2. 固定 val12 诊断集（S1/S4 使用，与 2026-10-14 审计同源）

- 选择：val split、非镜像、`num_frames>=122`，`RandomState(42).choice(12)`；起始帧必须等于
  `analysis/vq_code_stability_2026-10-14/audit/baseline_val12.json` 的 `starts`（脚本内断言）。
- 每个 source 取 `frame_start..+120`（121 帧，120 个转移区间）。
- 指标口径（单位：cm，全部来自反归一化后的 24 关节 local/world 位置）：
  - local：`p = motion_raw[:, 3:75].reshape(T,24,3) * 100`；
  - world：`reconstruct_selected_globals_from_375(zero_traj ‖ motion_raw) * 100`（84 维
    traj 槽为零块占位；该重建只读 motion 块。禁止把 normalized traj 与 raw motion 拼接）；
  - `P(x)`：GT 运动；`p_q = P(D(q))`，`p_c = P(D(z_e))`，`p_gt = P(x)`；
  - `r_t = p_q - p_c`；`Δr = r_t - r_{t-1}`；`Δ²r = r_t - 2r_{t-1} + r_{t-2}`；
  - `e_q = p_q - p_gt`；`Δ²e_q` 同理；全部取逐帧 `‖·‖`（对 24 关节 L2，再按帧记录）。
  - 官方 Jitter：`‖Δ²p‖`，逐帧值经 `Utils.motion_eval._jitteriness_cm` 同一公式（×100），
    probe 内断言 `mean(逐帧)=compute_sequence_metric_sums().jitter_sum/count`。
  - world 足滑：GT 接触掩码（原指标）与**预测接触掩码**（新增诊断）分别报告；
    接触定义沿用 `MotionEvalConfig`（趾高 ≤20cm 且趾速 ≤20cm/s，dt=1/60）。
  - local/world MPJPE、root ADE、速度方向误差：一律走 `compute_sequence_metric_sums`。
- 禁止："raw291×100=cm"、step1_* 冒充 Jitter、跨窗口协议比绝对阈值。

## 3. 活动性代理（新增，防"更平滑但不动了"）

逐 source 计算，B1 与 B0 配对比较：

| 代理 | 定义 |
|---|---|
| `moving_frame_fraction` | 帧均关节位移 > 0.5cm 的帧占比（GT 同值参照） |
| `strong_moving_frame_fraction` | 帧均关节位移 > 2.0cm 的帧占比 |
| `leg_travel_cm` | 4 个 foot/toe 关节逐帧位移路径长度均值 |
| `toe_height_range_cm` | 4 个 foot/toe 关节垂向高度的 (max-min) 均值 |
| `foot_lift_fraction` | 趾高 > 5cm 的帧占比 |
| `pred_contact_coverage` | 预测接触掩码为真的帧占比 |
| `pose_step_mean_cm` | 帧均关节位移（连续量） |
| `toe_speed_mean_cm_per_frame` | 趾部逐帧速度均值 |

GT-contact-count 不是活动性证据，只作参照记录。

规范修订记录（B0/B1 开跑前，未见过任何 B0/B1 数据）：

1. 初版脚本把已是 cm 的局部位置又乘了一次 100（jitter 值 ×100）；修正后 `jitter_*` 逐关节
   值均值与 `Utils.motion_eval` Jitter sum/count 在 1e-6 内一致（探针内置断言）。
2. `gt_switch_*` 命名与归档对齐：`gt_switch_slots_per_frame` = 相邻帧间换码槽计数均值，
   `gt_switch_fraction_per_slot` = 逐 (帧,槽) 对换码比例（= 前者/32，与归档 0.5608 同口径）。
3. 接触诊断的掩码函数期望**米**制输入（内部 ×100），初版误传 cm 导致接触全零；已改为 /100。
4. `strong_moving_frame_fraction` 阈值 2.0cm 在 60fps 下无区分度，定为 1.0cm。

## 4. 分层

suffix → 组（依据 100STYLE 命名 + 本库实测 root 速度：W=行走、R=奔跑、S=侧向、I=静止、T=转身）：

- idle = {ID}；walk = {FW, BW, SW}；run = {FR, BR, SR}；turn = {TR1, TR2, TR3}

分层结果只作诊断记录（逐组 mean/p95），不作为独立选型依据（小样本）。

## 5. 梯度标定（S2，不更新优化器）

- L_base = recon×3 + vq×1 + vel×15 + root×3 + code_align×2 + latent_l1×0.01 + latent_vel×0.1
  （B0/B1 的共同监督项）；L_acc = `compute_loss(acc_weight=1)['acc_loss']`。
- 4 个独立 train 小批 B16×T8（seed 4242+i，逐段内采样，不跨 segment 边界）；模型复制副本上
  前向/反向，EMA 用 `freeze_codebook=True` 隔离；**不调用 optimizer.step**；前后 state hash 相同。
- 记录：E/D 分组 `‖g_base‖`、`‖g_acc‖`、cos、每批比值；取 `ACC_WEIGHT=w` 使两组
  `w‖g_acc‖/‖g_base‖ ∈ [0.1, 0.3]`；若两组隐含 w 差异过大或任一组 cos<0（冲突）或统计极不稳，
  标风险并停止 pilot 分析（不硬跑）。

**S2 实测结果（2026-10-14，脚本 `s2_acc_gradient_calibration.py`，结果 `S2/gradient_calibration.json`）**：

- encoder：`‖g_acc‖/‖g_base‖ = 0.0759`（4 批 0.063–0.082），cos = 0.52（0.46–0.61）
- decoder：`‖g_acc‖/‖g_base‖ = 0.0162`（4 批 0.0146–0.0176），cos = 0.11（0.07–0.14）
- 两组的 10–30% 权重窗口不重叠：encoder w∈[1.32, 3.96]，decoder w∈[6.19, 18.56]。
- 判定：cos 全为正（无冲突）、批间统计稳定（无"极不稳"）；E/D 响应比 4.7×，属**中等不均衡**，
  未达"极不均衡"停跑条件。按本文档预注册的回退规则取缺口几何中点
  `w* = sqrt(3.995×6.218) = 4.9842` → encoder 37.8%、decoder 8.1% of base gradient。
  该选择保留了风险标记（写入 JSON `risk_flags`），S4 的活动性/质量门是防止过权重的保护。
- 与仓库既有注释（"3-5 is a useful range"，由 d2 误差比得到）独立吻合，但不是按其设置的。
- DoD：未调用 optimizer.step；前后 `state_fingerprint` 相同；批次来源逐条落盘。

## 6. B0/B1 实验契约（S3）

两 run 唯一变量 = `ACC_WEIGHT`；都从 ep90 完整 VQ 状态冷启（init，非 resume），都更新 E+Q(EMA)+D：

| 字段 | B0 | B1 |
|---|---|---|
| `seq_len` / `stride` / `batch` | 8 / 1 / 512 | 相同 |
| `acc_weight` | 0 | **4.9842**（S2 gap 中点） |
| 其余权重 | 3 / 1 / 15 / 3 / 2 / .01 / .1（recon/vq/vel/root/code_align/latent_l1/latent_vel） | 相同 |
| margin / roundtrip / teacher | 0 / 0 / 关闭 | 相同 |
| epochs / eval_every / save_every | 10 / 1 / 5 | 相同 |
| seed / loader_seed / lr | 42 / 4242 / 1e-4→1e-5 cosine | 相同 |
| OUT | `checkpoints/vq_temporal_B0_ep90seq8_10ep` | `checkpoints/vq_temporal_B1_ep90seq8_10ep` |

统一入口：`scripts/run_train_vq_temporal_ab.sh`（ARM=B0|B1、ACC_WEIGHT=…，其余 env 已按本表 pin 死）。

训练前后记录：config.json、history.jsonl（逐 epoch updates）、manifest.json（含源/数据/ norm
指纹、commit、初始与最终 state hash、真实每 epoch 更新数）。训练后量化向量必然改变：**新 VQ 身份，
不得复用旧 cache/旧 controller**。

**checkpoint 选型规则（预注册，修订记录见下）**：主分析 checkpoint = 各 run 的
`epoch_0010.pt`（= 固定预算终点，cosine 已衰减到 min_lr；两臂同一条规则、**零选择**）。
`epoch_0005.pt` 作为中途鲁棒性对照；门控 `best.pt` 一并报告但**不用于选型**——其
first-eval-always-accepts 语义在 10ep 短跑下可能把 best 冻结在 ep1，属于规则性假象而非
"事后挑有利结果"。若主/次 checkpoint 在任一预注册 gate 项上结论冲突 → 判 INCONCLUSIVE
（不推广）。禁止事后挑 epoch。

> 选型修订记录（B0/B1 开跑前，未见过任何 B0/B1 训练数据）：初版把 `best.pt` 定为主分析
> checkpoint。复核训练器 gate 语义（`check_best_gate`：reference=None 时无条件接受首个 eval）
> 后确认该规则在 EVAL_EVERY=1 的 10ep 跑里可能产生 best.pt=ep1 的结构性假象，故改为主规则
> = 固定终点 epoch_0010.pt。此修订只利用对训练器代码的先验阅读，不涉及任何本实验产出数据。

## 7. S4 验收（预注册门槛与否决规则）

> 预注册修订记录（发生在 B0/B1 开跑之前，未见过任何 B0/B1 数据）：style 代理从"val12 的 12 个
> source"扩为"全部 90 个 val source"——12 源的粒度为 8.3pp/源，2pp 门槛不可判定；90 源为 1.1pp。
> 其余判据/门槛不变。固定 val12（12 源）仍用于门 1–6、8 的配对比较。

主表 = val12 配对比较（B1−B0，按 `source_entry_id` 配对 bootstrap，1000 次重采样，seed 42，95%
百分位区间）：

| # | 判据 | 通过条件 |
|---|---|---|
| 1 | `‖Δ²r‖` mean 与 p95 | 两者都下降，且 95% CI 上界 < 0 |
| 2 | `‖Δ²e_q‖` mean | 不增（B1 ≤ B0×1.005） |
| 3 | 官方 quantized Jitter | 相对下降 ≥ 15% |
| 4 | world 足滑 | 不恶化（B1 ≤ B0×1.005；local 同报） |
| 5 | local/world MPJPE、root ADE | 相对退化 ≤ 3%（每项） |
| 6 | 活动性代理（§3，除 GT 参照外全部） | 相对下降 ≤ 5%（每项） |
| 7 | IND style 保真代理（冻结 100 类分类器，**全部 90 个 val clips** 前 240 帧 VQ 重建；1 source = 1.1pp） | 准确率下降 ≤ 2pp |
| 8 | 连续 AE 对照（r 与 c 差异） | 记录；若 B1 的连续路径同步变平滑，须在报告中区分，不得记为量化残差收益 |

**否决（直接 FAIL）**：只降换码率/只提 roundtrip/只降 MPJPE；以冻结运动换低 Jitter（活动性
或位移代理同步大幅下降）；明显退化（style 或活动性崩坏）→ STOP，不自动延长到 150ep。
小样本 CI 未跨 0 的判据记"未确认"；全部判据通过且无否决 → PASS；有明确反向证据 → FAIL；
预算内无区分度 → INCONCLUSIVE。负结果完整归档。

### 7.1 契约闸门 g0 强制化（审计后加固，2026-10-14 傍晚）

独立审计指出初版 `s4_acceptance.py` 的 manifest 检查是可选且只由 `same_total_updates` 决定，
不同初始化/数据只要更新数相同仍可能 PASS。已修复为**必填** `g0_contract`：两份 manifest + 各自
`config.json` 全字段核对（初始状态指纹、init 文件 sha、DB/split/VQ 状态指纹、norm sha、pinned
协议、acc 单变量模式、probe 来源目录、config 差异键 ⊆ {out_dir, acc_weight}），任一不符 →
**INVALID_EXPERIMENT**，不进入质量判定。

时点与影响：该加固发生在 B0/B1 完成之后；它收紧的是"契约不满足时的短路"，不改变任何质量门的
阈值或方向。用真实 manifest/config 重跑后 g0 全部 12 项 PASS，两臂判定仍为 FAIL，结论不变。
同批修复：`best.pt` epoch 改为从 `history.jsonl` 动态读取（更正初版报告"均停在 ep1"的错误，
实际 B0 best=ep1、B1 best=ep3）；连续 AE 对照的措辞更正（见 S4/REPORT.md §3）。

## 8. 边界

- 不读 test 做任何选择；历史 test 探针已用，不能称 untouched。
- 本轮结束即停：150ep 重训、cache 刷新、controller 训练、线上替换均需另行授权。

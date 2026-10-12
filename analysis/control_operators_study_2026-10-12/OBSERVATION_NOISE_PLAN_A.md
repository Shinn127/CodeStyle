# 方案 A：观测噪声增强（实现 + 代码 review）— 2026-10-12

对应 ControlOperators 学习报告的方案 A：把 ControlOperators"单步训练 + 输入噪声"的
鲁棒化机制引入我们的 TF 课程，替代/部分替代 rollout 的作用。
设计依据：[control_operators_study REPORT](REPORT.md)；量化 A/B 已证明反馈量化不是主因
（[QUANTIZATION_FEEDBACK_AB.md](QUANTIZATION_FEEDBACK_AB.md)）。

## 一、设计（来自实测校准）

校准数据（ODE4 ep100，真实 240f 协议 rollout，16 源）：
真实反馈误差的 latent 每维 RMS ≈ 1.1×transport_std（随机分量 ≈ 1.25，含采样分支切换；
系统分量 ≈ 1.0）。噪声增强应匹配"随机抖动"量级且不复制系统偏差，因此采用与
ControlOperators 相同的**相对幅度**形式（他们用 U[0,1]×0.5，单位=每维 std）：

```
shift = latent_scale · transport_std · (A · U[0,1]) · ε    # A = --observation-noise-std
motion_obs, latent_obs = feedback_state(z_gt + shift)      # clip → quantize → decode
```

- 施加位置：**TF 分支的滚动态观测**（step ≥ 1）：latent token 与 motion token 一起替换，
  保证两者严格一致（motion = D(Q(clip(z+shift)))），且落在码本流形上
- 不施加于：step 0（种子帧）、验证/评估（`_run_epoch` 里 `is_train` 门控）、
  feedback 分支（p=1 或门控选中 feedback 的行）
- RNG：仅 A>0 时消耗随机数；A=0 与旧路径**逐位一致**（回归测试覆盖）
- 成本：每应用步一次 decode（TF 阶段 ≈ +2.1ms/step，即每 update +6.1ms）
- 版本：objective 11（A=0）→ **12**（A>0）；resume 校验 `observation_noise_std` 字段

入口脚本 [run_train_flowmatch_history_obsnoise_100ep.sh](../../scripts/run_train_flowmatch_history_obsnoise_100ep.sh)：
50 TF + 20 ramp + 30 rollout、history、Euler4、`OBS_NOISE_STD=0.5`（默认）；
`OBS_NOISE_STD=0` 即干净基线（A/B 用）。

## 二、实现正确性验证

单测 10 项（`tests/test_observation_noise.py`）+ 冒烟 + 端到端：

| 检查项 | 结果 |
|---|---|
| 观测被替换且 motion=decode(Q(clip(·))) 严格一致 | ✅ |
| 轨迹块不动、history 跟随上一观测 | ✅ |
| A=0 时零 decode、零 RNG 消耗（纯 TF 不引入解码） | ✅ |
| 每应用步恰好一次 clip→quantize→decode（3次@steps4） | ✅ |
| 种子确定性；p=1 与 steps=1 完全惰性（RNG/梯度/解码数逐位一致） | ✅ |
| 禁用/启用 loss、梯度均有限且不同 | ✅ |
| 训练系数均值 0.2504 ≈ U[0,0.5] 期望；验证路径 = 0.0 | ✅ 实测 |
| feedback_state 顺序 = clip→quantize→decode（推理口径一致） | ✅ 源码核实 |
| 噪声单位 = scale·transport_std·A·U（与采样噪声同一相对尺度） | ✅ 源码核实 |
| resume 字段 + legacy 默认（旧 ckpt 可读、不误 resume） | ✅ |

## 三、Code review 结论（正确性 + 效率）

### 正确性
逐项核对 5 个不变量（顺序/版本/RNG/单位/门控）无问题；无 autograd 泄漏（no_grad 包裹、
反馈 detach）；无缓存张量原地修改（noised 观测经 cat 新建）；`source_latents` 未被写入。
未发现正确性缺陷。

### 效率（review 发现 5 项，3 项已落地，均实测验证）

| # | 位置 | 问题 | 处置 | 实测 |
|---|---|---|---|---|
| 1 | `basic_vqvae_375to291.py:157` `quantize_only` | 两遍范数式距离计算 | ✅ 改为单次增广 addmm（argmin 的 ||x||² 是行常数可省） | **1.77×**（1.47→0.83ms@B1024），与旧路径和库路径在 **6144 个真实 latent 上逐位一致** |
| 2 | `flowmatch_vqvae_controller.py:1169` dedup 门控 | 验证走逐样本 selector（4096 行/批） | ✅ 门控放宽到 eval（数学等价的 90 行表 + 加权正则） | val 指标差 ≤4.8e-7（fp32 归约序），flow/loss 逐位一致；省 ~8s/epoch |
| 3 | `motion375.py:137` `_integrate_root_motion` | 逐帧 Python 循环（每个指标调用×2） | ✅ 闭式向量化（float64 累加防漂移） | **200×**（7.6ms→0.0395ms/240f），与逐帧循环差 ≤2e-6m（fp32 舍入）；**MPJPE 探针 81.5s→18.0s** |
| 4 | `eval_rollout_mpjpe.py:104` | 每 case 重复计算 global/root（sums 已返回） | ⏸ 未做 | 需改指标口径（jitter/FS 的 slice 定义会变），收益 ~9s/次，#3 已拿走主要收益，**不做** |
| 5 | `controller_validation.py:200` 逐 case `_root_ade_cm` | 270×循环根积分 | ✅ 由 #3 自动受益 | 每次验证省 ~2.8s |

效率总账（obsnoise 100ep run）：#1+#2 约省 25–30 分钟墙钟；
验证每 5 epoch 由 ~25.4s 降到 ~22s（#3），MPJPE 探针快 4.5×。

### 已排除（明确检查后判定无需处理）
`torch.cat`/`where`/`clone` 开销总计 <0.15ms/update；训练路径无 GPU↔CPU 同步；
无"算了不用"的 decode；resident loader 已消除主要 H2D。

## 四、后续建议

1. **跑 A/B**：`OBS_NOISE_STD=0.5`（默认）vs `OBS_NOISE_STD=0`，其余完全一致，
   两个 100ep run（各 ~2.3h）后比较封存 matched/cross、MPJPE/jitter/FS、闭环滑行。
   若噪声版把 rollout 阶段的 jitter/FS 回升压住且质量不降，则考虑下一轮减少 rollout 占比。
2. 噪声幅度可调（0.25/0.5/0.75 扫）——校准显示真实随机误差 ≈1.25σ，0.5 起步合理。
3. **#4 若要做**：需同时定义清楚 jitter/FS 的 slice 口径并与历史对齐后再改，属独立小任务。

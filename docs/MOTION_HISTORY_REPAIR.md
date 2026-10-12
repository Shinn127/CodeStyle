# Label 动作连续性修订与短训入口

> **2026-10-12 晚更新：donor 训练路径已全面移除。** 10 epoch 短训对比证明 donor 的跨风格合成目标
> 损害闭环运动质量（tf10 donor-free：MPJPE 7.97 / jitter 0.57；phase10 含 donor：10.28 / 2.14；6tok：11.23 / 1.79）。
> 本页下方描述中的 phase-aware donor bank、`--cross-donor-source`、`--decouple-*` 已不存在；
> `run_train_flowmatch_history_phase10.sh` 在 2026-10-12 短暂以 donor-free 配置运行，随后于 2026-10-13 清理中删除（下方为历史记录）。
> 归因证据见 [TF10_DIAGNOSIS.md](../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)，
> 归档源码见 [docs/archive/donor_sources_2026-10-12/](archive/donor_sources_2026-10-12/README.md)。
>
> **2026-10-13 入口收敛**：本页实验用入口 `run_train_flowmatch_history_phase10.sh`（rollout）与
> `run_train_flowmatch_history_tf10.sh`（纯TF）已在清理中删除；label 主线入口为
> `scripts/run_train_flowmatch_history_obsnoise_tf100.sh`（100ep 纯TF + 观测噪声），
> 短测/变体可用基础入口 `scripts/run_train_flowmatch_6tok_curriculum100.sh` 加环境变量复现。

2026-10-12：代码修订及轻量验证完成，没有启动完整训练。原因及原始审计见 [诊断方案](../analysis/interactive_quality_2026-10-12/NO_EXTRA_LOSS_SOLUTION.md)。运动质量是否改善仍需短训后的固定 rollout 验证。

## 实际改动

- 保留六个 token、256 hidden、两层 transformer、target/normalized FM、冻结 VQ 与原 normalization。previous-latent token 增加两帧 normalized latent 差分的零初始化线性投影。相同输入/noise 下，投影初始化时输出与原模型一致；没有新增 loss。
- TF 用连续 source GT 的两帧 q；展开/离线/交互用实际过去两帧反馈 q。reset 初始差分为零，切换 style 保留历史。新入口所有 TF 观测为 D(source q)，验证 loss 也使用这一观测；GT/合成监督保持独立，反馈严格 clip→quantize→decode，detach。
- phase-aware train-only donor bank 支持整个四步目标窗口。suffix、root/yaw 门限不变，用16帧上下文匹配相位；脚部相对运动有置信度时拒绝反向候选。全窗口 donor/offset 固定，无候选回退 matched；retarget 保留 source root 并修正 joint velocity。
- 每批完全 TF 或完全展开：TF 批不做 Euler，展开批用完整八步。反馈只计算 latent，取消无用 continuous decode；条件投影在同次 Euler 内复用。
- train features、code indices、D(q) 使用只读 FP32 GPU 缓存。donor 对应关系按 source anchor/offset 缓存；查询矩阵形状固定，避免缓存冷热改变近乎并列的候选。构建查询库时只保留 root，修掉切片保留整块 motion 的显存浪费。
- 验证选型加入原始 jitter、matched local/world 足滑、contact 覆盖；matched/cross 分别报告姿态单帧跳变 p99/max。cross 不借用 source GT 的接触相位计算足滑。零 contact 不通过约束；全部失败时保存诊断 best，标记 `constraints_passed=false`。
- 交互新增显式 `--smooth-plan`；IK 用 filtered contact speed 加 raw speed 防滑行误锁，UI 区分开关、接触数量和完整锁定数量。显示后处理不写回生成器，不计入正式指标。

## 10 epoch 诊断微调

在 NewCode 下执行，正式训练由用户启动：

```bash
# 历史入口（已删除，记录保留）：phase10 以 donor-free 配置运行
DRY_RUN=1 bash scripts/run_train_flowmatch_history_obsnoise_tf100.sh   # 现行主线入口
bash scripts/run_train_flowmatch_history_obsnoise_tf100.sh
```

| 项目 | 默认值 |
|---|---|
| 初始化 | `flowmatch_label_6tok_curriculum100/best.pt`，兼容权重初始化；新增 history 投影为零 |
| 输出 | 独立 `checkpoints/flowmatch_label_history_phase10` |
| 训练 | seed42、10 epoch、FP32/highest、关闭 TF32、stride1 |
| batch/window | 1024 windows/update，seq_len6（一个额外历史帧），四步平均 FM，feedback detach |
| loss | flow1、prior CE .02、entropy .003；recon/root/style-consistency 均为0 |
| 优化器 | AdamW、lr5e-5→1e-5、cosine10、weight decay1e-4、grad clip1 |
| donor | matched_root 请求 .3，phase-aware bank、16帧上下文、32 windows/clip |
| feedback | ep1–2纯 TF；ep3 .08333、ep4 .16667、ep5–10 .25；按批确定性调度 |
| 采样 | 独立 Gaussian std1、Euler8、clip→Q→D |
| 缓存预算 | train cache9 GiB、donor matching2 GiB；CUDA allocator expandable segments |
| 验证 | ep5/10，固定270 matched+540 cross、240f/60fps rollout；validation loss 每epoch |

新选型诊断门限：matched acc≥.85、cross acc≥.80，Root ADE≤5/10cm；matched/cross jitter≤.5cm，matched local/world F.S.≤.8/1cm且有contact。先比较是否全过，再比较归一化误差与 accuracy 的联合分数。这是当前质量筛选门限，不是论文复现标准。`best_loss.pt` 另存 loss 候选；正式评价读取 best 的 `args.best_selection` 和 `history.jsonl`，不能仅看文件名。

日志新增 `donor_opposite_fraction`、`donor_first_pose_step_cm`、`donor_target_style_counts/coverage`、实际 cross 比例和峰值显存。真实100批 smoke：cross≈17.7%，候选均值≈2.09、反向代理比例0、首目标 pose差≈11.6cm。相位门限避免了一类冲突，但大幅 style 改变仍可能造成跳变，不保证充分解耦。

```bash
# 从零训练（现行入口；phase10 专用脚本已删除）
FROM_SCRATCH=1 OUT=checkpoints/flowmatch_label_history_obsnoise_tf100_scratch \
  bash scripts/run_train_flowmatch_history_obsnoise_tf100.sh
# 同目标/config/cache模式的 epoch 边界续训（示例：ODE4 历史 run）
RESUME=checkpoints/flowmatch_label_history_ode4_100ep/latest.pt \
  bash scripts/run_train_flowmatch_6tok_curriculum100.sh
# 短训后手动切换交互模型；默认交互 checkpoint 仍保留旧 Label150
FLOW_CKPT=checkpoints/<your_run>/best.pt \
  bash scripts/run_interactive_label_control.sh --smooth-plan
```

新 training objective version=10，旧状态不能直接 resume；旧模型仍可读取。`--init-ckpt` 启用 history 时只允许缺失新投影，其他权重/VQ 来源仍校验。resume 要求相同缓存模式/codec chunk；预算不足时有显式 fallback，查看 `[GPU train cache]`。可用 `MICROBATCH_SIZE` 降低训练峰值，donor 仍先在完整1024窗口上构造。

CUDA末位误差不等于 RNG 被改变：真实冷/热缓存的 targets 和 RNG 检查逐值一致，100更新模型 hash 在非确定性 CUDA backward 下仍可能不同；不声称所有GPU场景位级确定。CPU epoch续训及轻量CUDA续训测试通过。

不要靠减 noise 或强相关 noise 将冻结姿态变成低 jitter。十个 epoch 后若原始运动质量不改善，停止自动延长，按 matched/cross、历史 delta 实值/置零、GT轨迹/交互轨迹拆解，再决定是否修改目标过渡。

## 提速与验证

RTX5070Ti16GB / Torch2.13.0+cu132，FP32；100 updates、batch1024、unroll4、25% feedback、Euler8：

| 路径 | 100 updates |
|---|---:|
| CPU数据、旧 per-window 反馈执行 | 8.71s |
| 整批调度、CPU数据、在线 GT decode | 5.75s |
| 整批调度、GPU全缓存冷启动 | 4.99s |
| GPU缓存重复固定批次（热缓存上限） | 4.32s |
| 最终入口含新增诊断统计 | 5.27s |

最终入口约1.65×提速；热缓存对照约2×，**不是完整epoch的实测加速比**。batch gating 改变门控组织，这也不是精度对照。入口准备/缓存构建约5.8s，独立于 update 时间。最终峰值 allocated≈11.0GiB；训练段20次 GPU 利用率采样均值78.8%（54–87%，短测），排除setup/validation。GPU train缓存≈6.99GiB、对应缓存≈1.64GiB。

按2584 batches/epoch外推，25%反馈下主循环约2.3min/epoch，10epoch约0.38GPU小时；前两epoch纯TF更便宜。包含验证等的总成本暂估约0.4–0.5GPU小时，**尚未实跑完整epoch**，数据访问、保存和系统负载会改变时间。

327项CPU、21项CUDA回归通过；真实100-update入口、诊断checkpoint选型保存、60/600f headless、12f simulator TF、6-case240f验证和60帧GUI通过。VQ无梯度/权重不变，反馈 motion/latent 匹配。6-case检查未通过运动质量约束，属于 smoke checkpoint 的接口验证，不能据此声称训练有效或accuracy达标。

证据：[性能](../analysis/interactive_quality_2026-10-12/implementation/performance.json)、[缓存正确性](../analysis/interactive_quality_2026-10-12/implementation/cache_correctness.json)、[rollout检查](../analysis/interactive_quality_2026-10-12/implementation/rollout_verification.json)、[GUI截图](../analysis/interactive_quality_2026-10-12/implementation/gui.png)、[完整核验](../analysis/interactive_quality_2026-10-12/implementation/verification.json)。

```bash
python run_tests.py cpu
python run_tests.py gpu
# 最多3组各100次更新，不保存模型；热模式重复相同批次
python analysis/interactive_quality_2026-10-12/implementation/benchmark_training.py
```

# CodeStyle｜闭环反馈稳定性：禁止 Rollout 训练的执行规约

> **状态**：独立诊断与非 rollout 候选方案。**严禁恢复**历史 TF→rollout curriculum、scheduled sampling、DAgger、model-generated multi-step feedback 训练、donor。  
> **拟放置路径**：`docs/CLOSED_LOOP_NO_ROLLOUT_PLAN.md`。  
> **前提**：保留原 VQ(ep90)+对应 Controller 作为验证基线；VQ 源头新资产不与本轮 Controller 实验混用。  
> **边界**：允许**推理/验证时 rollout** 来评估闭环（这不是 rollout 训练），但训练期 `feedback_probability` 必须为 0，且不以模型自己先生成的多帧状态作为监督上下文。

## 1. 真实反馈通路与问题定义

在 `evaluation/simulate_realtime_flowmatch_control.py`，每步输入状态为 `x_curr=[traj84, generated_motion291]`、`prev_latent`、`history_latent`；`sample_latent` 预测 `z_hat`，`model.feedback_state` 执行 `clip → (quantize) → decode` 后返回**一致的 motion 和 latent**，并由生成结果更新下一帧历史。实时脚本还根据生成 root 速度/偏航积分世界轨迹。

状态误差可能经 `motion`、`q`、两帧 history、根轨迹与风格/规划条件共同传播。此前 GT 轨迹固定的 refresh-K 实验表明 code-to-GT match 随反馈长度快速下降，但匹配率低并不等于运动物理质量差（同一条件可有多种有效未来）。**不允许把周期性 GT 重锚定作为部署解法**：真实交互不存在未来 GT。也不能用 codec `decode→reencode` 同帧换码率充当闭环上限；当前 Controller 在运行中直接传递 feedback latent，不逐帧 re-encode。

## 2. 既有被验证方向：纯 TF 下的观测噪声，不是 rollout 训练

现役 `scripts/run_train_flowmatch_history_obsnoise_tf100.sh` 已明确：`FEEDBACK_PROBABILITY=0`，`OBS_NOISE_STD=0.5`，每次训练基于 GT history/target，而非模型多步生成历史。`training/train_flowmatch_vqvae_controller.py::_noised_gt_observation` 在 TF 分支对 GT latent 注入扰动后执行 `feedback_state(clip→Q→D)`，**motion token 与 latent token 始终相符**。

`analysis/control_operators_study_2026-10-12/OBSERVATION_NOISE_PLAN_A.md` 有实证：原 100ep 2L obsnoise 模型相较无噪声 ODE4 的 matched/cross/style、jitter、MPJPE 有一些改善，但 cross 仍明显不足；**不能说 0.5 是通用最优，也不能保证它只降抖动**。

禁止使用历史文档中已撤回的“更多 rollout / donor 才能解决”的建议。

## 3. 优先执行次序（先零训练，再有限单因素 TF）

### F0. 只读的闭环误差分解——允许 inference rollout，禁止训练

固定同一个 Controller/VQ、相同 val source/target/noise seed、240f@60Hz、solver 取 checkpoint；记录 GT 轨迹条件下与实际 planner 轨迹条件下两个**不同协议**。比较：

- 默认 `clip→Q→D`、`--no-requantize-z`（**仅旁路诊断，偏离正式协议**）；
- checkpoint Euler steps 与旁路 2/4/8（不修改官方主线步数）；
- noise std 选至少 `{0, 0.5, 1}`，但低噪声导致静止/滑行的风险必须与活动/接触同时记录；
- history delta 正常 vs 置零诊断（若现役模型含 `history_conditioning`；零值旁路不能冒充重训练）。

**指标**：正式 MPJPE、Root ADE、Jitter、local/world FS、style matched/cross、预测足部/关节活动性、冻结/滑行帧比例、接触覆盖、真实状态与规划的速度/heading 偏差、各帧误差增长曲线；**code match 仅诊断**。按 source_entry_id 配对分析，遇到合法多峰轨迹不将 code mismatch 自动归错。

**不可破坏的不变量**：任意 feedback 模式返回 `motion == D(latent)`；若改 latent 必须重新 decode，并同步更新 `prev_latent/history_latent/x_curr`；不得只滤 motion 而把原 latent 写回控制器，也不得让后处理世界 IK 姿态回写模型。

### F1. 无训练的推理旁路实验（不改默认）

1. 保持 VQ 和 Controller 权重固定，先测试现有 no-requantize 及 noise/solver 旁路；锁定 val 选型后才考虑默认开关变更。
2. 可选“latent 残差受限投影”仅限**实验开关**：输入 `z_hat`，对相对上一状态的 latent 增量施加有上界的修正，然后统一 `clip→Q→D`；不能对原始 `D(q)` motion 单独平滑。保留完整 raw 输出/修正量以监测冻结动作、迟滞、抹除转身、足滑；无独立质量收益则弃用。
3. 不要默认启用 per-slot hysteresis：GT 本身每帧大量换码，迟滞可能延迟步态。不要把重采样多个 seed 后选最平滑那条当“免费修复”；需要无监督选择规则和速度/风格质量门槛。
4. 显示 Smooth/IK 可以独立改善 UI，但不算 Controller 修复；不得用于正式模型指标或写回反馈。

### F2. 若仍需训练，只做**单步 TF 条件鲁棒化**（需另行授权）

在已有 obsnoise TF 配方上，先做严格同初始权重、同 optimizer updates 的 `OBS_NOISE_STD ∈ {0.25,0.5,0.75}` 单变量对照。不得只比较不同 ep 或 2L/6L；风格与运动指标都必须验收。若 0.5 无优势可直接停止。

若幅度扫描仍不足，**另立目标版本**测试“相关噪声 TF”：从 train GT 序列构造 `δ_t=αδ_{t-1}+σϵ_t`（而非模型自己 rollout 的状态），按相同 `clip→Q→D` 渲染每一步观测，并保持连续两帧 history/motion 与当前 latent 一致；监督目标始终是真实下一帧 GT，且训练不调用模型预测形成后续条件。与同预算独立高斯 TF 对照：单变量为噪声时间相关性 `α`，不能同时改幅度和 loss。

更远期的一步局部一致性损失：对同一 GT 下一帧目标，在**相同** FM 时间 `t`、transport `z0`、style/traj 下，比较干净状态与适度扰动态的 endpoint，约束局部过度敏感；仍是单步 TF 不做 rollout。此项可能压平真实动作条件，必须加活动/相位保护并作为**新独立实验**；不能把单步目标一致性等同于保证闭环全局收敛。

**严禁**：在训练 batch 中先预测 `z_t` 并将其作为下一步条件，不管 detach 与否；该做法仍是 model-generated feedback / rollout 训练，与用户弃用决定冲突。

## 4. F3. 若需要长期纠偏：先承认可观测性限制

真实交互没有每帧 GT 锚点，不能用 oracle refresh-K 做稳定器。当前状态只来自过去生成结果和意图轨迹；如果生成分布已经进入另一条合理轨迹，单靠码身份正则也不能知道应回到哪条 GT 路线。

可以单独研究**基于外部可观测信号**的 feedback correction（玩家速度命令、期望朝向/轨迹、实际环境脚接触、物理地面），但这些信息须实际存在且符合当前接口。新约束要以 raw Controller 输出的动力学、风格、动作幅度作联合验收；禁止读取未来 GT 或直接用规划轨迹硬覆盖模型 root 运动。

## 5. 代码边界 / 交付

- F0：新增 `analysis/closed_loop_no_rollout/**` 诊断脚本/JSON，现有模拟入口优先以 flags 跑；不改训练器。
- F1：必要时仅在 `evaluation/simulate_realtime_flowmatch_control.py` 和对应交互封装加**默认关闭**的旁路 flag（最好封装到单一 `feedback_state` API），以及 `tests/test_closed_loop_feedback_contract.py`。控制器训练目标不变。
- F2：经批准才在 `training/train_flowmatch_vqvae_controller.py`、其 wrapper 和 `tests/test_observation_noise.py` 加相关噪声/一致性；`FEEDBACK_PROBABILITY=0` 必须在 config 和运行记录中断言。应新增 objective version/resume guard，禁止从旧目标状态静默 resume。
- 禁止改 `arch/basic_vqvae_375to291.py` 或替换 VQ：与 VQ 主实验不同轴；数据与权重必须原样冻结。

**每个实验交付**：`contract.json`、source/ckpt/DB/norm hash、val case+noise manifest、metric JSON（per-source + 分活动）、quality/activity/CI、帧时间性能、`PASS/FAIL/INCONCLUSIVE`、原始日志和可复现指令。凡是只改善匹配率、低噪声后角色不动、足滑/风格恶化、状态不一致，均直接 FAIL。没有 GPU/真实 ckpt 时只能完成静态/fixture 检查，不能宣称闭环质量已验证。

## 6. Agent 固定执行顺序

```text
F0 只读冻结模型的官方闭环 + 旁路诊断（val）
 → 明确主要误差来源（量化/历史/采样/实际planner）
 → F1 无训练旁路、保留状态一致性、完整质量验收
 → 只有 F0/F1 不足且经单独授权，才运行 F2 的纯 TF 单因素短训
 → val 锁定后按照正式协议 test 一次，永不以 test 回调参
 → 保留旧模型和默认反馈方式，直到新方案所有质量门槛通过
```

### 参考

- `analysis/control_operators_study_2026-10-12/OBSERVATION_NOISE_PLAN_A.md`
- `docs/MOTION_HISTORY_REPAIR.md`（顶部修订声明优先，下面的 rollout/donor 方案是历史废弃内容）
- `analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md`（需阅读顶部撤回的归因）
- `analysis/codebook_divergence_2026-10-14/{REPORT.md,CAUSE_ANALYSIS.md,CODE_ANALYSIS.md}`（部分旧机制推断已撤回，先看文首审计限定）
- `arch/flowmatch_vqvae_controller.py::feedback_state`、`training/train_flowmatch_vqvae_controller.py::unrolled_flow_loss`
- `evaluation/simulate_realtime_flowmatch_control.py`、`docs/INTERACTIVE_LABEL_CONTROL.md`

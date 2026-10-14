# CodeStyle｜VQ 与 Controller 稳定性修复：Agent 执行规约

> **性质**：可交由编码/实验 Agent 分阶段执行的技术规约；**不是已验证的修复方案，也不代表所有训练已获授权**。
> **项目**：`Shinn127/CodeStyle`，建议入库路径 `docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md`。
> **参考 HEAD**：`12861afc4f15527eed3f8c89da501fbfeb2fa8dd`（2026-10-14 核查；实际执行前重新查询 HEAD）。
> **现役配套**：原始无镜像 VQ 150ep run 的 `best.pt`（checkpoint 内 ep90）及与其匹配的 Controller、norm、cache。不得假定任意新 VQ 权重可直接替换。
> **硬约束**：**禁止 rollout 训练**；允许闭环 **rollout 推理和评估**。禁止重新引入 donor、DAgger、scheduled sampling、多帧模型自生成反馈训练。

## 0. 给 Agent 的立即执行指令

在真实环境中按顺序做：

1. 阅读本规约和 §2 列出的仓库证据；核验最新 HEAD、已存在改动、checkpoint、norm、数据库和 GPU/依赖。发现既有修复不得重复改写。
2. 建立新工作分支与新的实验输出目录；完成 **G0（契约）→ C0（闭环诊断）→ C1（不训练的推理旁路）**。这些步骤不得改训练目标和现役默认值。
3. 单独准备 **C2（Controller 纯 TF 噪声 A/B）** 和 **V1（VQ Decoder-only 物理位置二阶损失 A/B）** 的代码、单测、dry-run、梯度标定；**训练开始之前**冻结完整实验协议、预算与评估清单。
4. **涉及 GPU 优化器更新的真实训练**，仅在用户明确批准具体阶段后执行。默认每个训练 pilot 至多 10 epoch/臂；V1 第一轮固定 5 epoch/臂。超预算、完整 100/150ep、cache 全链重建、Controller 正式重训和默认模型切换需要新授权。
5. 对每个阶段输出 `PASS / FAIL / INCONCLUSIVE / BLOCKED / INVALID_EXPERIMENT`，归档原始 JSON、命令、日志、校验和及结论。遇到否决条件即停止，不得自动升级到下阶段。

**权限解释**：本文件授权 Agent *理解、设计、审查和准备*实验；若上层委托明确说“执行 C2/V1 的限额 pilot”，该阶段才可运行训练。不得把本规划视为无限训练/部署授权。

## 1. 问题定义和已知结论：避免重复失败方向

### 1.1 VQ 问题

`z_e=E(x)`、`q=Q(z_e)`、`m_q=D(q)`、`m_c=D(z_e)`。相同验证协议下，原始 ep90 VQ 的连续 AE Jitter 约 **0.120 cm**，量化 AE 约 **0.412 cm**；连续 AE world 足滑约 **0.207 cm**，量化 AE 约 **0.488 cm**。这证明离散量化链路存在额外时间/接触误差，但 **不证明每次换码都错误**。

已结束的 S3/S4 全 VQ `ACC_WEIGHT=0` vs `4.9842` 10ep A/B（固定 val12）：

| 指标 | B0 | B1 | 结果 |
|---|---:|---:|---|
| `‖Δ²(p_q-p_c)‖` | 0.4170 cm | 0.3703 cm | −11.2% |
| 量化 Jitter | 0.4382 cm | 0.3952 cm | −9.8%（未达原 15% 门槛） |
| World MPJPE | 1.5060 cm | 1.6552 cm | **+9.9%** |
| Local MPJPE | 1.0832 cm | 1.3128 cm | **+21.2%** |
| 强运动帧比例 | 0.0708 | 0.0667 | −5.9% |
| 连续 AE Jitter | 0.1251 cm | 0.1277 cm | **+2.0%，并未同步变平滑** |

结论：**原 mixed normalized-motion291 的二阶损失方案 FAIL；不得继续放大该权重或自动延长到 150ep。** 最新 S4 报告已修正连续 AE 归因和 B1 best epoch（B0 ep1、B1 ep3），勿重新引入已撤回说法。

### 1.2 Controller 问题

每帧调用 `sample_latent`，随后 `feedback_state(clip→Q→D)`；返回的 `motion`、`feedback_latent`、双帧 history 进入下一帧。自反馈使观测分布偏离 GT teacher forcing 状态，但 **GT 码身份匹配下降不能单独证明生成运动失败**。codec `decode→encode` 的 ~37.8% 同帧保码率不是 Controller 的硬上限：Controller 正式路径直接传递 latent，通常不会再编码该 motion。

历史纯 TF + GT latent 观测噪声的 100ep 2L 结果曾出现 matched 84.9%、cross 20.9%、MPJPE 6.89cm、Jitter 0.474cm、world FS 0.610cm；**cross 不足并未由观测噪声解决**。这是历史条件下的经验参考，不能与不同层数、权重、数据/solver 的 run 拼作因果对照。

### 1.3 总体目标（优先级从高到低）

1. 闭环长时运动质量、速度/转向指令跟踪、合理活动性和目标风格。
2. 保住 local/world MPJPE、root ADE、world 足滑、接触及摆腿行为。
3. 在不牺牲上述目标的条件下，降低 **量化所附加** 的二阶物理运动误差。
4. `code match`、码本换码率、roundtrip、Voronoi margin 仅诊断；**禁止单独作为成功指标**。

## 2. 源文件与必读记录

先读取对应模块，不依赖旧文档中未更正的结论：

- `docs/NEXT_TASKS.md`，`analysis/vq_temporal_source/{COMMON.md,S4/REPORT.md}`，尤其 S4 预注册阈值及后续加固的强制 g0 契约。
- `analysis/vq_code_stability_2026-10-14/AUDIT.md`、`analysis/codebook_divergence_2026-10-14/{CAUSE_ANALYSIS.md,CODE_ANALYSIS.md}`；优先阅读顶部复审撤回说明。
- `analysis/control_operators_study_2026-10-12/OBSERVATION_NOISE_PLAN_A.md`、`docs/CLOSED_LOOP_NO_ROLLOUT_PLAN.md`。
- `arch/{basic_vqvae_375to291.py,flowmatch_vqvae_controller.py}`。
- `training/{train_basic_vqvae_375to291.py,train_flowmatch_vqvae_controller.py}`。
- `evaluation/simulate_realtime_flowmatch_control.py`、`Utils/{motion_eval.py,vqvae_eval.py,motion375.py}`。
- `scripts/run_train_flowmatch_history_obsnoise_tf100.sh`、`scripts/run_train_basic_vqvae_375to291.sh`。
- `docs/NOMIRROR_CHAIN.md`、`Utils/runtime.py::require_unmirrored_db`、相关测试。

**执行前先检查**：新 HEAD 是否有新报告、任务已完成或接口已重命名；如有，以实际源码为准，先更新 `contract.json`，必要时标 `BLOCKED` 而非猜测运行参数。

## 3. 全局实验与安全契约 G0（必须最先 PASS）

### 3.1 隔离原则

- 建立独立分支，保留原 `main`、原 VQ/Controller、旧结果及其缓存。不得在原 `checkpoints/` run 中覆盖文件。
- Controller 支线 **全程冻结原 ep90 VQ**；VQ 支线 **不修改 Controller**。未完成各自验收前不组合两条变化。
- 无镜像生成数据库 `database_100sty_375_memmap_nomirror`；train 用于训练，val 用于候选选择，test 仅在方案锁定后评估。OOD/test 不可用于训练或梯度标定。
- 使用 **原 VQ `norm_stats.npz`**，不重新估计。新缓存需要显式版本名及来源 SHA。
- 记录 commit SHA、tracked dirty 状态与 tracked diff SHA、每个被执行脚本的 SHA、全部源权重/数据/split/norm 指纹。**启动前**采集；训练结束时再采集以检测中途代码变化。
- FP32、TF32 off、确定性相关设置、GPU/驱动/PyTorch/`vector_quantize_pytorch` 版本、seed/loader seed、完整命令行/输出路径统一留痕。
- 如果 checkpoint / DB / norm / 精度或官方 metric 函数不能正确加载，立即 `BLOCKED`；禁止用合成数据产生实际训练/性能结论。

### 3.2 状态与语义不变量

1. 正式量化反馈：`q=Q(clip(z))`、`m=D(q)`；传递的 `prev_latent/history_latent/x_curr.motion` 必须同源。
2. 连续反馈旁路：`q=z_clipped`、`m=D(q)`（此 `q` 指传递的连续 latent，并非码字）；在 JSON 中写 `feedback_mode=continuous`，不可冒充正式协议。
3. 训练必须保持 `feedback_probability=0`；**不得**将 Controller 自己预测的 `z_hat/m_hat` 接回下一训练步，即使 `detach/no_grad`。
4. 训练窗口不跨 segment；`source_entry_id` 隔离。所有物理 cm 指标基于正确反归一化的 24 关节，不得 `raw291 × 100`。
5. 对照实验的**单一变量**、相同训练更新数、初始化、数据顺序、checkpoint 选型规则和指标协议，均在看到实验结果前冻结。
6. 验证/推理允许真实自回归 240f rollout；不得将“禁止 rollout 训练”误解为禁止测闭环。

### 3.3 统一评估口径

- VQ：冻结 val12（与 S1/S4 归档相同 12 source ×121f）+ 训练器固定 90 val windows + 90 val source 的冻结风格分类器代理；按 source 汇总，bootstrap CI，不只报告全帧总体均值。
- Controller：锁定固定 val case、输入指令、matched/cross 来源、风格组合、每次运行的采样种子；**240f@60fps**。若当前验证入口不支持某条件，在报告写 `NOT_MEASURED`，不可伪造。
- 两种协议必须分开：**GT 轨迹条件**（误差归因用 oracle）与 **真实交互 planner 条件**（部署真实性）；绝不把它们的绝对分数混比。
- 必须报告 local/world MPJPE、root ADE、local/world FS（同时报告 GT-contact mask、预测 contact coverage、有效 contact count）、官方 Jitter、p95、动作幅度、转身及步态行为、style matched/cross、时间开销。
- 足滑同时记录 **事件加权 pooled** 与 **逐 source 平均**。接触数 0 的 source 必须显式标记，不得让 `0/0→0` 假造“滑行改善”；逐 source 平均只在有效接触样本上计算并注明覆盖率。
- 同时报告 absolute 与 paired Δ、95% source-level bootstrap CI（推荐 >=1000 draws、固定随机种子），失败与负结果不省略。

**G0 交付**：`analysis/vq_controller_recovery/G0/{contract.json,AUDIT.md}`，包含已完成的旧修复、未修 bug、适用版本、测试可用性及 `PASS/BLOCKED`。无 G0 PASS，不启动后续训练。

## 4. 工作流 C：Controller 闭环优先（固定原 VQ）

### C0｜只读闭环因果分解（不训练；优先级最高）

**假设**：真正的质量衰退可能分别来自 (a) 单步 FM/solver 误差，(b) `Q` 的离散映射，(c) 双帧 history 或反馈分布，(d) root/planner 条件，不可仅由码匹配曲线推断。

在相同 val 源、风格目标、种子和 checkpoint 上至少比较：

| ID | 上一帧状态 | 轨迹/控制条件 | 目的 |
|---|---|---|---|
| C0-A | GT motion + 对应 GT VQ latent | GT 轨迹 | GT 单步条件能力，仅诊断 |
| C0-B | 从 GT latent 重建的一致 motion + 对应 latent | GT 轨迹 | 排除 GT raw/codec 观测流形差 |
| C0-C | 正式 `clip→Q→D` 自反馈 | GT 轨迹 | 单独量化闭环反馈误差 |
| C0-D | 正式 `clip→Q→D` 自反馈 | 实际 planner | 测真实交互行为 |
| C0-E | 连续 `clip→D` 自反馈 | **与 C0-C 相同** GT 轨迹 | 量化反馈贡献，仅旁路 |

注意：C0-A/B 属 teacher anchored；不可把它们与 C0-C 的 code match 差直接称成“闭环因果分解”。记录实际相同配置的 MPJPE/质量差，并控制轨迹/历史输入。

额外做**单次外加扰动恢复**：在固定时刻给上一帧 latent 注入事先确定的有界扰动，保持 motion/latent 一致，然后推理观察 8/16/32 帧后的速度方向误差、Jitter、FS、动作幅度和风格，不用未来 GT 纠正。与零扰动对照成对比较。

时序分桶：`[1,16]`、`[17,32]`、`[33,60]`、`[61,120]`、`[121,240]`；输出每源曲线和总体。风格识别须区分 matched/cross，不可把 code match 当 ground truth fidelity。

**允许改动**：仅新增 `analysis/vq_controller_recovery/C0/**` 及只读评估适配器/单测；必要时为 evaluation 加默认关闭的诊断 flag。**禁止改 VQ、Controller 模型参数与生产反馈默认值。**

**C0 DoD**：可复现五组输入状态；history/motion/latent 不变量 PASS；输出固定 seed、per-source 误差和真实运行配置。若诊断路径自身违反状态不变量，`BLOCKED`，先修评估器。

### C1｜不训练的推理旁路（C0 后执行，默认关闭）

保持所有权重冻结，单次只改变一项：

- `requantize` 开/关；**关闭仅诊断**，不得直接变更正式 Controller 反馈语义。
- Euler solver 2/4/8 步（如果实际 checkpoint 支持），以官方步数为基准；记录速度/延迟/质量。
- inference transport noise `std` 取 0/0.5/1（如当前代码支持）；必须同时报告静止/冻结比例与 style 准确率。
- history delta ablation（仅当 `history_conditioning=true`），明确为分布外诊断，不用于直接部署。

可选有界 latent 修正投影 **先写实验设计与单位测试**，不得擅自启用迟滞、per-slot 码保持、world IK 回写或输出 motion 平滑。凡改 latent 后必须重新解码，并正确更新整个 feedback state。

**停止条件**：任何候选出现风格准确率或速度响应下降、冻结比例上升、local/world FS 恶化，不能只凭 Jitter 改善推广。保留 C0 正式模式为默认。

**C1 DoD**：同条件成对结果、部署开销、识别最佳诊断方向或结论 `NO_USEFUL_INFERENCE_FIX`。如果 C0/C1 足以表明根因在 planner 或状态实现，应针对该 bug 单独提出修复审批，而不是跳到训练。

### C2｜纯 TF 观测鲁棒性 A/B（**需要训练授权**）

**只在 C0/C1 确认反馈扰动确实造成质量缺口时启动。** 固定原 VQ、同一个 Controller 架构/初始状态；训练阶段 `feedback_probability=0`，且下一帧监督永远来自 GT。先复核已有 `OBS_NOISE_STD=0.5` 机制及 Objective version、resume guard、history/motion 一致性。

**实验 C2a：只扫描观测噪声幅度。** 候选 `{0.25, 0.5, 0.75}`，同一初始化、相同 LR、batch、seed、data order、updates、solver 与 checkpoint 选型规则；默认最多 **3 臂 × 10ep**（若预算不足，须在开始前等比例减小每臂上限并预注册）。旧 100ep/不同层数 run **不是**严格 A/B 的控制臂。

扰动观测应严格符合：

```text
z_gt = frozen_VQ.encode(GT_frame)
z_obs = z_gt + calibrated_perturbation
q_obs = Q(clip(z_obs))
m_obs = D(q_obs)
condition = [GT_traj, m_obs, q_obs, consistent_previous_history]
target = GT_next_frame / GT_next_latent
# 不允许 model.generate -> next train condition
```

注意仓库实际 `observation_noise_std` 的单位是 `transport_std` 倍数、并含均匀抽样系数，**不能把候选数值当作绝对 latent std**。

**实验 C2b（仅 C2a 不足且单独授权）**：改时间相关性而不改边际噪声幅度。独立 AR(1) 创建设定相关历史：

$$\delta_t=\rho\delta_{t-1}+\sqrt{1-\rho^2}\,\sigma\epsilon_t$$

同一 GT 窗口生成 correlated / independent 两个条件。须固定相同有效单位/每维尺度、初始化分布、style/traj，并检验扰动后两帧 history 与 motion 一致；不将模型预测写入下一训练条件。`ρ` 和幅度不能在一次实验同时改变。新目标语义需升级 objective version、严格 resume 校验。

**C2 验收（建议在运行前写入冻结 `protocol.json`）**：与同预算对照相比，至少一个预注册的主要闭环物理误差或扰动恢复指标出现可证实改善；world/local MPJPE/root ADE 不得恶化超过 3%，FS 不得恶化超过 0.5%，活动性不得下降超过 5%，matched 与 cross style accuracy 分别不得下降超过 2pp；不允许通过静止动作改善 Jitter。所有百分比为工程试验闸门，不是理论保证。没有有意义收益则 `FAIL`；CI 无法区分则 `INCONCLUSIVE`。

**C2 交付**：`C2/PROTOCOL.md`、各臂 `config/manifest/history/ckpt_hash`、配对 val 结果与 CI、失败病例及 `DECISION.md`。**不自动运行 100ep 或替换默认 controller。**

### C3｜单帧 FlowMatch 终点监督（可选、后置，另需授权）

只在 C2 失败/不足且 C0 显示 **同一帧 Euler 终点** 的误差显著时实验。先核对仓库已有 `recon_inference_path` 与历史不稳定记录，不重复引入已经失败的辅助项组合。

冻结 VQ，训练 Controller，对 GT 条件下的**单帧**推理终点监督：

$$\hat z_{t+1}=\operatorname{Euler}_K(F_\theta;\ s_t^{GT},u_t,\xi),\qquad L_{end}=\operatorname{Huber}(\hat z_{t+1},z_{t+1}^{GT}).$$

- 这是 **单帧积分 K 次**，不是跨时间的 autoregressive rollout 训练；后续训练条件仍取 GT 或预注册的 GT 噪声观测。
- 先连续 latent endpoint，**不同时**加 code CE、STE、margin、量化输出重建损失或改 solver；这些是不同实验。
- 对该 loss 做有限差分/梯度流、边界与数值稳定测试；用训练小批测与原 FM 梯度的比例/余弦，单变量短训 `λ_end=0` vs 一个事先标定权重。
- 如果出现 mode collapse、风格下降、latency/显存超限或闭环指标劣化，立即停止。

### C4｜Cross-style 专项（独立问题；不得以噪声代替监督）

在 C0 中先按源风格/目标风格的 90×90 类型、transition 时长、history 风格泄漏程度分层，记录匹配与跨风格的性能差。不得把旧 donor 合成监督恢复到主线。

后续可提出使用**真实有监督风格过渡样本**或被验证有效的风格/动作内容解耦，但它们都是独立数据/建模项目，须证明没有 future GT、没有 train/test 源泄漏。没有可信 cross-style 真值时，使用冻结 style classifier 仅作代理，不能据其独自证明物理过渡正确。

**C4 本轮只做诊断与数据可用性审计；新 cross-style 模型训练需另行批准。**

## 5. 工作流 V：VQ 量化附加误差（与 C 完全隔离）

### V0｜复用基线并修复必要的评估/工件问题（无训练）

在原 ep90 VQ 上运行/核验固定 val12 + 90 val 结果，包含：

- `p_q=P(D(Q(E(x))))`、`p_c=P(D(E(x)))`、`p_gt=P(x)`，位置统一 cm；`r=p_q-p_c`、`e=p_q-p_gt`。
- `‖Δ²r‖` mean/p95，`‖Δ²e‖`，quant/continuous/GT Jitter，local/world MPJPE、root ADE、world/local FS、预测接触覆盖、移动帧和腿部活动。
- `Q` 码身份/usage 仅诊断；每次 eval 不得更新 EMA state。
- 对照 `analysis/vq_temporal_source/S1/S1_baseline_ep90.json`；不符先修单位和归一化口径。

已在新提交 `12861af` 修复的 S4 g0/best epoch/连续 AE 报告**不得重复当成未解决 bug**。其他尚未核实的问题（例如附属 wrapper 收尾或 metadata tag）先做定向回归测试再决定是否修复。

### V1｜Decoder-only 的**物理位置**二阶损失（需要训练授权；优先 VQ 试验）

**假设**：mixed normalized-motion291 二阶损失会与位置重建竞争；只监督真实局部关节位置的 `Δ²`，同时保留强重建约束，可能减少 `Q` 引入的 Jitter 且不显著牺牲姿态精度。该假设**尚未验证**。

**固定：** 原 ep90 checkpoint、原 norm、train split、`E` 和 `Q/EMA` 全冻结；只更新 Decoder，不改 K2048/32槽/4D/rotation trick、编码路径、Controller。正/负两臂从**相同 decoder state 冷启**，`seq_len` 预先设定（建议 8）、同 seed/updates。`margin/roundtrip/teacher=0`。不得引用先前“全 VQ B0”充当本轮 frozen E/Q 对照。

损失（不引入新的 world FK）：

$$P_t^q=100\,\operatorname{reshape}_{24\times3}\bigl(\hat m_{t,3:75}\odot\sigma_{3:75}+\mu_{3:75}\bigr),\quad P_t^{GT}=100\,\operatorname{reshape}_{24\times3}\bigl(m^{GT}_{t,3:75}\odot\sigma_{3:75}+\mu_{3:75}\bigr)$$

$$L_{posacc}=\operatorname{SmoothL1}_{\beta\text{ (cm)}}\left(\Delta^2P^q,\Delta^2P^{GT}\right),\quad \Delta^2P_t=P_{t+1}-2P_t+P_{t-1}.$$

这里的 `μ,σ` 是 **motion291 对应位置通道**；如完整归一化向量为 375D，则需取索引 `84+3:84+75`，不得忘记 84D trajectory offset。`SmoothL1 β` 以 **cm** 标定并记录，不得照搬归一化 motion loss 的默认 β。计算 Torch 可微图，**禁止转 NumPy 后把无梯度指标当 loss**。

**实施步骤**：

1. 在 `arch/basic_vqvae_375to291.py` 添加默认 `weight=0` 的物理位置二阶损失；在 trainer/wrapper 显式透传并把配置纳入 resume/objective guard；不改变现有 `acc_loss` 语义和默认值。
2. 单测：常速度运动损失接近 0、匹配 GT 加速度损失 0、已知振荡和时序错位损失为正、decoder 参数梯度非零、E/Q/EMA 状态完全不变、seq<3 安全处理、cm 与归一化 index 正确、FP32/EMA 不变量。冻结时如走 `model.train()` 必须验证 quantizer `freeze_codebook`/eval 实际状态而非只用 `requires_grad=False`。
3. **训练前**从 4 个独立 train B16×T8 估计 `∇_D L_base` 与 `∇_D L_posacc` 的范数/余弦、单项与总项的梯度竞争；标定 `λ`，不得沿用失败的 `ACC_WEIGHT=4.9842`。
4. 注册 **V1-B0（weight 0）/ V1-B1（标定 λ）**，两臂各 **5 epoch**，使用相同初始权重、norm、loader order、lr、训练 updates；保存 `epoch_0005.pt` 固定终点（选型规则冻结）。所有输出单独命名。
5. 用同 val12/90 val/活动性与接触代理验收，先比较 B1−B0，再单独与现役 ep90 VQ 作部署价值对照。禁止不同协议指标混用。

**V1 预注册默认验收门槛（可在训练前修订并冻结，不得事后改变）**：

| 门 | 要求 |
|---|---|
| V1-0 契约 | 源/数据/norm/hash 相同，E/Q/EMA 逐张量不变，两臂只差 `posacc_weight` 与 OUT，updates 相同 |
| V1-1 直接作用 | `‖Δ²r‖` mean + per-source p95 均下降且配对 95% CI 上界 <0 |
| V1-2 时序主目标 | 官方 quantized Jitter 相对 B0 下降至少 **15%**（工程筛选门） |
| V1-3 重建质量 | local/world MPJPE 与 root ADE 每项退化 ≤ **3%** |
| V1-4 足滑 | 事件加权 world/local FS 退化 ≤ **0.5%**；同时报告逐 source 均值与 CI、contact count，不允许只挑其中一个 |
| V1-5 防止冻结 | 移动/强移动帧比例、足部行程等相对下降 ≤ **5%**，contact coverage 不发生严重异常 |
| V1-6 风格 | 冻结分类器的 90 val source 风格准确率下降 ≤ **2pp** |
| V1-7 证据 | 连续 AE 对照、codec 换码率单独报告；不据其单独判 PASS |

只降低 `Δ²r`、只降低 code flip、或者用 pose 精度/活动性换更低 Jitter：**FAIL**。5ep 未明显区分：`INCONCLUSIVE`，需批准才延长，绝不自动继续。

**V1 DoD**：新增代码/单测/paired 5ep 结果（如获训练授权）、固定端点 checkpoint、状态指纹、`DECISION.md`。V1 PASS **只是 representation-val 候选**，不能直接用于现役 controller。

### V2｜全 VQ 物理损失联合训练（仅 V1 明确 PASS、且新授权）

- E/Q/EMA 解冻与 V1 是 **另一实验轴**。需一对新的相同 ep90 冷启全 VQ 实验，不可复用 V1 的 frozen-Q 对照，也不可把多个超参同时更改。
- 使用 V1 已验证的物理 loss，重新测 encoder/decoder 分组梯度范数及余弦；S3/S4 曾显示 E/D 对 mixed acc 梯度响应不均衡，不能简单套权重。
- 全部 V1 物理、活动、风格门槛仍适用；换码率/RT 保持率不是验收目标。
- PASS 后仍不得覆盖旧 VQ/latent cache；须另行评估下游兼容性。

### V3｜因果短历史 Decoder（仅独立审批，不与 V1/V2 混跑）

仅在证据显示单帧 `D(q_t)` 难以利用时间信息且 V1/V2 无效时比较：

$$D(q_t)\quad\text{vs}\quad D_{hist}(q_{t-2},q_{t-1},q_t).$$

同参数/预算、无未来帧、无 rollout 训练。必须评估 **GT 编码历史** 与 **Controller 预测历史** 的分布差；现实部署需要维护相同历史缓存及版本。它是**独立结构方案**，不是本轮默认开发范围。

**低优先级/本轮禁止**：重建 K/slots/dim、margin-only、roundtrip-only、contact-world 复杂物理约束、全 150ep 从零训练。只有完成独立证据后才可专门立项。

## 6. 最终集成 I：只有两条支线分别达标才进入

严格按 **单向版本化依赖**：

```text
VQ baseline ep90 ──────────────┬── Controller C0/C1/C2（VQ 冻结）
                              └── VQ V0/V1（Controller 不动）
VQ V1/V2 候选 PASS
  → 独立新 VQ 版本
  → 按变更范围重建失效缓存/统计
  → 重新构建或适配配套 Controller（仅纯 TF；禁 rollout 训练）
  → 固定 val 240f 闭环 matched/cross 与物理质量联合验收
  → 锁定版本后一次正式 test 验证 → 可回滚交付
```

缓存 invalidation：

- 若 **仅 D 更新**：原 E/Q 产出的 indices/quantized latent 可能仍可使用，**但 `D(q)` motion 观测缓存一定失效**；latent 指数统计是否可用需按依赖链审计，不可自动假定全部缓存可复用。
- 若 **E 或 Q/EMA 更新**：重建 VQ target cache、train-only latent stats、style prior、所有从 E/Q 派生的索引/表，及 D(q) motion 缓存。标明哈希、训练来源和模型身份。
- 改变 Encoder/Decoder 可能改变 Controller 条件分布；**旧 Controller 与新 VQ 直接串联只能做兼容性诊断，不能作为正式候选交付**。
- 新配套 Controller 只采用既有纯 TF/观测扰动方案，不恢复任何 model-generated feedback training。

**集成必过门槛**：同协议下 matched/cross style、local/world MPJPE、root ADE、官方 Jitter、world FS、预测接触/活动性、速度与转向响应、延迟/吞吐量。锁定验证集和选型规则，按 source/style 配对报告变化与 CI。只有 **同版本新 VQ + 新配套 Controller** 联合通过，才能提出默认切换；任何关键退化则保持旧默认并记录失败原因。

## 7. 文件修改白名单与禁止行为

| 阶段 | 允许新增或修改 | 必须避免 |
|---|---|---|
| G0 | `analysis/vq_controller_recovery/G0/**`；必要时独立只读检查脚本 | 旧资产、数据库、旧报告历史 JSON |
| C0/C1 | `analysis/vq_controller_recovery/C0/**`、`C1/**`；必要时 `evaluation/simulate_realtime_flowmatch_control.py` 的默认关闭 flag、`tests/test_closed_loop_feedback_*.py` | `arch/basic_vqvae_375to291.py`、Controller 训练/权重默认语义 |
| C2 | 经批准后 `training/train_flowmatch_vqvae_controller.py`、专用新 wrapper、`tests/test_observation_noise.py` 和新 `C2/**`，新增 objective version | rollout/DAgger/donor、跨步自生成条件、现役默认配置 |
| C3 | 经批准后 Controller loss/solver 的**单帧 endpoint** 分支及独立测试 | 与 C2 同时启用未经隔离的其他损失 |
| V0 | `analysis/vq_controller_recovery/V0/**`、只读探针和测试 | 改训练器、改旧 S4 原始结果 |
| V1 | 经批准后 `arch/basic_vqvae_375to291.py`、`training/train_basic_vqvae_375to291.py`、新 V1 wrapper、相应测试和 `V1/**` | E/Q/EMA 更新、覆盖既有 VQ best/旧 cache、改 Controller |
| V2/V3 | **另行授权**、新分支及独立配置 | 和 V1 做多因素比较、未审查就合并主线 |
| I | **另行授权**、版本化缓存工具/Controller 适配脚本/部署入口 | 静默重建覆盖、直接把新 VQ 注入现役 Controller、自动上线 |

禁止默认修改：`dataset.py`/DB 生成和 split 定义、风格分类器、无人批准的源权重和缓存、正式交互反馈开关、GT/test 评估协议；确有根因 bug 时另出变更方案、单测和影响分析申请授权。

任何跨白名单改动在提交前都必须先 `BLOCKED` 并向用户解释具体必要性，不得顺手“重构全仓”。

## 8. 自动验收与交付格式

### 8.1 每次改动必做

```text
[ ] HEAD / tracked-diff / 训练源文件 hash 已记录
[ ] pytest 定向测试通过（含负例/边界/不变量）
[ ] bash scripts/run_tests.sh cpu 通过（如环境满足）
[ ] CUDA 可用时 bash scripts/run_tests.sh gpu 通过；否则注明 NOT_RUN
[ ] 若有 wrapper，DRY_RUN 展开的完整 argv 能被训练器 argparse 接受
[ ] 单变量 config diff 只包含白名单字段；manifest 指纹验收为必填且 fail-closed
[ ] checkpoint 使用预注册的固定 epoch；不事后挑 best.pt 获益
[ ] train/val/test 来源隔离，旧 test 不用于模型选择
[ ] 质量/活动性/风格/FS 有逐 source JSON、CI、零接触处理
[ ] 没有 overwrite 原资产、没有默认配置回归、没有 rollout 训练
[ ] 提交记录附 exact commands + SHA；未运行的测试明确标 NOT_RUN
```

### 8.2 每一阶段的固定文件结构

```text
analysis/vq_controller_recovery/
  G0/{contract.json,AUDIT.md}
  C0/{PROTOCOL.md,manifest.json,per_source.json,REPORT.md}
  C1/{PROTOCOL.md,manifest.json,per_source.json,REPORT.md}
  C2/{PROTOCOL.md,arms/<id>/{config.json,history.jsonl,manifest.json},acceptance.json,DECISION.md}
  C3/{PROTOCOL.md,...}           # 仅获授权后生成
  V0/{baseline.json,REPORT.md}
  V1/{PROTOCOL.md,gradient_calibration.json,arms/<id>/...,acceptance.json,DECISION.md}
  V2/...                          # 仅获授权后生成
  integration/...                 # 仅获授权后生成
```

每份 `REPORT/DECISION` 必填字段：

```yaml
stage: G0|C0|C1|C2|C3|C4|V0|V1|V2|V3|I
hypothesis: ...
source_commit: ...
tracked_dirty: false
train_source_hash: ...
model_ckpt_hashes: ...
normalization_hash: ...
db_split_fingerprints: ...
model_version_compatibility: ...
protocol_and_seed: ...
architecture_and_optimizer: ...
control_and_candidate: ...
actual_update_counts: ...
unit_tests_and_regressions: ...
per_source_metrics_and_CI: ...
metric_definitions_and_units: ...
contact_count_and_activity: ...
unexpected_behaviors: ...
verdict: PASS|FAIL|INCONCLUSIVE|BLOCKED|INVALID_EXPERIMENT
reason: ...
next_stage_approval_needed: ...
```

### 8.3 判决优先级（Fail closed）

1. 契约、来源、指纹、数据污染不合法 → `INVALID_EXPERIMENT`，不看分数。
2. 权重/DB/GT/设备缺失，无法运行关键检查 → `BLOCKED`，不能伪称已运行。
3. 任一硬质量/风格/活动性否决门触发 → `FAIL`。
4. 未显著改善主目标、CI 不足或不同预注册 checkpoint 结论冲突 → `INCONCLUSIVE`。
5. 全部门槛满足且可复现 → `PASS`；此为**该阶段** PASS，不代表自动批准推广或下一阶段训练。

## 9. 阶段依赖与终止条件（最终决策树）

```text
START
  └─ G0 Contract
       ├─ INVALID/BLOCKED → STOP，提交缺失证据/修复建议
       └─ PASS
            ├─ C0（闭环误差与扰动恢复）→ C1（无训练旁路）
            │    ├─ 已能解释并修复状态 bug → 单独请求实现审批
            │    └─ 仍有分布偏移证据 → 请求 C2 纯 TF 限额训练授权
            │                            └─ C3 仅在终点误差诊断支持时独立申请
            └─ V0（ep90 诊断）→ 请求 V1 Decoder-only 物理 acc 5ep/臂授权
                                     ├─ FAIL → STOP VQ 训练；独立提出 V3 研究计划
                                     └─ PASS → 申请 V2 或配套 Controller 集成

任何阶段变成 FAIL/INCONCLUSIVE：不得自动扩增 epoch、换架构或串联两项新 loss。
任何跨分支集成：先批准新 VQ 身份与缓存依赖，再配套纯 TF Controller，再统一 val/test 验收。
```

**本计划第一轮预期交付范围**：G0、C0、C1、V0 的真实诊断报告，以及经用户批准后 C2/V1 中选择的**一个**限额 A/B。避免并行启动多个 GPU 训练造成结果难以归因。

## 10. 参考证据链接（对应本版本）

- [HEAD `12861af`](https://github.com/Shinn127/CodeStyle/commit/12861afc4f15527eed3f8c89da501fbfeb2fa8dd)：S4 强制契约与历史报告修订。
- [S4 全 VQ 负结果](https://github.com/Shinn127/CodeStyle/blob/main/analysis/vq_temporal_source/S4/REPORT.md)：精度/活动代价及真实验收结论。
- [S3/S4 预注册协议](https://github.com/Shinn127/CodeStyle/blob/main/analysis/vq_temporal_source/COMMON.md)：单变量、checkpoint、指标口径与后续加固。
- [纯 TF 观测噪声历史证据](https://github.com/Shinn127/CodeStyle/blob/main/analysis/control_operators_study_2026-10-12/OBSERVATION_NOISE_PLAN_A.md)。
- [不使用 rollout 训练的现有规约](https://github.com/Shinn127/CodeStyle/blob/main/docs/CLOSED_LOOP_NO_ROLLOUT_PLAN.md)。
- [无镜像数据/缓存重建依赖](https://github.com/Shinn127/CodeStyle/blob/main/docs/NOMIRROR_CHAIN.md)。

> **最终执行原则**：先诊断真实闭环，再用严格单变量寻找可复现收益；先保住姿态、运动活动和风格，再追求降低 Jitter；VQ 与 Controller 分开验收，最后才能合并。**任何模型训练、完整重训和默认替换都不得越过本规约的审批边界。**

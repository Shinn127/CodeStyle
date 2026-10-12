> **历史文档（donor 路线已于 2026-10-12 移除）**：本页记录的 donor/cross-style 合成监督在当前代码中
> 已不存在（objective version 11 起）。保留原文用于回溯当时的设计与结果；现行归因见
> [TF10_DIAGNOSIS.md](../../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

# 不新增loss的代码核验与修订方案

本轮针对实际交互使用的label controller及最新6-token curriculum100 run。保留冻结VQ、normalization、style prior/selector、6-token transformer、target prediction与原有FM/CE/entropy目标。2026-10-12已完成动态条件、目标配对、反馈、选型和效率修订；仅运行轻量回归及有界GPU smoke，完整训练仍由用户启动。实现与实测见 [修订说明](../../docs/MOTION_HISTORY_REPAIR.md)。以下问题/数值来自修订前审计，保留为历史证据。

## 已确认的问题与证据

1. **motion与latent不是两份独立动态观测。** `LabelControlSession.step()`的motion来自D(q)，下一步同时输入q和D(q)；它们数值对应正确，但motion中的joint velocity来自单帧decoder，不是实际生成位置的帧间速度。两者一致并不意味着运动学一致。

   同300f按键序列，将motion291的root-frame joint velocity变换到world，与实际生成位置的中心差分比较：std1的脚部均值误差1.545m/s，std0.25为0.525m/s；Neutral GT参考片段为0.033m/s。参考与生成并非同一轨迹，不比较步速优劣，只检查各自内部的速度一致性。单帧码本状态无法强制保证相邻生成帧动力学一致。

2. **每帧重新采样Gaussian，噪声在选择运动分支。** 同历史/轨迹/style的32个noise在2/4/8/16步下，连续输出pose离散程度为0.416/3.396/5.915/6.107cm。8和16接近，增加步数不能根治当前分支离散；2步反馈却很少呈现这些变化。

   target模式v=(f(z,t,c)-z)/(1-t)，Euler令t=n/N，有

   `z_(n+1) = ((N-n-1)/(N-n))*z_n + f(z_n,n/N,c)/(N-n)`。

   最后一步z_N=f(z_(N-1),(N-1)/N,c)。2步与8步最后分别查询t=.5与.875，并经历不同中间状态。这是实现正确的数学结果，不是除零/时间索引错误。

   强相关noise rho=.999、std仍1的单case，jitter降到0.206cm，但滑行升至78.48%，最长58帧。它能稳定姿态，却未建立步态推进。不能部署为根治措施或改正式noise manifest。

3. **cross目标缺少相位方向约束。** `matched_root_targets()`筛suffix、root speed/yaw，随后按脚的位置与root距离挑donor，没有脚部相对运动方向或接触状态条件。真实train-only、窗口均匀抽样的1024×5 batch（seed42）中：236个cross；227个脚部有明显相对运动，其中44.05% source/donor的左右脚相对运动方向相反。合成首目标相对source上一帧的全身pose差均值13.227cm；对应连续GT为0.510cm。

   13cm包含合理style变换，方向代理也不是真实phase标签，不能把所有这些样本称为错误。但当前目标不区分风格改变与步态相位重置，确实要求控制器在一个1/60秒内处理很大的跨风格姿态跳跃。在保持原抽中目标style与root门限时，仅60.59%的cross样本在该batch还能找到非反向候选，说明只加batch内过滤会降低实际cross覆盖。

4. **训练条件仍混用GT motion与D(q)。** `train_feedback_context=false`。即使开启，当前只替换window[:,0,84:]；纯TF及ramp的GT分支随后重新读取原始window[:,step+1]，其余teacher observations仍是GT motion。正式离线rollout首帧也用GT，交互首帧用D(q)。不是GT标签时间偏移：数据库明确x_i=[traj_i,motion_(i-1)]，预测目标x_(i+1)包含motion_i，当前相邻索引匹配。

5. **低比例rollout当前并不省积分。** p在(0,1)时，整个batch仍执行每个中间步骤的Euler，随后用where丢弃GT分支的结果。`compute_loss(return_prediction=True)`还decode连续latent，但`unrolled_flow_loss`反馈仅用pred_latent，再通过feedback_state量化decode一次。CPU真实入口hook验证u4/feedback p=.25产生6次decoder调用；中间反馈最多需要3次。

6. **验证和交互条件不同。** rollout训练/离线验证沿用GT轨迹条件，没有根据生成root重建过去轨迹；交互过去为生成root，未来为键盘预测。planner的中心速度平滑，但未来每一帧的goal_velocity直接使用当前command，按键方向改变会立刻重排长达50帧的未来目标。这个条件差异代码明确，尚未测定它的独立贡献。IK接触速度门限又会拒绝已在滑动的脚，不能替代生成稳定性。

## 推荐实施路线

### A. 补真实历史变化，保留6个token与现有loss

增加最近两帧量化latent的变化量：

`delta_h = normalize(q_(i-1)) - normalize(q_(i-2))`。

在现有previous-latent condition token里加一个小线性投影：

`previous_token = latent_proj(normalize(q_(i-1))) + history_delta_proj(delta_h)`。

`history_delta_proj: latent_dim→256`，权重和bias均零初始化。不要增加history transformer、递归隐藏状态或新loss；原6token、256hidden、2layers、FFN1024不变。这个投影提供实际最近输出的变化方向，现有velocity通道继续保留，feedback motion仍严格=D(q)。它比给静态decoder velocity加正则更直接补充缺失的观测，但有效性属于需训练验证的假设。

- TF使用连续两个source GT quantized states；rollout第二步起用实际生成的两个反馈q，delta默认detach。
- 第一训练步从同segment多读一个前帧/cache row；不跨segment/split。segment首帧与交互reset用零delta。style切换保留history。
- 条件投影每个模型步算一次，在同次Euler的8个查询间复用。无需额外decoder。
- 将新的history参数显式穿过compute_loss/sample_latent/field.prepare_conditions；不把它存成model的跨batch可变隐藏状态。
- 从现有best初始化时只新增零投影，其他权重完整复制；初始条件和固定noise输出应保持一致。`--init-ckpt`需明确允许这些新参数，resume必须校验history配置/实现版本，不能绕过strict校验。

暂不改变Gaussian base mean、不切换latent-delta目标。先保持绝对target FM的含义，避免同时改变观测与transport分布。

### B. 将matched_root升级为相位更一致的目标配对

复用已有StyleDonorBank的phase_descriptor（左右脚位置方向、相对velocity、relative contact），但当前bank只支持T=2，不能直接给unroll4启用；需扩展到整段目标窗口。选择依据读取train-only的12–16帧GT上下文，而目标仍是当前4步。

- root/yaw/suffix门限不变，合成目标仍保留source root、donor pose/rotation及现有velocity修正。
- 在root合格候选上增加相位/脚部运动方向cost，关注source**当前观测**与donor对应观测，避免只比较future目标位置。
- 有充分运动/接触置信度时拒绝明显相反的支撑/摆动方向；低速、非周期动作不机械套用cosine硬阈值。阈值用train数据诊断，不根据test调。
- 目标style仍从符合质量门限的styles中均匀选，再选择该style中最佳donor。同窗口donor/时间offset固定；禁止逐步重新挑选或串联置换。
- 无合格donor退回matched，记录真实cross比例、phase反向率、per-style覆盖、首目标腿部jump分布，不能强行为满足请求概率提供低质量目标。
- bank密度与上下文可换显存/内存提高命中率；先测量32个windows/clip的合格率。不要放松root门限掩盖phase不足。

这只改现有监督样本的构造，不加优化项。它也不保证全部content/style disentanglement。

### C. 对齐真实8步反馈，并控制成本

训练反馈、验证、交互均用Euler8+clip→Q→D；FP32/highest/noTF32，VQ冻结。

- 反馈专用接口只返回Euler latent，跳过无用continuous decode；aux全部0时每步只decode最终q。保留历史return_prediction行为用于其他调用。
- 初始/所有TF teacher observation统一D(source q)，batch内批量decode或做FP32只读缓存。GT/synthetic target motion、target latent必须先独立保存，不能因改observation而被view/in-place一起覆盖。
- 为省成本，使用**整批TF/整批rollout交替**，而不是当前计算全部再where的per-window低概率门控。新run可以明确按batch执行3TF:1rollout；TF批不积分，rollout批完整8步。所有批仍计算现有4步FM平均与每窗口一次prior/entropy。
- u4时，25%批×3中间反馈×8步=平均6个额外field forward/批，与100%批×3×2步在次数上相同。实际时间受decode、batch调度、kernel影响，需100-update smoke验证，不能预先保证等耗时。
- 交替schedule、loader/RNG、history q、有效batch统计均随checkpoint保存，避免续训改变批序列。按真实optimizer update裁剪梯度一次。

### D. planner与IK的辅助修订

- planner持有平滑的world目标velocity与yaw-rate；中心与future从同一状态预测，按键不直接重排整个future command。限制加速度/转向加速度应从train root分位数确定，并保留停止/起步响应。官方离线协议继续固定GT case；新增键盘场景验证记录区别。
- IK用多帧height/升降趋势与filtered speed组成接触置信度、迟滞及release原因；把enabled、candidate、active、fully blended区别显示。不直接将速度门限调高并强锁双脚。
- 这些是显示/控制辅助，不能算作原始generator质量达标。

### E. 将运动质量加入选型

继续固定270matched/540cross的240f rollout，不用test选型。保留原accuracy/root数据；同时记录原始生成local jitter、world足滑/contact count与覆盖、跳变p99和max，以及固定键盘序列的滑行比例。

候选先通过style/root与运动质量约束，再比较联合结果；所有候选未通过时明确是诊断模型。质量阈值先以相同case的GT/VQ与旧稳定模型作参照，不在存在零contact时用0通过。现有style-priority best保留，新增quality selection记录避免改写历史含义。

## 有界短训练建议（代码修订后由用户启动）

初始化当前epoch65 best兼容权重，新history projection为0；重新建optimizer/scheduler，不resume旧目标状态。第一轮只做10epoch诊断微调，不立刻重跑100epoch。

| 项目 | 设置 |
|---|---|
| 目标 | 原FM target/normalized；prior .02，entropy .003；recon/root/style-consistency/所有新loss=0 |
| VQ/数据 | 原冻结nomirror VQ150及统计/split/cache；stride1 |
| 优化 | FP32，有效batch1024，AdamW，lr5e-5→1e-5 cosine10，gradclip1，seed42 |
| 历史 | 两帧latent变化投影；所有GT observation=D(q) |
| 目标配对 | phase-aware matched_root，请求.3，实际比例与覆盖如实记录 |
| ep1–2 | 纯TF，先学新增动态条件和清理后的目标 |
| ep3–4 | 反馈概率按批升至.08333、.16667；完整8步rollout |
| ep5–10 | 3TF批:1完整8步rollout批 |
| 噪声 | 保持独立Gaussian std1，不用降幅/强相关噪声遮掩步态问题 |
| 验证 | ep5/10固定240f验证；短训后固定按键短/长场景，原始与显示后处理分开 |

先对Neutral/FW/急停/转向/侧移及少数强风格做轻量probe，再扩大到原validation。既看std1主设置，也检查std0.25是否存在冻结吸引子。不能只按低jitter判断成功；必须同时检查相位推进、抬脚与root速度。

短训未改善时停止自动延长，按matched-only/phase-aware cross、GT计划/键盘计划、真实delta/置零delta分解。不要同时引入新loss、conditional Gaussian base、额外深层history encoder使归因不可判断。

## 实施前后的关键测试

- 新投影0初始化时固定输入/noise与旧field输出一致，开启后能得到非零梯度；6token不变。
- history只来自过去、无跨clip读取；GT/rollout/reset/style-switch的delta时序正确，feedback motion=D(q)不变。
- TF统一decode后GT/synthetic监督与target latents保持逐值不变。
- donor整窗口一致、phase方向与contact规则可解释，无候选正确fallback；不使用val/test bank。
- latent-only feedback比旧return_prediction路径的latent、loss、梯度、RNG一致，减少decoder调用。
- TF批没有Euler调用；rollout批只有8步，batch schedule续训可复现；VQ无梯度/权重不变。
- 100-update FP32 GPU smoke实测时间/显存，再估训练成本。没有通过这些检查就不启动短训。

审计原始文件：`no_loss_audit/summary.json`（solver、donor）、`no_loss_audit/velocity_consistency.json`、`no_loss_audit/decode_calls.json`；相关noise反例见`best_std1_rho0999/summary.json`。这批结果是单batch/单style/单seed诊断，不是完整训练效果验证。

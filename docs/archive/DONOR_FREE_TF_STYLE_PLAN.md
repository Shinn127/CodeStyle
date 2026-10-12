# 无 donor、无 rollout：风格响应优先的 TF 修订计划

2026-10-12重新审订。状态：读取当前代码、训练日志与已有干预记录后形成计划；本轮未修改训练/推理程序，也未运行新训练。主线先保持单帧、六token结构，四帧联合生成暂作后备。保留现有实验资产，不恢复已经删除的donor代码。

## 1. 已确认事实和应撤回的归因

当前 `flowmatch_label_history_ode4_100ep` 从零训练，objective11、FP32、VQ冻结、六token、双帧history。前50epoch纯TF，51–70升至全rollout，随后全rollout至100。

固定270 matched / 540 cross、240f / 60fps / Euler4的validation rollout：

| Epoch | 反馈比例 | matched acc | cross acc | matched/cross jitter cm | matched/cross ADE cm |
|---|---:|---:|---:|---:|---:|
| 50 | 0 | 63.70% | 0.74% | .436 / .468 | 5.77 / 7.21 |
| 55 | .25 | 67.04% | 2.22% | .538 / .574 | 4.13 / 5.49 |
| 100 | 1 | 74.07% | 5.56% | .792 / .905 | 3.05 / 4.78 |

20次验证全部未过联合约束，`best.pt`对应epoch55，是诊断候选。曲线说明本轮rollout课程改善部分追踪，却没解决cross，并伴随jitter升高。训练步数/LR也在变化，所以不能据此证明一切rollout训练必然有害。

旧 `TF10_DIAGNOSIS.md` 的TF10从donor-era模型初始化，同时取消donor/rollout再训练。其65.68% cross可能继承旧模型的能力；不能推出“从零matched-only TF足够学习cross”，也不能把多个同时变化的因素分别归因给history、donor和rollout。旧文中“必须保留donor”“增加rollout压抖动”的建议不再作为主线依据。

当前交互代码 `set_style→current_feature→sample_latent(style_feature=...)` 连通。已有一source、六目标style、三个checkpoint的同noise干预记录显示：style feature各不相同，blend完成后等于目标feature，但所请求的五种cross风格均未被生成分类识别，第一帧pose差仅约.06–.46cm。它支持“条件影响很弱”，不能把单source结果当总体accuracy；总体失败证据来自540-case验证。

独立的确定缺陷：Python交互入口缺省会跟随checkpoint，但shell仍传入 `--ode-steps ${ODE_STEPS:-8}`，覆盖本轮Euler4。修这个问题只能统一协议，无法解释和解决Euler4验证中的低cross。

## 2. 失败机制重新推导

### 2.1 完整历史提供了绕开style的路径

同风格TF训练只观察 `(h_A, trajectory, style_A)→q_next_A`，cross却请求 `(h_A, trajectory, style_B)`。当前完整absolute q、完整291维motion以及delta q都进入field。历史已包含姿态、风格和运动趋势，忽略显式style的函数仍能取得很低FM loss。继续训练matched-only并不直接约束未观察到的cross组合。

prior CE监督的是selector接近各style的code分布，不是生成器必须使用该条件：这项CE对field参数的直接梯度为0。selector没有塌缩、prior KL低，都不能证明生成style正确。单纯增大prior权重不是本问题的主解法。

### 2.2 rollout输入和原GT目标可能错相位

目前展开用生成历史替代观测，目标仍是原GT的绝对时间索引。一旦生成状态的相位、支撑脚或姿态偏离，训练就在要求它一步回到原GT时刻。detach只取消跨步梯度，不能消除输入/目标的这种冲突。这是抖动与滑足的合理机制假设，需要以同一初始化的对照才能严格定量归因。

### 2.3 删除全部姿态历史也不是无代价解法

如果过滤后的条件C把两个不同姿态m_a和m_b映射成相同内容状态，固定目标style/noise得到同一输出y，则

`max(||y-m_a||, ||y-m_b||) ≥ 0.5 ||m_a-m_b||`。

因此只保留相位而彻底丢弃姿态边界，无法保证对任意当前姿态都连续。prior选出的8个slot只是统计相关slot，不是已验证的纯style因子。简单缩小history幅度也不是去风格，网络可以补偿缩放。纯residual预测也不能自动解决静态姿态切换：ArmsAboveHead的同风格GT速度接近0，并不告诉模型怎样从放下手臂过渡到举手。

所以先用明确的history信息遮蔽训练减少捷径，同时保留干净history任务与真实姿态边界，不立即把所有历史彻底删除。

## 3. 第一轮：六token、纯TF、风格注入与history联合遮蔽

### A. 每层接收style

保留 `label→selector→style code/prototype→e_s`、原style token和原loss。两层transformer的生成query再接受小型style scale/shift或residual bias，输入统一为e_s；调制线性层零初始化，scale有界。同一次Euler的style投影缓存一次。不新增classifier loss或label-only旁路。

目的：缩短style到预测输出的路径。但更强注入本身不能保证使用style，因此必须与B的训练干预一起验证。

### B. 30%窗口同时遮蔽两条absolute-history路径

默认70%窗口保留完整history；30%窗口同时屏蔽absolute q token中的静态分支和motion token中的完整pose/rotation分支。共享mask在整个TF窗口保持一致，不能只mask其中一条、让另一条继续泄漏相同信息。遮蔽作用于condition projection，不覆盖原始q、GT targets、归一化统计或真实反馈状态。

历史token建议写为：

`h_q = P_delta(delta_q) + keep * P_abs(normalize(q_prev))`

`h_motion = P_phase(phi) + keep * P_motion(motion_prev)`

其中keep在训练按窗口采样，正常rollout默认keep=1。保持六token，不新增时序encoder。phi只从过去实际状态构造：root vx/yaw/vz、归一化左右脚方向、由两帧位置及真实root位移/yaw补偿得到的相对运动方向、软contact和history有效性。不要用decoder单帧velocity冒充实际位移差分。GT条件来自同一冻结VQ/cache，训练和live使用同一个phi函数。

保留delta q是保相位的起点，不声称它完全无style；主实验不强行把8个selected slots全部清零。若还存在明显捷径，先用诊断验证泄漏位置，再考虑更强内容过滤。

从epoch1即使用30%遮蔽，避免先花几十epoch强化复制捷径；本轮不再叠加独立motion/latent Gaussian noise、low-t重加权或复杂课程。少量reset案例（建议5%窗口）令older=current并标记history无效，模拟实际冷启动；目标仍为同一真实GT。reset案例也计入原窗口权重，不重复添加loss。

这仍只用本style真实相邻帧，既没有跨style合成目标，也没有自生成历史。它让模型在部分样本上必须更多依赖style/trajectory/dynamics，是归纳偏置而非cross可辨识性的数学保证。

### C. 正常推理保留完整实际history

主评估keep=1，切换style时保留实际姿态、root和两帧history，明确改变style条件。不得通过reset到目标风格GT、偷换source、改变noise、强制覆盖root来伪造成功。

另外做keep=0的只读诊断：若只有遮蔽history时才能生成正确style，而正常keep=1仍接近chance，说明捷径没有真正解决，本轮不能验收。不能把诊断路径较好的accuracy替换主结果。feature blend=0用于辨别语义响应，正常显示可单独比较.25s；插值e_s本身没有真实transition监督保证。

## 4. 至多50 epoch的单条主实验

| 配置 | 值 |
|---|---|
| 初始化 | 新结构从零；旧checkpoint冻结作对照，不沿用rollout optimizer/state |
| 数据 | 当前nomirror90-style split、stride1、冻结VQ150 best、原norm和train-only latent stats |
| 模型 | 单帧query、六token、hidden256、2层、FFN1024；双帧动态条件及逐层style调制 |
| 训练 | 50epoch上限、seed42、FP32/highest、TF32关，feedback_probability固定0 |
| 批次 | 1024窗口/update、seq_len6，4个独立GT相邻转移/window；不是4帧联合生成 |
| 遮蔽 | epoch1–50固定30%静态history联合遮蔽；5%reset-history案例 |
| FM | target、normalized loss、原Gaussian std1、t~Uniform[0,1) |
| loss | flow1、prior CE .02、entropy .003；recon/root/style-consistency全0 |
| optimizer | AdamW、lr5e-4→1e-5、cosine50、wd1e-4、gradclip1/update |
| 主推理 | Euler4、noise std1、clip→Q→D；8步仅同noise诊断 |
| 验证 | 每5epoch固定270 matched+540 cross，240生成帧；仅validation选型 |

先审计各style/suffix的可用窗口与root速度支持范围，确认训练采样不过度偏向长clip；若需要改为style均衡采样，应作为明确的新配置记录，而不是悄悄更改现有split/prior。cross完整manifest保留，缺少共同速度支持的案例单独分层分析，不删除困难案例提高accuracy。

epoch5/10做固定24次跨style切换诊断（12个validation sources各2个目标，覆盖静态姿态风格和步态风格）；同时比较正确目标、请求source style、history遮蔽三种干预，同case共享noise。

**epoch10决策门槛**：若完整540-case cross仍≤10%，且epoch5→10没有改善、正常history下的语义干预仍失败，则停止该run，最多使用已发生的10epoch，不默认续到50。这个10%只是“明显失败”的止损线，不是成功标准；跨过它也需检查matched/root/运动质量后才继续。

最终目标仍是同checkpoint/主协议下matched≥85%、cross≥80%，并兼顾原始运动质量。50epoch不保证达标。label模型不具备unseen接口，不能用本轮label结果推断unseen70%；clip/recipe另行接入同一e_s接口并独立训练验证。

## 5. 效率和代码修订顺序

1. **协议与初始化**：shell仅在显式设置ODE_STEPS时覆盖checkpoint；UI显示checkpoint epoch、实际步数、目标style/blend进度和joint约束状态。显式init缺失直接报错，禁止以微调lr静默从零。
2. **模型**：在`arch/flowmatch_vqvae_controller.py`增加phase projection、共享history mask及逐层style调制；原VQ/selector/FM参数化不变。目标版本独立于当前11，新训练不能resume旧状态。
3. **纯TF入口**：`training/train_flowmatch_vqvae_controller.py`禁止主run feedback>0。q[0]/q[1]→q[2]等四组真实转移可展平为[B*4,...]，一次并行field前向；这是计算合批，不是联合时序模型。prior/entropy每窗口计一次、clip梯度每update一次。显存不足使用microbatch累积。
4. **一致历史**：共享phi构造函数供缓存训练、离线rollout和交互调用。复用现有FP32数据/GT decoder缓存，不再建立donor cache。训练没有Euler或预测decoder；GT缓存准备的冻结decode单独计时。
5. **验证**：记录各FM时间段误差、condition干预、cross输出被分为source style的比例、切换前后完整240f accuracy、过渡峰值和达到目标的时间。分类器只作评估。保留jitter/足滑/contact、MPJPE/Root ADE、低速冻结代理及600f结果，避免“低jitter但不走路”。
6. **入口**：计划新建`run_train_flowmatch_style_tf50.sh`和独立输出目录；本轮尚未创建或启动。原入口保留作历史对照。

先通过索引/梯度/cache及真实入口100-update smoke，实测新速度/显存再估50epoch成本。现有纯TF约54s/epoch只是旧结构基线，不能直接作为新方案承诺。

关键回归：clean/masked targets完全相同；两个静态history入口共同被mask且无旁路；style调制获得FM梯度、VQ无梯度/权重不变；固定世界落地脚的root补偿差分正确；训练没有Euler调用；rollout不读取future GT motion；motion=D(q)；reset/切换history正确；固定case/noise可比较；保存/加载、epoch续训和validation RNG隔离。

## 6. 四帧联合FM何时值得做

现有`DONOR_FREE_TF_BLOCK50_PLAN.md`保留为后备。先要求上述模型在正常history下有明确目标style响应，再判断独立单帧采样是否仍主导jitter。若响应解决但时间连续性仍差，才申请下一轮4帧联合FM：一次预测[B,4,D]、块内共享t、帧间query attention，明确新增token/块边界/交互重规划/按帧noise协议。

“把四个TF转移batch起来”不会学习联合时序分布；“输出四帧”也不会自动修复块边界或跨style过渡。后备设计必须保留足够真实边界状态，专门检查相同phase、不同起始pose情况下的第一帧跳变，不能把所有静态历史一删了之。

若第一轮连style依赖都未建立，不应靠延长horizon掩盖问题，应先定位remaining history/trajectory捷径、数据共同支持和style注入梯度。真实跨style过渡监督缺失带来的限制需要诚实保留。

证据：`checkpoints/flowmatch_label_history_ode4_100ep/{config.json,history.jsonl}`；`analysis/donor_free_replan_2026-10-12/style_intervention.{py,json}`；旧TF10记录`analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md`（比较混有初始化及多因素变化）。本轮复用这些已有实验，不声称重新执行了其GPU干预。

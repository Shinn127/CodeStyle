# 无 donor、无 rollout 的下一轮训练与代码修订计划

> 重新审订：此四帧联合生成方案暂作后备。优先执行 [六token纯TF风格响应方案](DONOR_FREE_TF_STYLE_PLAN.md)，先验证style依赖；直接删除全部静态姿态历史会损失过渡边界信息，不能假定四帧生成自动消除跳变。本页其余内容保留供后续评估，尚未实施。

2026-10-12。状态：审计和方案完成，训练/推理代码尚未按本方案修改，没有启动新训练。此方案取代继续执行50 TF→20 ramp→30 rollout的建议；不恢复已清理的donor内容。修订前原始实验及checkpoint保留。

## 1. 当前结果说明了什么

读取 `checkpoints/flowmatch_label_history_ode4_100ep/history.jsonl`，这些数值来自固定270 matched、540 cross、240f/60fps的validation rollout，不是teacher-forced指标：

| Epoch | 训练反馈比例 | matched acc | cross acc | matched/cross Root ADE cm | matched/cross jitter cm |
|---|---:|---:|---:|---:|---:|
| 50 | 0 | 63.70% | 0.74% | 5.77 / 7.21 | .436 / .468 |
| 55 | .25 | 67.04% | 2.22% | 4.13 / 5.49 | .538 / .574 |
| 70 | 1 | 70.00% | 3.52% | 4.66 / 5.90 | .828 / .922 |
| 100 | 1 | 74.07% | 5.56% | 3.05 / 4.78 | .792 / .905 |

Rollout阶段改善了部分root追踪和matched分类，但没有解决cross，jitter明显升高。训练阶段还同时改变了epoch/LR，因此曲线支持本轮课程不值得继续，不能严格证明所有rollout训练必然有害。

现有 `best.pt` 是epoch55，所有20次验证均未通过联合约束。这是失败候选中最低联合分数的诊断模型，不是已经达标的best。选择它相对latest会进一步降低style accuracy，不能仅凭文件名判断模型。

只读干预：3个checkpoint，1个Neutral/FW validation source，6个目标style，共用相同Gaussian noise，240f/Euler4/FP32/seed42：

- prior风格名称与数据库顺序完全一致；所有90个style feature的最小两两距离约.76–.84，未塌成一个向量。
- `set_style()` 后，交互feature能准确达到目标feature，代码路径有响应。
- GT和quantized AE在这个source均分类为Neutral；生成的ArmsAboveHead、ArmsFolded、Robot、HighKnees、Old没有分类为所请求style。
- 改变style会改变continuous latent和部分code，但首帧平均pose变化仅约.06–.46cm。这不是完全无数值响应；语义响应不足和闭环漂移才是关键。单source不能代替总体评估，540-case cross曲线提供了更广的失败证据。
- wrapper仍默认传入 `--ode-steps 8`，覆盖Python入口跟随checkpoint的逻辑；训练/验证实际为4。这个协议问题要修，但它解释不了Euler4验证本身的低cross。
- `history_tf10` 等微调脚本在init缺失后静默退回从零、仍用5e-5。新计划必须明确初始化策略，禁止这个低LR回退。

证据：[干预结果](../analysis/donor_free_replan_2026-10-12/style_intervention.json)、[验证曲线](../analysis/donor_free_replan_2026-10-12/validation_curve.png)、[CSV](../analysis/donor_free_replan_2026-10-12/validation_curve.csv)。

## 2. 重新推导失败机制

### 风格条件在训练中是冗余信息

清理donor后，每条监督都是同style的真实相邻帧：

`(历史motion/latent来自s，目标style=s) → 下一帧GT也来自s`。

记完整历史为h、轨迹为T、style为s。若h已经足够辨识s并预测下一帧，模型 `F(Q_t,t,h,T,s)=G(Q_t,t,h,T)` 可以获得很低的FM loss。训练集没有约束“历史来自A、请求B”这一组合，无法从matched监督直接确定正确cross行为。

现有prior CE主要监督selector的分布；它不直接要求flow field使用style，更不保证240帧输出具有该语义。增大prior权重、看到低KL、重复传style、增加训练epoch，都不能消除这个问题。给style增加旁路但保留全部历史捷径也可能仍被忽略。

trajectory和gait dynamics也可能与style相关，不能声称过滤静态pose后就实现了充分解耦。新数据采样保持style均衡，并统计suffix/root速度/yaw的跨style共同支持范围；能在相近内容范围中重采样的，仍只使用该style自身的真实GT。缺少共同支持不能凭空合成目标，须记录覆盖缺口。原prior/normalization不因重复采样而重算。

### 当前rollout监督会要求错误相位回到原GT时刻

当前展开把生成历史送回模型，但下一步仍监督原GT时间索引的目标。若生成步态相位已经偏移，输入是相位φ_hat，目标却是原φ_GT。detached feedback阻止了跨步梯度，不能消除这个输入/目标矛盾。

它可以改善root追踪，同时让模型学会快速拉回GT姿态，产生抖动和脚接触不一致。去掉donor移除了另一类错误目标，没有自动修复此问题。此次训练彻底关闭自生成feedback。

### 单帧建模仍没有直接学习连续输出的联合分布

目前每帧独立Gaussian采样、单帧endpoint loss、单帧VQ decode。D(q)与q一致，不意味着相邻生成帧的世界足速度和root运动一致。把noise强相关化或减到很小可能将运动变成冻结姿态，也不能作为根治。

因此需要同时让style成为必要条件，并让FM直接看见真实GT的短时序相关性。推荐小幅扩展为4帧联合latent FM，保留冻结VQ、style prior/selector、conditional flow matching的核心。

## 3. 推荐主线：4帧联合TF，真实GT监督

### 3.1 过滤静态风格历史，保留真实动态状态

完整motion/q继续保存在闭环状态，供decode、root积分和可视化使用；不修改反馈code，不在推理后覆盖root。

field的历史条件改为：

1. **latent动态token**：最近两帧normalized q的差分，屏蔽prior选出的8个style slots。通过独立小线性层投影；不再额外叠加完整absolute q的直接条件旁路。
2. **motion/content token**：root平面速度/yaw、相对脚部方向、由真实历史位置差分计算的运动方向、软接触和历史有效标记。姿态幅度作规范化；不继续把完整291维静态pose/rotation直接交给field。
3. 保留trajectory、style、time条件。

不简单删除motion信息：支撑/摆动方向必须可辨。decoded joint velocity只是单帧输出，不能替代实际两帧运动；差分前要将旧脚位置变换到当前root frame，用现有root积分约定补偿平移/yaw。

history初始化只有一帧时，使用重复q并标记历史无效；训练加入少量相同的reset条件，不能在评估额外提供未来GT帧。接触信息是软参考，不硬锁source接触时序，因为target style可能改变cadence/support。

slot saliency是统计相关性，8个slots不是已经证明的纯style因子。差分和幅度规范化也不能完全去style。实施前要做保留相位的检查；训练中用wrong-style干预确认剩余历史不会继续压过目标条件。

### 3.2 style进入每层生成计算

保留 `label→selector→style codes/prototype→e_s` 和原prior优化。保留style token，再给两层transformer加入由e_s生成的轻量scale/shift或残差bias。使用小线性层，不堆新encoder；modulation可缓存于同次Euler的各次查询之间。

不添加classifier loss，不新增label-only identity旁路作为通用recipe方案。接口只接受e_s，未来clip/recipe模型可以用独立训练的encoder接入同一field接口；不能默认把label e_s替换成recipe e_s就能泛化unseen。

### 3.3 联合预测未来4个latent

数据约定：`X_k=[trajectory_k,motion_(k-1)]`，q_k=冻结VQ(X_k)。一个6-row window使用：

```text
q[0]：older history
q[1]：current history
q[2:6]：4个真实GT target，保持完整128维
trajectory[1:6]：current及future条件；不读取future GT motion作为条件
```

设归一化GT块为Q_1，形状[B,4,128]；Q_0仍采用现有Gaussian/transport参数，四个query各自采样，t在同一块内共享：

\[
Q_t=(1-t)Q_0+tQ_1,
\qquad L_{FM}=\mathbb E\left[\frac14\sum_{h=1}^4\frac1D\|F_\theta(Q_t,t,C)_h-Q_{1,h}\|^2\right].
\]

总loss仍为 `L_FM + .02 L_prior + .003 L_entropy`。reconstruction/root/style-consistency loss均0。prior/entropy每窗口一次。没有donor、retarget目标、模型生成历史或跨步反传。

四个noisy-latent query加五个条件token，共9个token；hidden256、2 layers、FFN1024、cosine保持。小幅改变时序query数量，让模型学习块内相关性，核心仍是VQ latent conditional FM。不是单帧模型多跑四次。

这项推荐涉及此前6-token边界的改变，必须明确视为时序结构升级。若继续严格固定6-token，则只实施历史过滤/逐层style注入，保持单帧TF；4帧块方案作为后备。两条方案不同时自动开跑，不把单帧生成四次称作联合建模。

Euler target转换保持相同，只将公式作用于整个块：

\[
Q_{n+1}=\frac{N-n-1}{N-n}Q_n+\frac1{N-n}F_\theta(Q_n,n/N,C).
\]

训练一次field forward，无Euler和prediction decoder；GT history/content来自冻结codec缓存。4帧联合分布不能保证块边界稳定，因此边界跳变是强制验收项。

## 4. 一条有界50 epoch训练

从零训练新结构，不续训当前rollout状态，不静默复制不兼容权重。旧模型仅作冻结对照。先代码smoke，再由用户启动正式训练。

| 参数 | 计划值 |
|---|---|
| 数据/VQ | 当前nomirror90-style split、冻结VQ150 best、原normalization和train-only latent stats |
| 训练 | 50 epoch、seed42、FP32/highest、TF32关闭 |
| batch | 1024 windows/update，4个joint targets/window；不足时512×2或256×4累积 |
| 窗口 | seq_len6、stride1、prediction_horizon4；autoregressive training feedback=0 |
| FM | target、normalized loss；Gaussian std1；25% t来自[0,.25]，其余Uniform[0,1) |
| 优化 | AdamW、lr5e-4→1e-5 cosine50、weight decay1e-4、grad clip1/update |
| style | 原8 slots/top-k4/temperature1/dense prior，逐层style modulation |
| loss | flow1、prior .02、entropy .003；recon/root/SC均0 |
| ep1–10 | clean真实GT条件；小比例reset历史增强用于支持冷启动 |
| ep11–20 | 冗余phase cue的group dropout由0升至.10 |
| ep21–50 | group dropout .10；root/trajectory/style保持；不同时清空全部phase线索 |
| 不默认加入 | 独立motion/latent Gaussian噪声、persistent noise、donor、rollout、额外loss |
| 主推理 | Euler4、noise std1、clip→quantize→decode；8步仅固定noise诊断对照 |
| 选型 | 每5epoch固定240f validation rollout；不使用test/OOD选型 |

low-t采样增强早期transport阶段对条件的学习；它是时间采样分布调整，不是新增loss，也不能替代历史过滤。

这里group dropout作用于derived content cues，目标始终真实GT。若以后加codec-history noise，必须从同一扰动q重新构造匹配motion/phase，先证实不改变支撑/相位；本轮不把该分支叠加进主实验。

epoch10必须检查固定wrong-style/target-style干预。如果cross仍接近当前0–6%且没有上升趋势，停止这条诊断run并分析，不默认把余下epoch当补救。这里只规划一条至多50epoch的label主实验，不自动追加sweep、多seed或另一轮recipe训练。

正式开跑前补测同一固定validation manifest的GT/quantized AE分类与运动质量，给出按style/action分解的参考值。这些是数据/codec保真参考，不是严格数学上界；若对应动作本身无法被classifier可靠识别，不能把缺口全部归因于generator，也不能用classifier loss硬优化出高报告分数。

本轮50epoch是验证预算，不是对matched85+/cross80+的承诺。无真实跨style过渡数据时，平滑切换和任意cross组合仍有可辨识性限制；结构过滤是要检验的假设。

## 5. 交互与评估协议

- wrapper只在用户明确提供 `ODE_STEPS` 时传该参数，默认从checkpoint读取；界面显示实际NFE/horizon/checkpoint epoch、选型是否通过约束。
- 每次生成4帧，实际逐帧commit，逐帧积分root并更新两个真实历史state。移动/朝向/style请求变化时丢弃尚未消费的预测块，从实际已commit的state重规划，不reset实际姿态或历史。
- 持续按住同一个键不反复触发全量replan；最长缓冲响应66.7ms。style blend保留显式开关，纯label干预用blend0，正常显示可用.25s；未训练的feature插值不能被当作保证平滑。
- 每块clip发生在Q之前，最终motion严格=D(最终q)。处理后姿态不写回模型。未来trajectory仅由历史和用户plan构造，不读取future GT motion。
- noise按case/global frame索引；变batch、ODE步数、重规划和未消费buffer不能偷换已比较帧的noise。干预时不同style共用同一noise。
- 正式style accuracy始终使用240个生成帧；切换指标单独统计transition峰值、达到目标style的时间及blend结束后的完整240f分类。
- 增加边界/块内jitter、world足滑、contact覆盖、moving时的冻结姿态proxy、600f误差曲线。零contact不能以0分通过，低jitter且冻结姿态也不能算改善。
- 过滤静态历史可能损失逐帧paired reconstruction细节；matched MPJPE、过渡跳变也必须报告。4帧联合训练和feature blending都不提供跨块/跨style连续性的保证。
- 保存motion合格候选、style候选与joint选择记录；所有joint约束失败时明确保存diagnostic候选，不能让`best.pt`或UI隐藏该状态。最终目标仍由同一checkpoint、同一主设置联合满足。
- 仍按source_entry_id聚类CI；不拿一个source的干预分类结果替代总体matched/cross。

## 6. 最小代码修改清单与测试

| 路径 | 需要修订 |
|---|---|
| `arch/flowmatch_vqvae_controller.py` | horizon queries、动态history/content编码、逐层style注入、块FM与Euler；旧模型仅保持可评估 |
| `training/train_flowmatch_vqvae_controller.py` | 真GT块监督、正确q/trajectory索引、每窗口regularizer、phase cue增强、独立版本与严格初始化/保存 |
| `Utils/clip_recipe_resident.py` | 复用FP32 train cache，按需缓存只读history/content；记录来源/完成状态 |
| `Utils/interactive_control.py` | 4帧buffer、真正commit历史、事件replan、逐帧noise索引 |
| `Utils/motion_eval.py` | 块autoregressive rollout、末尾截断、禁止future GT motion条件泄漏 |
| `Utils/controller_validation.py` | 块边界质量、switch干预、diagnostic选型状态 |
| `run_interactive_label_control.sh` / UI | checkpoint步数默认值、请求/生效style与buffer状态 |
| 新独立训练wrapper | `run_train_flowmatch_block_tf50.sh`；从零50epoch，独立输出，禁止missing-init回退 |

不重建被删除的donor文件，不写新的对齐损失或通用训练框架。新objective/condition schema独立版本；同版本resume恢复optimizer/scheduler/RNG，旧checkpoint不能当新训练状态。

必须先通过：

1. q[0]/q[1]/q[2:6]及trajectory索引、clip/split边界、reset历史有效性。
2. style mask只作用history条件；完整query/targets/feedback不被mask。
3. root平移/yaw补偿后，固定世界落地脚的phase velocity接近0；没有用decoded velocity冒充实际差分。
4. field/style modulation有梯度，VQ无梯度/权重不变；总loss无新项。
5. GT cache条件和live条件一致；训练没有Euler、逐步预测decode或自生成历史。
6. 将future GT motion替换为垃圾值但保留trajectory，rollout必须不变，防止teacher forcing伪装成评估。
7. 每帧motion/latent匹配；style/方向变化丢弃buffer且保留已commit历史；块末尾/末尾截断正确。
8. 固定noise下保存/加载与同环境epoch边界resume检查；验证隔离训练RNG。
9. 真实入口100 updates、60/600f与切换smoke；全FP32，无NaN/Inf；contact/边界统计齐备。

性能先测100-update默认配置，显存不足才启用microbatch。单次块field替代四次独立TF field、取消训练Euler、复用GPU缓存和条件投影，预计减少开销，但不沿用当前54s/epoch或此前1.65×的数字。代码实施后按新实测速率重新估算50epoch成本。

clip/recipe的后续模型复用content/block/e_s接口，独立训练其style条件；不对齐label feature，不自动宣传unseen已支持。先完成label的正确style响应和原始运动质量验收，再规划该分支。

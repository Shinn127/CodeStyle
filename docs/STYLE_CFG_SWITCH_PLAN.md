# Style-only CFG与交互风格切换

2026-10-12。状态：代码核对与推导，尚未修改模型或启动训练。CFG可作为风格强度/切换控制，不等于已经学会平滑跨style运动，也不能保证替代合格的过渡数据。

## 1. 当前代码实际情况

- 当前label `FlowMatchVQVAEController`没有训练style-null分支；现有checkpoint不能通过把e_s随意置零直接获得有效CFG。
- `arch/clip_recipe_v2.py`已有可学习`null_style`及`null+w*(conditional-null)`的target融合。`training/train_clip_recipe_v2.py`默认style dropout=.1、guidance=1.5；实际dropout还受paired/unpaired资格影响。这是已存在的代码能力，不是本轮新的效果证据。
- V2将条件/null分支合批，保持相同z/history/content/trajectory。可复用思路，不能把整个V2训练目标、权重或style feature直接搬给label模型。

CFG原始方法联合训练有条件与无条件预测；它也已被用于Flow Matching，见 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 和 [Guided Flows](https://arxiv.org/abs/2311.13443)。下文是针对当前运动控制代码的设计推导，论文并不保证本任务的运动质量或准确率。

## 2. 对target-prediction FM的接法

固定同一个flow状态z_tau、时间tau、真实历史h和轨迹T，令F_B是请求目标B的target预测，F_0是只去掉style的target预测，两者使用同一normalized latent坐标：

`F_cfg = F_0 + w * (F_B - F_0)`

`v_cfg = (F_cfg - z_tau)/(1-tau)`。

由于两条分支在同一个z_tau求值，这等价于 `v_0+w*(v_B-v_0)`。w=1是原条件模型，w=0是不指定目标style的模型，w>1外推style方向。后者仍条件于完整history/trajectory，不是删除所有控制信息的unconditional模型，也不保证一定延续某个已知style。

每个Euler步都应在当前同一个guided z_tau上重新计算两条预测，不能分别生成两段motion后混合，也不能先分别quantize再相减。完成积分后才clip→Q→D，motion与反馈q仍一致。不要假定8个prior-selected slots是纯style维度而只指导它们。

两条分支必须仅style不同：保留相同motion、latent历史、trajectory、flow时间和输入noise。若移除history或trajectory来构造baseline，差值就包含内容/控制变化，难以保持root轨迹。

## 3. style切换与CFG强度是两个控制量

运动帧时间k与ODE内部tau必须区分。设A为此前生效的style命令，B为新命令，alpha(k)在约.3–.5秒候选时长内从0平滑到1。可以在同一z_tau上定义：

`F_switch = F_0 + w * [(1-alpha)*(F_A-F_0) + alpha*(F_B-F_0)]`。

- alpha=0对应CFG-A；alpha=1对应CFG-B。
- w=1时null项抵消，只需要A/B两条分支，这是条件向量场插值，不应单独称为CFG。
- w>1时过渡需要A/B/null三条分支，稳定后只需B/null两条。
- alpha控制切换进度，w控制条件强调程度；增大w不等于更平滑的切换。

用零端点导数的smoothstep安排alpha，使控制条件的改变平缓。这个公式不能保证生成pose/velocity连续：每帧noise、状态变化和quantization都可能制造跳变。它也没有提供接触或稳定步态约束。

第一版应分别测清“固定B的CFG”和“w=1的A/B向量场插值”，有效后才组合三分支。不同时叠加当前e_s插值与向量场插值。快速再次切换需要保留当前混合命令状态，不允许把实际history重置；多目标混合的分支数量/延迟必须有界。先用单次切换基准确认机制，再扩展完整UI。

## 4. label模型需要怎样训练

1. 增加一个可学习null_style，约10%训练窗口将style条件替换为null；同窗口共享mask。所有style路径（token、若有逐层调制等）一起替换，不能残留目标label旁路。
2. 目标motion/latent、history和trajectory不变。conditional/null样本混在同一个batch，一次普通FM前向；不新增loss，不需要每个训练样本计算两遍。
3. selector仍按真实style监督prior CE，null mask只用于field条件；记录实际dropout和conditional/null各自的FM误差。
4. 保持VQ冻结、FP32、纯TF。推理history遮蔽策略应与选定训练方案一致，CFG两分支之间不得不同。
5. 新配置记录null训练、guidance约定和采样策略；旧labelcheckpoint只可作为显式初始化，不能当作已经具备CFG能力的模型。null分支未经训练时拒绝w≠1的标准CFG调用。

如果同时采用过渡donor，标准style dropout应清楚定义应用的数据分布。若只对真实matched数据训练null、而conditional还训练过渡数据，则两者不再是同一训练分布的条件/边缘预测，应明确称为continuation-baseline实验并单独验证，不悄悄当作标准CFG。

## 5. 对当前失败原因的限制

matched-only完整history已经能识别style，可能使 `F_B≈F_0`。CFG无法从几乎为零或方向错误的差值中创造正确风格。因此style dropout本身不解决历史捷径，甚至可能使模型更依赖history。

应先测同history/noise下的条件差异是否与目标语义一致。可采用之前讨论的历史条件训练干预，或质量合格的过渡donor，让field真正学到目标命令的作用；CFG是调节已学习方向的手段。两者各自效果先测，避免一次叠加所有改动。

大w可能放大误差、把latent推到数据范围边界、导致clip饱和和更多code切换，进而增加jitter、改变root输出或损失步态推进。低多样性/低jitter也不能证明运动正常。不要默认用CFG5–10去挽救接近chance的cross结果。

## 6. 最小验证顺序和成本

- 现有label模型可先测试A/B双条件插值及小幅对比外推；这是诊断，不是标准CFG，也不能用未训练的零style冒充null。
- 新null训练完成后，固定同一个checkpoint、同一case/noise，对比w=1/1.25/1.5/2。先小型validation干预，出现风格收益且质量未恶化再扩大；不自动扫大量参数。
- 对稳定target B与切换A→B分别报告accuracy、目标响应时间、过渡峰值、Root ADE、world足滑/contact、腿部活动/冻结proxy、clip比例及code切换率。正式matched/cross仍用原固定240f生成窗口，切换后额外240f只能作为另列指标。
- 若所有有用的w都显著恶化运动质量，或只是把动作推向别的错误style，则停止增加w，回到训练数据/条件依赖问题。
- 训练style dropout开销小；推理双分支/三分支的field样本计算量约2×/3×。合批和缓存投影可减少调度开销，但不能承诺wall time不变。Euler4稳定CFG每帧相当于8个分支样本评估，三分支过渡为12个；只做一次最终quantize/decode。

建议起点是style dropout=.1、验证w=1.25或1.5，以及独立的w=1场插值诊断。这些是待验证超参数，不是已证实适合当前checkpoint的最佳值。label/clip/recipe可共享“按条件预测做CFG”的接口，各自style编码器和null分支需在本模型中训练，无需把recipe feature对齐label feature；OOD提升仍需单独验证。

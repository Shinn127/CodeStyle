> **历史文档（donor 路线已于 2026-10-12 移除）**：本页记录的 donor/cross-style 合成监督在当前代码中
> 已不存在（objective version 11 起）。保留原文用于回溯当时的设计与结果；现行归因见
> [TF10_DIAGNOSIS.md](../../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

# 下一版：针对 rollout matched≥85%、cross≥80%、unseen≥70%

2026-10-10，条件注入实施版。反馈训练域、reference表达、每层条件注入与量化后的输出风格监督已实现为独立V2入口；原6-token/2-layer主干不再是硬约束。本文件保留方法推导与验证目标，实际参数/运行方式以[V2运行说明](../../docs/CLIP_RECIPE_V2.md)为准。正式Φ与controller训练尚未执行，不能承诺指标已可达到。

## 已确认的差距

当前50-epoch模型的epoch35 checkpoint，在统一test cases上240帧/60fps autoregressive rollout：matched **62.90%**、cross **41.72%**、unseen **1.62%**，RootADE **32.93/33.28/33.04cm**。完整结果见 [REPORT](../../eval_results/clip_recipe_practical_rollout/REPORT.md)。这些与85/80/70分别相差22.10、38.28、68.38个百分点。

本次另外对三个既有checkpoint进行相同validation对照：每style一个未镜像validation clip，共90个source窗口，90 matched + 180 cross；240个生成帧、Euler8、固定case seeds、FP32/highest/noTF32、clip→quantize→decode。Recipe模型使用同一批每style五个独立train exemplars，label模型使用label条件。

| 冻结checkpoint | Matched | Cross | Matched RootADE | Cross RootADE |
|---|---:|---:|---:|---:|
| 当前clip_recipe_gt_history50 best35 | 61.11% | 37.78% | 29.28cm | 28.35cm |
| 旧clip_recipe_scratch150 best145 | 67.78% | 53.33% | 10.70cm | 11.34cm |
| label150 best | 77.78% | 72.22% | 11.37cm | 14.52cm |

这是270-case validation对照，不替代正式test。训练时长、增强和监督同时不同，不能把退步唯一归因于mask、flow-time或噪声。它支持两个判断：现有flow/VQ有更好的稳定性和seen cross潜力；仅换回旧recipe模型仍不足以达到85/80，更没有证明OOD70。

原始结果与来源指纹：[共享validation](shared_rollout_validation/summary.json)、[case manifest](shared_rollout_validation/rollout_style_validation_cases.json)、[复现脚本](compare_rollout_validation.py)。三个模型的checkpoint与VQ状态均核验不变，每模型评估约35秒。

另有16个固定validation case的冻结推理干预：

| 当前模型的干预 | Matched命中 | 动态cross命中 | OOD命中 | Matched RootADE |
|---|---:|---:|---:|---:|
| 原路径 | 5/8 | 0/4 | 0/2 | 42.17cm |
| E([trajectory,D(z)])再投影、解码反馈 | 7/8 | 0/4 | 0/2 | 32.49cm |
| 输出8slots加入reference prior，强度.25 | 8/8 | 0/4 | 0/2 | 90.14cm |
| 输出8slots加入reference prior，强度1 | 7/8 | 0/4 | 0/2 | 96.60cm |

这些是小样本研究诊断，不是新成绩。Cycle projection平均改变73.23%的slot code，但部分pose-cross root误差反而变大；它说明decoder一致性不等于encoder一致性，不能证明每次投影都更好。输出prior干预提高部分matched同时损害root，不部署这两个开关。它使用完整reference histogram，若采用还需扩展recipe接口，不能声称原π接口已实现这种控制。最初诊断的prior距离使用未裁剪latent；按主路径修正为先clip后计算距离的复测得到相同结果，表中prior两行以[修正复测](rollout_interventions_clipped_prior.json)为依据。其他对照见[原始记录](rollout_interventions.json)、[脚本](rollout_interventions.py)。

## 为什么prior本身不能保证风格

现有prior loss约束的是selector：`KL(p(code|style) || q_selector(code|reference))`。它不直接约束 `Style(D(Q(z_pred)))`。此外，统计slot差异可能包含style与action/root采样分布的共同变化，不能将8个slot当作已经解耦的style坐标。提高CE不会自动补齐生成端监督或任意cross条件的训练覆盖。

当前reference经过无序histogram、TopK4、每slot的4D均值，再进入P。对任意两个概率配方π、ρ，若 `Σπ_k E_k = Σρ_k E_k`，现有P得到完全相同输入；P无法恢复丢失的分布信息。无序histogram也丢失时间顺序。这些是确定的表达边界，但不是已证明的accuracy上限。

已有很小的OOD真实motion/VQ AE对照能保留若干OOD风格，说明不必先重训VQ；那批31例AE的77.42%不是generator的理论上限。当前OOD1.62%→70%需要验证新的条件泛化和输出约束，不能只靠seen accuracy高推断可行。

## 1. 先恢复反馈稳定性，继续便宜的单步训练

默认观测从 `GT motion + q_GT` 改为 `D(q_GT) + q_GT`，trajectory仍为原source命令；下一帧endpoint仍是真实GT编码。冻结VQ，只做无梯度decode，可缓存观测。现有`--train-feedback-context`提供了这条基本路径，但目前不能与practical_history同时开启，实施时必须处理这个限制，不能仅叠加命令参数。

少量样本使用成对的codec扰动：

```text
q_GT → 小幅normalized latent噪声 → clip → Q → D
                                            └─ 同一个q作为previous latent
```

这样motion与latent对应同一个反馈状态，不分别给两路输入无关联的噪声。起始候选：概率.2，噪声强度在[0,.15]间取值；从epoch1起、前5epoch升至满强度。记录实际换码比例、root/body扰动和恢复误差，不能把.15当作已验证最优强度。

先停用目前30%foreign history+20%全body mask的组合，恢复uniform flow-time。这是撤回整体效果不佳的配置，不是宣称已找出单项因果。不要每次推理强制cycle projection或直接覆盖root。无需Euler展开、生成历史反传或联合更新VQ。

## 2. 为clip恢复时序表达，recipe仍是唯一共同接口

建议下一版恢复论文已有的轻量temporal Es，而非继续只调histogram残差。这会调整最早“不新增clip encoder”的实施边界；保留的是论文的选码、聚合、P、conditional flow和冻结VQ核心。

```text
独立reference clips
  → cached frozen-VQ sequence（32×4D/frame）
  → 2-layer temporal Es（候选hidden128，240帧内降采样为60tokens）
  → Gj dense logits → TopK4 probabilities π
  → nonlinear selected-code aggregation → 8个slot tokens及global style
  → 每层AdaLN + slot cross-attention → conditional flow

exported/manual recipe π ────────────┘
```

Es显式携带实际采样间隔/位置；每个独立exemplar单独编码再合并1/3/5个clip，不能拼成虚假连续运动。Gj不再限制在`log(hist)+[-4,4]`；dense prior CE保留作辅助统计约束，不直接把π写进生成latent。Label只用于train采样、prior和辅助表征学习，不输入flow，不对齐label模型特征。

当前代码先把每个slot的分布压成4D均值，再把8slots压成一个64D style feature。仅保留均值会使部分不同配方不可区分；后续P无法恢复这些信息。这是表达风险，不是已测得的准确率上限。

修订为共享的小型非线性code映射`φ: R4→R64`，每slot构建：

`μ_j=Σπ_jk E_k; a_j=Σπ_jk φ(E_k); h_j=−Σπ_jk logπ_jk`。

将`[μ_j,a_j,h_j,slot_id]`投影成一个128D slot token，共8个；另由全部slot tokens得到256D global style。保留原μ，同时通过非线性code特征减少均值碰撞。有限维聚合仍不是无损编码。φ从冻结code向量计算，不引入style-label embedding；φ参数可训练，不能跨optimizer updates缓存其输出。Manual recipe中所有非零项均参与加权，不偷偷再裁成TopK4。

所有条件通路必须只依赖最终π：不把Es隐藏特征另行注入flow。Clip导出π与pure recipe输入必须逐值得到相同slot/global条件和固定noise rollout。

### 每层注入，而非仅依靠一个style token

当前field是pre-LayerNorm Transformer，只有一次style token输入，最后只读取z_t token。将style feature整体乘常数，其尺度效应会被后续LayerNorm部分抵消，不能当作可靠的控制旋钮。Label模型同一主干的cross更好，也说明单token不是唯一原因。

新主干候选固定hidden256、4 blocks、4heads、FFN1024、dropout0。每个block包含：

1. Global style与flow time预测AdaLN的shift/scale/residual gates，在LayerNorm之后施加`(1+γ(s,t))*LN(h)+β(s,t)`；self-attention和MLP均受调制，输出head也有条件调制。
2. z_t主token直接cross-attend到8个slot tokens，通过残差加入，避免所有配方只能经由单一global向量。
3. trajectory、previous motion、previous latent仍为独立条件输入；不把8个统计style slots当作纯body坐标，也不按这些slots直接覆盖输出latent或root。

AdaLN使用稳定的identity/zero-gate初始化；最初一步style梯度可能为零，梯度验收须覆盖若干真实optimizer更新之后，不能只验证初始化瞬间。参考[DiT官方实现](https://github.com/facebookresearch/DiT/blob/main/models.py)，其效果证据来自图像生成，移植到此控制器仍须验证。

### 有训练依据的style CFG

在有GT的FM分支按窗口以0.1概率关闭全部style通路（slot tokens、global modulation与output modulation都切换到一致的learned-null条件）；保留history与trajectory。无配对cross辅助分支保持style条件开启。Style-null样本不计算要求特定reference的输出style loss。

推理在同一z_t、time、history、trajectory下计算条件/无style两路target prediction：

`F_guided = F_null + w*(F_style-F_null)`。

然后按现有target→velocity转换与Euler积分。两路共享Gaussian输入。主设置预先固定`w=1.5`；validation额外固定`w=1`作为依赖诊断，不按matched/cross/OOD偷偷分别选择强度。推理每个ODE step需要两路field evaluation，可沿batch合并，不能称其免费。旧checkpoint没有受训的null style通路，不能直接套用这条公式并宣称得到合法CFG。

这是对[Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)的flow target参数化适配；其收益、轨迹影响和实时成本均须实测。

**效率实施约束**：每个optimizer step按8个完整source窗口×64个转移组织，reference、Es、selector、φ聚合及slot K/V投影按window去重，再广播到512个转移；不能对512个anchor重复运行reference网络。推理可缓存不随状态改变的条件与K/V；固定Euler时间网格下，AdaLN仅依赖style与time，也可预计算全部N步系数，在style/solver设置变化时刷新。训练中的随机time仍正常计算。缓存frozen VQ indices/latents与冻结Φ reference embeddings，不跨参数更新缓存Es/Gj/φ/P输出。CPU reference裁剪及元数据按窗口prefetch；batch starts仍stride1，epoch update数固定。只有8个独立window会降低单update的style多样性，应按style/action均衡采样并记录覆盖。现有真实新样本流66.85→47.09ms/update的修复见[性能记录](../pipeline_speed_2026-10-09/REPORT.md)；该数字不适用于尚未实现的新网络。

## 3. 让生成动作受到reference风格约束

增加独立的、train-only风格度量Φ。正例为同style不同独立clip/尽可能不同action，困难负例为同action不同style；输入用root-frame pose及时间差分，可结合rotation，不直接读root3。训练同时包括GT与冻结VQ AE，减少编码误差被当成风格差异。Φ不能使用10个OOD styles训练，不复用包含OOD训练的分类器，更不优化正式报告的100类classifier。

先检查Φ在固定controller-IND held-style子集上的reference检索与GT/AE一致性，再决定是否作为监督；VQ曾见过这些IND styles，所以这只是Φ表征泛化诊断，不是全系统OOD结果。Φ通过检查后冻结；reference embeddings可缓存。

对flow的**一次target endpoint预测**施加：

`L_style = 1 − cosine(Φ(decoded predicted sequence), Φ(independent reference))`。

`L_root = L1(decoded predicted root3, source GT root3)`。

不恢复全291D reconstruction loss，不调用8-step Euler。VQ参数仍冻结，但这里需要允许decoder的输入梯度回到flow。辅助loss采用量化前向：`c=clip(z_pred); z_ST=c+stopgrad(Q(c)-c); motion=D(z_ST)`。它在前向与实际clip→quantize→decode一致，反向是明确的straight-through近似，不是量化的真实导数。否则连续latent中的style变化可能没有跨过code边界，部署后会消失。另保留continuous endpoint诊断，用同一case检查量化前后的损失与code变化。

Φ与reference独立、冻结、train-only。对生成序列增加同action不同style的负例对比，检查它是否比错误reference更接近目标；不只要求“换recipe时数值变化”。权重须根据梯度与运动质量短测设定，不能只追高classifier分数。

当前已新增`compute_loss(return_endpoint=True)`，直接返回这次FM evaluation的可导`endpoint_latent`，不调用Euler或decoder。下一版辅助loss应使用它；不要使用旧`return_prediction=True`或直接开启旧root/recon分支来实现本段方案，因为这些入口会执行Euler求解。

风格是序列属性，不能对512个随机单帧伪造一个时间窗口。每个source窗口连续读取240帧，从0..238均匀取64个anchor，每个anchor预测其真实相邻下一帧；anchor间距为3或4帧，Φ使用实际时间间隔，reference采用同一规则。8窗口×64转移保持有效512，辅助序列覆盖约4秒。window starts仍stride1，epoch update预算固定。实现显式排除不足240帧source并记录数量，没有伪造padding。辅助loss至少覆盖全部unpaired windows，再补足约四分之一完整windows。

有效cross配对继续检查root、相位、动作与整段连续性；增加一致retarget的target-history维持样本。对没有可信GT cross endpoint的请求，不将source GT或任意donor冒充target风格GT：允许uniform target采样的unpaired辅助分支，只计算style/root约束，FM loss显式关闭并单独统计。建议起点为70%matched/维持、10%可信转换、20%unpaired cross；这属于需要验证的新训练目标，而非已证明正确的解耦数据。Unpaired分支从Gaussian、t=0的单次endpoint预测开始，不用未知target构造假FM桥。它不能保证与完整ODE解完全一致，需用rollout验证发现差距。

上述比例与λ是起始研究参数；先记录各项梯度与成本再固定。Prior仍是辅助，输出style loss不是prior CE的替代名称。不要用prior CE的常数下界占total比例选权重。

## 单一有界50-epoch验证计划

从零初始化新的controller/Es/P，保留冻结VQ、既有normalization、FP32/highest/noTF32、seed42、有效batch512、stride1、normalized target FM、AdamW和cosine50。旧checkpoint只作冻结对照，不作为初始化或续训状态。

接口/梯度回归和真实数据100-update smoke已完成：recipe等价、VQ无梯度且不变、辅助loss不调用Euler、反馈motion/latent匹配、无配对样本FM显式关闭、train-only references、null style清除全部条件通路、CFG共享噪声、ST量化前向。完成版约48ms/update、1374MiB allocated，600帧ODE8+CFG约5.13秒。Φ为随机资源fixture，不能据此声称style有效；完整epoch和正式Φ成本仍待训练实测。

最多50epoch，包含warmup，不自动追加。前5epoch以真实维护/反馈校准和Es/prior为主；辅助风格任务逐步开启。Φ短预训练也须单独记step和GPU时间，不隐瞒成本或自动追加一轮50epoch。

Epoch5/10为诊断节点。固定history、Gaussian noise和source trajectory，比较正确reference、同action错误reference、null style和导出recipe：记录条件间距、continuous endpoint差异、量化后code-change、目标style距离/accuracy及Root ADE。Attention权重本身不作为因果证据。检查动态target分组、source retention、reference检索和window级实际任务比例。参考晋级门槛可预先设为epoch10 matched≥75%、cross≥55%、RootADE≤15/20cm；它们是预算筛选标准，不是训练结果预测。没过就停止定位，不默认烧完余下epoch。若用户希望无条件跑满50，应显式选择该行为。

每5epoch使用既有270 matched/540 cross、240帧rollout manifest选择checkpoint；沿用root约束，不混用TF best状态。选型仍记录cross80目标是否通过，不把旧的`constraints_passed`当成85/80/70验收。Unseen10类只在锁定模型后正式评估，不用其test accuracy逐epoch挑模型；此前对OOD的探索性诊断须保留披露。

最终目标是同一个checkpoint、同一个clip→quantize→decode设置、同一240f classifier上达到85/80/70。Root ADE、world foot sliding、contact覆盖和运动质量同时报告，不能靠生成固定夸张姿势或牺牲轨迹追高accuracy。达不到时按反馈稳定性、条件检索、转换/维持、动态风格、OOD保持分解，不自动延长训练。

## 本次完成与未完成

已完成：Temporal Es、标准化code几何的非线性聚合、AdaLN/slot cross-attention、受训null-style CFG接口、独立Φ短预训练入口、quantized endpoint损失、完整window配对/无配对任务、统一recipe/export/eval以及有界50epoch入口。轻量回归及真实数据100-update+60/600帧反馈smoke通过，见[实现记录](../clip_recipe_v2_implementation/smoke_final.json)。正式Φ预训练、50epoch训练和85/80/70联合验收尚未执行；smoke使用随机Φ，不能当作style效果证据。

2026-10-10追加效率修订：相同训练目标下合并反传、GPU只读数据驻留、冻结Φ缓存及固定shape CUDA图、去掉无用量化诊断与统计同步。新run默认启用；FP32与采样/损失语义保留，梯度累积安全回退eager。最终交替新样本短测35.46→23.48ms/update（33.8%时间下降）；完整训练成本仍需实际epoch测量。见[性能报告](../clip_recipe_v2_speed/REPORT.md)。这次不增加训练epochs，不启动正式训练。

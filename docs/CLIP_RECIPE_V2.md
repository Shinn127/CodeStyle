> **历史文档（donor 路线已于 2026-10-12 移除）**：本页记录的 donor/cross-style 合成监督在当前代码中
> 已不存在（objective version 11 起）。保留原文用于回溯当时的设计与结果；现行归因见
> [TF10_DIAGNOSIS.md](../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

> **2026-10-10 状态**：镜像段已确认导致 mesh 扭曲，训练数据一律不使用镜像；含镜像库上训练的
> VQ/controller/classifier 与派生 cache 已全部删除，代码默认路径与防呆见 [NOMIRROR_CHAIN.md](NOMIRROR_CHAIN.md)。
> 本文下列内容为修订前的记录，其中涉及 `--mirror`/`--no-mirror`、`--include-mirror`、`--unmirrored-sources`
> 等开关与已被删除的 checkpoint 的说明不再适用。

# Clip/recipe V2：实现与有界训练

2026-10-10。代码已实现并完成轻量回归和真实数据GPU smoke。正式Φ预训练和50-epoch controller训练尚未执行，matched85/cross80/unseen70仍是待验证的联合目标。

## Pipeline与方法边界

Clip经固定VQ150编码，独立exemplar各自进入2层temporal Es（hidden128，240帧采样60tokens），再合并1/3/5个exemplar特征。Gj输出dense logits并产生TopK4 π。Manual/exported recipe直接提供π，全部受控slots必须存在。

每slot保留原code均值，并聚合共享MLP对标准化code几何的64D映射、entropy及slot identity，得到8个128D tokens；其组合产生256D global style。最终π是所有style通路的共同输入，Es隐藏特征没有旁路。Manual dense recipe的非零项全部参与，不额外裁剪。

Field为hidden256/4blocks/4heads/FFN1024；global style/time对每层及输出head做AdaLN，主latent token直接cross-attend到slot tokens。全部参数/前后向FP32、highest、TF32=False。原VQ权重与normalization固定；classifier只用于validation/test。

有GT的FM分支按window以0.1概率关闭全部style通路，训练learned-null condition。推理固定Euler8、std1、style CFG1.5，条件/无style两路共享状态与噪声。两路沿batch合并；slot K/V和固定时间网格的AdaLN系数可在连续rollout中复用，style/权重/solver变化后刷新。

## 训练组织与损失

8个240帧source windows，每window在0..238均匀取64个anchor，各预测真实相邻下一帧；有效batch512。开始位置stride1，source按style/action均衡抽样。每epoch update预算仍为原seq_len2的train transitions数量除512，当前数据约11158；不会按window batch8把epoch放大64倍。

不足240帧的source segments显式排除，数量写入config；没有用padding伪造动作。Reference必须来自足够长的独立未镜像train clips，排除原source与donor的source_entry_id。Es/Gj/聚合及slot K/V按window计算一次，不跨optimizer更新缓存学习模块。

默认任务请求：10%完整window matched-root合成配对、20%unpaired cross、其余matched。配对要求相同action、整个window平面root速度差≤max(.25m/s,20% source speed)、yaw差≤30deg/s，归一化phase距离≤.25；一个window固定一个donor。Synthetic motion保留source root并修正joint velocity，重新编码target。第一个anchor从原source历史过渡，其余维持目标历史。无合格donor回退matched，记录请求/实际比例，不称为真实transition GT。

Unpaired target style均匀取另一个seen style；从Gaussian、t=0单次预测，**FM loss为0**，只用style/root辅助约束。真实OOD数据不进入训练。

历史条件是匹配的`D(q_prev)+q_prev`；以.2概率加入幅度U(0,.15)的normalized latent噪声，clip→quantize→decode后反馈，前5epoch ramp。Motion与latent始终对应同一个状态。训练不运行Euler/rollout。

主要损失：FM=1、dense prior CE=.02、entropy=.003、output style=.1、root=.1。前5epoch关闭unpaired与output style，epoch6—10线性开启。Root loss和style loss仅在辅助windows上对单次endpoint计算，至少覆盖全部unpaired，再补足约四分之一windows。

辅助前向`clip→Q→D`，通过明确的straight-through近似回传到flow/Es/P；VQ参数完全冻结，关闭全291D reconstruction。Φ也冻结，输入为root-frame position/rotation及按实际时间差校正的差分；排除root3和含root运动项的joint velocities。Style loss为cosine距离加不同目标style的reference对比，不优化正式报告classifier。

## 两个有界训练阶段

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
conda activate mcc
DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
RUN_ROOT=checkpoints/clip_recipe_v2 bash scripts/run_train_clip_recipe_v2.sh
```

1. `training/train_motion_style_metric.py`：最多1000 optimizer updates，IND90中保留10个styles做Φ检索诊断，仅用另外80类训练。每batch同style不同原clip为正例，优先不同action；混合GT与冻结VQ AE。Held-style support/query原clip不重叠，报告GT/AE检索。Φ曾共享见过90类的VQ，因此此诊断不是系统OOD结果。检索门限默认.60，失败不启动controller。
2. `training/train_clip_recipe_v2.py`：从零初始化controller/Es/φ/P，最多50epoch，AdamW lr5e-4、weight decay1e-4、clip-grad1、cosine到1e-5、seed42。Controller可训练全部IND90；真实OOD10只提供最终评估reference。

默认输出`checkpoints/clip_recipe_v2/style_metric`及`controller`。已有best/latest不会以新run覆盖；不接受旧checkpoint初始化。已完成Φ阶段可复用同内容fingerprint的独立Φ文件，仍会检查检索门限与来源。

```bash
# 同版、同目标、同资产与配置的epoch边界续训
RESUME=1 RUN_ROOT=checkpoints/clip_recipe_v2 bash scripts/run_train_clip_recipe_v2.sh

# 接口smoke：2个Φupdates + 2个controller updates；独立_smoke目录。
# 仅这个显式模式将Φ门限设为0，不能据此声称Φ有效。
SMOKE=1 RUN_ROOT=checkpoints/clip_recipe_v2 bash scripts/run_train_clip_recipe_v2.sh
```

有效batch512支持512或256×2/128×4；objective OOM时恢复同一随机状态后缩小microbatch，128仍OOM则停止。梯度裁剪/optimizer每512个anchors执行一次。实际microbatch持久化用于续训。

## GPU利用率与空间换时间

新run默认`--backward-mode combined --reference-cache-device gpu --gpu-data-cache-gb 10 --cuda-field-graph`。训练目标、240帧窗口、8×64 anchors、stride1、FP32/highest/noTF32及任务比例保持不变。

- FM、prior与辅助loss合并一次反传，消除原先同一个field的重复反传。与旧split模式的loss/RNG一致，参数梯度只允许FP32累加顺序的舍入差异；同一执行模式的CPU/CUDA epoch边界续训逐值一致。
- 将train-only normalized FP32 motion375和只读int32 VQ indices驻留GPU。当前数据共8.83 GiB；normalization沿用原NumPy FP32操作，输入逐值不变。Sampler仍使用原CPU generator与style/action权重，完整驻留后不启动DataLoader workers、不逐batch传motion。Indices表包含固定数据库全部索引，但训练motion、references与donors仍只从train取样。
- 冻结Φ reference embedding保留GPU，LRU容量262144；1/3/5个独立exemplar的lookup→VQ decode→Φ分别捕获固定shape图。可学习Es/Gj/φ/P每次更新重算。Φ对已验证的完整固定长度clips使用等价dense路径，减少mask筛选同步。
- Dropout-free field前/后向使用CUDA graph，包含history/trajectory投影，降低Python与kernel launch开销。只在512整批时复用；梯度累积和OOM回退走eager，避免重复forward覆盖同一图的反传缓冲区。推理不使用训练graph。
- Frozen quantize-only保留同一library最近邻输出，跳过perplexity/bincount诊断；donor统计留GPU后按epoch汇总。

10 GiB是只读数据的预算；初始化按当时GPU空闲量再预留2 GiB，空间不足时自动降为indices-only/CPU，分配OOM也回退。当前默认单模型短测峰值约10.3 GiB allocated；不包含其他进程、CUDA上下文或driver占用。可在新run显式调整：

```bash
GPU_DATA_CACHE_GB=10 CUDA_FIELD_GRAPH=1 DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
# 禁用数据驻留或field图进行诊断
GPU_DATA_CACHE_GB=0 CUDA_FIELD_GRAPH=0 DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
```

同版续训自动恢复checkpoint内的上述四项执行参数；旧V2缺少字段时恢复split/CPU-cache/无驻留/无field图。不能用`resume`静默改变梯度累加顺序；新训练直接使用当前默认即可。

交替相同新样本/RNG的100-update成对短测：35.46→23.48 ms/update，耗时下降33.8%，吞吐约1.51倍；进程记录仅有本短测与桌面服务。较早另一项目占用GPU时的48.83→36.25 ms对照保留。它们不是完整epoch耗时或GPU利用率的保证。合并反传的FP32累加顺序会产生微小训练轨迹差异，100-update对照的平均total loss差异最大0.000127。PyTorch2.13的graph stream边界产生一条AccumulateGrad同步warning；梯度/续训测试通过，warning未屏蔽，可用`CUDA_FIELD_GRAPH=0`诊断。详见[本次性能与一致性记录](../analysis/clip_recipe_v2_speed/REPORT.md)。

## Validation、选择与正式test

每5epoch使用90个未镜像validation clips×3固定windows，270 matched/540 cross，240个生成帧、60fps、相同case Gaussian seeds、ODE8/CFG1.5；只在初始帧读取GT motion/latent。Reference仅用train clips，正式100类classifier冻结，排除GT初始帧。

选择先检查matched≥85%、RootADE≤5/10cm，再比较cross acc、matched acc及归一化ADE。`seen_targets_passed`另外记录matched85/cross80是否同时通过；`constraints_passed`不等于unseen70验收。保存`best`、`best_loss`、`latest`及完整选择依据/RNG，先更新best状态再写latest。

Epoch5/10额外用4个固定validation cross cases，对正确reference、同action覆盖的错误reference、null style做240帧rollout；记录条件距离、原目标accuracy、错误目标命中、RootADE、首步连续latent差异及量化后code变化。Clip与导出π的条件逐值核验，所有variants共用noise。

Epoch10诊断门限matched≥75%、cross≥55%、RootADE≤15/20cm，未过则保存`stopped_diagnostic.json`后停止，不自动追加训练。明确希望跑满50时可设`DIAGNOSTIC_EPOCH=0`；总上限仍50。

锁定validation选择的checkpoint后，复用统一test入口：

```bash
FLOW_CKPT=checkpoints/clip_recipe_v2/controller/best.pt \
OUT_ROOT=eval_results/clip_recipe_v2_locked \
bash scripts/run_evaluate_style_rollout.sh
```

Seen recipe builder、OOD recipe exporter、共享模型加载和rollout已接入V2。合并多个exemplar时保留clip_lengths，避免伪造连续reference。旧histogram不能代替V2 temporal序列。Test保持原固定case/noise/cluster CI协议，OOD不用于逐epoch挑模型。达到85/80/70必须由同一个checkpoint、同一个推理设置满足，并同时检查运动质量/轨迹。

## 已测资源与当前限制

真实数据临时模型：10 updates预热+100新样本updates，完成版约48.21ms/update、allocated峰值1373.89MiB；600帧ODE8+CFG约5.13秒，60帧约.52秒。无NaN/Inf、VQ与随机冻结Φ不变、反馈latent/motion匹配。见[完成版smoke](../analysis/clip_recipe_v2_implementation/smoke_final.json)。此前同条件版本44.03ms/update的[初测](../analysis/clip_recipe_v2_implementation/smoke.json)保留；完成版对Es和冻结Φ的输入梯度使用FP32 math attention，并用规则repeat汇总窗口梯度，CPU/CUDA入口均验证了精确续训。这次Φ是随机资源fixture，未验证style质量。

此次实际合成cross覆盖**1.25%**，请求9.75%；unpaired为19.63%。完整window匹配严格，不能把它当作70/10/20已经实现。首先观察实际覆盖与condition diagnostics，不能静默放宽物理门限。8个独立windows也降低每update的style多样性，已用style/action均衡采样控制该风险。

上述48ms是本次效率修订前的历史短测，也不是旧47ms的严格成对提速对照，目标/模型/采样均不同。历史8.97分钟/epoch、7.47小时/50epoch外推不适用于当前默认实现；最新成对结果及显存见上面的效率说明，完整训练主体与总GPU小时仍需实际epoch计时。

回归入口：`bash scripts/run_tests.sh core`、`bash scripts/run_tests.sh gpu`。原实现验收为257 CPU + 16 CUDA，见[代码验收记录](../analysis/clip_recipe_v2_implementation/REPORT.md)；本次修订后**262 CPU + 21 CUDA通过**，见[性能回归记录](../analysis/clip_recipe_v2_speed/REPORT.md)。代码通过不代表三项accuracy已达到；正式训练、完整test与最终BVH结果仍由新run产生。

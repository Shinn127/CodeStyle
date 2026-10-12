# 当前cross/unseen失败的证据与解释

**评估口径更正**：用户已明确正式matched/cross/unseen全部使用autoregressive rollout。以下数字、bootstrap和冻结输入诊断来自teacher-forced，只能用于局部诊断。当前clip/recipe完整rollout已完成：matched62.90%、cross41.72%、unseen1.62%，RootADE约33cm，见[正式结果](../../eval_results/clip_recipe_practical_rollout/REPORT.md)。另有同case validation对照及下一版建议，见[NEXT_SOLUTION](NEXT_SOLUTION.md)。以下历史TF数据保留，不能与rollout成绩混用。训练继续teacher-forced。

2026-10-09。依据50-epoch训练日志、现有test逐样本结果，以及当前best（epoch35）的16-case冻结validation诊断。没有改训练主程序或更新模型权重；扩展了已有诊断脚本的输入/输出参数。所有诊断使用FP32、highest、TF32=False、seed42、240帧/60fps，无rollout。

## 结果没有实质突破，root误差增大

| Test | 上轮clip_recipe150 | 本轮GT-history50 |
|---|---:|---:|
| Matched acc | 98.40% | 98.50% |
| Cross acc | 39.26% | 41.32% |
| Unseen acc | 0.36% | 0.50% |
| Cross RootADE | 7.664cm | 10.632cm |

对相同5000个source/target cases，按90个source_entry_id聚类、1000次paired bootstrap（seed42）：cross acc变化+2.06pp，95%CI [-0.29,+4.40]；RootADE变化+2.967cm，CI [+2.255,+3.710]。准确率增益没有超出本次重采样的不确定性，root退步明确。这个CI只反映case采样，不能替代多训练seed的不确定性；两轮同时改变训练任务、flow-time分布和训练周期，不能把变化单独归因于mask。

Validation cross从epoch5的21.30%升至35的40.56%，随后40/45/50为38.52/38.33/38.15%。所有候选matched ADE均超过5cm门限，所以选出的best为诊断候选，constraints_passed=false。50与150训练量不同，训练不足尚不能完全排除；当前曲线没有支持“原样延长就能达到80%”。

## 历史增强生效，但任务不足以提供可靠的转换监督

Epoch50实际mixed/masked比例30.02%/19.98%，GT target fraction=1，flow-time均值.2999。因此不是增强开关未启用。

Clean/mixed/masked FM loss=.03988/.31150/.31030。后两类约为clean的7.8倍；从epoch1明显下降，但仍是困难任务。这些loss难度不同，比例不能解释为梯度支配或某模块失效。

当前任务将另一风格历史当作扰动，恢复原真实GT下一帧。它保持目标真实，但同action、近似脚部相位不保证精确的一步对应；masked样本又缺少明确body相位。它不是已经验证的、任意source内容和root命令下的目标风格转移配对。FM有风格监督信号，但也有条件不确定性；折中预测是风险，不能仅由MSE证明发生了平均化。

训练仍只有seen目标，foreign history与endpoint均来自同action训练集合；它没有补齐随机cross评估的全部action组合，也没有增加OOD目标监督。

## 失败具有结构：静态姿态较好，复杂动态较差

本轮5000个cross cases中，90个target styles有28个准确率低于10%。例如Superman100%、LawnMower96.08%；BigSteps1.52%、March2%、Morris0%。每style约40–80 cases，且按target选择存在差异，不能将这个划分当作严格因果证明，但结果支持先检查动力学/时序表达。

Reference逐帧VQ后合并为无序histogram，再每slot选4个codes，最终只保留4维加权平均，8slots拼成32维再投影64维。当前100份seen/OOD profile的learned TopK平均保留原hist质量29.41%，与raw TopK支持重合87.62%。保留质量少不等于丢掉同等比例的风格信息，但该路径确实没有保留完整分布和时间顺序。

8个style slots按train style的统计差异选出，不等于已解耦的8个style-only变量：原VQ负责motion/trajectory重建，codes还可携带content、phase与root。不同配方也可能产生接近的码向量均值。P无法恢复已经在统计/聚合时丢掉的信息。

## OOD不只是复制source，更是controller缺少泛化

5000个OOD输出有92.44%被分类为seen style，37.24%保留source style。因此source依赖是因素，却不是全部错误；大量输出转向其他seen模式。

已有31个真实OOD reference的classifier准确率100%、quantized VQ AE为77.42%。这说明VQ能保留其中许多OOD风格，但31例很少，不能将77.42%当作generator上限。

当前16-case诊断包含8 matched、4动态cross、2姿态cross、2 OOD（WildArms/Zombie）；每case固定noise，各种方法不更新权重：

| 输入/解码对照 | matched | 动态cross | 姿态cross | OOD |
|---|---:|---:|---:|---:|
| Source history，clip+quantized | 8/8 | 0/4 | 1/2 | 0/2 |
| Source history，continuous+clip | 8/8 | 0/4 | 2/2 | 0/2 |
| Source history，continuous无clip | 8/8 | 0/4 | 2/2 | 0/2 |
| Source history，quantized无clip | 8/8 | 0/4 | 1/2 | 0/2 |
| Raw TopK绕过learned selector | 8/8 | 0/4 | 1/2 | 0/2 |
| Joint history mask | 7/8 | 0/4 | 1/2 | 0/2 |
| Target GT history直接VQ AE | 8/8 | 3/4 | 2/2 | 2/2 |
| Target GT history经controller | 8/8 | 1/4 | 2/2 | 0/2 |

GT AE加clip/re-quantize后命中数相同。裁剪或quantization影响部分root误差与姿态case，但没有救回这些动态/OOD case。Oracle GT latent与预测latent在所检查的train min/max边界内。

Oracle每帧提供GT，不是可用生成方案；4个动态oracle有3个改变了action/content，所以不能唯一归因于style。上述两个OOD的AE成功、经过controller又失败，至少说明该controller不能可靠保持这些OOD特征。不能继续将全部失败解释成source历史泄漏，也不能将这两例外推为所有OOD的理论上限。

## Prior监督的边界

Prior CE约束selector dense分布；entropy约束选码概率。两者都不直接约束生成动作或生成latent的style统计。Epoch50 KL约5.20，在bounded histogram residual下可有非零可达下界；不能拿label selector的接近零KL作预期，更不能由CE占total比例判断梯度支配。

FM endpoint确实含有GT风格，但当前模型可以通过强历史预测取得很高matched，也可以在未见配方下退回seen输出模式。当前loss没有额外阻止这种OOD退回行为。冻结VQ本身合理，现有证据没有要求联合训练VQ或默认恢复昂贵decoder反传。

## 优先级判断

下一版应先提升reference条件对动态风格/分布的表达，并验证输出是否保持独立reference的风格信息；随后修正转换与维持任务的实际对应关系。保留选码→code aggregation→P→flow→冻结VQ核心，不对齐label feature、不直接优化报告classifier。具体新增约束仍须设计并短训检验，这份诊断不证明某个新loss必然解决80%/OOD目标。

当前不支持仅继续增加mask/noise、替换raw recipe、关闭quantization或原样追加训练作为主要补救。正式test清单和指标口径保持不变。

原始记录：[逐样本及日志审计](audit.json)、[冻结诊断](probe.json)。诊断复现：

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
/home/shinn/miniforge3/envs/mcc/bin/python analysis/clip_recipe_failure_2026-10-09/teacher_forced_probe.py \
  --history-only --flow-ckpt checkpoints/clip_recipe_gt_history50/best.pt \
  --seen-report eval_results/clip_recipe_practical_test/minor_revision_generalization_summary.json \
  --unseen-manifest eval_results/clip_recipe_practical_unseen/case_manifest.json \
  --out-dir analysis/clip_recipe_failure_2026-10-09
```

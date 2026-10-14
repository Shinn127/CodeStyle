# Clip/recipe rollout评估结果

2026-10-09。完整评估已完成，使用clip_recipe_gt_history50的epoch35 best.pt。该checkpoint此前按teacher-forced validation选出，本次是其rollout复评，不称为rollout validation最优模型。

配置：FP32、highest、TF32=False、seed42，60fps/240个生成帧（4秒），Euler8、noise std1，clip→re-quantize→冻结VQ decode。只在初始帧读取GT motion/latent，此后使用生成反馈；trajectory按source提供。classifier不包含GT初始帧。

| 设置 | Cases | Style accuracy | 95% cluster CI | Root ADE | Root ADE 95% CI |
|---|---:|---:|---|---:|---|
| Matched | 1000 | 62.90% | [54.27, 70.53]% | 32.93 cm | [29.10, 37.42] cm |
| Cross | 5000 | 41.72% | [39.41, 43.84]% | 33.28 cm | [29.76, 36.78] cm |
| Unseen | 5000 | 1.62% | [1.16, 2.16]% | 33.04 cm | [29.76, 36.54] cm |

CI按原始source_entry_id聚类，1000次bootstrap；单seed训练的不确定性不包含在这些CI中。

本次沿用1,000个固定、未镜像test source windows。Matched/cross与既有TF评估共用cases；OOD共用这些sources，每source5个固定OOD targets，每style5个独立240帧exemplar，其余clips作real参考。原manifest和checkpoint SHA256核验未变化。执行与最后核验共约8.98分钟。

结果未达到matched≥85%、cross≥80%、unseen接近cross的目标。与同checkpoint的TF结果（98.50%/41.32%/0.50%）相比，rollout matched显著退化，cross大致持平，unseen仍低。各类约33cm Root ADE也表明当前生成反馈下的轨迹保持不足；原因仍需进一步诊断，不能仅凭这些结果唯一归因于某个模块。

Cross被分类为source style的比例3.34%；OOD保留source style比例7.02%，被分类为seen style比例85.80%。低source retention不能单独证明成功转换。

已有label rollout记录为81.90%/74.02%、Root ADE12.53/16.27cm，但其case manifest不同，不进行严格paired差值比较。本次没有额外运行label模型。

本次没有重新训练、扫描checkpoint、改变模型或覆盖历史TF结果。Formal rollout评估结果与以前的TF诊断分别保留。

原始输出：[Seen汇总](seen/minor_revision_generalization_summary.json)、[Seen逐样本](seen/minor_revision_generalization_per_sample.csv)、[Unseen汇总](unseen/summary.json)、[Unseen逐样本](unseen/per_sample.csv)、[执行与核验](execution.json)、[固定cases](case_manifest.json)、[日志](evaluation.log)。

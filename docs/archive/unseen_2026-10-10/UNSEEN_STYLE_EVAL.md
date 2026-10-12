> **2026-10-10 状态**：镜像段已确认导致 mesh 扭曲，训练数据一律不使用镜像；含镜像库上训练的
> VQ/controller/classifier 与派生 cache 已全部删除，代码默认路径与防呆见 [NOMIRROR_CHAIN.md](../../NOMIRROR_CHAIN.md)。
> 本文下列内容为修订前的记录，其中涉及 `--mirror`/`--no-mirror`、`--include-mirror`、`--unmirrored-sources`
> 等开关与已被删除的 checkpoint 的说明不再适用。

# Rollout、recipe与unseen评估

当前主入口为 `run_evaluate_style_rollout.sh`。默认模型 `clip_recipe_gt_history50/best.pt`，固定240生成帧/60fps、Euler8、noise std1、clip→quantize→decode。只在初始帧使用source GT motion/latent，此后反馈生成状态。

Matched1000、cross5000、unseen5000；同一source/target/noise case manifest，5个独立240f exemplars/style，classifier为100类/240f/60fps。分类输入排除GT初始帧，source_entry_id聚类bootstrap1000次。当前结果62.90%/41.72%/1.62%，见[正式报告](../../../eval_results/clip_recipe_practical_rollout/REPORT.md)。

## Conditioning

`clip→frozen VQ indices→完整histogram→learned selector→TopK4 π`，导出的recipe保存最终π，不是raw histogram。π→同一码向量聚合/P→flow；pure recipe提供所有受控slots、strength1、无label embedding。任意recipe不保证可生成物理合理动作；元数据须匹配VQ指纹、slots顺序及selector。

Seen reference只从train、未镜像、独立source_entry_id选择。OOD每style固定5个独立240f exemplar，其余clips用于real参考；OOD不进入generator训练。不会将OOD名称作为generator label，也不将label模型的manual接口当作训练好的recipe模型。

## 执行

```bash
DRY_RUN=1 bash scripts/run_evaluate_style_rollout.sh

FLOW_CKPT=checkpoints/clip_recipe_gt_history50/best.pt \
OUT_ROOT=eval_results/new_locked_rollout \
bash scripts/run_evaluate_style_rollout.sh
```

旧结果受目录检查保护；换模型须新OUT_ROOT。先在validation锁定checkpoint，再test；不按test挑选。Label seen-only可显式设置label checkpoint和 `EVAL_UNSEEN=0`。

单独OOD入口：`evaluate_unseen_style.py`，默认同一个clip/recipe模型，默认rollout。显式reconstruct只作teacher-forced诊断。共享recipe、逐样本recipe、label和切换接口的回归见 `run_tests.sh core`。

Style accuracy、Root ADE、运动质量、local/world foot sliding与FID口径分开；不同extractor的global/style/clip FID不能混用。统计与normalization来自固定VQ/train数据，不从test重估。当前histogram仍丢失时序；[下一版temporal方案](../../../analysis/clip_recipe_failure_2026-10-09/NEXT_SOLUTION.md)未实现。

旧label-based OOD样例、TF训练计划已归档，不能作为当前OOD入口。

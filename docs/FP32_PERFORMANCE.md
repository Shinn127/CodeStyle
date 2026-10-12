> **2026-10-10 状态**：镜像段已确认导致 mesh 扭曲，训练数据一律不使用镜像；含镜像库上训练的
> VQ/controller/classifier 与派生 cache 已全部删除，代码默认路径与防呆见 [NOMIRROR_CHAIN.md](NOMIRROR_CHAIN.md)。
> 本文下列内容为修订前的记录，其中涉及 `--mirror`/`--no-mirror`、`--include-mirror`、`--unmirrored-sources`
> 等开关与已被删除的 checkpoint 的说明不再适用。

# 当前FP32性能

## V2当前默认：2026-10-10

当前主线为temporal clip/recipe V2（256/4blocks/FFN1024、8windows×64anchors）。FP32/highest/noTF32、冻结VQ、无全motion reconstruction及无rollout训练保持不变。

新run默认合并一次反传、GPU reference Φ缓存、10 GiB只读数据预算与CUDA field前后向图。当前train FP32 features及int32 code indices实际8.83 GiB驻留；单模型峰值约10.3 GiB allocated。CPU sampler顺序、reference裁剪、normalization与最近邻code选择均保留。可学习条件模块不跨更新缓存。

交替同样本/RNG的四组25-update新样本流对照：平均35.46→23.48 ms/update，耗时下降33.8%、吞吐1.51倍；进程记录仅有短测和桌面服务。较早另一训练任务占用GPU时48.83→36.25 ms/update的对照也保留。不能据此保证完整epoch速度或外推50epoch总GPU小时。262 CPU + 21 CUDA通过；拆项测量、训练数值差异、冻结状态及精确续训检查见[详细报告](../analysis/clip_recipe_v2_speed/REPORT.md)。

```bash
DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
GPU_DATA_CACHE_GB=10 CUDA_FIELD_GRAPH=1 DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
python analysis/profile_clip_recipe_v2.py --compare --cuda-field-graph \
  --gpu-data-cache-gb 10 --out /tmp/v2_speed_compare_new.json
```

短测限定10个预热和100个测量updates/variant，不保存正式checkpoint；包含真实新sample/reference。Graph仅覆盖单次512整批调用，256×2/128×4和OOM回退使用eager。PyTorch2.13会报告graph stream边界的AccumulateGrad同步warning，未屏蔽；梯度与精确续训均测试通过。`CUDA_FIELD_GRAPH=0`可禁用训练field图。

旧V2同版resume会恢复原执行参数；没有新字段的旧run默认split/CPU reference cache/无驻留/无field图。升级提速策略使用新run，避免静默改变累加顺序。

## 旧六token模型：保留历史记录

六token/hidden256/2layers/FFN1024、有效batch512、highest、TF32=False、冻结VQ、reconstruction关闭。

最新真实新样本流：66.85→47.09ms/update，吞吐约1.42倍，peak allocated 778MiB；recipe命中率同为40.36%，训练指标逐值一致。249 CPU + 13 CUDA回归通过。见[批处理/recipe性能报告](../analysis/pipeline_speed_2026-10-09/REPORT.md)。

此前25.82→17.40ms/update属于重复预热样本的热缓存短测，不能用于完整训练耗时保证；固定270-case rollout validation batch8→64为35.05→9.72秒。原始记录与backend数值差异见[前次性能报告](../analysis/training_speed_2026-10-09/REPORT.md)。

CLI默认 `--worker-batches`、`--fp32-math-attention`、`--style-val-batch-size 64`。Worker输出整批tensor并预读VQ indices；recipe只读受控slots，直接构建稀疏全histogram；VQ encode仅运行encoder/quantizer。Math SDPA仅用于六token训练field，eval保持原后端；不同batch的FP32/quantized feedback可能改变边界case，模型比较固定batch。VQ参数不更新，非有限梯度检查保留。

```bash
python analysis/profile_training.py --streaming --name speed_stream --warmup 100 --updates 100
python analysis/profile_training.py --name speed_warm_cache --warmup 100 --updates 100
```

短测用真实cache/train-only references，controller随机初始化，限定updates、不保存正式权重；顺序执行，GPU不与其他任务共享。`--streaming`使用新的固定shuffle样本；默认模式重放预热样本，通常100%命中recipe bank。此前约2.70小时/50epoch是热缓存训练主体外推，遗漏recipe首次构建成本，应撤回作为实际训练预算的参考。当前未测完整epoch，未来temporal Es/Φ也需另测。

V2已按8window×64transition组织512，并按window去重reference/Es、广播条件；其单次endpoint辅助损失不触发Euler。旧性能数据不代表V2耗时。

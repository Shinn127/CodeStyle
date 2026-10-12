> **历史文档（donor 路线已于 2026-10-12 移除）**：本页记录的 donor/cross-style 合成监督在当前代码中
> 已不存在（objective version 11 起）。保留原文用于回溯当时的设计与结果；现行归因见
> [TF10_DIAGNOSIS.md](../../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

# V2训练效率修订：2026-10-10

代码修订及回归完成。当前真实新样本成对短测为 **35.46→23.48 ms/update**，耗时减少33.8%，吞吐约1.51倍。采用FP32/highest/TF32=False，训练目标、采样单位、有效batch512和50epoch上限保持不变。没有执行正式Φ预训练或controller训练，没有新增style accuracy结论。

## 已确认瓶颈与修改

修订前8-update trace有16次backward、约18118次kernel launch和625次CUDA同步调用；这是带profiler的诊断，不能直接当作正常训练耗时或GPU利用率。

| 瓶颈 | 本次修改 |
|---|---|
| 同一field先FM backward，再辅助loss backward | FM/prior/auxiliary合并一次反传，旧split保留用于明确对照 |
| 240帧motion经过worker IPC、pin-memory、H2D | GPU驻留train-only normalized FP32 features；保留原CPU sampler及其generator |
| reference codes反复CPU复制和传输 | 只读int32 code indices驻留GPU，按原裁剪计划gather，padding不变 |
| 冻结Φ逐reference解码/attention，cache存CPU再上传 | GPU embedding LRU262144；1/3/5 exemplar分别捕获lookup→decode→Φ图，结果clone独立保存 |
| 大量小field kernels逐个由Python启动 | 512整批field前/后向CUDA graph，包含history/trajectory projections |
| 冻结量化仍计算perplexity/bincount | 同一个library quantizer的quantize-only路径，最近邻code和VQ状态逐值一致 |
| GPU donor统计转Python scalar，Φ反复mask过滤 | 统计保留Tensor、固定完整clip用等价dense Φ路径；正常padding入口保留 |

Es、Gj、可学习code映射φ及slot/global投影每次更新重算，不缓存学习模块。没有移除cross/style/root任务、降低FP32精度、扩大有效batch或减少epoch内updates。

GPU field图的保存缓冲区只能支持单次forward后接backward；256×2/128×4明确走eager。OOM缩小microbatch后也走eager，避免重放覆盖前面microbatch的反传输入。CPU模式不使用CUDA图。

## 内存与默认参数

新run默认 `combined / GPU reference cache / GPU data budget10 GiB / field graph enabled`。

当前5,714,320个train帧的FP32 motion375与固定数据库int32 indices，共 **9,476,250,048 bytes（8.83 GiB）**。Normalization沿用原NumPy FP32 subtract/divide，输入逐值相同。只有train motion驻留；虽然只读code表保存整个固定数据库，references、source与donors仍按train split筛选。

预算同时受初始化时空闲显存减2 GiB的限制；空间不足则indices-only或CPU，分配OOM也安全回退。缓存预算不等于模型总显存。最终单模型smoke峰值 **10586.94 MiB allocated（约10.34 GiB）**，不含driver/上下文/其他进程。成对对照同时保留两份临时模型/optimizer，不能据其总峰值推断正式单模型需求。

```bash
DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
GPU_DATA_CACHE_GB=10 CUDA_FIELD_GRAPH=1 DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
```

同版resume恢复checkpoint保存的执行参数；缺少新字段的旧V2恢复split/CPU-cache/无驻留/无field图。其余版本检查仍严格，不能借resume偷偷切换目标/资产/精度或累加策略。

## 测量及限制

设备RTX5070Ti；Torch2.13.0+cu132，seed42。临时随机controller与随机冻结Φ只测成本、接口及状态；真实cache、train sources与独立train references。辅助任务按epoch10的完整权重测量，不是只测最便宜的warmup。

最终[成对原始记录](paired_verified.json)：每variant预热10updates，随后四组25updates；每组同样本/RNG交替执行，顺序每组交换。

| block | 原执行策略 ms/update | 新默认策略 ms/update |
|---|---:|---:|
| 1 | 35.86 | 23.23 |
| 2 | 35.26 | 23.81 |
| 3 | 35.10 | 23.30 |
| 4 | 35.61 | 23.57 |
| 平均 | **35.46** | **23.48** |

对照的GPU进程清单前后均只有本短测及225 MiB桌面服务，另一训练任务已不在清单中；没有暂停或终止其他项目进程。较早[受共享GPU影响的成对记录](paired_final.json)为48.83→36.25ms/update，也保留以披露环境变化。

拆项顺序短测保留：`before.json`40.96、`combined_gpu_cache.json`34.51、`field_graph.json`32.35、`resident.json`31.72、`full.json`35.72ms/update；它们在不同时间测量、部分还在中间代码版本，不能把差值当作各项严格独立收益或挑最小值相加。`dense_graph_resident.json`48.44受到另一GPU任务干扰。最终报告优先使用交替成对测量。

[最终单模型smoke](smoke_final.json)为10次预热+100新样本updates，22.71ms/update；60帧反馈0.461秒、600帧4.584秒，无NaN/Inf，motion严格对应返回latent的decode，VQ/Φ状态无变化。Rollout只是有限的接口检查，没有用于训练或style性能评分。

这些短测不覆盖完整epoch的缓存增长、Φ正式预训练、完整rollout validation和checkpoint I/O，不能保证50epoch总GPU小时。未把profiler的CUDA busy时间或进程显存占用当作SM利用率；本次确定的收益是实测训练吞吐提高。

## 精度、测试与剩余限制

**262 CPU + 21 CUDA通过**，另有5个CPU subtests。日志见[cpu_tests.log](cpu_tests.log)和[gpu_tests.log](gpu_tests.log)。覆盖真实CPU/CUDA入口精确续训、单批graph/eager梯度、梯度累积的安全回退、CPU/GPU缓存一致性、normalization/裁剪/sampler RNG、冻结码本、显存分配失败回退和旧执行参数恢复。脚本dry-run及语法检查也通过。

相同初始权重/输入/RNG的objective loss容差1e-6，梯度容差atol2e-5/rtol3e-4。合并反传改变FP32累加顺序，因此不同执行策略的长期训练不能宣称逐bit相同。成对100-update对照中，四个block平均total loss最大差0.0001264、flow loss0.0000241、style loss0.0002933、root loss0.000729；任务比例与codec code-change统计一致。同一策略的连续训练与epoch边界resume，模型权重和RNG逐值匹配。

Torch2.13在graph与eager的stream边界报告一条AccumulateGrad可能引入同步的warning；没有屏蔽。当前梯度、optimizer及精确续训检查均通过，且成对吞吐仍提高。需要诊断时可设`CUDA_FIELD_GRAPH=0`，GPU只读数据/Φ缓存及合并反传仍可使用。

Style accuracy、完整训练稳定性与85/80/70目标仍需正式训练和锁定checkpoint后的rollout评估；代码/提速验收不替代研究效果验收。

```bash
# 有界成对复测，选新输出路径，不保存正式模型
python analysis/profile_clip_recipe_v2.py --compare --cuda-field-graph \
  --gpu-data-cache-gb 10 --out /tmp/v2_speed_compare_new.json
bash scripts/run_tests.sh
bash scripts/run_tests.sh gpu
```

主要源文件指纹和机器可读摘要见[verification.json](verification.json)。

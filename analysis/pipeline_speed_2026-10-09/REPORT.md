# 训练批处理与recipe首次构建提效

2026-10-09。FP32/highest/TF32=False、冻结VQ150、有效batch512、stride1、原6-token flow网络、原target FM目标。所有测量使用临时随机controller，没有更新正式训练权重。改动不增加rollout训练或reconstruction loss，不解决风格准确率本身。

## 更接近真实训练的测量

RTX5070Ti、Torch2.13.0+cu132；seed42，GT history augmentation与low-t配置保持一致，真实train cache、独立train-only references。每版100 updates预热后，用新的确定性shuffle和随机输入计时100 updates，而非重放预热的样本。计时包括DataLoader与recipe准备，排除随后8步profiler开销。

| 新样本流 | Recipe优化前 | 最终版 |
|---|---:|---:|
| ms/update | 66.85 | **47.09** |
| updates/s | 14.96 | **21.24** |
| recipe bank命中率 | 40.36% | 40.36% |
| 初次预热ms/update | 87.78 | 60.30 |
| 初次预热recipe命中率 | 15.875% | 15.875% |
| peak allocated显存 | 778.46MiB | 778.46MiB |

耗时下降29.56%，吞吐提高41.96%。这里的“优化前”已经启用了本轮较早的worker批处理及encoder-only修复，比较只反映后续recipe首次构建优化。全部返回训练指标逐值一致，包括loss、prior KL、histogram/selector统计、实际history比例和cache命中率。原始记录见[前](streaming_before.json)、[后](streaming_final.json)、[汇总](comparison.json)。

这些是有限新样本短测，包含不同程度的缓存未命中；完整epoch的bank覆盖、LRU淘汰和系统调度还会影响速度，未把此结果外推为50epoch GPU小时保证。未来temporal Es/Φ也必须另测。

## 主要改动

1. `VQ.encode()`只执行encoder与quantizer，保留原latent/indices/perplexity和输入形状。原代码调用完整forward，获取latent时会运行decoder；只监测公开decode的旧测试漏掉了它。现在直接监测decoder模块，并禁止实际TF入口调用内部decode。
2. DataLoader worker直接输出整批x/labels/start/segment tensors，不再创建512份样本字典与视图再stack。worker同时预读对应cache indices，pin memory后异步上传；主线程直接codebook lookup。cache只以路径/shape/dtype传入，worker懒打开memmap，序列化时清除文件映射。其他训练器仍使用原Dataset默认路径。
3. 纯clip/recipe、GT目标label未改变时，使用已有CPU labels组织references，避免GPU→CPU回读。目标decoupling路径继续读最终labels。保留原GPU随机draw顺序和候选/clip/source排除语义。
4. 构建reference recipe仅读取受控slots，240f crops改为连续slice，避免复制全部32个slots及临时frame ID数组。全histogram直接存储每slot实际出现的code与FP32概率，避免构建8×2048稠密表后两次扫描nonzero。选码、码槽顺序、exemplar数量、私有seed、TopK控制、LRU及style prior均保持原语义。
5. 性能脚本增加`--streaming`、初次预热时间/命中率及Python profiler，区分首次构建与热缓存field速度。core回归纳入新批处理测试。

单独1200帧/8slots/2048 codes的1000次参考profile构建，dense+两次nonzero约169ms，直接sparse约92ms；这是局部微测。profile位于[前端优化前](crop_python_profile.txt)与[最终](final_python_profile.txt)，有profiler及多线程开销，不能作为训练wall time。

## 热缓存测量的局限

最早重复预热样本的短测26.89→16.35ms/update；后续复测旧collate19.51、新路径18.94ms/update，存在明显系统波动。不能把首轮64%的吞吐提升作为稳定承诺。它们都100%命中recipe bank，遗漏真实训练首次构建开销。此前约2.70小时/50epoch的热缓存外推也不再作为实际预算参考。

尝试的独立CUDA transfer stream没有额外收益，已移除；全slot合并bincount的局部微测也更慢，没有保留。保留实测有益的受控slot读取与稀疏profile实现。

## 正确性验证与使用

最终**249 CPU + 13 CUDA通过**，另有5个subtests；只使用临时小数据。见[CPU日志](cpu_tests.log)、[CUDA日志](gpu_tests.log)。

- VQ encoder-only输出与原forward的三项输出CPU/CUDA逐值一致；VQ冻结、无梯度、状态不变。
- 单步、长窗口拆成TF transitions、三步展开：原cache查表与worker预读路径的全部指标、梯度、权重、CPU/CUDA RNG逐值一致，包含microbatch累积。
- sparse与dense reference全histogram在不同帧数、码本规模、非顺序slots、cache miss与LRU淘汰下逐值一致，recipe采样/RNG不变；缓存与在线编码、train/source隔离原测试通过。
- 实际CUDA parser/optimizer/save/load入口：连续训练与epoch边界续训的模型、optimizer及CPU/CUDA RNG逐值一致。

默认`--worker-batches`，可`--no-worker-batches`对照原collate；语义一致，不要求重建VQ cache/prior或重训已有模型。已有backend/目标版本/数据指纹的resume限制继续适用。

```bash
python analysis/profile_training.py --streaming --name fresh_stream --warmup 100 --updates 100
python analysis/profile_training.py --name warm_prefix --warmup 100 --updates 100
bash scripts/run_tests.sh core
bash scripts/run_tests.sh gpu
```

脚本结果使用独立名称，拒绝覆盖。正式训练仍从用户的既定入口启动；这次性能修复不构成重跑已失败history50方案的理由。

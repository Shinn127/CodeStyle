# FP32训练效率修复与验证

**后续核验更正**：本报告训练计时是重复预热样本的热缓存结果，recipe命中100%，不能代表完整训练成本；2.70小时/50epoch不再作为预算参考。另发现原`VQ.encode()`调用完整forward，会运行内部decoder；当时只禁止了公开`decode()`，因此先前“零decoder”的检查范围不足。该冗余计算已修复，并新增内部decoder检查。最新新样本流、真实cache miss开销与回归结果见[后续报告](../pipeline_speed_2026-10-09/REPORT.md)。下述历史数字保留，不作为实际总训练速度承诺。

2026-10-09。本次已修改训练/验证主路径并完成短测，没有运行完整epoch、训练新的正式模型或更新历史checkpoint。提速不代表matched/cross/unseen准确率提高；新temporal Es与独立输出风格约束仍按[NEXT_SOLUTION](../clip_recipe_failure_2026-10-09/NEXT_SOLUTION.md)待实施。

## 实测收益

RTX5070Ti/16GB，Torch2.13.0+cu132，FP32、matmul_precision=highest、TF32=False；有效batch512、seed42、原6-token/hidden256/2layers/FFN1024。真实train cache、train-only donor与recipe，临时随机controller。优化前快照与最终版均256updates预热、256updates计时，计时区域recipe cache命中100%；不包括profiler开销。

| 项目 | 修改前 | 最终版 |
|---|---:|---:|
| ms/optimizer update | 25.82 | **17.40** |
| updates/s | 38.74 | **57.46** |
| peak allocated memory | 1084.19MiB | **778.46MiB** |
| 按11158updates外推训练主体min/epoch | 4.80 | **3.24** |
| 外推50epoch训练主体GPU小时 | 4.00 | **2.70** |

吞吐约1.48倍、计时耗时下降32.6%。这些是短测与外推，不含完整epoch I/O、保存、冷缓存、验证或未来Es/Φ，不能作为新模型总GPU小时保证。较早相同配置短测为18.84ms/update，正常运行有波动。最终benchmark预热期间曾与小fixture回归测试并行，**计时区域在其结束后独占GPU**；该次cold_ms不作前后成本比较。

相同90 validation clips、每clip一个240f窗口、90 matched/180 cross的rollout验证：batch8为35.05秒，batch64为**9.72秒**，约3.61倍。每5epoch正式810-case验证仍需首轮实测，不能简单把270-case时间当作实际810-case耗时。

Manifest、reference clips、target、Gaussian case seeds与模型权重相同，VQ状态不变。不同batch的FP32 GEMM/quantized反馈产生数值差异：matched判定90例不变，cross有1/180例改变（37.78%→38.33%）；RootADE为29.28/28.35→29.25/28.25cm。**这不是风格能力改善，不宣称输出逐值一致**。新训练固定validation batch64并写入日志，模型比较应固定该参数；老记录保留。

原始记录：[修改前256](fp32_reference256_20261009.json)、[最终256](fp32_final256_20261009.json)、[profile](fp32_final256_20261009_profile.txt)、[batch64验证](validation_batch64.json)。PyTorch allocated显存不是nvidia-smi reserved/进程占用。

## 修改内容

1. GPU历史donor检索改为一批距离矩阵，保留原action分组的每次随机draw形状、顺序及style均匀选择。缓存bank相位范数与action掩码；缓存用Tensor版本检查，在原地修改/fixture修改group后刷新。原CPU路径保留，GT endpoint、root retarget与实际task比例未改。
2. Pure bank-recipe的CPU随机整数批量生成。以候选数与bank size的最小公倍数采样再取模，在当前小整数CPU RNG下保留每个旧scalar结果及完整RNG状态。非bank/非pure/大范围分支继续原路径。候选排除集合缓存有4096条上限；独立clip/source/donor隔离不变。
3. 冻结codebook的geometry、style-slot身份和prior entropy缓存，按buffer修改/设备/存储变化刷新。复用dense CE计算真实KL与共享selector entropy，优化项的CE与梯度不变；entropy/KL各自保留1e-8/1e-12 clamp语义。
4. Recipe sparse profile和donor数据H2D使用non_blocking；仅减少传输等待，必须读取GPU结果的D2H仍同步。原fused AdamW、grad clipping与逐update非有限梯度检查均保留。首次训练与resume均显式使用`foreach=False`的fused配置，修正二者此前`None/False`配置记录不一致。
5. 六token的FP32 **训练field**使用math SDPA，减少memory-efficient attention 64×64 tile的浪费。推理/eval继续原路径；不改变网络宽度、layers、token数或TF32。此优化适用于当前短token网络，不将未来60-token Es全局强制成math attention。
6. Style validation batch可配置，默认64，不改变240帧、rollout、初始GT策略、ODE8或recipe条件。支持较小显存机器显式设为32/16/8。
7. 新增`compute_loss(return_endpoint=True)`：返回单次FM field evaluation的可导raw endpoint latent。target和velocity都支持，不调用Euler或decoder。下一版style/root辅助loss可只decode这个终点；旧`return_prediction=True`仍保留其Euler路径，不应混用。

同样16个profiled updates：cudaStreamSynchronize **945→293**（约59.1→18.3/update），cudaLaunchKernel **20409→9847**，CPU randint **9430→16**。计数包含profiling内部运行，不是100% GPU utilization的保证；主路径仍有必要D2H、CPU recipe组织和数据读取。

## 正确性检查

16个真实512 batch，优化前后固定输入与RNG：donor选取、root/motion、target相关历史latent、recipe与CPU/CUDA RNG **逐值一致**。保留原attention后端时全部controller gradients最大差为0，返回字段最大差9.54e-7（KL/entropy统计重排）；冻结VQ不变。

开启math SDPA后仍保持上述样本与RNG一致。全部返回字段（含norm日志）最大差1.72e-5；梯度最大分量差3.19e-6、全梯度相对L2最大差8.39e-6，均属FP32后端数值差异，不宣称训练轨迹逐位不变。配置/resume显式检查attention选项。

测试：相关CPU套件123 passed、5个CUDA测试跳过；宿主CPU/CUDA相关套件47 passed。新增endpoint接口后的相关套件33 passed、3个CUDA测试跳过。最终新增效率组在宿主环境**12 passed**，包含target/velocity各一次field、零Euler/decoder及输入梯度，及真实CUDA入口连续训练/epoch边界恢复的model、optimizer、CPU/CUDA RNG逐值一致。套件有重叠，不累加为独立测试数。覆盖缓存失效、scalar RNG等价、真实训练/validation/保存、epoch边界续训、recipe/OOD接口、冻结VQ与展开。Python编译、shell语法、相关tracked diff检查通过。

原始核验：[原后端逐值检查](equivalence.json)、[math梯度检查](equivalence_math.json)、[CPU套件](fp32_pipeline_cpu_tests_20261009.log)、[宿主相关套件](fp32_pipeline_cuda_tests_20261009.log)、[最终效率/续训测试](fp32_efficiency_final_cuda_20261009.log)。

没有采用额外CUDA graph：实测没有加速。当前Inductor导入有duplicate flex_attention template错误；未改全局Torch环境或默认依赖compile。限制主进程4个CPU线程的试验未改善steady-state，不作为默认配置。

## 使用与下一版成本边界

训练CLI新增并默认启用`--fp32-math-attention`，可用`--no-fp32-math-attention`保持原后端；`--style-val-batch-size`默认64，wrapper可设`STYLE_VAL_BATCH_SIZE`。既有clip/recipe wrapper已显式传参，`DRY_RUN=1`可检查。

这是性能修复，不建议为了这次提速重跑已经失败的原history50方案。正式下一版仍须新RUN_ROOT、从零训练、最多50epoch。历史checkpoint保持可读；resume不能悄悄改变attention/validation batch：旧config缺失分别按原后端与batch8解释。原目标版本检查仍适用，旧TF最佳选型状态不能混入新rollout选型。

下一版Es/Φ的主要效率要求是**8个完整window×64个transition**组成有效512，reference与Es/Gj/P按window去重并广播。仅缓存冻结VQ及冻结Φ；可学习Es输出不得静态缓存。保持stride1和11158个update/epoch预算，避免长窗口按旧batch定义使每epoch多出64倍工作。

风格/root辅助loss只在选定完整窗口上读一次预测终点，初始四分之一窗口；不使用`return_prediction=True`，不做Euler或rollout训练。新Es/Φ吞吐必须另做100-update实测后再估算50epochGPU小时，本次2.70小时不能直接用于那个尚未实现的新模型。

复现当前短测：

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
python analysis/profile_training.py --reference --name repeat_before --warmup 100 --updates 100
python analysis/profile_training.py --name repeat_after --warmup 100 --updates 100
python analysis/training_speed_2026-10-09/verify_equivalence.py --math-attention
```

当前benchmark读取checkpoint配置与固定统计，不继承controller权重/optimizer；先后顺序运行，不能同时争用GPU。若要复用本报告的原始统计配置，显式设置`--config-checkpoint checkpoints/clip_recipe_scratch150/latest.pt`。原2026-10-07 benchmark脚本及历史分支已归档；本报告的当前结果JSON与profile表保留在本目录。

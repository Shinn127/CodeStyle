> **历史文档（donor 路线已于 2026-10-12 移除）**：本页记录的 donor/cross-style 合成监督在当前代码中
> 已不存在（objective version 11 起）。保留原文用于回溯当时的设计与结果；现行归因见
> [TF10_DIAGNOSIS.md](../../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

# Clip/recipe V2代码验收

2026-10-10。方法实现、回归与有界真实数据smoke已完成；正式Φ预训练、50epoch controller训练及85/80/70联合指标尚未验证。详细配置与启动命令见[运行说明](../../docs/CLIP_RECIPE_V2.md)。

新增时序clip→π接口、非线性code聚合、8 slot/global条件、4-block逐层AdaLN/cross-attention、受训null-style CFG、独立train-only Φ入口、quantized单次endpoint辅助loss、完整window配对与unpaired任务、同版续训及rollout选型。模型前后向FP32/highest/noTF32，VQ与Φ冻结。没有按OOD test结果选择checkpoint。

完整回归：**257 CPU + 16 CUDA**，另5 subtests；[CPU](cpu_tests.log)、[CUDA](gpu_tests.log)。覆盖clip导出π与manual条件、时序敏感性、CFG共享输入及null分支、预测latent/motion反馈一致、辅助梯度而不执行Euler、VQ/Φ冻结、unpaired FM=0、train/source/donor隔离、完整window root保留、真实导出/加载/rollout验证入口和CPU/CUDA精确epoch续训。旧路径回归保留。

真实数据10 updates预热+100新样本updates，完成版[smoke](smoke_final.json)：48.21ms/update，allocated峰值1373.89MiB，effective512/micro512；60帧.52秒、600帧5.13秒，ODE8/CFG1.5，反馈匹配、无NaN/Inf，VQ与随机Φ状态不变。该Φ是随机资源fixture，不是有效style度量或训练效果证据；没有保存正式模型。

完整window请求paired cross9.75%，实际1.25%，unpaired19.63%；未把配对短缺伪装成请求比例已经实现。约四分之一auxiliary windows和全部unpaired会计算量化后style/root约束。较低的配对覆盖需结合实际训练诊断判断，不自动放宽物理门限。

资源数字不保证完整epoch或未来训练曲线。初测44.03ms/update的[原始记录](smoke.json)保留；完成版使用FP32 math attention覆盖Es/冻结Φ输入梯度，并用规则repeat汇总共享窗口条件，已验证GPU精确续训。

[实施状态与来源SHA256](implementation.json)记录本次代码版本。新入口`run_train_clip_recipe_v2.sh`先最多1000 updates的Φ预训练，通过IND-held-style检索门限后controller从零最多50epoch；epoch10未通过预设rollout诊断门限则停止。输入统计仅train估计，现存VQ cache/prior继续使用并校验内容指纹。

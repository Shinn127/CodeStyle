# 训练与交互质量诊断（2026-10-12）

追加深入代码审计与不新增loss方案见[NO_EXTRA_LOSS_SOLUTION.md](NO_EXTRA_LOSS_SOLUTION.md)：真实训练batch的cross相位错配、decoder velocity与生成位置差分的不一致、2/4/8/16步轨迹，以及强相关noise导致滑行的反例。优先修条件与配对，原第6节的新增时序loss建议暂不作为第一轮方案。

结论：最新训练提升了风格识别，但未解决运动质量。抖动和大跳变主要在 flow 输出量化前就已存在；低噪声又会诱发姿态冻结滑行。训练/推理反馈求解器不一致和偏重 accuracy 的选型使问题加重。显示层 IK 能减轻部分滑动，但接触识别依赖已经不稳定的生成脚速，无法完整兜底。

这次仅运行四条 300 帧只读 GPU 探针、已有录制 CPU 重放与 32 项轻量回归测试，没有训练、完整 benchmark 或改动采样默认值/VQ/模型权重。

## 1. 实际训练与选型

新 run：`checkpoints/flowmatch_label_6tok_curriculum100`，从零 100 epochs，6-token transformer，batch1024，FP32，target/normalized，noise std1，unroll4，前30epoch TF、31–40 ramp、随后 rollout。训练反馈 Euler2、验证/交互 Euler8；recon/root/style consistency 权重均0。

| validation（240f rollout，270 matched/540 cross） | matched acc | cross acc | matched ADE cm | cross ADE cm |
|---|---:|---:|---:|---:|
| best，epoch65 | 92.96% | 90.56% | 8.53 | 10.91 |
| epoch80 | 91.48% | 90.56% | 7.78 | 9.65 |
| latest，epoch100 | 91.85% | 90.19% | 6.85 | 9.37 |

20次验证均未通过 matched ADE≤5cm 的约束。best 是未过约束的诊断候选；选型先比较 cross acc，再 matched acc，最后 root error，因此 cross acc相同的epoch65优于epoch80，即使后者root更好。选型不看 jitter、足滑、步态冻结或姿态跳变。风格准确率本身不能验收视觉质量。此处均为 validation，不是新增 test 或 unseen 结果。

交互 `.sh` 与 Python 默认仍加载旧 `flowmatch_label_nomirror_150ep/best.pt`。只有显式 `FLOW_CKPT`/`--flow-ckpt` 才用新模型；默认 noise std1、requantize ON、postprocess ON、IK ON。启动终端 `[Run]`、JSON 的 checkpoint/noise_std/feedback 可以核实实际配置。没有擅自切换默认模型，因为新模型运动质量未证实更优。

已有相同seed42、90个240f窗口的 rollout 探针：新best std1 的 local jitter/F.S.=1.624/1.507cm，旧150ep=0.802/0.811cm。std0.25时分别0.456/0.618与0.283/0.448。每style一个窗口，仅作同case诊断；原始数据在 `analysis/motion_quality_diagnosis/probes/{6tok_ns10,6tok_ns025,tr6_ns10,tr6_ns025}/`。

## 2. 跳变在量化前发生，噪声会驱动不同运动分支

新best，Neutral，seed42，固定按键序列：idle30f→前进120f→前进转向60f→前进60f→侧移30f，300f/60fps/Euler8。风格全程不变，不涉及切换条件插值。每一步同时保存 D(clip(z)) 与 D(Q(clip(z)))，实际反馈始终为后者。

| 同case设置 | 连续 local 二阶差分 cm | 量化 local 二阶差分 cm | 最大 local 单帧步长 cm | 滑行占比（行走/转弯段） |
|---|---:|---:|---:|---:|
| best，独立 noise std1 | 2.753 | 2.700 | 16.51 | 0.42% |
| best，独立 noise std0.25 | 0.330 | 0.344 | 1.90 | 10.55% |
| best，std1，诊断 AR(1) rho0.95 | 0.774 | 0.812 | 10.61 | 9.28% |
| latest，独立 noise std0.25 | 1.471 | 1.418 | 9.89 | 9.28% |

连续和量化在同一步使用相同历史/trajectory/noise，所以第一行直接反驳“跳变主要由最近邻量化造成”。不代表量化完全无影响，也不等于 continuous反馈闭环的全序列结果。

现有同case录制中，std0.5在恒定前进段 frame61 就出现平均关节位置13.21cm跳变，另一处frame124为12.92cm；既无style切换，也不在按键改变边界。旋转单帧最大约69.85°。6D输入第一轴norm最小0.936、两轴cross norm最小0.878，未出现接近零的6D退化，因此这些大跳变不是quat符号、NaN或退化旋转转换造成的。

AR(1)探针保持标准高斯边缘，只改时序相关性，jitter下降但大跳变和滑行仍存在。因此不能用“低噪声/相关噪声即可根治”替代步态稳定性训练。它是单case假设验证，不是已可推广的新推理设置。

## 3. 训练反馈 Euler2 与推理 Euler8 不一致

冻结 best/std1 在frame60后的同一历史、轨迹、style，抽32个固定noise，只改变求解步数：

| 步数 | 连续输出pose离散程度 cm | 量化输出pose离散程度 cm | 同端点连续→量化pose差 cm |
|---|---:|---:|---:|
| 2 | 0.416 | 0.441 | 0.165 |
| 8 | 5.915 | 5.954 | 0.341 |

离散程度为跨noise的3D关节位置std范数均值。这是同一固定历史的直接反事实对照，显示便宜2步反馈没有暴露8步输出的同等多分支变化；单次FM loss虽不依赖solver，训练看到的后续条件分布却依赖solver。不能由此宣称所有帧都放大14倍，也不能把推理改2步当作质量已验证的修复。

训练 unroll4只覆盖4/60秒，反馈detach，没有显式跨时间的运动学一致性/支撑脚约束。起始帧还使用GT motion，而闭环反馈使用D(q)（`train_feedback_context=false`）。这些是代码确认的覆盖缺口；它们对最终质量的各自贡献尚未单独消融。

## 4. 为什么滑行、IK不够有效

std0.25时new best仍有10.55%行走段满足“root>0.4m/s、去掉Hips平移后的pose变化<0.3cm/frame”；std0.05的已有同case录制为55.27%，最长80帧。恢复motion token并未消除低噪声吸引子。旧模型与新模型还同时改变了训练课程/batch/lr/epoch，之前把现象归为“无motion token决定性原因”并没有独立消融支撑，应收回该强因果结论。

实际Geno后处理在生成root上的同case重放：std1时IK active占40.67%脚帧，完全blend占29.67%；std0.25时54.83%/46.50%。输入脚接近地面但水平速度≥0.7m/s、会被接触入锁速度门限拒绝的比例，分别68.90%和48.83%。过滤、锁定18cm修正上限、20cm拉扯解锁、腿长可达范围进一步限制补偿。模型已经滑行或脚部剧烈抖动时，仅启用IK不等于可靠检测支撑。

std1的mesh姿态world二阶差分代理3.815→1.750cm，仍有19.37cm的最大显示单帧步长；std0.25为0.351cm，但不能因此宣称脚稳定或步态正确。采用同一显示脚底高度掩码的地面代理速度也仍很高。这个掩码受地面修正影响，不能当成真实contact或论文F.S.，仅帮助辨认显示滑动。原始位置与固定FK的ankle差均值约0.18–0.22cm，说明骨长重建不是这一case的主要误差。

Root积分已与motion375现有定义验证一致（前一帧vx/yaw/vz，deg/s，60fps）；反馈motion严格对应返回latent；平滑/IK不写回生成器。没有发现坐标正负、root直接覆盖或VQ更新造成当前问题。

## 5. 训练日志缺陷已修正

`_run_epoch`累积顺序是 `[matched_loss_sum,cross_loss_sum,matched_count,cross_count]`，读取时误用交错顺序，导致 matched_samples成为cross loss sum，cross_flow_loss成为count比。已经修正，32项feedback/interactive/postprocess测试通过，含真实epoch日志断言、同环境续训一致性；不改变loss、梯度、RNG或checkpoint权重。

历史记录可从已有task指标恢复正确值，不能覆盖历史日志：ep100 matched/cross flow MSE实际为0.03739/0.25854，样本数2,073,742/572,274，实际cross21.63%，总flow0.08522。cross目标仍明显更难。原日志3.62等“cross loss”不能用于讨论梯度支配。

## 6. 建议顺序

1. **先修训练/选型一致性，再重训**：feedback solver与主推理统一8步（可用较小真实反馈子集抵消成本，不能默认认为2步等价）；起始观测使用冻结D(q)；验证新增相同noise下的jitter、接触覆盖与world足滑、跳变尾部、固定按键步态测试。style/ADE联合约束都未过时明确保留诊断身份。
2. **补时序/运动学约束，不启用全部291维recon**：优先小权重支撑脚world velocity一致性、短时pose差分/旋转一致性或输出joint velocity与相邻pose变化一致性。VQ继续冻结；如要decode生成latent计算loss，需保留输入梯度而不更新VQ。也可先做latent增量正则的低成本诊断，但它不直接保证脚锁稳，不应先承诺有效。
3. **训练覆盖实际采样条件**：若继续使用独立std1正式协议，应让模型在std1下时序稳定；若交互采用低幅/相关noise，则训练反馈同步覆盖该路径并单独评估低噪声闭环。检查真实GT未来轨迹与键盘外推条件错配，尤其起步/急转/侧移。当前没有独立测出planner错配的贡献。
4. **显示侧辅助**：基于多帧高度/运动/抬脚趋势的接触置信度和迟滞，而不是仅依赖已滑动脚的低速门限；分辨IK enabled/接触候选/完全锁定与释放原因。不要直接抬高速度阈值强锁双脚，否则会误锁摆动相、扭膝或积累拉扯跳变。flat-ground IK不能替代正确步态生成。

不建议直接增加epoch、继续加大平滑、单独关闭量化，或据风格accuracy选最新模型作为视觉质量改进。现有证据尚不支持这几种替代。

## 复现与产物

```bash
/home/shinn/miniforge3/envs/mcc/bin/python analysis/motion_quality_diagnosis/check_interactive_quality.py \
  --checkpoint checkpoints/flowmatch_label_6tok_curriculum100/best.pt --noise-std 1 \
  --output analysis/interactive_quality_2026-10-12/recheck_std1
```

`best_std1/`、`best_std025/`、`best_std1_rho095/`、`latest_std025/`各有summary与逐帧trace；checkpoint内容SHA写在每份summary，VQ状态前后相同。`render_replay.json`保存显示门限与FK检查，`tests.log`保存32项测试。两个best独立噪声录制与先前已存session的raw_motion逐值一致。单style/单seed/5秒探针没有证明跨style/长时/实际显示帧率的全面质量；未执行原生GUI或正式accuracy测试。

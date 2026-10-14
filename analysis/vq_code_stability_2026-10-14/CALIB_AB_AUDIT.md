# Decoder 校准 A/B 独立复核

2026-10-14。对象：提交 `ab05397` 的代码和三份实际 best checkpoint。
本轮只做诊断，没有 optimizer update、正式训练、VQ/controller 替换。原结果报告保留。

**结论：不推广 A/B 的决定正确；“已证明单帧 decoder 无法解决，必须改表示/加历史”的归因不成立。实验还存在 normalization 与采样顺序两处偏离预设的控制变量。** 这些偏差需要修复，但现有证据也不支持“修掉它们就能成功”。

## 1. 实际验证

- 基线 best 实际 epoch90；A/B best 都是 epoch5。
- 三份 encoder+quantizer 参数/缓冲的含 dtype、shape 的内容指纹相同；复核前后完整模型 state 未变化。
- 冻结 teacher 目标确实是 `D0(E0(x))`，没有误用量化 teacher，student loss 对 `D_student(q)` 有效。
- 现有 `tests/test_decoder_only_calibration.py`：CPU **9 passed**。这些单元测试没有覆盖下面的 normalization、采样顺序与真实 resume 入口问题。
- 独立脚本：`audit_calibration.py`，结果 `calibration_independent_audit.json`。
- CPU FP32/highest、两线程、TF32 关闭；与原探针相同 seed42 的12个 validation clip，每片段122帧（120个二阶差分）。原报告写“120帧”不够准确。
- 梯度检查：真实 train 数据，4个独立 B32×T4 batch，无 optimizer update。不代表完整训练梯度或所有样本。

## 2. 两处实验契约偏差

### 2.1 没有固定 normalization

A/B `config.json` 的 `norm_stats=null`。数据管线按 seq_len4 的 train windows 重算，而基线是 seq_len2 的统计值。

实际数组比较：

- mean 最大绝对差约0.001219（各通道单位不同，不是统一厘米）；std 最大绝对差约0.014915。
- 更有意义的相对量：最大 mean 差 / 基线 std = 0.000291；std 最大相对变化约0.0546%。
- 相同12个val片段、同一冻结 E/Q、同设备与编码 batch，换用 A/B norm 后 **0.158% slot 选码改变，4.713% 帧至少一个 slot 改变**。

所以权重冻结并不等于完整映射冻结：`q(x)=Q(E((x-mu)/sigma))` 仍依赖 norm。A/B 共用新 norm，彼此这项一致；但与基线的比较不满足“仅 decoder 变化”。

原 residual probe 硬编码基线 norm；训练内 eval 用 A/B 自己的 norm。二者不属于相同 pipeline。补测 B 的 GT 二阶误差：用基线 norm 为0.40017cm，用本次训练 norm 为0.39937cm，基线0.39770cm。**这个偏差很小，修正后也没有明显改善，不能把整体失败都归咎于 norm。**

### 2.2 seed 相同不代表相同采样顺序

`dataset._build_loader_for_split()` 没有独立 `generator`。B 在开始迭代前额外 `build_frozen_teacher()`，内部随机初始化网络后加载权重，消耗全局 Torch RNG；A 没有这一步。

独立复现证实 teacher 构建改变 RNG，随后相同 `randperm` 的序列不同。正式运行未保存 sample-order manifest，不能恢复声称完全配对的数据顺序。A/B teacher 构建还可能改变随后 worker 的 seed。

修正方式：隔离 teacher 构建 RNG，并给 train/val sampler/loader 独立且显式的 generator；如支持严格续训，保存其状态。不要只在日志写 seed42。

## 3. 补测真正的冻结 teacher 残差与 GT 误差

原 probe 的 `r_self=P(D_theta(q))-P(D_theta(z_e))` 是合法的当前模型量化残差，但不是 B 的冻结目标误差。

新增 `r_fixed=P(D_theta(q))-P(D0(z_e))`，以及 `e_GT=P(D_theta(q))-p_GT`。以下统一用基线 norm，local24关节，cm，固定同一 q 和 z_e，均值：

| 指标 | 基线 | A | B |
|---|---:|---:|---:|
| 原 self 残差二阶范数 | 0.38990 | 0.39443 | 0.39350 |
| 冻结 teacher 位置残差 | 0.25644 | 0.32464 | 0.31577 |
| 冻结 teacher 残差二阶范数 | 0.38990 | 0.39380 | 0.39296 |
| GT 位置误差 | 1.06965 | 1.04979 | 1.05236 |
| GT 一阶误差 | 0.26356 | 0.26478 | 0.26437 |
| GT 二阶误差 | 0.39770 | 0.40105 | 0.40017 |
| 冻结 teacher normalized291 MSE | 0.001201 | 0.001716 | 0.001625 |

位置拟合略好，时间误差没有改善；改用正确的固定 teacher 探针，仍然是负结果。

但不能说 teacher “任何方面都无增益”：B 相对 A 的 teacher MSE 低约5.3%，原90窗口 RootADE 低约2.2%，后者也并非报告所称的“全部指标≤1%”。这些收益不足以满足抖动/足滑验收。

按 source_entry_id 配对、1000次 bootstrap、seed42：B−A 的 GT 二阶误差差值为−0.00088cm，95%区间[−0.00197, +0.00012]；B−基线为+0.00247cm，区间[−0.00439,+0.00856]。这里支持“未观察到明确时间收益”，不支持证明严格相等或必然变差。仅12个source，且不包含训练 seed 的不确定性。

## 4. Teacher 梯度与训练目标

对 decoder 参数计算实际加权梯度：

`g_GT = grad(3 L_recon + 15 L_vel + 3 L_root)`

`g_teacher = grad(2 L_teacher_MSE)`

| 检查模型 | mean ||g_teacher|| / ||g_GT|| | mean cosine(g_teacher,g_GT) |
|---|---:|---:|
| 基线初始化 | 1.63% | 0.111 |
| A 最终 | 2.99% | 0.087 |
| B 最终 | 2.63% | 0.082 |

本次实际检查支持“teacher 在这些 batch 上偏弱”，但不能由 loss 占比0.4%直接推出。余弦接近0也不支持用“梯度高度重合”解释它。不能据此机械放大几十倍：teacher 有自身误差，强拉 teacher 可能牺牲 GT/风格。

B 的训练日志 teacher loss：epoch1 0.001068 → epoch5 0.000934（约−12.5%）；validation：0.003483 → 0.003452（约−0.9%）。没有 epoch0 同口径全val记录，不能推断整个训练相对初始化的全val变化；能看到的是后续训练下降未明显转化为 val 收益。

更根本的目标问题：normalized291逐帧MSE混合root/位置/旋转/速度；它并不直接强调位置的高频误差和接触行为。`||Delta^2 r||_2 <= 4 ||r||_2` 仅是上界，不能推出训练MSE略降就能降低jitter。相同大小的逐帧残差可以是平滑偏移，也可以是逐帧交替的抖动。

我先前给 lambda=2 作为起点，并建议训练前检查梯度；现在看这个候选对当前主监督很弱，且逐帧目标对高频问题不够直接。这是方案假设未兑现，不能全部解释成执行问题。

## 5. 报告中不成立的推论及代码隐患

- `q-z_e` 固定不意味着 `D(q)-D(z_e)` 固定；改变 decoder 仍会改变局部 Jacobian，甚至有限差分。非线性网络上 `J_D(z_e)(q-z_e)` 也只是局部近似。5ep失败不能证明函数类或表示的信息论上限。
- 原方案§7要求检查完整q对应的动作/相位歧义；本轮没有提供此证据，因此不能说§7触发条件已证实。即使存在歧义，也还要验证历史是否提供可泛化的信息。
- 原 probe `jitter_*` 实际是一阶位移范数，`d2_*` 才是二阶差分。文档的“一阶jitter”不应与项目正式Jitter混用。
- 原 probe 把 normalized traj 与 raw motion 拼接；当前世界坐标重建只读取后291维，因此这次不会改变world结果，但混合单位输入应修正，注释“GT trajectory用于积分”也不准确：实际积分的是每条输出的root motion。
- teacher fingerprint 只覆盖 encoder+quantizer，遗漏真正决定 `D0(E0(x))` 的 decoder，换teacher decoder不会被发现。
- `--decoder-only` 强制 `--init-ckpt`，而 init/resume 互斥，导致真实 decoder-only resume 无法通过入口。
- teacher resume 检查在当前 `teacher_fingerprint` 赋值前运行，B resume 会先因 None 与已有hash不等而失败。单元测试只测试手动设置hash的 helper，未覆盖这个入口问题。

后面三项没有造成已完成 fresh A/B 的结果，但“resume已验证可用”的说法需要撤回。当前只登记问题，未改训练代码。

## 6. 建议的下一步

1. 停止推广这批 decoder，保留原 VQ/controller 配套。不自动延长teacher训练、不切换码本、不据此直接加入历史decoder。
2. 先修正固定norm、独立RNG、完整teacher身份、resume入口和探针口径；这些是任何后续有效对照的前提。旧结果保留并注明偏差，不能改配置伪装原实验。
3. 如果继续做一个最小训练验证，优先讨论 **关闭teacher，使用已有GT二阶差分acc分支作为唯一新变量**，而不是再堆teacher/margin/roundtrip。权重先用真实train小batch梯度标定，不能由loss数值照搬。这是已有loss开关的新实验，不是本轮已修复或已经证明有效；用户若坚持acc=0则不启用。
4. 若不愿启用acc，短历史decoder只能作为独立结构假设：先比较相同参数预算的单帧与因果短历史拟合，对齐val数据、活动/风格与world接触指标；不能把整体平滑或腿不动当成功。历史q可以提供变化信息，但也会带来GT编码历史与controller预测历史的分布差异。
5. 表示层改善最后仍需配套controller原始rollout验证。当前仅是GT编码的AE诊断，无法承诺交互式的抖动/滑足、matched/cross/unseen同时改善。

训练效率上，冻结E/Q时可一次缓存正确norm下的q（或indices），后续合批decoder训练；这能节约重复编码/最近邻搜索，但不改变上述负结果和方法判断。本轮不启动新训练或重建大缓存。

复核命令：

```bash
/home/shinn/miniforge3/envs/mcc/bin/python analysis/vq_code_stability_2026-10-14/audit_calibration.py
/home/shinn/miniforge3/envs/mcc/bin/python -m pytest tests/test_decoder_only_calibration.py -q
```

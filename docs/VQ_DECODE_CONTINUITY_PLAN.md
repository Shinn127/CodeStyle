# VQ跳变问题：从解码连续性推导下一步方案

> **执行结果（2026-10-14，已按本文 §5 完成 A/B）**：decoder-only 校准**未降低**量化残差
> r/Δr/Δ²r（+1~5%，jitter/足滑持平），B 相对 A 无增益（≤1%，噪声内）→ 按 §5 预设**停止
> teacher 路线**，不推广；结论与归因见 [CALIB_AB_RESULT.md](../analysis/vq_code_stability_2026-10-14/CALIB_AB_RESULT.md)。
> 实现与正确性检查见 [DECODER_CALIB_IMPL.md](../analysis/vq_code_stability_2026-10-14/DECODER_CALIB_IMPL.md)；
> 下一步按本文 §7 的优先级（短历史 decoder / 码本容量）另立预注册方案。

2026-10-14。状态：推导和实验计划，未修改训练目标、未启动新训练。依据为 [独立审计](../analysis/vq_code_stability_2026-10-14/AUDIT.md) 的固定validation比较。

## 1. 应优化的目标

目标是“量化输出相对于真实运动没有额外的时间跳变和足滑，同时保留动作与风格”，不是“码ID尽量不变”。目前原VQ的GT相邻帧约56%的slot会换码；这既包含真实动作推进，也可能包含量化扰动。

设z_t=E(x_t)、q_t=Q(z_t)，位置输出分别为p_q(t)=P(D(q_t))、p_c(t)=P(D(z_t))，P包含反归一化和取24关节位置。令r_t=p_q(t)-p_c(t)，则严格有：

`Δ²(p_q-p_GT) = Δ²(p_c-p_GT) + Δ²r`。

因此应检查逐帧残差r及其时间差分，而不是仅看换码次数。标量jitter是范数的平均，**J_q-J_c不能直接当成J(r)**；应显式计算Δ²r，并与GT二阶变化对照。

现有固定12个val clip中，pilot的continuous AE jitter约.114cm，而quantized AE约.507cm；原VQ分别约.120/.412cm。它支持优先校准量化解码分支的假设，但不能保证该问题完全可由decoder解决。

## 2. 为什么margin、roundtrip、减K不是直接答案

跨越一个量化边界时，输出差为 `D(q_right)-D(q_left)`。提高进入某个cell的余量能减少特定扰动导致的换码，却不控制这个解码差。如果所有相邻cell输出完全一样，连通码图上的decoder会退化为常量；离散、无记忆的decoder不可能既消除所有边界差，又保留丰富动作。

raw margin还存在尺度自由度：E'=aE、C'=aC、D'(z)=D(z/a)保持输出和码身份不变，却扩大raw边界距离。本次pilot的raw margin中位数上升，而相对码间距的余量未提高，不能据此认定抗扰性增强。

roundtrip是同帧表示一致性，当前controller携带q，并非每帧重新E(D(q))。它不能代替对实际时间误差的监督。减K可能降低换码次数，也可能增加每次解码跳幅；迟滞量化可能延迟必要的步态更新。两者都不应先于运动质量对照默认启用。

## 3. 首选低风险实验：冻结E/C，只训练D

选原150epoch run的best（实际epoch90）作为统一初始化，复制D0为冻结teacher，D_theta为可训练student。保持原2048码、32×4 slots、原norm和数据split。

```text
真实x → 冻结E0 → z_e ──────────→ 冻结D0 → continuous teacher
                  └→ 冻结Q0 → q → 可训练D_theta → quantized output
```

训练期不更新encoder、EMA码本或任何code assignment。teacher也冻结。相同输入/精度/chunk下q应保持一致，因而能隔离“换码后的解码后果”。**冻结参数不足以冻结EMA**：每次model.train()之后仍要保证encoder/quantizer为eval，并通过buffer hash检查。

现有alignment是 `||D(z_e)-sg(D(q))||²`，它让连续分支靠近离散分支。这符合原有两路对齐目的，并不是数学bug，但不是当前“改善离散分支连续性”的最直接目标。

改用同一源checkpoint的冻结连续teacher：

`L_align_new = ||D_theta(q)-sg(D0(E0(x)))||²`。

不要用不断变化的student D_theta(z_e)作teacher，也不要混用另一个encoder/码本的latent。teacher较平滑不代表绝对正确，因此真实GT重建和GT时间差分仍是主监督。

在既有normalized motion291空间复用当前损失：

`L_D = 3 L_recon(GT) + 15 L_vel(GT) + 3 L_root(GT) + lambda_align L_align_new`。

第一轮lambda_align取原入口已有的2.0作为待验证起点。VQ、margin、RT、latent L1/velocity均设0：冻结表示时这些不再是本轮要优化的目标。acc仍为0，不新增classifier/contact等loss，也不训练controller。

如果对冻结teacher的逐帧残差RMS能压低，其二阶误差也受到约束：离散二阶差分算子的L2范数不超过4，即 `||Δ²r||_2 <= 4||r||_2`。这只是上界，不是对平均MSE、尾部接触或自然度的保证；仍要直接验收Δ²r、GT时间误差和world足滑。

## 4. 为什么保留GT运动差分，不直接追求“平滑”

当前vel项为 `||Δ m_hat-Δ m_GT||`。若确有需要，已有acc项为 `||Δ² m_hat-Δ² m_GT||`。二者监督真实运动变化，不是 `||Δ m_hat||` 或 `||Δ² m_hat||` 的零变化目标。

对动态GT，静止输出仍有差分误差，因此没有直接奖励“完全不动”。但有限loss权重和有损表示仍可能产生折中，必须检查腿部活动、移动时冻结姿态、接触覆盖和style保真。

第一轮不加acc。若decoder校准后高频残差仍明显，再单独验证已有acc分支：seq_len至少3、目标是真实GT二阶差分、按decoder梯度量级标定小权重，不能直接抄一个很大的数字，也不与改K/margin同时进行。

## 5. 最小对照与验收

不立即重训整个VQ/controller链。建议两组各最多5epoch，采用相同初始化、同一采样顺序/updates、seed42、FP32/highest、TF32=False：

| 项目 | A：decoder校准对照 | B：连续teacher校准 |
|---|---|---|
| E / codebook / norm | 冻结且相同 | 冻结且相同 |
| student D初始化 | 原D0 | 原D0 |
| GT recon / vel / root | 3 / 15 / 3 | 3 / 15 / 3 |
| alignment | 0 | 原有项改为冻结连续teacher，初始权重2 |
| margin / RT / acc | 0 / 0 / 0 | 0 / 0 / 0 |
| 序列 | seq_len4，stride1 | 相同 |
| batch | 512 windows；不足用相同有效batch累积 | 相同 |
| optimizer | decoder-only AdamW，lr1e-4→1e-5 cosine5 | 相同 |

这些是建议值，不是已验证的最佳超参。先100-update smoke测量耗时/显存与梯度，再决定是否正式执行；本轮没有运行。对照A用于区分额外decoder训练本身的收益，B相对A才体现teacher校准的增益。

对照使用独立的固定validation manifest，原D0、A、B都重测同样病例；不要直接混用历史60f、当前121f及其他240f数字。至少包括：

- Quantized/continuous AE的MPJPE、Root ADE、GT时间差分误差、jitter。
- 显式量化残差r、Δr、Δ²r的均值/P95/max，定位最差片段。
- world足滑、contact数量/覆盖、移动时腿部活动与冻结姿态。
- 固定IND validation的量化AE style fidelity；分类器只评价，不训练或筛选训练样本。
- 固定真实片段的原始可视化，关闭显示smooth/IK；另报完整显示链路，不能混为模型收益。
- E/C、norm及q不变；decoder确有梯度且teacher无梯度；数据/源码指纹完整。

验收需要运动质量联合改善，不能仅MPJPE或jitter一个指标更好。零contact不以0通过；低jitter但腿不动不通过。IND验证选定模型后才看OOD，不能用OOD挑decoder。若B没优于A，停止把teacher方案当作解法，不默认延长训练。

## 6. 实施边界与效率

- 模型推理结构保持单帧E→Q→D，所有损失复用已有项；只在VQ训练器增加decoder-only和明确的alignment目标模式。
- 冻结teacher放在trainer外部，不注册进正式VQ模型state_dict，避免破坏现有加载器。checkpoint记录teacher源及hash，resume要求训练部分/teacher/norm/目标一致。
- q和continuous teacher可预计算。**现有teacher_motion缓存存的是D(q)，不能误当D0(z_e)**；新缓存明确teacher_kind及来源，并只构建train。
- decoder阶段可绕过每batch的E和NN搜索，合批解码序列；保持FP32。缓存预处理和训练时间分别报告，不沿用margin微基准或历史epoch速度。
- 同码本不等于旧controller无需验证：D(q)变化会改变其motion-history输入。新VQ必须作为明确版本处理，旧controller不能无记录override替换。
- 若E/C确实逐值未变，code indices、latent stats、prior在数学上可复用，但须显式验证并重新建立来源关联；不能绕过现有完整VQ指纹守卫。所有D(q)观测缓存必须重算。是否需要controller短TF适配由后续有记录的rollout对照决定。

## 7. 若decoder-only不足，再做什么

先辨认表示上限，而不是继续堆loss：如果相同完整q经常对应不同真实姿态/相位，单帧确定性decoder无法恢复被丢掉的信息；MSE最优解是条件均值，可能损害运动。应测真实碰撞/条件误差，不能用稀疏n-gram或1NN直接宣布信息论上限。

只有证据支持后才考虑：更合适的VQ容量/维度、短历史decoder或时序表示。它们都改变方法与训练/推理契约，需要独立方案；软量化、迟滞或persistent noise不能作为未经验证的替换开关。

若将来重训E/C并保留稳定性项，应优先研究尺度归一化：用局部code间距s_i归一化真实边界rho和RT误差，关注rho/s_i与||E(D(q))-q||/s_i，而非单看raw距离。小间距需要合理floor，归一化后梯度尺度改变，不能直接沿用raw项的权重1.0。本轮不把这条研究分支混入decoder校准实验。

**优先顺序：先固定码本验证能否改善离散解码的运动质量，再决定是否需要改变表示；最终用配套controller的原始rollout验收。** 减少码ID变化不是最终目标，也不能替代matched/cross/style及轨迹指标。

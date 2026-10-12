# 保留跨风格监督的连续过渡 donor 策略

2026-10-12，方案设计，尚未恢复donor代码或启动实验。修订建议：无donor纯TF保留为对照；优先验证“合格的连续过渡donor + 纯TF”，第一轮不同时叠加history遮蔽、逐层style调制或rollout训练，以便归因。既有VQ、style prior/selector、六token、target FM和FP32保持。

## 1. donor有效的部分与旧实现的问题

donor提供了matched-only训练缺失的组合：历史显示风格A，命令要求B，后续运动向B转变。你的实验支持保留这一监督信号；不能因旧实现质量差，就推断一切donor策略不可行。现有对照同时涉及初始化和rollout等变量，因果幅度仍需同初始化A/B验证。

归档实现 `docs/archive/donor_sources_2026-10-12/style_donor_bank.py` 仍可核验：筛选suffix/root/phase，读取donor的未来pose/rotation，替换source root并修正velocity，再编码target。TF历史仍来自原source。即使同窗口donor固定，每个TF转移仍可能要求原source姿态在一帧内跳到donor姿态。相位相近不足以保证姿态边界接近。

接触问题可直接由运动学解释。忽略记号中的坐标转换，一只脚的世界速度为：

`v_foot = v_root + R (omega × p_foot + d(p_foot)/dt)`。

在初始朝向对齐、保留donor局部脚运动的情况下，将donor root改为source root，会增加约 `delta_v_root + R(delta_omega × p_foot)` 的世界脚速度。仅修改motion291里的velocity通道，不能把这个位置轨迹变成固定落地脚。单看平移，差0.25m/s在60fps约为0.42cm/帧；这是运动学量级示例，不是新的实测结果。

“静止姿态但root移动”可能来自近静止donor/错误phase组合、不同目标的平均化或闭环吸引子，不能仅据日志断言是哪一种。筛选必须检查较长时间的实际运动状态和脚部活动，不能只用短窗口root速度/单帧phase。

## 2. 推荐改成完整过渡片段，而不是替换某一帧target

构造流程：

`真实source A前缀 → 连续A到B过渡 → 接触修正后的B后缀`。

目标style从命令发出时就是B。过渡早期仍像A是正常响应过程，不能给每一帧强加“立即被分类为B”的要求。训练不使用style classifier loss。

关键约束：**history、next target、velocity和trajectory必须来自同一条最终过渡序列。** TF训练历史使用预生成序列的冻结codec状态，不使用模型自己的输出，也不能继续用原source历史搭配过渡/旧donor target。

### 2.1 先匹配连续片段

- 两端均来自train，style不同、source_entry不同，动作suffix兼容；不使用val/test补候选。
- 用约60–120帧上下文判断moving/idle、步态推进和接触事件；局部phase匹配再用短窗口。真实静止只与静止配对，不把站立pose移植到持续移动root。
- 比较完整接触序列、左右支撑角色、相位推进方向、root速度/转向，而不只比较脚的瞬时位置。
- 可做小幅、单调的时间重参数化来对齐步态；先从接近1的倍率范围试验，超出范围拒绝，不能靠任意拉伸满足命中率。所有速度在最终时间轴上重算。
- 一个过渡片段固定一个donor和时间映射。不在相邻帧/相邻TF转移重新随机选donor。

### 2.2 先得到适配source轨迹的donor，再处理姿态边界

保留source的root控制计划时，必须同步调整donor的脚部位置轨迹。根据可靠的donor接触区间设置世界空间落脚锚点，在统一骨架上进行有限的下肢IK/骨盆补偿，随后FK重建所有joint positions。摆动脚保持摆动，不把双脚一直锁住。

IK只用于离线构造监督，不是训练loss，也不是推理后覆盖root。若需要大幅拉腿、破坏骨长、出现不可达支撑、过度改变donor姿态，则拒绝该pair。几何修正不提供动力学/平衡保证。

真实SlideFeet等风格不能机械按“所有低脚都应固定”处理：只约束原序列中有证据的稳定接触，其余按该动作/style的真实运动特征验收。零接触时足滑不可用，不能用0证明合格。

### 2.3 使用有边界条件的渐变修正

将适配后的donor记为D(t)，对它添加随时间衰减的姿态修正，使过渡起点匹配source当前的姿态、线/角速度，必要时匹配加速度；到终点修正及一、二阶导数为零。平移分量可用满足六个端点约束的五次多项式。旋转在统一的旋转切空间处理并验证角速度边界，不对6D rotation或375维feature直接线性插值。

过渡时长初始考察24–48帧（0.4–0.8秒）；这只是候选区间，实际由距离、速度/加速度和接触检查决定。幅度大的动作若在预算时长内不合格，就拒绝或显式记录需要更长过渡，不强迫一帧完成。

单纯smoothstep混合仍不够。位置混合的导数包含 `alpha_dot * (p_B-p_A)`；如果velocity通道只混合两端原velocity，会再次与实际位置不一致。应从最终连续运动重新提取position/rotation/root velocity/joint velocity/trajectory等全部375维特征，使用原坐标、fps和提取约定。姿态边界修正与接触修正之后都要复查，后一次操作可能破坏前一次约束。

### 2.4 最终监督必须经过codec后的检查

对最终序列构造 `X_k=[trajectory_k,motion_(k-1)]`，用原冻结VQ编码、量化并缓存q。TF条件是D(q)，目标latent来自同一序列下一行；不要复用原source/donor的旧velocity或旧latent。

同时检查合成运动和quantized AE重建。原序列很平滑、但经过VQ后抖动/接触明显恶化的样本不能直接进入训练。保留失败原因，确认冻结VQ是否能表示此类过渡；不能默认VQ能编码任意插值运动。

## 3. 数据质量门槛先于训练

先离线构造数百个候选，覆盖若干静态姿态风格、步态风格、起步/停止/转向；不先生成巨大cache或开50epoch训练。

必须记录：

- source到过渡、过渡到后缀的pose/线角速度/加速度跳变；完整序列的峰值而非只有均值。
- source控制轨迹误差、world contact slip、接触覆盖、脚穿地、IK修正量、骨长/可达性。
- 移动时的腿部活动、步态推进和静止滑行proxy；阈值结合真实style/action分布，不能把低jitter等同于运动良好。
- 过渡末端与接触适配后donor的差距、适配前后donor变化量，防止把目标style修没了。
- 上述指标经VQ量化后的变化。
- 请求数、接受数、按source/target style与action的接受矩阵。无合格pair回退原matched样本，绝不回退旧的瞬时拼接。

阈值先参照train真实运动及其VQ重建误差确定，不根据test调节。不使用最终报告classifier作为生成数据的优化目标或筛选器。先证明目标数据本身质量可靠，再讨论模型训练的风格/运动指标。

## 4. 训练方式

- 纯TF，feedback_probability=0；VQ冻结、FP32，loss继续FM + .02 prior CE + .003 entropy，不加reconstruction/root/classifier loss。
- 以约80%真实连续片段、20%合格过渡片段为起始配比；过渡片段按完整window抽样，history不能跨来源拼接。实际比例由质量通过率决定，不强凑20%。
- 原始matched数据保证稳定维持风格；过渡数据学习“当前A、命令B、逐渐到B”。B的稳定后缀也应出现，避免只会改变而不会保持。
- 第一轮不同时加入上一计划的30%history遮蔽或新style调制，保持架构一致，测清过渡donor的独立收益。需要时再单独修改条件注入。
- 对clip/recipe分支，参考应来自目标B的其他train clips，并排除source及当前donor entry，避免把精确目标片段作为style参考泄漏。先验证当前label主线，不自动启动recipe训练。

建议最小A/B：两组从同一个纯TF checkpoint（如现存epoch50）初始化，只继承相同权重、分别重置optimizer/RNG，训练相同updates和LR；A继续真实TF，B增加合格过渡TF。先各5epoch，正式运行仍由用户授权/启动。不拿不同初始化或不同solver的结果归因给donor。

固定240f matched/cross rollout之外，加入“切换前稳定 → 发出B命令 → 过渡 → 维持B”的场景。分别报告过渡峰值、响应时间、切换后的连续240f目标style准确率以及停止/再次切换能力；主评估的cross完整240f窗口仍保留，不能删除困难过渡帧提高原指标。

诊断时style feature blend=0，让模型直接收到B；过渡由训练数据学习。显示端.25秒feature插值作为单独对照，避免将两种过渡机制混为一谈。不以降noise或持续相关noise遮掩冻结姿态。

## 5. 为什么不优先只扰动donor历史

“目标仍是B的真实GT，只用donor A轻扰动B的历史”可作为小剂量抗扰增强。但扰动大时，仍会教模型一步拉回B，复现跳变；扰动小时，不一定覆盖完整A→B切换。这条策略要以一步运动可行性约束扰动幅度，不能把任意cross历史替换包装成安全增强。本轮不作为主donor机制。

## 6. 代价与实施边界

相位匹配、边界拟合、接触修正、完整特征提取和VQ编码全部离线完成；训练只读取同格式FP32 features/code cache，不在每个update搜索donor、跑IK或在线retarget。分别统计预处理时间、通过率、缓存大小和训练吞吐，不能仅报告训练加速而隐藏预处理成本。

新产物独立目录，记录原始两端entry/start/style、时间映射、过渡时长、root计划、接触/IK策略、生成器版本、VQ/norm/split指纹与完成状态。先复用现有FK/特征提取和缓存工具；不恢复旧的单步target移植训练入口。

这属于需要验证的合成过渡数据方法，会改变训练数据构造，但保留VQ latent + style prior + conditional FM核心。它比旧拼接更直接消除监督里的几何冲突，不能保证任意style pair都可行，也不能保证所有闭环抖动都由donor解决。若可接受过渡太少，须报告覆盖缺口；部分风格组合可能需要真实过渡数据或更强运动适配，不能靠放宽质量门槛冒充解决。

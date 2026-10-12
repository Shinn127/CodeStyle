# ControlOperators（SIGGRAPH Asia 2025 参考实现）学习报告

来源：https://github.com/gouruiyu/ControlOperators（本地副本 `revision/ControlOperators-main/`，
与 GitHub 最新结构一致）。作者：Daniel Holden 等（"Control Operators for Interactive Character
Animation"）。目标：回答"为什么它不做 rollout 训练也能稳定实时交互"，以及我们能借鉴什么。

## 一、它的训练协议：单步 + 输入噪声，无 rollout

`train.py` 主循环（约 500k 迭代、batch 2048）：

```python
Zprev = Z[I-1]; Znext = Z[I]                       # 相邻两帧，单步 pair
Ztil  = Zdist * randn_like(Znext)                  # 起点：拟合高斯（非标准正态）
Zprev = Zprev + Zdist * 0.5 * rand() * randn_like(Zprev)   # ★ 输入噪声增强
t     = rand(); Zbar = (1-t)*Ztil + t*Znext
loss  = |Zdist*denoiser([Zbar, Zprev, Cnext, t]) - (Znext - Ztil)|²
```

三个关键点：

1. **每个训练样本只有一步**（Zprev→Znext），从未展开多步、也没有对齐推理求解器。
2. **对反馈输入 Zprev 加随机幅度噪声**（`pose_noise=0.5`，幅度为 `U[0,1]*0.5` × 各维 std）。
   模型每步都见过"上一步带误差的状态"，学会从有噪声的历史中恢复——这是暴露偏差（exposure bias）
   的廉价近似，替代了 rollout 训练的作用。
3. **latent 起点与速度尺度都用数据拟合的统计**：噪声从 `Zdist*randn` 采样（贴近数据分布）、
   denoiser 输出乘 `Zdist`（相当于按维反归一化），让 ODE 只走短距离。

## 二、结构差异：无 VQ 的连续 latent（与我们的根本区别）

- 自编码器是普通 MLP（2 层，hidden 512），latent 256 维**连续**，无量化、无码本。
  AE 损失含**速度一致性项** `15*|ΔX_gnd - ΔX_rec|` 和 latent 平滑项。
- 推理反馈：`Zprev = clip(Znext_积分结果)` —— 直接把自己上一步的连续输出喂回去，
  **反馈回路里没有任何量化阶梯**。我们的 VQ 反馈每帧有 99% 概率换码（此前诊断），
  等于每帧给回路注入一次离散噪声，他们从结构上不存在这个噪声源。
- denoiser：`SkipCatMLP(inp=769, hidden=1024, depth=10)` —— 10 层宽 concat-skip MLP，
  输入 `[Z, Zprev, C, t]`，无 attention、无 trajectory token、无 style token（控制向量
  由 Control Operator schema 编码而来）。

## 三、推理回路：稳定性的另一半来自运行时

`controller.py` 每帧：`inference(S=4)` → decode → **输出惯性化** → **脚锁定+IK** → 渲染。

- **推理仍每帧采新噪声**（`Zdist*randn`）并 S=4 欧拉积分 + clamp 到数据 min/max。
  即"每帧随机性"不是他们回避的东西——靠下面的机制吸收。
- **输出惯性化平滑 `alpha=0.75`**：`out = 0.25*新预测 + 0.75*(上帧 + dt*速度外推)`。
  显示输出 75% 是上一帧的惯性外推——比我们的因果低通（6Hz 基频）激进得多。
- **脚锁定 + IK 用的是模型自己输出的 contact 位**（姿态向量末 2 维，训练时由
  toe 速度阈值+中值滤波标出），而不是运行时从高度/速度反推。锁定/释放用
  `InertializeCubic` 三次混合过渡（blendtime 0.2s），触发/解锁条件是"contact 位 +
  与目标的偏移距离 <0.2 / >0.01"——不依赖脚是否已经滑动。
- **控制目标是弹簧平滑的**：trajectory 模式由目标速度/朝向经弹簧动力学（halflife 0.2）
  外推 3 个未来点（20/40/60 帧）；velocity 模式用平滑速度+可选朝向。键盘/手柄输入
  从不直接进网络。

## 四、与我们项目的对照

| 维度 | ControlOperators | 我们（label controller） |
|---|---|---|
| 训练步数 | 单步 pair，无 rollout | 50 TF + 20 ramp + 30 rollout（ODE4 run） |
| 反馈噪声 | **每步对 Zprev 加随机噪声** | 无（曾有过，切换 rollout 时全部移除） |
| latent | 256 维连续，无量化 | 128 维 32×4 VQ（2048 码本，冻结） |
| 反馈回路 | clip → 直接用（无量化） | clip → requantize → decode（量化阶梯） |
| 上下文 | 单帧 Zprev | 双帧历史 + trajectory + motion token + style |
| 网络 | SkipCatMLP 10 层×1024 | 6-token Transformer 2 层×256 |
| 运行时平滑 | 惯性化 75%（激进） | 因果低通 6Hz + IK（保守） |
| 脚接触 | **模型输出 contact 位**驱动 IK | 高度/速度启发式 + 滤波（易误锁/漏锁） |
| 控制输入 | 弹簧平滑的轨迹/速度 | planner 平滑（`--smooth-plan` 可选） |
| 评估 | 演示级（视频） | 240f 自动协议（acc/ADE/MPJPE/jitter/FS） |

## 五、可借鉴项（按性价比排序）

### A. 重新引入"反馈输入噪声"（单步 TF 内），**不恢复 rollout**
- 做法：TF 训练时对 `prev_latent`（以及可选 feedback motion）加 `U[0,1]*σ*randn`，
  σ 参考其 0.5×latent std；保留我们现有的双帧历史/课程不变。
- 目标：用廉价方式获得"从自身误差恢复"的能力，替代 rollout 的一部分作用，
  且避免 rollout 阶段实测到的 jitter/FS 回升（我们的 rollout 30ep 使 jitter 0.44→0.84）。
- 成本：训练器 ~30 行 + 10 epoch 短训（~15 min，若纯 TF 更快）+ 封存评估。
- 风险：此前历史噪声实验（donor 时代）未改善质量，但当时噪声施加对象/幅度/训练阶段
  都与这里不同（那时是结构噪声+rollout 并存），值得以他们的配方（单步、Zprev、
  随机幅度、幅度较大）重新做一次干净对照。

### B. 模型预测 contact，替代启发式接触估计
- 做法：给 controller 加一个小辅助头（在 latent 或 decoded motion 上）预测左右脚
  contact（标签按他们的方案：toe 世界速度 <0.15 m/s + 中值滤波），推理时用它驱动
  IK 的锁定/释放；显示侧仍不写回模型。
- 价值：直接针对我们诊断出的"IK 在已滑动脚上误锁/漏锁"问题。
- 成本：标签生成 ~1h，辅助头训练并入现有 TF 流程；显示侧接入半天。

### C. 显示侧惯性化对照实验（零训练成本）
- 把我们的显示平滑从 6Hz 低通换成他们的"0.75×惯性外推+0.25 新预测"混合，用
  `record_control_video.py` 对比滑行/脚感。若有效，说明我们此前的平滑太保守。

### D. 深宽 SkipCatMLP（备选，不优先）
- 他们的 denoiser 是 10 层×1024 的 concat-skip MLP；我们此前 50ep 归因里
  skip_cat_mlp 输给 transformer，但那是在 VQ latent + 不同训练制度下。若 A–C 后
  仍需提升闭环上限，可用他们的确切结构（depth 10/hidden 1024）重测一轮。

### E. 诊断性对照：连续反馈 vs 量化反馈
- 他们整个回路无量化。我们已有 `--no-requantize`（AE 连续反馈）交互路径，
  可用其做 jitter/滑行的对照，用来估计"量化阶梯"在闭环噪声中的实际占比
  （此前结论 ~20%，值得在 ODE4 模型上复核）。注意：正式论文口径仍必须 requantize。

### 不建议照搬的
- **放弃 VQ**：论文管线（复现目标）以 VQ 为核心，不能整体替换；只做诊断对照。
- **去掉 trajectory/style 条件**：他们的控制 schema 是另一套设计目标（无风格标签），
  与我们"风格迁移 + 键盘轨迹"的任务不对应。
- **单帧上下文**：我们的双帧 history 经短训验证有效，不回退。

## 六、直接结论

1. 他们"无 rollout 也稳定"**不是魔法**，而是把三件事同时做到位：单步训练+
   输入噪声增强（替代 rollout 的鲁棒化作用）、无量化反馈（除掉了量化噪声源）、
   强运行时惯性化+模型 contact 驱动的 IK（吸收剩余抖动与打滑）。
2. 我们与此对照的缺口正好在两个点上：**(1) 训练侧没有输入噪声增强；
   (2) 显示侧 contact 来自启发式而非模型**。
3. rollout 训练在我们这里实测有反效果（jitter/FS 回升），而它恰恰是 ControlOperators
   刻意不用的东西——这支持把我们 ODE4 课程里的 rollout 部分换成"TF + 输入噪声"的
   变体做 A/B，这是下一步最值得做的实验（方案 A，成本半天内）。

## 附：关键源码位置

| 内容 | 文件:行 |
|---|---|
| 单步训练 + 输入噪声 + 损失 | `train.py:925-945` |
| 推理 S=4 + 噪声起点 + clamp | `train.py:860-884`、`controller.py:974-983` |
| 运行时惯性化平滑 alpha=0.75 | `controller.py:1112-1117` |
| 脚锁定/IK（模型 contact + InertializeCubic） | `controller.py:462-760` |
| 控制 schema（velocity/trajectory 运算符） | `control_encoder.py:82-130` |
| 运行时弹簧轨迹 | `control_encoder.py:200-320` |
| contact 标签生成（速度阈值+中值滤波） | `train.py:318-336` |
| AE 损失（速度一致性） | `train.py:738-744` |
| SkipCatMLP | `networks.py:37-72` |

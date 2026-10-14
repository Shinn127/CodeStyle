# C2 预注册协议：纯 TF 观测鲁棒性 A/B（需要训练授权）

依据 [docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md) §4 C2。
本文件在 C2 训练启动前冻结。**训练只在用户明确批准后执行。**
C0 证据（`../C0/REPORT.md`）：自反馈分布偏移使 local MPJPE 1.17→4.43cm、strong 活动 −44%；
C1 证据（`../C1/REPORT.md`）：更低推理噪声改善 Jitter 但以活动性为代价 → 观测鲁棒性只能靠训练。

## 1. 假设

训练期把"上一帧观测"替换为**病态的 GT 观测**（`z_gt + U[0,A]·transport_std·ε` → clip→Q→D，
motion/latent 同源一致），能让 Controller 在推理期接自生成观测时更稳，从而缩小 C0 发现的
闭环缺口；且不引入任何 rollout / model-generated feedback。

## 2. 实验轴（单变量）

**C2a：只扫描观测噪声幅度 A ∈ {0.25, 0.5, 0.75}**（3 臂），固定其余一切：

| 项 | 值 |
|---|---|
| VQ | 原 ep90，全冻结 |
| Controller | 从**同一份**已训练 100ep 权重 `--init-ckpt` 冷启（不是 resume） |
| 架构 | 2L/256/heads4、history 双帧、6 token（与现役一致） |
| 课程 | 纯 TF：`FEEDBACK_PROBABILITY=0`、`FEEDBACK_RAMP_EPOCHS=0`（无 rollout 训练） |
| 目标 | `--observation-noise-std A`（既有 v12 机制，U[0,A]×transport_std） |
| epochs | **10 /臂**（默认上限） |
| batch/lr | 512 / 5e-4（sqrt 规则），cosine 随 EPOCHS 收尾 |
| seed / data order | 42 / `--seed 42`（controller 训练器每 epoch 以 seed 重播 loader） |
| solver | 训练 `--solver-steps 4`、`--feedback-solver-steps 4`（与现役一致） |
| 选型 | 固定终点 `epoch_0010.pt`（零选择）；训练内 240f rollout 验证（motion-quality selection）照常记录 |
| OUT | `checkpoints/c2_obsnoise_A<A>_10ep` |

**控制臂**：A=0.5 即现役配方的一次 10ep 复算。三臂共享同一 init 与同一数据顺序。
**不得**把旧 100ep/不同层数 run 当严格对照。

臂间唯一差异 = 噪声幅度；wrapper 固定 `OBS_NOISE_STD`，禁止另有改动。

## 3. C2b（暂不执行）

仅当 C2a 不足且单独授权：改时间相关性（AR(1) ρ）而不改幅度；新 objective version + 严格
resume 校验；本轮**不实现**，等 C2a 结论。

## 4. 验收（预注册）

对锁定 val 协议（240f@60fps、matched/cross、固定 case/指令/采样种子）与 C0-C 同口径：

| 门 | 要求 |
|---|---|
| C2-0 契约 | 三臂同一 init（state sha 相同）、同 updates、除 `observation_noise_std` 外 config 相同 |
| C2-1 主目标 | 至少一个预注册闭环指标（local/world MPJPE、root ADE、jitter、FS、速度方向误差、扰动恢复）相对 A=0.5 有**可证实**改善（配对 95% CI 上界 < 0） |
| C2-2 质量 | world/local MPJPE、root ADE 相对退化 ≤ 3%；FS 退化 ≤ 0.5% |
| C2-3 活动性 | moving/strong/腿行程下降 ≤ 5%；contact coverage 无异常 |
| C2-4 风格 | matched 与 cross accuracy 各下降 ≤ 2pp（240f 分类器） |
| C2-5 反静止 | 不得以冻结/滑行改善 Jitter |

无实质收益 → FAIL；CI 无法区分 → INCONCLUSIVE。**不自动扩到 100ep、不替换默认 controller。**

## 5. 执行（代码已就绪，训练需授权）

```bash
# 三臂串行（避免并行 GPU 训练混淆归因）
PYTHON_BIN=... ARM=C2-A025 bash scripts/run_train_c2_obsnoise_ab.sh
PYTHON_BIN=... ARM=C2-A050 bash scripts/run_train_c2_obsnoise_ab.sh
PYTHON_BIN=... ARM=C2-A075 bash scripts/run_train_c2_obsnoise_ab.sh
```
验收用评测入口（`scripts/run_evaluate_style_rollout.sh` / controller_validation）与 C0 逐源口径，
写入 `C2/{arms,acceptance.json,DECISION.md}`。

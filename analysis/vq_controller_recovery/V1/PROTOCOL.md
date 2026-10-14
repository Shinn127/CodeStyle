# V1 预注册协议：Decoder-only 物理位置二阶损失 A/B

依据 [docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md) §5 V1。
本文件在 V1 训练启动前冻结；标定结果写 `gradient_calibration.json`，结果写 `V1/arms/*` 与
`acceptance.json` / `DECISION.md`。

## 1. 假设（尚未验证）

S3/S4 的失败是 **mixed normalized-motion291** 空间二阶项与位置重建竞争所致（local MPJPE +21%）。
只监督**真实 24 关节位置（cm）** 的 `Δ²`、同时保留强重建约束，可能降低 `Q` 引入的 Jitter
与量化二阶误差，而不显著牺牲位置精度。

## 2. 固定项（与 S3/S4 明确区分）

| 项 | 值 |
|---|---|
| 初始化 | 原 ep90 VQ `best.pt`（两臂同一 state，逐张量相同） |
| 冻结 | **E 与 Q/EMA 全冻结**（`--decoder-only`，quantizer 保持 eval、EMA/死码路径关闭） |
| 只更新 | Decoder 参数 |
| norm | 原 `norm_stats.npz` 显式传入（不重算） |
| 损失 | **训练器 decoder-only 实际 base**：recon=3 / vel=15 / root=3（`--decoder-only` 会自动将 vq/code_align/latent_l1/latent_vel 归零，见 trainer 的 `[Info] decoder-only zeroed`）；margin/roundtrip/teacher=0 |
| `posacc_beta_cm` | 0.5 cm（SmoothL1 拐点） |
| 序列 | `seq_len=8`、`stride=1`、batch 512、seed 42、`LOADER_SEED=4242` |
| lr | 1e-4 → 1e-5 cosine |
| epochs | **5 /臂**（首轮上限） |
| 选型 | 固定终点 `epoch_0005.pt`（零选择）；`best.pt` 一并报告但不用于选型 |
| OUT | `checkpoints/vq_posacc_V1-B0_<id>` / `vq_posacc_V1-B1_<id>` |

唯一变量 = `POSACC_WEIGHT`（B0=0，B1=标定 λ）。**不得引用先前全 VQ B0 当本轮对照。**

## 3. 梯度标定（训练前，不更新优化器）

- 4 个独立 train 小批 B16×T8（seed 4242+i，段内采样，不跨 segment）。
- 目标：`∇_D L_base`（上表除 posacc 的项）与 `∇_D L_posacc` 的范数、cos；**只按 decoder 分组**
  （E/Q 冻结，无意义）。
- 选 λ 使 `λ‖g_posacc‖/‖g_base‖ ≈ 0.1–0.3`。若 cos<0（冲突）或统计极不稳，标风险并停止；
  若窗口与 S2 的 gap-midpoint 情形类似，用同一回退规则（缺口几何中点）并记录。
- 与 S2 的 4.9842 **无继承关系**：那里是 mixed 空间 + 全 VQ，这里是 cm 位置空间 + frozen E/Q。
- **协议更正（训练前，未见过任何 V1 数据）**：初版标定误用未归零 base（`gradient_calibration.json`，
  λ=1.6278）。训练器在 decoder-only 下会把 vq/code_align/latent_l1/latent_vel 归零，因此按实际 base
  重算（`gradient_calibration_decoder_only.json`）：decoder ratio 0.1046、cos 0.056、
  窗口 [1.059, 2.508] → **λ=1.6295**（两者差 0.1%，归零项对 decoder 梯度几乎无贡献，但以实际 base 为准）。

## 4. 验收门槛（预注册，训练前冻结，事后不得改）

| 门 | 要求 |
|---|---|
| V1-0 契约 | 两臂只差 `posacc_weight`/OUT；updates 相同；E/Q/EMA 逐张量不变（manifest 指纹）；init 相同 |
| V1-1 直接作用 | `‖Δ²r‖` mean 与 per-source p95 均下降且配对 95% CI 上界 <0（val12，与 S4 同口径） |
| V1-2 时序主目标 | 官方 quantized Jitter 相对 B0 下降 ≥ 15% |
| V1-3 重建质量 | local/world MPJPE、root ADE 每项退化 ≤ 3% |
| V1-4 足滑 | 事件加权 world/local FS 退化 ≤ 0.5%；同时报逐 source 均值/CI 与 contact count |
| V1-5 防冻结 | moving/strong 帧占比、腿行程等下降 ≤ 5%；contact coverage 无异常 |
| V1-6 风格 | 90 val source 冻结分类器准确率下降 ≤ 2pp |
| V1-7 证据 | 连续 AE 对照、codec 换码率单独报告；不据其单独判 PASS |

**否决**：只降 `Δ²r`、只降 code flip、或以姿态精度/活动性换更低 Jitter → FAIL。
5ep 未区分 → INCONCLUSIVE，需批准才延长。
V1 PASS 仅是 representation-val 候选，**不得**直接注入现役 controller。

## 5. 执行步骤（代码已就绪，训练需授权）

```bash
# 标定（读-only）
python analysis/vq_controller_recovery/V1/v1_gradient_calibration.py --device cuda
# 训练（需用户批准后执行，B0 后 B1 串行）
PYTHON_BIN=... ARM=V1-B0 bash scripts/run_train_vq_posacc_ab.sh
PYTHON_BIN=... ARM=V1-B1 ACC_WEIGHT=<calibrated> bash scripts/run_train_vq_posacc_ab.sh
# 验收（探针 + 门槛）
python analysis/vq_temporal_source/run_probe.py --ckpt <arm>/epoch_0005.pt ...
python analysis/vq_controller_recovery/V1/v1_acceptance.py --b0 ... --b1 ...
```

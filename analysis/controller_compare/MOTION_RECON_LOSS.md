# Motion 空间监督：监督推理实际显示的 motion（2026-10-11）

## 问题（用户观察 + 数据确认）

控制器训练只监督 latent 空间（flow matching 预测目标 latent）。但实时交互/rollout 实际显示的是
**`decode(requantize(clip(z_hat)))`** —— 这条离散化管线训练时从未被监督过。后果在 90-clip
探针上直接可见（同 seed42、同协议）：

| checkpoint | RootADE | **MPJPE** | localMPJPE | Jitter | F.S. | style hits |
|---|---:|---:|---:|---:|---:|---:|
| tr6（旧 label150，latent-only 监督） | 15.2cm | 10.58cm | 4.88cm | **0.96cm** | **0.91cm** | 79/90 |
| tr5（50ep，latent-only） | 14.4cm | — | — | 1.19cm | 0.96cm | 69/90 |
| cur100 best(ep70) | 9.77cm | 12.02cm | 4.66cm | 1.42cm | 1.36cm | — |
| cur100 ep75 | 8.09cm | 10.58cm | 4.63cm | 1.37cm | 1.31cm | 87/90 |
| **cur100 ep80** | 8.40cm | **10.72cm** | **4.41cm** | 1.29cm | 1.27cm | — |
| cur100 best TF（teacher-forced 上界） | 3.72cm | **4.20cm** | **1.07cm** | 1.44cm | 1.25cm | — |

MPJPE 口径：90-clip 探针（seed42），全球坐标（root 积分后）与 local（root-frame）两种；与
`Utils.motion_eval.compute_sequence_metric_sums` 同定义。关键读法：

- **local MPJPE 反而更好**（4.41 vs tr6 的 4.88）——单帧姿态本身没问题，问题在 root 轨迹
  （Global MPJPE − local ≈ root 漂移贡献）与逐帧时序噪声（jitter/F.S.）。
- **TF 上界 MPJPE 4.20 / local 1.07**：模型单步预测很准；rollout 累积 + 量化台阶把它放大到
  10.7/4.4。这与"必须监督推理路径 motion"的结论一致：差距主要在后处理管线（requantize 跳码）
  和自回归累积，而不是 field 本身。

即：rollout 课程把轨迹跟踪和风格都显著改善了（ADE 15.2→8.1、hits 79→87），但**显示出来的
motion 平滑度变差了**（jitter +43%、foot sliding +44%）——latent 误差最小的解不一定 decode
得最干净（预测 latent 贴近码本边界时，requantize 会跳码，解码后就是台阶）。

## 实现

新增 `--recon-inference-path`（配合 `--recon-weight > 0`，默认关，旧行为不变）：

- 监督对象从"连续 decode" 改为 **`decode(requantize(clip(z_hat)))`**，与推理
  (`feedback_state`) 完全同序；
- 量化查找是冻结的最近码（值运算，无梯度），用 straight-through 估计器把梯度留在预测 latent 上：
  `z_st = z_clipped + (z_quantized − z_clipped).detach()` —— 前向值=显示的量化 latent，
  梯度 d(z_st)/d(z_clipped)=I；
- unroll 训练里 aux motion loss 与反馈共用同一个中间步预测（"监督的正是训练 rollout 用的"），
  因此放开了旧的 "fb solver ≠ solver 时禁止 aux loss" 限制；末步（无反馈）用完整 solver，
  中间步用 `--feedback-solver-steps`；
- TF 阶段（p=0）同样逐 step 监督 motion（回归目标仍是 GT motion，与 decouple 一致）。

## 验证

- 新增 3 项单测：decode 输入逐点等于 requantize 后的预测（STE 前向精确、浮点残差 ≤1e-8）、
  梯度确实到达 field、默认（不带 flag）保持连续 decode 旧语义；`test_auxiliary_loss_rides_the_feedback_solver_on_intermediate_steps`
  验证中间步=fb solver、末步=full solver。
- 全量 CPU 套件 **310 通过**。
- 真实数据冒烟（u4+fb2+recon 1.0）：6 updates 正常，`train_recon=0.77`。

## 成本

每步都要反传 `decode + Euler 积分`，实测 300 updates：**16.7s → 31.1s（1.86×）**。
即 u4+fb2 从 ~2.2min/epoch 升到 ~4.1min/epoch。可接受的取舍：这是第一次让 field 收到
渲染管线的梯度。

> **2026-10-11 晚更新（重要）**：深度诊断（[motion_quality_diagnosis/REPORT.md](../motion_quality_diagnosis/REPORT.md)）
> 表明显示抖动的主因是**每帧采样噪声被模型放大的倍率**（TF-only 3.3× → rollout 5.1-6.4×），而不是
> 量化台阶（残差/跳码率全程平稳）。**不加 recon、不动权重**，仅把交互采样噪声降到 std=0.25，即可让
> jitter 回到 VQ 水平（0.326 vs 0.328）且 ADE/MPJPE 更好。因此本文的 recon 微调降为可选项；
> 若仍要跑，用修正后的剂量（0.25 + 锚 0.1 + 仅末步），且仅在上面的备选路线之外单独评估。

## 建议的后续训练

curriculum100 完成后（latent-only 基线，约 ep79/100 仍在跑），用它做**微调**（同架构可直接
`--init-ckpt`，不需要从零）：

```bash
$PY training/train_flowmatch_vqvae_controller.py \
  --db database_100sty_375_memmap_nomirror \
  --vqvae-ckpt checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt \
  --norm-stats .../norm_stats.npz --vq-target-cache cache/vq_targets_vq150ep_nomirror_repair_v2 \
  --style-code-prior cache/style_code_prior_vq150ep_nomirror_repair_v2.npz \
  --latent-stats checkpoints/flowmatch_label_nomirror_150ep/latent_stats.npz \
  --init-ckpt checkpoints/flowmatch_label_curriculum100/best.pt \
  --out-dir checkpoints/flowmatch_label_motionrecon_ft30 \
  --field-arch transformer --no-use-motion-token \
  --epochs 30 --lr-schedule-epochs 30 --lr 1e-4 --min-lr 1e-5 \
  --seq-len 5 --unroll-steps 4 --feedback-solver-steps 2 \
  --recon-weight 1.0 --recon-inference-path \
  --decouple-prob .3 --decouple-latent matched_root --noise-std 1 --solver-steps 8 \
  ...
```

判定：同一 90-clip 探针的 **jitter / F.S. 下降**（目标 ≤ tr6 的 0.96/0.91）且
ADE 不退化（≤8–10cm）、style hits 不降。若 recon 权重导致 flow 退化，把权重从 1.0 降到 0.3–0.5
再试；观测 `w_recon` 与 flow loss 的相对份额。

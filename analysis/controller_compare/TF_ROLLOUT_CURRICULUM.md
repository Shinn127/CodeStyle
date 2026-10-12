# TF → rollout 衔接训练（curriculum，2026-10-11）

目标：解决 exposure bias 的同时避免"硬切换"到 rollout 训练带来的两个问题——
(a) 训练分布与前期 TF 阶段突变导致 loss 跳变/风格退化；(b) 学习率曲线与阶段不匹配。
实现为**同一 run 内的按 epoch 课程**，不需要换 checkpoint、不需要重配 optimizer。

## 机制

新增三个参数（仅 `--unroll-steps > 1` 有效，resume 强校验）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--feedback-start-epoch E` | 1 | 从第 E 个 epoch（1-based）起引入模型自反馈历史；之前为纯 GT 多步 TF |
| `--feedback-ramp-epochs R` | 0 | 从 0 线性爬升到目标比例所需的 epoch 数；0 = 到 E 直接全量 |
| `--feedback-probability P` | 1.0 | 目标反馈比例（比如 0.8 = 20% 窗口永远用 GT 历史） |

每个**窗口**（不是每步）抽一次门控：窗口要么整段用自己的反馈（`D(q̂)`+反馈 latent），
要么整段用 GT 行。这样单窗口内历史来源一致，不会出现"第 1 步自反馈、第 2 步 GT"的混叠。
`p=0` 的 epoch **完全不调用 Euler/decode**（中间步预测不需要），所以纯 TF 阶段速度与单步训练同级。

调度的实际比例每 epoch 记录在 `history.jsonl` 的 `feedback_curriculum`（`probability` 与
`actual_fraction`），val 恒为 `steps=1` 不受课程影响。

## 权重与优化稳定性（"过渡/权重变化"处理）

1. **优化器状态不重置**：AdamW 动量跨阶段连续，避免切换点二次收敛震荡。
2. **学习率单条 cosine**：与 label-150 一致把 `--lr-schedule-epochs` 设为总 epoch 数，课程只是
   在其中按 epoch 展开，不需要分段重启。
3. **课程期用微调级 lr**：从已有 checkpoint（`flowmatch_label_nomirror_150ep/best.pt`）续训时
   用 `--lr 1e-4`（不是 5e-4）；从头训练可保持 5e-4。
4. **比例只在 epoch 边界变化**：单 epoch 内固定，梯度尺度连续；ramp 步长 = P/R，R≥2 时每级
   ≤0.5（推荐 R=2–4）。
5. **噪声已按上一版决定关闭**（unroll>1 强制 0），不存在"噪声 + 课程"叠加。

## 推荐配方（label controller 微调，从 6-token best 起）

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
PY=/home/shinn/miniforge3/envs/mcc/bin/python

$PY training/train_flowmatch_vqvae_controller.py \
  --db database_100sty_375_memmap_nomirror \
  --vqvae-ckpt checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt \
  --norm-stats checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz \
  --vq-target-cache cache/vq_targets_vq150ep_nomirror_repair_v2 \
  --style-code-prior cache/style_code_prior_vq150ep_nomirror_repair_v2.npz \
  --latent-stats checkpoints/flowmatch_label_nomirror_150ep/latent_stats.npz \
  --init-ckpt checkpoints/flowmatch_label_nomirror_150ep/best.pt \
  --out-dir checkpoints/flowmatch_label_curriculum50 \
  --conditioning-mode label --epochs 50 --lr-schedule-epochs 50 --lr 1e-4 --min-lr 1e-5 \
  --seq-len 8 --unroll-steps 3 --train-feedback-context \
  --feedback-start-epoch 6 --feedback-ramp-epochs 4 --feedback-probability 1 \
  --feedback-solver-steps 2 \
  --decouple-prob .3 --decouple-latent matched_root \
  --prediction-mode target --flow-loss-space normalized --noise-std 1 --solver-steps 8 \
  --batch-size 512 --microbatch-size 0 --num-workers 8 \
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1 \
  --style-proto-normalization none --style-selector-init random \
  --style-hist-weight .02 --style-entropy-weight .003 \
  --recon-weight 0 --root-recon-weight 0 --style-consistency-weight 0 \
  --tf-val-every 5 --val-every 1 --save-every 5 \
  --matmul-precision highest --no-tf32 --device cuda --seed 42
```

时间线：ep1–5 纯 GT TF（每 epoch ~54s，等价单步训练）→ ep6–9 反馈 25/50/75/100% 爬升
（u3+fb2，~3.7min/epoch）→ ep10–50 全反馈。实测冒烟（4 epoch，ramp=2）：
p=0.00/0.50/1.00/1.00，实际比例 0.00/0.49/1.00/1.00，与调度一致。

从零训练的场合：去掉 `--init-ckpt`，lr 保持 5e-4、cosine=150，课程建议
`--feedback-start-epoch 51 --feedback-ramp-epochs 10`（对齐 label 150 recipe 的噪声起点）。

## 验证记录

- 单元/集成测试 24 项（含 ramp 调度值、p=0 无 Euler/decode、p=1 与默认逐字节一致、
  每窗口二值门控、observation_latents 走 GT 分支、CLI 三阶段记录、resume 校验、unroll=1 拒参）；
  全量 CPU 套件 326 通过。
- GPU 真实数据冒烟（上面命令的 4-epoch 版）：`history.jsonl` 的
  `feedback_curriculum.actual_fraction` 随调度按 0 / 0.49 / 1.00 / 1.00 变化。

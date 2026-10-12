# Motion token 消融：controller 去除 291 维 motion 输入（2026-10-11）

## 背景与动机

controller 的 velocity field 原本有 6 个 token：`[z_t, z_prev, motion, traj, style, time]`。
其中 `motion` token 是 291 维的上一步 motion（经 `motion_proj` 投到 hidden）。

推理时喂入的 motion 恒等于 `decoder(z_prev)`，即上一帧反馈 latent 的确定性函数——这条信息
已经通过 `z_prev` token 可达。motion token 在训练时提供的是"更精确的 GT 上一帧"（GT history
或加噪历史），与推理时的可用信息分布不一致（exposure bias 的一个具体来源）；去除它可以：

- 移掉推理期无法精确复现的冗余输入通道（训练/推理条件对齐）；
- 结构更小（5 token），并使 `z_prev` token 成为上一帧状态的唯一通道；
- 作为一个干净的消融，检验 motion 直通是否真的贡献动力学质量。

## 实现（默认关闭，不影响旧模型）

| 位置 | 变更 |
|---|---|
| `arch/flowmatch_vqvae_controller.py` | `ConditionalVelocityField(use_motion_token=...)`；False 时 `motion_proj=None`，token 序变为 `[z_t, z_prev, traj, style, time]`，`position_embeddings` 5 行。`FlowMatchVQVAEController` 与两个 factory 同步透传（config key `use_motion_token`，默认 True）。 |
| `training/train_flowmatch_vqvae_controller.py` | 新增 `--use-motion-token / --no-use-motion-token`（默认 True，写入 checkpoint args）；resume 校验该字段；从 6-token checkpoint `--init-ckpt` 时自动重映射：丢弃 `field.motion_proj.*` 和 position-embedding 第 2 行，其余权重逐字节保留（`remap_token_state_dict`，5→6 反向不支持）。 |
| `Utils/motion_eval.py` | 通过 config 自动构建 5-token 模型（已测 6-token 旧 checkpoint 不受影响）。 |

旧的 6-token checkpoint（如 `flowmatch_label_nomirror_150ep/best.pt`）行为完全不变：
`use_motion_token` 缺省 True，加载、评估、交互路径与之前一致。

## 用法

从现有 6-token best 初始化做 5-token 微调（推荐，remap 自动生效）：

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
  --no-use-motion-token \
  --out-dir checkpoints/flowmatch_label_notoken_ft30 \
  --conditioning-mode label --epochs 30 --lr-schedule-epochs 30 --lr 1e-4 --min-lr 1e-5 \
  --seq-len 2 --unroll-steps 1 \
  --decouple-prob .3 --decouple-latent matched_root \
  --prediction-mode target --flow-loss-space normalized --noise-std 1 --solver-steps 8 \
  --batch-size 512 --microbatch-size 256 --num-workers 8 \
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1 \
  --style-proto-normalization none --style-selector-init random \
  --style-hist-weight .02 --style-entropy-weight .003 \
  --recon-weight 0 --root-recon-weight 0 --style-consistency-weight 0 \
  --matmul-precision highest --no-tf32 --device cuda --seed 42
```

可与 P1 的 unroll 回放组合（已冒烟验证可跑）：

```
--no-use-motion-token --seq-len 8 --unroll-steps 3 --train-feedback-context
```

评估/交互无需改动：所有入口从 checkpoint args 读取 `use_motion_token` 自动构建。

## 验证记录

- 新增 `tests/test_motion_token_ablation.py` 5 项：默认 6-token 不变；5-token 前向+反向+
  Euler 采样+feedback decode；6→5 重映射逐字节一致（除丢弃项）；5→6 被拒；eval loader
  按 config 构建 5-token 模型。
- 相关回归 74 项、全量 CPU 套件 305 项通过。
- 真实冒烟（6-token best.pt → 5-token，`--init-ckpt`）：remap 日志正常，4 updates 训练正常；
  保存的 checkpoint 经 `Utils.motion_eval._build_flowmatch_from_checkpoint` 重建（num_tokens=5、
  motion_proj=None）并完成 rollout 前向；unroll3+feedback-context 组合 6 updates 正常。

## 效果预期与判定

去除 motion token 后，flow 输出不再看到"精确的上一帧 291 维"，动力学只能从量化 latent 还原——
训练初期 flow loss 会明显回升（冒烟中 0.22→0.20/epoch 量级），需看 30 epoch 后是否收敛到与
6-token 相当。判定用既有探针（同 seed42 90-clip rollout）：

```bash
$PY analysis/controller_compare/render_controller_compare.py --mode rollout --backend none --seed 42 \
  --flow-ckpt checkpoints/flowmatch_label_notoken_ft30/best.pt
```

关注：RootADE 均值/尾部是否 ≤ 6-token 基线（7.89cm / p90 16.0cm），style acc 是否不退化
（85/90）。若 5-token 持平或更好，则交互/评估默认切到该模型并省去一个输入通道；若明显变差，
说明 motion 直通在训练分布下确实被依赖，保持 6-token 并把结论记入报告。

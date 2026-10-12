# Controller 架构切换：skip_cat_mlp 取代 transformer（2026-10-11）

按最新要求，label controller 的 velocity field 从 6-token self-attention Transformer 改为
**concatenative skip MLP（`ControlSkipCatMLP`，与 ControlOperators `networks.SkipCatMLP` 同构）**；
transformer 保留为可选项，仅用于加载旧 checkpoint 与对照实验。

## 实现

| 位置 | 变更 |
|---|---|
| `arch/flowmatch_vqvae_controller.py` | 新增 `ConditionalSkipCatMLPField`：输入拼平为 `[z_t, z_prev, motion?, traj, style, time]`（共 128+128+(291?)+84+64+64 维），经 `ControlSkipCatMLP(inp→latent_dim, hidden=flow_hidden_dim, depth=flow_layers)`。`prepare_conditions` 缓存 Euler 求解内恒定的条件向量（无 attention，无 position embedding）。Controller 新增 `field_arch ∈ {transformer, skip_cat_mlp}`。 |
| `training/train_flowmatch_vqvae_controller.py` | `--field-arch`，**新 run 默认 `skip_cat_mlp`**；resume 强校验该字段；跨架构 `--init-ckpt` 明确拒绝（报错提示）。 |
| `Utils/motion_eval.py` / 交互与离线入口 | 从 checkpoint args 自动构建对应架构；旧 checkpoint（无该字段）按 `transformer` 加载，行为不变。 |

`--flow-layers` 此时是 skip-cat 的 depth（默认 2；ControlOperators 原版常用 3）、`--flow-hidden-dim`
是 width（默认 256；原版常用 512）。`--flow-heads/--flow-ffn-dim` 对 skip_cat 无效（接口保留）。

## 兼容性与迁移

- **旧模型照常可评估/交互**：`flowmatch_label_nomirror_150ep/best.pt` 等按保存的 args 解析为
  transformer，无需改命令。
- **不能跨架构热启动**：`--init-ckpt` 6-token transformer → skip_cat 被拒绝（拓扑不同，无可映射
  语义）。skip_cat 的 field 权重只能**从头训练**；selector/prior、冻结 VQ、数据与 loss 配置不变。
- resume 一致性：transformer run 用默认（skip_cat）续训会报 `Resume configuration mismatch:
  field_arch`；续训旧 run 必须显式 `--field-arch transformer`（已实测两种行为）。

## 用法

新 controller 训练（默认即 skip_cat_mlp，可显式写出）：

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
  --field-arch skip_cat_mlp \
  --out-dir checkpoints/flowmatch_label_skipcat_150ep \
  --conditioning-mode label --epochs 150 --lr-schedule-epochs 150 --lr 5e-4 --min-lr 1e-5 \
  --seq-len 2 --unroll-steps 1 \
  --decouple-prob .3 --decouple-latent matched_root \
  --prediction-mode target --flow-loss-space normalized --noise-std 1 --solver-steps 8 \
  --batch-size 512 --microbatch-size 256 --num-workers 8 \
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1 \
  --style-proto-normalization none --style-selector-init random \
  --style-hist-weight .02 --style-entropy-weight .003 \
  --recon-weight 0 --root-recon-weight 0 --style-consistency-weight 0 \
  --tf-val-every 5 --val-every 1 --save-every 5 \
  --matmul-precision highest --no-tf32 --device cuda --seed 42
```

（epochs/lr/历史噪声起点按 150-epoch 全量预算；若走 [IMPROVEMENT_PLAN](IMPROVEMENT_PLAN.md) 的
微调路线，同样因为不能热启动，合并为一次训练：可在上述命令上加
`--seq-len 8 --unroll-steps 3 --train-feedback-context` 直接把回放训练纳入本轮。）

评估与可视化不变，换 `--flow-ckpt` 即可：

```bash
# 单窗口对照抽查
$PY analysis/controller_compare/render_controller_compare.py --styles March Roadrunner \
  --flow-ckpt checkpoints/flowmatch_label_skipcat_150ep/best.pt
# 90-clip 探针
$PY analysis/controller_compare/render_controller_compare.py --mode rollout --backend none --seed 42 \
  --flow-ckpt checkpoints/flowmatch_label_skipcat_150ep/best.pt
```

## 已验证（冒烟，非正式指标）

- 单测：`tests/test_motion_token_ablation.py` 扩到 9 项（skip_cat 前向/反向、Euler 步内条件只算一次、
  5-token 变体、跨架构权重拒绝、eval loader 按 config 构建）；全量 CPU 套件 **309 通过**。
- skip_cat 真实训练冒烟（from scratch，6 updates）：loss 正常下降，保存/加载正常；
  eval 加载后 `ConditionalSkipCatMLPField` + rollout 前向有限值；交互 headless 30 帧通过。
- 推理速度（单步 sample，含 decode 反馈）：**4.25ms → 2.77ms（约 -35%）**，无 attention 的收益直接
  体现在交互窗口；训练吞吐在 20-update 短测中持平（1.0s vs 0.9s，噪声范围内）。
- 训练/续训语义：transformer run 用默认架构续训被 `field_arch` 明确拦截；显式
  `--field-arch transformer` 可续。

## 待办

- 从零训练 skip_cat_mlp label controller（150 epoch 或按 P1 合并 unroll），用既有 240 帧
  rollout validation 选型；与现 transformer best（matched 90.30 / cross 86.48 test）对比后决定
  是否整体切换。
- 若效果相当或更好，交互/评估默认资产切到新 checkpoint；transformer 模型保留为对照。

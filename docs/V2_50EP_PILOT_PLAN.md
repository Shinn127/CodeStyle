> **历史文档（donor 路线已于 2026-10-12 移除）**：本页记录的 donor/cross-style 合成监督在当前代码中
> 已不存在（objective version 11 起）。保留原文用于回溯当时的设计与结果；现行归因见
> [TF10_DIAGNOSIS.md](../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

> **2026-10-10 状态**：镜像段已确认导致 mesh 扭曲，训练数据一律不使用镜像；含镜像库上训练的
> VQ/controller/classifier 与派生 cache 已全部删除，代码默认路径与防呆见 [NOMIRROR_CHAIN.md](NOMIRROR_CHAIN.md)。
> 本文下列内容为修订前的记录，其中涉及 `--mirror`/`--no-mirror`、`--include-mirror`、`--unmirrored-sources`
> 等开关与已被删除的 checkpoint 的说明不再适用。

# V2 controller：50 epoch 训练计划与 10 epoch 试跑

2026-10-10。本文件是 V2 两阶段训练的**执行计划**（模型与损失设计见 [`CLIP_RECIPE_V2.md`](CLIP_RECIPE_V2.md)，诊断依据见 [`analysis/clip_recipe_failure_2026-10-09/DIAGNOSIS.md`](../analysis/clip_recipe_failure_2026-10-09/DIAGNOSIS.md)）。本轮先只跑 10 个 epoch 看效果，再决定是否续到 50。

正式达标口径（不变）：同一 checkpoint、240 帧/60fps **autoregressive rollout** 上 matched≥85%、cross≥80%、unseen≥70%。当前最好 rollout 成绩为 62.90%/41.72%/1.62%（clip_recipe_gt_history50 ep35）。

## 阶段划分

| 阶段 | 入口 | 预算 | 产物 |
|---|---|---|---|
| 0：Φ style metric | `training/train_motion_style_metric.py` | ≤1000 updates，IND90 留 10 个 style 做检索诊断 | `checkpoints/clip_recipe_v2/style_metric/best.pt` |
| 1：controller | `training/train_clip_recipe_v2.py` | 50 epoch，cosine T_max=50 | `checkpoints/clip_recipe_v2/controller/{best,best_loss,latest}.pt` |

阶段 0 的 held-style 检索门限 `.60`，未过则 controller 阶段直接拒绝启动（这是防止"风格度量本身无效还继续训练"的硬门）。两阶段都从零初始化，不接受旧 checkpoint。

## 阶段 1 固定配置

| 项目 | 设置 |
|---|---|
| 初始化 | controller / Es / φ / P 全部从零；VQ150 与 normalization 冻结 |
| 数据 | 90-style IND，原 clip-level split，`seq_len=2` 语义不变；stride1 起点、按 style/action 均衡采样 |
| Batch | 有效 512（对应 8 个 240 帧 window × 64 anchors）；microbatch 512；梯度裁剪每 512 anchors 一次 |
| Optimizer | AdamW，lr 5e-4 → 1e-5，weight decay 1e-4，grad clip 1，seed 42 |
| Scheduler | 单条 cosine，T_max=**50**（试跑时仍是 50，不改 horizon） |
| 模型 | flow hidden256 / 4 blocks / 4 heads / FFN1024；Es hidden128 / 2 layers；slot 128D×8 + global 256D |
| 任务混合 | 10% 配对合成（严格 window 级 root/yaw/phase 门限，一 window 一 donor）、20% unpaired cross（FM=0，仅 style/root）、70% matched |
| 辅助损失 | style cosine=.1、root=.1；前 5 epoch 关闭，epoch 6–10 线性升满；FM=1、prior CE=.02、entropy=.003 |
| 历史条件 | `D(q_prev)+q_prev`；20% 概率加 U(0,.15) normalized latent 噪声，前 5 epoch ramp |
| 推理条件 | style CFG 1.5、10% learned-null style dropout（训练时） |
| 效率选项 | 默认 combined backward + GPU 数据驻留（10 GiB）+ Φ 引用 LRU + field CUDA graph |
| 精度 | FP32 / highest / TF32=False |

## 验证、checkpoint 与选择

- **validation 已是 rollout 口径**：每 5 epoch（以及 epoch==50、epoch==10）用 90 个未镜像 val clips × 3 固定 windows = 270 matched + 540 cross，240 生成帧、Euler8、CFG1.5、固定 Gaussian；只在初始帧读 GT。
- epoch 5 与 epoch 10 额外跑 4-case condition diagnostics（正确/错误/null reference 的 240 帧 rollout）。
- 每个改进 epoch 保存 `best.pt`（选择键：先约束 matched≥85% 且 RootADE≤5/10cm，再比 cross acc、matched acc、归一化 ADE）、`best_loss.pt`、`latest.pt`，`history.jsonl` 逐 epoch 追加。选择记录含 `seen_targets_passed`（matched≥85 且 cross≥80 是否同时成立）与 `protocol='rollout_60fps_240f'`。
- epoch 10 的诊断门限：matched≥75%、cross≥55%、RootADE≤15/20cm。正式 50-epoch 流程里不过即写 `stopped_diagnostic.json` 并停；试跑用 `--stop-after-epoch 10` 提前收尾，门限判据由人工按同一阈值核对。
- test/OOD 只在锁定 checkpoint 之后跑（`run_evaluate_style_rollout.sh`），OOD 不参与选型。

## 试跑（本次执行）与续训

试跑命令等价于正式入口，只多一个停止点（`--stop-after-epoch` 不参与严格 resume 校验，续训时可改）：

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
PY=/home/shinn/miniforge3/envs/mcc/bin/python
# 阶段 0
$PY training/train_motion_style_metric.py --device cuda --seed 42 \
  --out-dir checkpoints/clip_recipe_v2/style_metric --updates 1000 --held-styles 10
# 阶段 1：50-epoch 计划，只跑前 10
$PY training/train_clip_recipe_v2.py --device cuda --seed 42 \
  --out-dir checkpoints/clip_recipe_v2/controller \
  --style-metric-ckpt checkpoints/clip_recipe_v2/style_metric/best.pt \
  --epochs 50 --stop-after-epoch 10 \
  --flow-hidden-dim 256 --flow-layers 4 --flow-ffn-dim 1024 \
  --style-guidance-scale 1.5 --style-dropout .1 \
  --microbatch 512 --workers 8 --val-every 5 --val-batch-size 64 --diagnostic-epoch 10
```

续到 50（同版、同目标、同资产，只改停止点）：

```bash
$PY training/train_clip_recipe_v2.py --device cuda --seed 42 \
  --out-dir checkpoints/clip_recipe_v2/controller \
  --style-metric-ckpt checkpoints/clip_recipe_v2/style_metric/best.pt \
  --epochs 50 --stop-after-epoch 0 \
  ... # 其余参数与上面完全一致
```

## 试跑要看的四件事

1. **实际配对覆盖率**。V2 的合成配对门限是 window 级（root/yaw/phase），smoke 里实际只命中 **1.25%**（请求 9.75%）。历史证据表明配对监督是唯一把 cross 推上去过的机制，所以 ep10 时实际覆盖若仍 ~1%，cross 期望值要下调。
2. **rollout matched 是否仍远低于 TF**。上一轮 TF matched 98.5% 而 rollout 只有 62.9%；V2 的 CFG + AdaLN 条件是针对这个问题设计的，ep5/ep10 的 rollout validation 是第一手证据。
3. **unpaired 分支（FM=0）是否有贡献**。它只用 style/root 约束，ep5 起才开启；若 cross 不随其开启而改善，说明间接约束仍不够。
4. **每 epoch 实测耗时**，用于校正 50 epoch 的 GPU 小时（当前按 23.5 ms/update 外推约 4.4 min/epoch，但含 GPU 驻留初始化与 rollout validation 的真实数字还没测过）。

## 决策规则（试跑结束后）

- 若 ep10 validation 同时满足 matched≥75%、cross≥55%、RootADE≤15/20 → `RESUME` 续到 50（同一日志与 checkpoint 链）。
- 若 matched 或 RootADE 任一项明显出界（例如 matched<70 或 RootADE>20）→ 停在此处，先按 DIAGNOSIS 的清单定位（条件距离、null vs correct reference 差异、逐 style 分解），不再盲目加 epoch。
- cross 只有 40–50 区间且覆盖率仍 ~1% → 视为"配对监督不足"，优先解决 donor/覆盖问题而不是延长训练。

## 成本预估

- 阶段 0：1000 updates × ~25–50 ms ≈ 1–2 分钟（另加数据准备）。
- 阶段 1：试跑 10 epoch ≈ 45–55 分钟（含 GPU 驻留初始化、2 次 rollout validation、2 次 condition diagnostics）。
- 续训到 50：约 3.5–4.5 小时（按试跑实测校正）。

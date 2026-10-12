# Controller 效果改善方案（2026-10-11）

对象：label controller（`flowmatch_label_nomirror_150ep/best.pt`，交互窗口与正式评估同一模型）。
目标：压缩 autoregressive rollout 相对 teacher-forced / VQ 上界的差距，改善实时交互与 240 帧 rollout
的运动质量（轨迹漂移、尾部崩坏），保持正式协议与指标口径不变。

## 诊断结论（证据）

1. **三段分解（90 clips，seed42，240 帧）**：VQ 上界 RootADE **1.06cm** → TF **3.11cm** → rollout
   **7.89cm**（median 5.54，p90 16.04，max 31.95）。误差大头是 rollout 累积（+4.8cm），不是表征或单步。
2. **漂移构成（20 clips）**：|err| 按四分位 1.35→4.83→7.93→**13.67cm**；其中横向分量
   0.97→3.99→6.08→**10.77cm**（占终段 ~79%），heading 误差 2.25°→**9.84°**，纵向（沿行进方向）几乎不增
   （-0.01→0.53cm）。即：**朝向漂移 → 横向发散**是主模式，不是"走不动"。
3. **推理侧旋钮无效**：noise_std 1.0/0.7/0.5/0.25 → ADE 6.98/6.39/6.91/6.90cm，ODE 8→16 无实质变化。
   非单调、单 seed，不能作为可靠杠杆。
4. **训练配置缺口**：当前 label 模型 `unroll_steps=1`、`train_feedback_context=False`，历史为 GT
   motion + structured noise（ep51 起 0.3）。推理时历史是自生成的 `D(Q(z_pred))`——典型的 exposure
   bias；trainer 已内置两条直接对应此缺口的路径：
   - `--unroll-steps K`：回放自回归 K 步，每步都在反馈自生成历史下算 loss；
   - `--train-feedback-context`：首帧观测从 GT 换成 `D(q_GT)`（与反馈状态同源）。
5. **成本冒烟（同批数据、同 microbatch=256、20 updates）**：
   - 基线 unroll=1/seq2/solver8：**1.1s**
   - P1 unroll=3/seq8/solver8 + train-feedback-context：**3.0s** → **≈2.7×**
   - 组合已验证可跑通（unroll=3 + feedback-context + decouple .3 + structured noise 正常出 loss/val）。
   按现有 ~42s/epoch（solver8、seq2）粗估：**~110s/epoch → 30 epochs ≈ 55–60min**。
   （反馈积分在 unroll 步各跑一次 Euler8，是主要开销；如预算紧张可试 unroll=2 或训练反馈用
   solver4 的近似，但会与推理 Euler8 的反馈分布产生偏差，默认不做。）

## P1（主方案）：回放微调 unrolled + feedback-context

从 `best.pt` 继续微调（不是从头），只改训练分布使其覆盖推理时的自生成反馈：

| 项目 | 值 |
|---|---|
| init | `checkpoints/flowmatch_label_nomirror_150ep/best.pt` |
| unroll-steps / seq-len | 3–4 / 8（窗口需 ≥ unroll+1） |
| solver-steps（反馈积分） | 8（与推理一致） |
| train-feedback-context | True（首帧观测=D(q0)） |
| decouple-prob / latent | 0.3 / matched_root（保持） |
| history noise | 无（机制已移除；训练只做 TF + rollout 课程） |
| lr / scheduler | 1e-4 → 1e-5，cosine horizon=epochs（微调不用 5e-4） |
| epochs | 30–50（观察曲线，最多 50） |
| batch / microbatch | 512 / 先 256，OOM 再 128 |
| 精度 | FP32/highest/noTF32（不变） |
| 选型 | 每 5 epoch 现有 240 帧 rollout validation（270 matched + 540 cross），同协议、seed42 |
| 验收 | 同 seed42 的 90-clip rollout 探针：均值/中位/p90 ADE 下降；240帧尾部(181–240)不崩；style acc 不退化 |

预期不确定性：目标是收窄 TF→rollout 的 4.8cm 差距，文献中 scheduled sampling/unrolled 通常显著
降低漂移，但增益幅度不能预付；success 判定按上面探针口径，不承诺具体数值。

风险：unroll 训练会让训练变慢（GPU 小时 ×2–2.5）；若 5 epoch 内 validation 的 rollout ADE 无改善
或 style acc 掉 >1pp，停止并回退（保留原 best.pt，不动）。

历史噪声机制已**整体移除**（2026-10-11，`--history-noise-*`/`--prev-*-noise-std`/`--prev-*-dropout`/
`--history-style-probability`/`--history-mask-probability` 及相关代码全部删除）。原因：反馈步的历史
已是模型自己的 `D(q̂)`，与推理同分布，再叠加高斯/扰动只会让训练条件偏离推理；全面切换到
TF + rollout 课程后不再需要"模拟自生成历史"的替身。旧含噪声 checkpoint 仍可评估，但不能再续训。

**衔接（已实现）**：TF→rollout 课程 `--feedback-start-epoch/--feedback-ramp-epochs/--feedback-probability`
（同 run 内按 epoch 从纯 GT 多步 TF 爬升到全反馈，含过渡/权重稳定性处理与推荐配方），
见 [TF_ROLLOUT_CURRICULUM.md](TF_ROLLOUT_CURRICULUM.md)。

**并行候选（可选叠加）**：controller 去除 291 维 motion token 的消融（`--no-use-motion-token`），
动机是训练期 GT motion 直通与推理期 `decoder(z_prev)` 的分布差；实现、用法与判定见
[MOTION_TOKEN_ABLATION.md](MOTION_TOKEN_ABLATION.md)。与 P1 组合已冒烟通过。

**架构已切换**：新 run 默认 `--field-arch skip_cat_mlp`（放弃 transformer），因此 P1 不能基于
`flowmatch_label_nomirror_150ep/best.pt` 热启动——回放训练需与从零训练合并为一次（在 skip_cat
命令上加 `--seq-len 8 --unroll-steps 3 --train-feedback-context`）。细节与验证见
[FIELD_ARCH_SKIPCAT.md](FIELD_ARCH_SKIPCAT.md)。

## P2（若 P1 后交互仍差）：交互条件对齐

交互入口的轨迹条件是 planner 合成的"意图"（目标速度 + 未来外推），而训练全部使用真实 GT 轨迹——
条件分布存在系统性差异（静止起步、恒速外推都是训练未见模式）。两条路：

- 先做廉价诊断：把 planner 风格轨迹（由 GT root 速度平滑得到意图 + 外推）喂给模型跑 rollout，与
  GT 轨迹条件对比 ADE，量化条件错配对交互的实际影响（几分钟，无训练）；
- 若显著：在 P1 的微调里替换/混合一部分样本的 future traj 为 planner 风格合成计划，或收敛 planner
  到数据统计（例如 intent 平滑系数、起步策略）。改变的是条件分布，不改推理数学。

## P3（暂不推荐先做，记录备选）

- 增大 structured noise 概率/幅度：过往诊断已说明"只加噪声"证据不足；
- 推理减弱噪声/增大 ODE：本轮扫描不显著；
- 架构层改动（更长历史 token、更大模型）：成本高、与"达标模型"对比基线重置，留待 P1/P2 后。
- 显示 3-tap 平滑：交互窗口已有后处理平滑；那是显示层，不改变生成本身，不能计入指标改善。

## 运行方式（P1 实施时）

新建独立输出目录，不复用旧 run。以下命令已在 CPU/GPU 上实测可跑通（20 updates 冒烟，FP32）：

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
  --out-dir checkpoints/flowmatch_label_unroll3_ft30 \
  --conditioning-mode label --epochs 30 --lr-schedule-epochs 30 --lr 1e-4 --min-lr 1e-5 \
  --seq-len 8 --unroll-steps 3 --train-feedback-context \
  --decouple-prob .3 --decouple-latent matched_root \
  --prediction-mode target --flow-loss-space normalized --noise-std 1 --solver-steps 8 \
  --batch-size 512 --microbatch-size 0 --num-workers 8 \
  --style-num-slots 8 --style-selector-topk 4 --style-selector-temperature 1 \
  --style-proto-normalization none --style-selector-init random \
  --style-hist-weight .02 --style-entropy-weight .003 \
  --recon-weight 0 --root-recon-weight 0 --style-consistency-weight 0 \
  --tf-val-every 5 --tf-val-classifier checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth \
  --rollout-val-every 0 --val-every 1 --save-every 5 \
  --matmul-precision highest --no-tf32 --device cuda --seed 42
```

正式启动前复核：OOM 就把 microbatch 降到 128；若首 epoch 明显超过 ~2 分钟就评估 unroll=2；
`--tf-val-*` 的其他参数（clips/windows/tolerance）沿用原 label run 的值以保持选型可比。

验证命令（P1 训练中/后，不改权重）：

```bash
# 同 seed42 的 90-clip rollout 探针（每 style 随机 1 clip，240 帧）
$PY analysis/controller_compare/render_controller_compare.py --mode rollout --backend none --seed 42
# TF 对照
$PY analysis/controller_compare/render_controller_compare.py --mode reconstruct --backend none --seed 42
# GenoView 抽查
$PY analysis/controller_compare/render_controller_compare.py --styles March Roadrunner --seed 42
```

## 追加：rollout/unroll 训练提速（2026-10-11）

`--feedback-solver-steps K`：unroll 训练里，**唯一**的 Euler 积分就是构建逐步反馈 latent
（FM loss 本身是单次 field 评估，不含积分）。该参数把中间步的反馈积分从 `--solver-steps`
（默认 8）降到 K；监督的辅助 loss 若启用则自动回到全步数（K 与 aux 不一致会被拒绝）。
`0`（默认）= 与 solver-steps 相同，即旧行为；resume 会校验该字段。

实测阶梯（transformer、batch 512 整批、8 workers、1000 updates 稳态、seed42；换算 epoch=5172 updates）：

| 配置 | s/1000up | 每 epoch | 相对旧 P1 配置 | 50ep 预算 |
|---|---:|---:|---:|---:|
| u1（现行 TF 训练） | 10.5 | 54s | — | — |
| u3, 反馈=solver8（旧行为） | 55.8 | 4.8min | 1.00× | ~4.0h |
| u3, fb=2 | 42.5 | 3.7min | **1.31×** | ~3.1h |
| u3, fb=1 | 40.5 | 3.5min | 1.38× | ~2.9h |
| u2, fb=2 | **22.5** | 1.9min | **2.48×** | ~1.6h |
| u2, fb=1 | 21.7 | 1.9min | 2.57× | ~1.6h |

其他杠杆：整批 512 比 microbatch 256 快 1.18×（u3fb2：42.5 vs 50.2s/1000up）；worker 数 8→16 无增益
（非数据瓶颈）；skip_cat 架构在 u1 下快 ~24% 但质量落后（见 ATTRIBUTION_50EP.md）。

推荐：rollout 微调用 **u3 + fb2 + 整批 512**，或预算紧时 **u2 + fb2**（回放视界只有 2 步）。
fb<8 意味着训练反馈 latent 与推理 Euler8 有近似偏差（量化后常落回同一码，偏差有限），
这是显式的速度/保真取舍。

**2026-10-11 进一步优化**（反馈预测跳过 autograd 图 + 重写 quantize_only 消除库 one-hot
物化，逐位一致）：u3+fb2 已到 **25.3 s/1000up（2.2 min/epoch，50ep ≈ 1.8h）**，u2+fb2 到
18.0 s/1000up（50ep ≈ 1.3h）；相对最初 u3/solver8 累计 **2.2×/3.1×**。交互推理同步
4.25→3.76 ms。细节与复现见 [ROLLOUT_SPEED_2026-10-11.md](ROLLOUT_SPEED_2026-10-11.md)。

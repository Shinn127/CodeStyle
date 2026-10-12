> **2026-10-10 状态**：镜像段已确认导致 mesh 扭曲，训练数据一律不使用镜像；含镜像库上训练的
> VQ/controller/classifier 与派生 cache 已全部删除，代码默认路径与防呆见 [NOMIRROR_CHAIN.md](NOMIRROR_CHAIN.md)。
> 本文下列内容为修订前的记录，其中涉及 `--mirror`/`--no-mirror`、`--include-mirror`、`--unmirrored-sources`
> 等开关与已被删除的 checkpoint 的说明不再适用。

# 数据层规范与重建手册

本文档描述 375 维 motion memmap 数据集的格式契约、划分协议、重建与校验流程，以及从数据集到
VQ-VAE 的标准化训练链路。适用于 `NewCode/` 下所有训练与评估脚本。

## 1. 数据来源与目录

原始 100STYLE 数据：`/home/shinn/桌面/Projects/data/100style/`
- `Frame_Cuts.csv`：每个 style 一行，列为 `{SUFFIX}_START/_STOP`
- `<Style>_<Suffix>.bvh`：810 个文件，10 种后缀（BR/BW/FR/FW/ID/SR/SW/TR1/TR2/TR3）

数据根目录的解析顺序（`--data-root` 未指定时）：`$MOTION375_DATA_ROOT` →
`<NewCode>/data/{100style,100sty}` → `/home/shinn/桌面/Projects/data/{100style,100sty}` →
`/home/shinn/桌面/{multimedia/,}revision/data/{100style,100sty}`。

生成两个 memmap 库：

| 库 | 内容 | 帧数 | style 数 | segments |
|---|---|---|---|---|
| `database_100sty_375_memmap` | IND（前 90 style，CSV 顺序），含镜像增强 | 7,068,516 | 90 | 1458 |
| `database_100sty_375_memmap_zeroshot` | OOD（后 10 style），不镜像 | 423,710 | 10 | 81 |

后 10 个 style 固定为：`TwoFootJump, WalkingStickLeft, WalkingStickRight, Waving, WhirlArms,
WideLegs, WiggleHips, WildArms, WildLegs, Zombie`。

## 2. 特征格式（375 维，schema v4）

`meta.json` 的 `feature_layout` 强制为 `trajectory: offset 0 dim 84` + `motion: offset 84 dim 291`，
`dataset.py` 在加载时校验，任何改动都会直接报错。

### 轨迹块 84 = 7 块 × 12 采样点

上下文采样 `arange(-window, window, context_stride)`，默认 `window=60`、`context_stride=10`
→ 12 个采样点，覆盖 ±60 帧，步长 10。

| 偏移 | 内容 |
|---|---|
| 0:12 | root 位置 x（相对当前帧，root 坐标系） |
| 12:24 | root 位置 z |
| 24:36 | 朝向 x |
| 36:48 | 朝向 z |
| 48:60 | 速度 x |
| 60:72 | 速度 z |
| 72:84 | 速率（≥0） |

朝向块满足 `|(dx,dz)| = 1`（单位向量，y 分量为 0）。

### 动作块 291 = 3 + 4 × 72

**关键设计：动作块取自第 t−1 帧**（`previous_ids = frame_ids - 1`），轨迹块取自第 t 帧。
因此单帧向量本身就是「上一帧状态 + 当前轨迹上下文」，下游无需再做时序对齐。

| 偏移 | 内容 |
|---|---|
| 0:3 | root 平面速度 (vx)、偏航角速度（度）、vz |
| 3:75 | 24 个选中关节的 local_pos（24×3） |
| 75:147 | local_rot 的 xform 第 0 列（24×3，单位向量） |
| 147:219 | local_rot 的 xform 第 1 列（24×3，单位向量，与上者正交） |
| 219:291 | 24 个关节的 local_vel（24×3） |

选中关节索引见 `Utils/motion375.py::SELECTED_24_INDICES`，名称见 `SELECTED_24_NAMES`。

### 有效帧与 clip 长度

每段 clip 两端各裁掉 `window` 帧（上下文采样需要），输出帧数 = `clip_len - 2*window - 1`。
`clip_len < 2*window + 2 = 122` 的 clip 被丢弃。当前数据 810 个 entry、0 个短 clip。

## 3. 划分协议

### IND / OOD 划分

按 CSV 顺序取前 `100 - K` 个 style 为 IND，末尾 `K` 个（默认 10）为 OOD。两个库的 style
名集合不相交，OOD 库的标签独立从 0 重新编号。

### IND 内部的 train/val/test

clip 级分层划分（`assign_entry_splits` → `assign_style_splits`），保证：

- **每个 IND style 都出现在三个 split 中**（实测 90/90；每个 style：train 6–8 clips，
  val 1 clip，test 1 clip）
- 划分单位是**源 clip**（`source_entry_id`），镜像变体与其原 clip 同 split，因此镜像增强
  不会造成泄漏
- 每个 style 内部按帧数贪心配平：train 77.6–84.3%，val 8.0–12.4%，test 5.2–11.2%

全局帧占比：train 80.8% / val 10.6% / test 8.6%（7,068,516 帧）

构建时会强制校验上述性质：若有 style 拥有 ≥3 个可用 clip 却缺失任一 split，生成脚本直接报错。

## 4. 重建数据集

```bash
cd NewCode
conda activate mcc

# 先看会产出什么，不写盘
python tools/generate_database_100sty.py --dry-run

# 正式重建（12 进程约 95 秒）
python tools/generate_database_100sty.py --num-workers 12
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--zero-shot-style-count` | 10 | 末尾 K 个 style 归入 OOD |
| `--val-ratio` / `--test-ratio` | 0.1 / 0.1 | 每个 style 内按帧数配比 |
| `--split-seed` | 42 | 划分随机种子 |
| `--window` / `--context-stride` | 60 / 10 | 必须满足 `window/context_stride = 6`，否则轨迹块不是 84 维，脚本会提前报错 |
| `--coordinate-convert` | 关 | ZeroEGGS 坐标系转换 |
| `--overwrite` | 关 | 覆盖已存在文件 |

输出的 `meta.json`（schema v4）除维度与切分摘要外，还记录 `context_stride`、`trajectory_samples`、
`savgol_window`、`segment_counts`、每个 style 的 clip 覆盖表，以及
`generation.frame_cuts_sha256`（输入指纹，用于追溯是哪份 CSV 生成的）。

## 5. 校验数据集

```bash
python tools/verify_memmap_db.py                        # 两个库全量检查
python tools/verify_memmap_db.py --db <新库> --compare <旧库>   # 额外做数值对比
python tools/verify_memmap_db.py --json /tmp/report.json
```

覆盖 35 项检查，包括：schema 与文件尺寸、标签连续性、segment 不重叠/无空隙/不跨 style、
镜像对同 split 同长度、**split 之间无帧重叠**、**每个 style 出现在三个 split**、
特征不变量（有限值、朝向单位性、速率非负、旋转基正交）。退出码非 0 表示有 FAIL。

## 6. 加载数据集（`dataset.py`）

```python
from dataset import build_train_val_test_dataloaders

bundle = build_train_val_test_dataloaders(
    db_path="database_100sty_375_memmap",
    seq_len=2, stride=1, batch_size=512, num_workers=8,
    split_mode="auto",                          # 默认
    norm_stats_path="checkpoints/<run>/norm_stats.npz",
)
```

`split_mode` 取值：

| 值 | 行为 |
|---|---|
| `auto`（默认） | 有离线 split 就用（IND 库），零样本库给单一 split，其余情况走 clip 级划分 |
| `precomputed` | 强制要求 `segments.json` 的离线 split；有 segment 缺 split 时直接报错 |
| `clip` | 强制按 clip 重新划分（窗口绝不跨 split） |
| `none` | 全部窗口合成一个数据集，不产出 val/test |

返回的 item 字段：`x [T,375]`、`traj [T,84]`、`motion [T,291]`、`label`、`start`、`segment_id`。

归一化统计只在 **train 窗口**上估计，并广播给 val/test；传入 `norm_stats_path` 时，文件存在则
加载、不存在则计算后写入，因此续训与评估可以复用完全相同的统计量。日志会打印 16 位
fingerprint 便于比对。

## 7. 标准化训练链路

```bash
# ① VQ-VAE（论文 MotionVQ 配置：见下方“超参口径”说明）
OUT=$PWD/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048
python training/train_basic_vqvae_375to291.py \
  --db database_100sty_375_memmap \
  --out-dir "$OUT" --split-mode auto --norm-stats "$OUT/norm_stats.npz" \
  --device cuda --input-mode traj_motion \
  --encoder-arch skip_cat_mlp --decoder-arch skip_cat_mlp \
  --encoder-layers 6 --decoder-layers 6 \
  --epochs 50 --batch-size 512 --seq-len 2 --stride 1 \
  --amp-dtype fp32 --no-tf32 --matmul-precision highest --no-compile \
  --recon-weight 3.0 --vq-weight 1.0 --vel-weight 15.0 \
  --latent-l1-weight 0.01 --latent-vel-weight 0.1 --root-weight 5.0 \
  --eval-every 10 --eval-metric mpjpe \
  --best-metrics mpjpe,jitter,foot_sliding,root_ade --best-min-delta 0.0 \
  --eval-clips 90 --eval-window-frames 60 \
  --num-workers 8 --seed 42 \
  --use-rotation-trick --kmeans-init --kmeans-iters 10 --dead-code-threshold 2
```

### 训练期表征层评估

每 `--eval-every`（默认 10）个 epoch 在**固定的** val 窗口集上做教师强制整段重建，指标直接
调用离线评估同一套代码（`Utils/motion_eval.compute_sequence_metric_sums`），因此与论文表格同源：

| 指标 | 含义 | 论文（32×4, K=2048） |
|---|---|---|
| `MPJPE (cm)` | 关节位置误差，含积分 root 轨迹 | 3.653 ± 0.187 |
| `Jitter (cm)` | 时序平滑度 | 0.252 ± 0.006 |
| `F.S. (cm)` | 脚部滑动 | — |
| `rootADE (cm)` | root 轨迹误差 | — |

**论文那一行的口径**（已逐位复现，见 `FID_EXTRACTOR_AND_PAPER_PARITY.md` §2）：权重
`basic_vqvae_375to291_32_4_2048/best.pt` 训练于**只有 10 个风格、无镜像**的
`database_100sty_375_memmap_head10`，测试集是同口径的 905 个 60 帧窗口，FID 用的是
**10 类**提取器。用当前生成器重建同口径库
（`tools/generate_database_100sty.py --style-limit 10 --zero-shot-style-count 0`）
后，该 checkpoint 的输出与论文表格逐位一致；把它丢到带镜像的 90-style 库上则失真
（见同文档 §2.1）。

**评估窗口必须保持 ~60 帧**：该指标会对重建出的 root 速度做积分，窗口越长 MPJPE 越被累积
漂移主导（同一模型 60 帧 2.33 cm vs 300 帧 188 cm）。60 帧是唯一能与论文 3.653 cm 直接比较的量纲。

`--eval-metric` 是**记录用**的主指标（默认 `mpjpe`，写进 `history.jsonl` 的 `best_metric`），
它不再单独决定 best。真正决定 best checkpoint 的是 `--best-metrics`：**列出的每个指标都必须
比当前 best.pt 上的取值更好**，才覆盖 best.pt。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--best-metrics` | `mpjpe,jitter,foot_sliding,root_ade` | 逗号分隔，全部变好才保存；`codebook_usage` 为越大越好，其余越小越好。`none` 退回单指标规则（只用 `--eval-metric`）|
| `--best-min-delta` | `0.0` | 每个指标要求的改善量（cm）。0 = 严格变好；负数允许同等程度的变差，例如 `-0.02` 容忍噪声 |

行为细节：

- 主指标（`--eval-metric`）总是被自动并入判据集合，所以 `best.pt` 与 `best_metric` 永远同步；
  首次评估（还没有参照 checkpoint）一定接受。
- 判据的参照值就是**当前 best.pt 上的指标**，因此被接受的 checkpoint 构成一条全指标单调变好的链，
  而不是「某一项刷新纪录就覆盖」。`best.pt` 里同时保存了 `best_metrics` / `best_metric_values`，
  续训时自动恢复参照；判据集合变化时会打印 warning 并重置参照。
- 每次评估打印 `[Gate] accept: mpjpe 12.34->11.98, ...` 或 `[Gate] reject, best.pt kept: jitter 1.01->1.05`，
  完整判据写入 `history.jsonl` 的 `best_gate` 段。
- **副作用要注意**：四项严格变好时 `best.pt` 可能长时间不更新（例如 MPJPE 一路下降但 Jitter 抖动
  上升），此时 `best.pt` 会停在较早的 epoch。脚本连续 3 次拒绝后会打印 `[Warn]`，并按 `--save-every`
  （默认 10）保存 `epoch_XXXX.pt` 作为兜底；需要放宽就用 `--best-min-delta -0.02`，或把噪声大的
  指标从 `--best-metrics` 里去掉。

评估集由 `--eval-clip-seed` 确定且全程固定，每次评估输出写入 `history.jsonl` 的 `eval` 段。

### 训练期 eval 走量化路径，离线 eval 走 AE 路径（重要）

两条评估路径测的不是同一个东西，比较论文数值时务必对齐：

| 路径 | 调用 | 是否过码本 | 用途 |
|---|---|---|---|
| 训练期 eval（`Utils/vqvae_eval.py`）| `model.encode()` → `decode(z_q)` | **是**（z_q 由码本查表得到）| best 判据、训练监控 |
| 离线 eval（`evaluation/evaluate_motion_375to291.py` → `Utils/motion_eval.reconstruct_motion_batch`）| `model.encode_pre_vq()` → `decode(z_e, from_indices=False)` | **否**（`_resolve_latents_btd` 只做 reshape，直接解码 z_e）| 论文表格数值 |

后果：同一个 checkpoint 在两条路径上的差别很大，量化误差完全体现在训练期 eval 上。
100k steps 那次训练（skip_cat_mlp 6/6，L_root=3.0）实测：

| 指标 | 离线（AE 路径，论文口径）| 训练期（量化路径）| 论文 MotionVQ |
|---|---|---|---|
| MPJPE (cm) | 1.486 | 1.429 | 3.653 |
| Jitter (cm) | 0.237 | 0.710 | 0.252 |
| F.S. (cm) | 0.289 | 0.565 | 0.277 |
| FID | 0.542 | — | 3.060 |

也就是说论文表格里的 MotionVQ 数值（3.653 / 0.252 / 3.060 / 0.277）是**AE 路径**的结果
（`MM-2025-11-0162/5 - code/arch/basic_vqvae_375to291.py` 里 `encode_pre_vq` 返回 z_e、
`decode(from_indices=False)` 不做量化），量化带来的 jitter / F.S. 代价不在论文数字里。
要报告「量化表征」的真实质量，必须用 `model.encode()` → `decode(z_q)` 这条路径。

⚠️ **上表 FID 一列已作废**：那两次数值（0.542 / 3.060）是用 10 类提取器
`fid_feature_extractor_head10` 在 90-style 库上算的，而该提取器只见过 10 个风格
（在 10 个 head 风格上、镜像+非镜像混合只有 57.8% 准确率：无镜像 100%、镜像 3.2%，
见 `FID_EXTRACTOR_AND_PAPER_PARITY.md` §1.3），属于分布外特征空间，与论文的
FID 不可比。现行口径：用 100 style 提取器
`checkpoints/fid_feature_extractor_full100_v4/best.pt`，两个模型都测同一套无镜像测试集
（`analysis/run_paper_parity.sh`），结果见 `FID_EXTRACTOR_AND_PAPER_PARITY.md` §3。

参考下限：同一套 metric 代码下，**真值自身**在 90 style 测试集上 jitter = 0.201 cm、
F.S. = 0.213 cm（prediction = reference），即任何模型的 jitter/F.S. 都不可能低于该值。

### `evaluation/evaluate_motion_375to291.py` 的两个口径开关

| 参数 | 默认 | 说明 |
|---|---|---|
| `--allow-legacy-quantizer` | 关 | 允许加载量化器用旧键（`quantizer.embedding.weight` / `quantizer.ema_*`）的 checkpoint：丢弃那 3 个张量、只走 AE 路径。论文期权重（`basic_vqvae_375to291_32_4_2048`、`..._head10` 等）必须带这个开关才能评测；不带时报错并提示。量化路径对这类 checkpoint **不可用**（码本保持在初始化值）|
| `--decode-smooth-taps` | 1（关）| 解码端对 291 维动作块做二项式时序滤波（3 = `[1,2,1]/4`，5 = `[1,4,6,4,1]/16`，须为奇数），轨迹 84 维原样保留。用于把「逐帧换码」噪声压掉，属于**带时序解码器的表征**档位，报告时不能与论文的逐帧解码口径混用 |

### FID 特征提取器（口径要求：CAMDM 实现 + 完整 100 style 全量训练）

- **现行主口径 = CAMDM 官方提取器**（`training/train_fid_feature_extractor_camdm.py`，与
  https://github.com/AIGAnimation/CAMDM 的 `Evaluation/feature_extractor/train_classifier.py`
  逐项对齐：同网络、45 维输入（15 个 ybot 关节）、60 帧窗口、`clip_offset=10`、
  batch 1024、Adam lr=1e-3、50 epoch、CE、**无训练/验证划分**）。
  产物：`checkpoints/fid_feature_extractor_camdm100/100style_position_classifier.pth`
  （100 个风格、非镜像、391,373 个窗口，训练准确率 99.90%）；
  当前 240 帧默认分类器为 `checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/`（训练窗口不含镜像）。
- 评测脚本会**自动**按 checkpoint 的 `joint_set`/`in_feat` 选择 45 维（ybot15）或 72 维
  （本项目 24 关节）输入（`--fid-joint-set auto`，会打印实际口径）。
- 备份口径：`checkpoints/fid_feature_extractor_full100_v4/best.pt`（本项目 24 关节、72 维、
  100 风格；作为交叉验证与旧实验的连续口径）。
- 重建目标变体：`checkpoints/fid_feature_extractor_recon100/100style_position_recon.pth`
  （同款 conv 主干 + 512 维瓶颈，但训练目标是**重建 45 维关节位置窗口**、不用风格标签，
  `training/train_fid_feature_extractor_recon.py`）。它对应论文稿里 "motion reconstruction objective"
  的表述；加载器按 checkpoint 的 `objective` 字段自动分发，评测链路无需改参数。三种口径
  （CAMDM 分类器 / 本项目 72 维 / 重建）在同一测试集上给出同一排序，见
  `FID_EXTRACTOR_AND_PAPER_PARITY.md` §3.4。
- **不要再用 `fid_feature_extractor_head10`** 评测 90-style 库：它只见过 10 个风格，
  在其余 80 个风格上属于分布外特征空间，FID 数值与论文口径不可比。
- ⚠️ 跨提取器的 FID 不可比（45 维 vs 72 维、特征层不同量纲）；CAMDM 自己的 FID 还额外
  按 style 分组、除以特征维度，与本项目评测脚本的池化 FID 不是同一个数。
- 同口径复测：`./analysis/run_paper_parity.sh all` + `python analysis/summarize_paper_parity.py`
- 先验校验：`python tools/verify_style_code_prior.py --expect-entropy-floor 4.834`（21 项：溯源/几何/**独立重算**/
  划分纯度/镜像敏感性/信息量/熵/与 selector 一致性；失败退出码非 0）。本先验实测：counts 重算 0 误差、
  $\mathbb E_\Omega[\mathcal H(\bar\pi)]=4.8342$、selector dense KL 0.364 nats、TopK 硬选择与先验 top-4 重合 42.4%
  （见 `RUN_STYLE_PRIOR_50EP.md` §8）
  （切换提取器用 `EXTRACTOR=... TAG_PREFIX=...`）。

### style prior 监督口径 `--style-prior-loss`

控制器训练里 style-code prior 的监督方式（详见 `Utils/style_code_prior.py::style_prior_divergence`）：

| 值 | 含义 | 备注 |
|---|---|---|
| `dense`（默认）| 对 `softmax(logits/τ)`（TopK 之前的分布）算 CE/KL | 良定，KL=0 可达，梯度 `p−π̄`；实测该项停在先验熵下界 H(π̄)≈4.8 nats |
| `js` | 对字面稀疏 π 用有界 JS 散度 | 有界（≤ln2），但存在不可达下界 |
| `sparse_eps` | `log(π+ε)`，ε=1e-3 | 最小补丁，仍以覆盖度为主 |
| `legacy` | 原 `log(clamp(π,1e-8))` | 复现旧 checkpoint 口径（该项 ≈18，占 loss ~86%） |

为什么必须改：论文式(6) 的 `KL(π̄‖π)` 在 π 为 TopK 稀疏时恒为 +∞（π̄ 全支撑）；旧实现的 clamp 让该项被
`-log(1e-8)×(top-K 外质量)` 支配——实测占控制器训练 loss 的 ~86-90% 且几乎不随训练下降。
新的默认权重 `--style-hist-weight 0.02`、`--style-entropy-weight 0.02` 是按 dense 口径的数值尺度重标的
（旧 0.2 / 1e-3 是针对 ~18 的伪常数与"失效的熵项"设的）；实测 flow 项占比回到 80.6%、prior 15.7%、熵 1.2%。
wrapper 对应 `STYLE_PRIOR_LOSS` / `STYLE_HIST_WEIGHT` / `STYLE_ENTROPY_WEIGHT`。

### 二阶（加速度）平滑项 `--acc-weight`

`--acc-weight` 在 291 维动作块上追加 `|Δ²pred − Δ²target|`（`--temporal-loss-type` 决定 L1/MSE）。
一阶的 `--vel-weight` 只匹配「步长」，逐帧输出噪声仍会留在二阶差分里，而 jitter / F.S. 量的正是
二阶差分；量化路径下 32 个码本的组合在相邻帧之间 100% 会变化，把二阶误差从 AE 路径的 0.013
抬到 0.055（归一化单位）。该顶直接压这个频段。

- **需要 `--seq-len ≥ 3`**（否则二阶差分没有定义），脚本启动时会校验并报错。
- 默认 0（论文配方不变）。实测 λ_acc=3.0 时该项贡献约 0.10（同期 vel=0.35、recon=0.21），
  建议 3–5。
- 与论文的 vel 项共用 `--temporal-loss-type`。

### root 轨迹约束

`--root-weight` 在 291 维动作块的**前 3 维**（root 平面速度、偏航角速度、竖直速度）上追加一个
L1 项。这 3 维在 291 维平均里只拿到 3/291 的重建梯度，但它们的误差正是积分成轨迹漂移的那部分。

默认 0（保持论文配方不变），建议取值 3–10。实测 `--root-weight 5.0` 的效果：

| epoch | train_root | eval rootADE | eval MPJPE |
|---|---|---|---|
| 1 | 0.2075 | 4.44 cm | 12.07 cm |
| 5 | 0.0632 | 1.98 cm | 7.33 cm |

### 超参口径提醒

`training/train_basic_vqvae_375to291.py` 的**脚本默认值就是论文值**：
$(\lambda_{rec},\lambda_{vel},\lambda_{z1},\lambda_{zv})=(3.0,15.0,0.01,0.1)$、$\lambda_c=0.25$、50 epochs。
而 `run_train_basic_vqvae_375to291.sh` 的 wrapper 默认值（recon 1.0 / vel 3.0 / z_l1 0 / z_vel 0）
来自后续实验，会覆盖成非论文配置——用 wrapper 时要显式传回论文值。

### 量化器：vector_quantize_pytorch

量化器使用 **`vector_quantize_pytorch` 1.31.4** 的 `VectorQuantize`，由
`arch/basic_vqvae_375to291.py::LibraryVectorQuantizer` 适配。原先手写的 `BasicVectorQuantizer`
及其自定义 rotation trick 已删除，量化路径只有这一个实现。

配置映射（`arch/basic_vqvae_375to291.py::LibraryVectorQuantizer`）：

| 我们的概念 | 库参数 | 值 |
|---|---|---|
| latent 维度 | `dim` | 128（= 32 × 4） |
| 码本大小 | `codebook_size` | 2048 |
| 每个 codebook 的维度 | `codebook_dim` | 4 |
| codebook 数量（token 数） | `heads` | 32 |
| 是否各 head 独立码本 | `separate_codebook_per_head` | `False`（**共享**一个码本） |
| EMA | `decay` / `ems_update` | 0.99 / True |
| 死码阈值 | `threshold_ema_dead_code` | 见下 |
| rotation trick | `rotation_trick` | True |
| commitment | `commitment_weight` | 0.25 |

因为 `dim == codebook_dim × heads = 128`，库不会插入投影层，`project_in` / `project_out` 都是
`Identity`，量化器**可训练参数为 0**（码本是 EMA 缓冲），与原实现一致。

已验证：eval 模式下 `encode → indices → quantizer.lookup(indices)` 往返**逐位相等**
（`max|Δ| = 0.0`），因此 VQ target cache 的 `lookup_latents` 路径是安全的；`vq_loss /
commitment_loss` 恒为 0.25；EMA 会更新码本；梯度经 rotation trick 正常回传到编码器。

#### ⚠️ `threshold_ema_dead_code` 的单位是"每 batch 的 EMA 分配计数"，不是步数

它的单位是"每 batch 的 EMA 分配计数"，**不是"连续未使用步数"**，所以不能沿用旧实现里
patience=5 这类数值。在 batch=512、32 heads、2048 codes 下，每步共 16384 次分配，
均匀时**每码每步约 8 次**。
真实模型上的实测稳态重置率：

| threshold | 稳态每步重置码数 | 占码本 | 结论 |
|---|---|---|---|
| 2 | 0–1 | 0.01% | 健康（库的上游默认值） |
| 8 | 0–4 | 0.35% | 健康 |
| 16 | 约 1900 | 94% | 码本几乎每步被重置 |
| **512** | **恒为 2048** | **100%** | **码本永不收敛，训练退化为随机码本** |

所以 `--dead-code-threshold` 的默认值是 **2**。若传入的值导致连续 3 步重置超过码本的一半，
适配器会打印告警。

#### rotation trick 正确性校验

`--use-rotation-trick` 由库实现（`vector_quantize_pytorch.py` 的 `rotate_to` /
`efficient_rotation_trick_transform`）。校验结果：

- **前向值保持**：`rotate_to(z, q)` 与 `q` 的差异仅 float32 噪声（~1e-6，且与 `|q|/|z|` 无关），
  所以解码器收到的仍是真正的码本向量。
- **梯度映射**：反向得到的映射为 `(|q|/|z|) · R`，其中 `R = I − 2wwᵀ + 2q̂ẑᵀ`（`w` 为 `ẑ`、`q̂`
  的角平分线单位向量）。数值验证 `R` **正交且 det = +1**（正交性误差 4.4e-16），即真正的
  **旋转**（两次 Householder 反射复合），把 `z` 方向映射到 `q` 方向。那个 `|q|/|z|` 标量是
  有意的幅度缩放，与前向的模长变化一致（实测 det(J) = 0.51⁴ 精确吻合 `(|q|/|z|)⁴`）。



### MotionVQ 网络架构（ControlOperators）

MotionVQ 的编解码器采用 ControlOperators 的拼接式跳连网络，**当前配置为编解码器各 6 层
`skip_cat_mlp`**：

| 项 | 值 |
|---|---|
| 编码器 | `ControlSkipCatMLP(inp=375, out=128, hidden=512, depth=6)` — 6 个 Linear，5 个 LayerNorm |
| 解码器 | `ControlSkipCatMLP(inp=128, out=291, hidden=512, depth=6)` — 同上 |
| 参数量 | 编码器 2.58 M + 解码器 1.90 M + 量化器 8 K = **4.49 M** |
| 激活 | GELU（`skip_cat_mlp`）；`mlp` / `skip_mlp` 用 ELU |
| 跳连形式 | 每块 `LayerNorm → cat(h, x) → Linear(hidden+inp, hidden) → GELU`，输出层同样接收 `cat(h, x)` |
| 吞吐 | fp32 / batch 512 约 **77.6 k samples/s**（`skip_mlp` 同深度为 91.1 k） |

选 `skip_cat_mlp` 而非 `skip_mlp` 的原因：ControlOperators 的 `networks.py` 明确注释，去掉
LayerNorm 后**网络较深时未训练输出会过大、训练不稳定**，而 `SkipMLP` 不含 LayerNorm。6 层的
`skip_cat_mlp` 每块都有 LayerNorm，规避了该问题。

三个网络类与上游 `gouruiyu/ControlOperators` 的 `networks.py` 已做数值等价验证：36 组维度×深度
组合加边界情况（`inp == hidden` 时首块残差被激活、`depth=1`）在同权重下输出逐位相同，参数量、
`state_dict` 键与默认初始化也一致。

`control_operators.py` 的 typed operator DSL（`Vector`/`Rotation`/`And`/`Or`/`Set`/...）未使用：
VQ-VAE 输入是单一同构 375 维向量，等价于对若干 `Vector` 做 `Struct` 拼接，没有架构损失。但注意
我们的 6 维旋转编码是**列优先** `[col0(3), col1(3)]`，上游 `Rotation` 算子按行优先展平成
`[a00,a01,a10,a11,a20,a21]`，只在需要接上游算子管线时才需转换。

**论文正文待同步**：论文写的是"2-layer transformer encoder with 256 hidden units and a
512-dimensional feedforward layer"，与本实现不符（VQ 路径没有 transformer）。用 `skip_cat_mlp`
的话，编解码器描述都应改成 ControlOperators 的拼接式跳连网络。

历史背景：论文已发表的数字来自旧代码 `Code/VQEMA.py` / `Code/VQBase.py`，那里用的是
`TransformerEncoderModel`（4 层 / nhead=4 / hidden=512 / GELU / 位置编码 / dropout=0.2）+
`MLPDecoder`（SiLU + Dropout 0.2）。换架构后有两处实质变化：**dropout 0.2 → 0**、
**激活 SiLU → GELU**，新旧结果不可直接对比。

```bash
# ② VQ target cache（用 ① 的目录，norm stats 自动取 ckpt 同级）
python tools/precompute_vq_targets.py \
  --db database_100sty_375_memmap \
  --vqvae-ckpt "$OUT/best.pt" \
  --out-dir "$PWD/cache/vq_targets_vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048" \
  --overwrite

# ③ style code prior（split=train，仅用训练帧统计）
python tools/precompute_style_code_prior.py \
  --db database_100sty_375_memmap \
  --vq-target-cache "$PWD/cache/vq_targets_vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048" \
  --out "$PWD/cache/style_code_prior_vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048.npz" \
  --split train --style-num-slots 16

# ④ flow-matching 控制器
python training/train_flowmatch_vqvae_controller.py \
  --db database_100sty_375_memmap \
  --vqvae-ckpt "$OUT/best.pt" \
  --norm-stats "$OUT/norm_stats.npz" \
  --vq-target-cache "$PWD/cache/vq_targets_vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048" \
  --style-code-prior "$PWD/cache/style_code_prior_vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048.npz" \
  --use-style-code-selector
```

**②③④ 必须使用同一份 `--db` + `--vqvae-ckpt` + `--norm-stats`**。`VQTargetCache.validate()`
会对三者的绝对路径与模型维度做全等校验，不匹配直接报错而非静默降级。

## 8. 环境注意事项

- **`torch.compile` 在当前环境不可用**：torch 2.13.0+cu132 的 inductor 在导入
  `flex_attention` 模板时抛 `AssertionError: duplicate template name`，三种 compile-mode 均复现。
  一律加 `--no-compile`。实测关闭编译反而更快（batch 512 + bf16 达 93k samples/s，
  单 epoch 约 61 秒），因此这只是配置项调整，不影响精度。
- `--load-into-memory` 需要约 11.9 GB 常驻内存（X.f32 全量）。本机 30 GB 内存，默认关闭。
- 数据管道只接受 **schema v4** 的 memmap 库；v3 库（含旧的 `*_legacy_v3`）会被直接拒绝并提示重建。
- 数据层已清理：旧 v3 库、失效的 `cache/vq_targets_*`（其 meta 指向已消失的 DB 与 checkpoint、
  帧数与当前库不符）与派生的 `style_code_prior_*.npz` 均已删除。`cache/zeroshot_style_recipes/`
  里的 5 个 JSON 是论文 demo 的风格配方输入，体积仅 28 K，故保留。
- `checkpoints/` 下的模型权重（含论文已发表的 head10 / `_full` / `-full_new` 变体）未删除。

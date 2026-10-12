> **历史文档（donor 路线已于 2026-10-12 移除）**：本页记录的 donor/cross-style 合成监督在当前代码中
> 已不存在（objective version 11 起）。保留原文用于回溯当时的设计与结果；现行归因见
> [TF10_DIAGNOSIS.md](../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

# 无镜像链路的 VQ 与 label controller 训练规划

2026-10-10。前提已就绪：数据全部重建为无镜像并逐字节验证（见"阶段 0"）；本文件规划 VQ 表示层
与 label 条件控制器的训练、验证、选型和评估。**所有旧数字都来自含镜像链路，不能与新结果直接对比。**

目标口径不变：同一 checkpoint、240 帧/60fps **autoregressive rollout** 上 matched≥85%、cross≥80%、
（unseen/OOD 不属于 label 模型范围，无对应条件通道）。含镜像链路的历史最好 rollout 成绩为 label 模型 **81.90% / 74.02%**（RootADE 12.5/16.3 cm），
它是本次复现的对照基线（数值会因数据口径变化而不同）。

## 阶段 0：数据（已完成）

| 数据集 | 路径 | 规模 | 校验 |
|---|---|---|---|
| IND 90 风格 | `database_100sty_375_memmap_nomirror` | 729 段 / 3,534,258 帧（train 549/2,648,792；val 90/441,354；test 90/444,112） | `tools/verify_memmap_db.py` 35 项 0 失败（1 warning：单帧 traj 单位范数 2.2e-3，属既有离群） |
| OOD 10 风格 | `database_100sty_375_memmap_zeroshot` | 81 段 / 423,710 帧 | 30 项 0 失败 |

由 `tools/generate_database_100sty.py` 从源 BVH 重新生成（无镜像路径，`augmentation: None`，meta 带
`frame_cuts_sha256` 指纹）。生成结果与先前"从旧库剥离镜像"的产物**逐字节一致**（X.f32/Labels.i32
全等、segment 身份与 split 分配全等），说明镜像移除没有改变原始片段的任何数值。

**2026-10-10 重新划分（split 元数据修正）**：原划分按"帧数配平 + 最长 clip 优先"，而 clip 时长与
locomotion 类型强相关（FW/BW/TR1 ≈7.5k 帧、BR/FR ≈4k、SR 2.4k、ID 0.7k），导致 val/test 系统性
偏向跑动类（val 88% = BR/FR，test 92% = FR/SR/BR，且 90/90 个 style 的 train 都缺 FR/SR）。
新规则 `suffix_balanced_rotation_per_style`：**按整段 locomotion 类型（suffix）留出，并按 style
轮转**，使 val/test 在 8 个后缀上各约 11 个 style；每个 style 的 train 保留 6–8 个动作。仅重写
`segments.json`/`meta.json` 的 split 字段（帧数据未动），并用 `tools/verify_memmap_db.py` 复验通过。
10 个 OOD 风格整体留作 zero-shot，**不参与 train/val/test**（OOD 库 81 段无 split 字段）。
受影响的下游：VQ/controller 的 train/val/test 全部按新划分重跑；style prior 按 train 重算；
合并的 100-style classifier 库需重建（其 split 字段来自旧划分，虽然两个提取器默认的 temporal
划分不读该字段）。回归测试见 `tests/test_generate_database_parallel.py::SuffixBalancedSplitTests`。

## 阶段 1：VQ 150 epoch（表示层）

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
PYTHON_BIN=/home/shinn/miniforge3/envs/mcc/bin/python EPOCHS=150 bash scripts/run_train_basic_vqvae_375to291.sh
```

输出 `checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/`
（含 `best.pt`、`latest.pt`、`norm_stats.npz`、逐 epoch history）。

| 项目 | 设置 |
|---|---|
| 结构 | skip_cat_mlp encoder/decoder 各 6 层，32 slots × 4 dims = 2048 codes |
| 序列 | `seq_len=2`、`stride=1`、`split_mode=auto`（用库内 precomputed split） |
| 损失权重 | recon 3.0、VQ 1.0、velocity 15.0、latent L1 0.01、latent-vel 0.1、root 3.0、acc 0（论文配方）+ **code-align 2.0**（本轮增强，见下） |
| 量化 | EMA + rotation trick + kmeans 初始化（10 iters）+ 死码重置（阈值 2） |
| 精度/批 | FP32、`highest`、TF32 关、无 compile、batch 512 |
| 验证 | 每 10 epoch 在 90 个 clip / 60 帧窗口上评估；`best.pt` 仅在 mpjpe、jitter、foot_sliding、root_ade **全部不劣化**时替换 |
| 预算 | 每 epoch 2,857,160/512 ≈ **5,579 updates**；真实数据实测 **~33.5k samples/s → 约 85 s/epoch → 150 epoch ≈ 3.6 小时** |

### 连续/离散 code 对齐（本轮新增）

旧的 VQ 只有 commitment（`||z_e − sg[e_q]||²`，经 `vq_weight=1.0`）与 rotation trick，且 EMA 下
`codebook_loss` 恒为 0——离散码与连续 latent 只在 latent 空间被间接拉近，在**函数空间**没有任何约束。
本轮加入两项（`arch/basic_vqvae_375to291.py::compute_loss`）：

1. **`code_align_loss`（新损失，权重 `--code-align-weight`，入口默认 `CODE_ALIGN_WEIGHT=2.0`）**：
   `MSE(D(z_e), sg[D(e_q)])` —— 用**离散路径**的输出作目标，要求连续 latent 经过冻结解码器后复现
   离散码的输出，使两条码路可互换（对下游"缓存离散索引 / 反馈重新量化"的闭环最相关）。目标侧 detach，
   避免退化为常量解码；梯度只到 encoder/decoder，EMA codebook 仍是 buffer（不接收梯度）。
2. **`latent_code_gap`（新增日志指标）**：`mean((z_e − z_q)²)`，每次前向零成本，用来直接观察
   latent 空间的对齐程度。

真实数据实测（batch 512、nomirror 库，训练开始阶段）：`align≈0.0054→0.0040`、`z_gap≈0.0140→0.0088`，
两者随步数下降；同批 `recon≈0.24`，所以权重 2.0 时该损失占总量不到 1%，但梯度占比约 10%——
它的作用是持续缩小两条码路的**功能差**，而不是主导重建。若首个 epoch 后 `z_gap`/`align` 不再下降，
可把 `CODE_ALIGN_WEIGHT` 提到 5 再观察（每 epoch 记录，便于对照）；`0` 可完全回到论文配方。

**验收**（不是论文达标）：loss 四段（recon/vel/root/latent）单调下降且无 NaN；MPJPE 收敛到与前一代
同一量级（前代 60 帧窗口 MPJPE 约 3.6 cm 是**含镜像训练**的数，新数允许有偏移，但不应显著变差）；
死码率正常、codebook 使用不塌缩。若 MPJPE 明显劣于前代，先查数据/精度差异再继续。

## 阶段 2：VQ target cache 与 style prior

```bash
PYTHON_BIN=/home/shinn/miniforge3/envs/mcc/bin/python bash scripts/run_precompute_vq_targets.sh
PYTHON_BIN=/home/shinn/miniforge3/envs/mcc/bin/python bash scripts/run_precompute_style_code_prior.sh
PYTHON_BIN=/home/shinn/miniforge3/envs/mcc/bin/python tools/verify_style_code_prior.py \
  --db database_100sty_375_memmap_nomirror \
  --vq-target-cache cache/vq_targets_vq150ep_nomirror_repair_v2 \
  --prior cache/style_code_prior_vq150ep_nomirror_repair_v2.npz
```

产物：`cache/vq_targets_vq150ep_nomirror_repair_v2/`、`cache/style_code_prior_vq150ep_nomirror_repair_v2.npz`。
校验器逐项重算计数与 JS saliency，并通过 `--expect-entropy-floor` 检查 L_prior 常数下界（历史值 ≈4.834 nats）。
**cache/prior 必须与 VQ 内容指纹一致**；入口会拒绝不匹配的旧产物。

## 阶段 3：两个 FID 特征提取器（评估口径，不参与生成训练）

数据：`database_100sty_375_memmap_fid_full100_nomirror`（已建好：810 段 / 3,957,968 帧 / 100 style）。

```bash
PYTHON_BIN=/home/shinn/miniforge3/envs/mcc/bin/python bash scripts/run_train_fid_feature_extractor_camdm.sh
PYTHON_BIN=/home/shinn/miniforge3/envs/mcc/bin/python bash scripts/run_train_fid_feature_extractor_recon.sh
```

| 项目 | 分类器（CAMDM 家族） | 重建版 |
|---|---|---|
| 输出 | `checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/` | `checkpoints/fid_feature_extractor_recon100_nomirror/` |
| 目标 | 100 类 CE（15 ybot 关节 / 45 维 / 240 帧@60fps） | 窗口重建 L1（512 维 bottleneck = FID 特征） |
| 规模 | 50 epoch、batch 1024、Adam 1e-3 | 同左（可用 `--recon-type/--vel-weight` 调目标） |
| 预算 | 约 15 s/epoch（实测）→ **13–15 分钟** | 约 30 s/epoch → **25 分钟** |

**本轮相对 CAMDM 协议的两处有意偏离**（都记录在 checkpoint provenance 里）：

1. **dropout 调整**：每个提取器 **2 个 dropout**——conv 特征处（`--dropout`）与 1024 维 hidden 到
   FID 特征之间（`--head-dropout`），两者本轮都设为 **0.3**（CAMDM 原本只有前者、为 0.5）。两者都写入
   checkpoint；旧 checkpoint 缺字段时按原 0.5/0 构建，行为不变。
2. **train/val 划分 + 按 val 选 best**（`--val-ratio 0.1`、`--split-by`）：
   - `temporal`（默认）：每条 clip 留出**尾段**做 val，并在两段之间留 `window_size` 帧的 guard band，
     保证窗口不重叠。**每个 (style, suffix) 同时出现在 train 与 val**——因为同一 style 的不同 clip
     是不同 locomotion 类型，按 clip 抽会把"动作差异"混进 style 指标；`temporal` 让两边的动作构成一致。
   - `clip`：整条 clip 留出（unseen clip，但各 style 的动作类型不匹配，即上述问题的口径）。
   - `suffix`：按后缀整组留出（所有 style 留出同一个动作；unseen clip 且动作一致，需显式
     `--val-suffixes`）。
   - 选型规则：分类器取 **val accuracy 最高**，同分取 val loss 更低，再取更早 epoch；重建版取
     **val 重建误差最低**。`100style_position_classifier.pth` / `100style_position_recon.pth`
     （所有评估入口加载的文件名）保存的就是 val 选出的模型；`final.pt` 保存最后一个 epoch，
     `best.pt` 单独留档。provenance 记录 `split_by / val_ratio / dropout / best_epoch / best_val_*`
     与 `selection_rule`。
   - clip 太短、无法容纳 guard band 时，训练**不会静默跳过验证**：打印 `[Warn]` 并在 checkpoint 里
     记 `val_split_warning='empty_val_split'`、`selection_rule='no_val_split'`。
   - `--val-ratio 0` 可完全恢复 CAMDM 的 all-clips 协议（无 val、无 best 选择）。

实测（真实数据、3 epoch、temporal）：train_windows 324,443 / val_windows 22,213（810 条 clip 全部
两边参与）；val_acc 41.6% → 60.4%，train_acc 99.8%。训练窗口不含镜像，因此所有 seen/OOD 数字与
旧 240f 分类器**不可直接比**，跨口径比较必须在同一分类器内部进行。

## 阶段 4：label controller 150 epoch

复现历史最优 label 配方（`run_train_flowmatch_tf150_staged_noise.sh`，默认路径已全部指向无镜像资产）：

```bash
PYTHON_BIN=/home/shinn/miniforge3/envs/mcc/bin/python DEVICE=cuda bash scripts/run_train_flowmatch_tf150_staged_noise.sh
```

输出 `checkpoints/flowmatch_tf150_staged_noise/`（若同名目录已存在，入口会拒绝；需要时用 `--out-dir` 指定新目录，例如加 `_nomirror` 后缀）。

| 项目 | 设置 |
|---|---|
| 初始化 | 从零；固定 VQ150ep(新) 与 normalization；6-token/layer256/2 层/FFN1024 |
| 条件 | label → learned selector（prior 初始化、8 slots、top-k4、raw prototype）→ style token |
| 损失 | flow 1.0、dense prior CE 0.02、entropy 0.003；recon/root/style-consistency/recipe 全 0 |
| 目标 | normalized target prediction；Euler8 仅用于生成 |
| 配对 | `decouple_prob=0.3`、`matched_root`（同 action suffix、root/yaw 合格 donor，在完整 batch 内选，source root 保留） |
| 历史增强 | epoch 1–50 干净 GT 历史；epoch 51–150 起 30% 概率加噪（latent .3、pose/vel .2、root .05、rotation 不加噪） |
| 优化 | AdamW lr 5e-4→1e-5（cosine horizon 150）、wd 1e-4、grad clip 1、seed 42、有效 batch 512（默认完整 512 microbatch，实测比 128×4 快约 1.5×） |
| 验证 | 每 5 epoch 固定 810-case TF 验证（90 未镜像 val clip × 3 窗口：270 matched + 540 cross，240 生成帧）；选型先过 matched≥85%、matched ADE≤5 cm、cross ADE≤10 cm，再比 cross acc、matched acc、归一化 ADE |
| 预算 | 每 epoch 2,857,160 个训练 transition ≈ **5,579 updates**；含镜像时为 11,158 updates/146 s，按同吞吐外推 **约 75 s/epoch → 150 epoch ≈ 3.2 小时**（加 30 次 TF 验证 ≈ 15 分钟） |

**训练中要看的**：`matched_flow_loss` 与 `cross_flow_loss` 分开记录、实际 `decoupled_fraction`
（历史为请求 0.3 中约 0.20–0.23 合格）、`history_noise_fraction`（epoch 51 起 0.3）、
每 5 epoch 的 validation matched/cross 与 ADE。

## 阶段 5：锁定与评估（正式口径）

训练结束用 validation 选型锁定 checkpoint（`best.pt`，选择依据记录在 checkpoint 的 `selection` 字段），
然后跑**官方 rollout 协议**：

```bash
FLOW_CKPT=checkpoints/flowmatch_tf150_staged_noise/best.pt \
OUT_ROOT=eval_results/label_nomirror_rollout \
  bash scripts/run_evaluate_style_rollout.sh         # seen(1000×5)，240 帧 rollout（该入口只评 seen）
```

报告：matched / cross 的 style accuracy（按 source_entry_id 聚类 bootstrap CI）、RootADE、
cross 的 source retention、逐 target style 分解（历史失败集中在 28/90 个 style <10%）。同时用
（OOD/recipe 评估工具 `evaluate_unseen_style.py` 与对应文档已归档到 `docs/archive/unseen_2026-10-10/`；label 模型不评 unseen。）

## 顺序、决策点与成本

| 阶段 | 预计耗时 | 决策点 |
|---|---|---|
| 0 数据 | ✅ 已完成 | — |
| 1 VQ 150ep | 1.5–3 h | 首 epoch 校正；MPJPE 与前代同量级才继续 |
| 2 cache + prior | 20–40 min | `tools/verify_style_code_prior.py` 必须全过 |
| 3 classifier 240f | 10–15 min | 上界检查（real motion 分类准确率应接近 100%） |
| 4 controller 150ep | 3.5–4 h | ep10/20 看 cross 与 decoupled 覆盖；ep50 看加噪阶段是否破坏干净 val |
| 5 评估 | ~20 min | matched/cross 与目标对比、逐 style 分解（seen only） |

总 GPU 约 **5–8 小时**。中途每完成一阶段暂停汇报，不自动追加实验；若跨阶段需要停机，用各自入口的
`RESUME` 路径续训（strict resume 校验配置与指纹，不能改 batch/microbatch 等被校验参数）。

## 阶段 5 结果（2026-10-10 完成）

见 [最终评估报告](../eval_results/label_nomirror_rollout/REPORT.md)。**目标达成**：
封存 test + 240 帧 rollout 下 matched **90.30%**（CI [85.98, 93.84]）、cross **86.48%**
（CI ±0.92，source retention 0.38%）；validation 曲线 ep150 为 85.93 / 84.81。
label controller 训练 150 epoch / 1.91 h，
速度优化后 45 s/epoch（见 [提速报告](../analysis/training_speed_2026-10-10/REPORT.md)）。

## 与旧链路的已知差异（预期）

1. 训练数据少一半（无镜像），所以每 epoch 更快但**有效样本多样性降低**；历史增强的比例/幅度建议先用
   原值，若 matched 掉得明显再单独讨论（不要与本轮同时改）。
2. 分类器换成无镜像 240f，历史 accuracy 数字整体不可比；跨链路的绝对值差异要归因到分类器口径。
3. OOD/unseen 不适用于 label 模型（无 OOD 条件通道）；recipe 类模型的历史结论（histogram/TopK 表示瓶颈）
   记录在归档的 `docs/archive/unseen_2026-10-10/`，不作为 label 模型的研究项。

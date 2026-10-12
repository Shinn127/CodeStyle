# NewCode：当前运行说明

2026-10-10。label controller 的目标为同一checkpoint、240帧/60fps autoregressive rollout上的matched≥85%、cross≥80%（已达成，见下）；unseen/OOD 不属于 label 模型评估范围（无对应条件通道），仅 recipe 类模型涉及。

新版V2已实现：`clip→冻结VQ序列→temporal Es→TopK4 π→非线性code聚合→8 slot/global条件→逐层AdaLN/cross-attention→CFG flow→clip→quantize→decode`。Recipe从最终π进入同一条件路径；输出style/root约束作用于量化后的单次FM终点。已通过小数据回归和真实数据GPU smoke，正式Φ预训练和controller训练尚未执行。详见[V2运行说明](docs/CLIP_RECIPE_V2.md)和[主线方案](analysis/clip_recipe_failure_2026-10-09/NEXT_SOLUTION.md)。

## 目录结构（2026-10-13 重组）

```text
NewCode/
├── README.md  dataset.py  run_tests.py  pytest.ini   # 门口文件
├── training/      # 14 个 train_* 入口（controller/VQ/classifier 训练）
├── evaluation/    # 7 个 evaluate_*/simulate_realtime_* 入口（封存评估、实时模拟）
├── tools/         # 8 个数据/缓存工具（generate_database、precompute_*、verify_*）
├── scripts/       # 全部 shell 包装（38 个 run_*.sh，含 run_tests.sh）
├── docs/          # 规划与运行说明（VQ_LABEL_TRAIN_PLAN、NOMIRROR_CHAIN 等）
│   └── archive/   # 历史方案与已下线源码归档
├── Utils/  arch/  Visualization/  tests/             # 库代码与回归
├── analysis/      # 实验记录与报告（只增不改）
├── checkpoints/  cache/  database_*/  eval_results/  # 生成资产（gitignore）
└── .git/          # 独立仓库（2026-10-13 起，与父 revision 仓库解耦）
```

所有 shell 入口以 `ROOT_DIR` 定位资源，可从任意目录执行：`bash scripts/<name>.sh`。

## 环境与保留产物

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
conda activate mcc
```

数据为 `database_100sty_375_memmap_nomirror`（90 IND、原 clip-level split、**已剥离镜像**）与 `database_100sty_375_memmap_zeroshot`（10 OOD，本身无镜像段）。**镜像段已确认会导致 mesh 扭曲，训练数据一律不使用镜像**：在含镜像库上训练的全部 VQ / controller / classifier 及派生 cache 已于 2026-10-10 删除；训练与评估入口默认路径已切到无镜像资产，并加 `require_unmirrored_db` 防呆，重建顺序见 [NOMIRROR_CHAIN.md](docs/NOMIRROR_CHAIN.md)。

已清理25.10GiB中间权重、重复派生数据和旧cache；原始IND/OOD、各run的best/latest、关键统计和历史评估保留。旧规划与sweep集中在[历史源码归档](docs/archive/README.md)，清单见[清理记录](docs/cleanup_2026-10-09/README.md)。

**2026-10-10 无镜像重做**：含镜像库上训练的检查点/缓存/eval 产物已删除，代码默认路径与防呆已切到无镜像资产，重建顺序见 [NOMIRROR_CHAIN.md](docs/NOMIRROR_CHAIN.md)。

**2026-10-10 结果**：label controller（无镜像链路，`checkpoints/flowmatch_label_nomirror_150ep/best.pt`）在封存 test 的 240 帧 rollout 上达到 **matched 90.30% / cross 86.48%**（目标 85/80 达成）；报告见 [REPORT](eval_results/label_nomirror_rollout/REPORT.md)。

**2026-10-12 新最优**：6-token transformer + TF→rollout课程（`checkpoints/flowmatch_label_6tok_curriculum100/best.pt`，选型 ep65）在同一封存 test 上达到 **matched 96.40% / cross 91.42%**（vs 上一代 +6.1/+4.9pp，ADE 7.96/9.93cm）；训练配方见 [RETRAIN_6TOK_PLAN.md](analysis/controller_compare/RETRAIN_6TOK_PLAN.md)，评估报告见 [REPORT](eval_results/6tok_rollout/REPORT.md)。交互窗口推荐该模型 + `NOISE_STD=0.5`（低抖动且无滑行）。

文档索引（当前有效）：[VQ_LABEL_TRAIN_PLAN.md](docs/VQ_LABEL_TRAIN_PLAN.md)（VQ 与 label controller 训练规划）、[NOMIRROR_CHAIN.md](docs/NOMIRROR_CHAIN.md)（无镜像链路与重建顺序）、[CLIP_RECIPE_V2.md](docs/CLIP_RECIPE_V2.md)（V2 模型与训练设计）、[V2_50EP_PILOT_PLAN.md](docs/V2_50EP_PILOT_PLAN.md)（50 epoch 计划与试跑）、[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)（数据集格式与生成）、[STYLE_CLASSIFIER_240F.md](docs/STYLE_CLASSIFIER_240F.md)（风格分类器）、[INTERACTIVE_LABEL_CONTROL.md](docs/INTERACTIVE_LABEL_CONTROL.md)（实时交互）、[FP32_PERFORMANCE.md](docs/FP32_PERFORMANCE.md)（性能记录）。历史检查点清单已移入 [docs/archive/](docs/archive/)。

## 测试与短测

```bash
bash scripts/run_tests.sh                  # 全部CPU回归，排除GPU用例
bash scripts/run_tests.sh core             # 控制器/prior/recipe/OOD核心回归
bash scripts/run_tests.sh gpu              # 只跑GPU用例；要求CUDA可见
bash scripts/run_tests.sh all              # CPU+GPU；要求CUDA可见
bash scripts/run_tests.sh gpu --collect-only
bash scripts/run_tests.sh core -k recipe
```

回归只用disposable fixtures，不读取真实训练数据或启动完整训练。测试保持FP32/highest/noTF32，限制小fixture的CPU线程，并输出慢项。真实训练吞吐单独测试（现役工具；旧的 `profile_training.py` 已随 donor 清理归档）：

```bash
python analysis/interactive_quality_2026-10-12/implementation/benchmark_training.py --updates 100
```

该短测读取当前真实cache/train references，随机初始化临时controller，限定updates，不保存正式checkpoint。`--streaming`评估新样本及recipe首次构建，省略它则重复已预热样本，不能代表完整训练成本。最新新样本流66.85→47.09ms/update；实测与一致性核验见[批处理/recipe性能报告](analysis/pipeline_speed_2026-10-09/REPORT.md)，原field优化见[前次报告](analysis/training_speed_2026-10-09/REPORT.md)。

## 训练

采用FP32、高精度matmul、无TF32、冻结VQ、normalized target FM。现有teacher-forced入口关闭reconstruction，stride1；正式style validation使用rollout。

```bash
DRY_RUN=1 bash scripts/run_train_clip_recipe_practical.sh

# 新主线：先Φ最多1000 updates，通过检索门限后controller从零最多50 epoch。
DRY_RUN=1 bash scripts/run_train_clip_recipe_v2.sh
```

`practical`入口是已验证失败的history50配置，保留作诊断。新主线使用`run_train_clip_recipe_v2.sh`，默认独立目录`checkpoints/clip_recipe_v2`；训练阶段、门限、续训和测试命令见[V2运行说明](docs/CLIP_RECIPE_V2.md)。Label历史基线入口为 `run_train_flowmatch_tf150_staged_noise.sh`。

V2新训练默认合并反传、GPU冻结Φ缓存、10 GiB只读数据预算及512整批CUDA field图；当前驻留数据约8.83 GiB，单模型短测峰值约10.3 GiB。交替相同新样本的对照35.46→23.48ms/update，耗时下降33.8%；短测不能保证总训练时间。`GPU_DATA_CACHE_GB`与`CUDA_FIELD_GRAPH=0/1`可调整新run；旧V2续训自动保留原执行参数。详见[FP32性能](docs/FP32_PERFORMANCE.md)。

旧六token训练入口默认`--worker-batches`、`--fp32-math-attention`、`--style-val-batch-size 64`，保留原collate对照。Resume需保持相同backend、batch、目标版本与数据指纹；旧模型可评估，不混用TF/rollout选型状态。

## 正式评估

2026-10-12 Label动作连续性修订：两帧latent历史、FP32只读GPU缓存及按批八步反馈；不新增loss。
**2026-10-12晚 donor 训练路径已全面移除**：短训归因显示 donor 的合成跨风格目标损害闭环运动质量
（无donor纯TF的tf10短训 MPJPE 7.97 / jitter 0.57 / F.S. 0.72 vs 含donor 10.28 / 2.14 / 2.10）；
`matched_root_targets`、StyleDonorBank、`--decouple-prob/--decouple-latent`、`--donor-*` 及全部
donor 指标均已删除（源码归档于 [docs/archive/donor_sources_2026-10-12/](docs/archive/donor_sources_2026-10-12/README.md)），
训练 objective 统一为 version 11，donor 时代 ckpt 不能 resume。history 短训入口仍为
`run_train_flowmatch_history_phase10.sh`（rollout）与 `run_train_flowmatch_history_tf10.sh`（纯TF），
见 [修订说明](docs/MOTION_HISTORY_REPAIR.md) 与 [TF10 归因](analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md)。

**2026-10-12 checkpoint 清理**：旧 controller / 特征提取器 checkpoint 已全部删除（9.9G → 175M），
仅保留 VQ150ep-nomirror（`best.pt`+`norm_stats.npz`）、240f 风格分类器与 150ep `latent_stats.npz`。
训练脚本从零或显式 `INIT_CKPT` 启动；交互/评估入口需要 `FLOW_CKPT=<新训练模型>`。
历史指标与封存报告（`eval_results/`）不受影响。详见 [清理记录](docs/cleanup_2026-10-12/CHECKPOINT_CLEANUP.md)。

无镜像Label150模型曾以 `flowmatch_label_nomirror_150ep/best.pt` 接入实时键盘控制（该 checkpoint 已在
2026-10-12 清理中删除；交互入口保留，用 `FLOW_CKPT=<新训练模型>` 运行）：`bash scripts/run_interactive_label_control.sh`。WASD移动、QE转向，右侧搜索列表或`[`/`]`切换style，Space暂停、R重置；逐帧autoregressive、Euler步数跟随checkpoint（ODE4模型为4，旧模型为8；`ODE_STEPS=`可覆盖）/FP32、默认clip→quantize→decode反馈（`--no-requantize`可切到连续latent直接decode的对照路径），实际root由模型输出积分。默认启用显示平滑/固定骨长/foot IK，P对照原始姿态、I切换IK；后处理不写回模型反馈或正式评估。详见[交互控制说明](docs/INTERACTIVE_LABEL_CONTROL.md)与[2026-10-11复测记录](analysis/interactive_label_nomirror/verification.json)。原有实时模拟入口（默认路径同为nomirror资产，`--no-requantize-z`可关闭量化）仍用于离线生成/播放。

```bash
DRY_RUN=1 bash scripts/run_evaluate_style_rollout.sh

# 锁定验证集checkpoint后写新目录，不覆盖旧报告（checkpoint 用当前训练产物）。
FLOW_CKPT=checkpoints/<your_run>/best.pt \
OUT_ROOT=eval_results/new_locked_rollout \
bash scripts/run_evaluate_style_rollout.sh
```

240个生成帧、60fps、std1、clip→quantize→decode；Euler步数跟随checkpoint（`--flow-ode-steps`缺省即ckpt，ODE4模型为4）；只在初始帧读GT motion/latent。Matched1000、cross5000，固定source/target/noise。Classifier排除GT初始帧，CI按source_entry_id聚类。该入口只评seen；label模型不评unseen。OOD/recipe评估工具已移入 [docs/archive/](docs/archive/)。

## 数据与来源

[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)保留motion375布局、坐标、fps、split与构建细节。VQ已固定，现存cache不需重建；确有需要时：

```bash
bash scripts/run_precompute_vq_targets.sh
bash scripts/run_precompute_style_code_prior.sh
```

默认已对齐VQ150/repair_v2；不要盲目使用 `--overwrite`。自定义模型/数据库须显式指定VQ_CKPT/OUT并重新核验指纹。

Full100 classifier数据库是IND+OOD合并副本，已逐字节核验后删除。重训classifier前先重建：

```bash
python tools/merge_memmap_dbs.py --primary database_100sty_375_memmap_nomirror \
  --secondary database_100sty_375_memmap_zeroshot \
  --out database_100sty_375_memmap_fid_full100_nomirror --secondary-split train
```

OOD仅进入报告classifier的数据，不进入generator训练。Classifier说明见[STYLE_CLASSIFIER_240F.md](docs/STYLE_CLASSIFIER_240F.md)。

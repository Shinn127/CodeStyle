# 无镜像（unmirrored）链路：数据、代码修订与重建顺序

2026-10-10。镜像段已确认会导致 mesh 扭曲，因此**训练数据一律不使用镜像**。含镜像库上
训练的全部 VQ / controller / classifier 与派生 cache 已删除；代码默认路径已切到无镜像资产，
并加了强制校验，避免任何入口静默使用含镜像数据。

## 数据

| 数据集 | 路径 | 内容 |
|---|---|---|
| IND 90 风格 | `database_100sty_375_memmap_nomirror` | 729 段 / 3,534,258 帧（原库按段剥离 mirror，帧逐字节保留；train 549 段/2,857,160 帧，val 90/373,192，test 90/303,906） |
| OOD 10 风格 | `database_100sty_375_memmap_zeroshot` | 本身无 mirror 段 |
| 合并 100 风格（仅 classifier 用） | `database_100sty_375_memmap_fid_full100_nomirror` | 由前两者合并（`tools/merge_memmap_dbs.py`），**尚未生成** |

剥离工具：`filter_memmap_db.py`（按段剥离、帧逐字节复制、写 provenance 与自校验报告，
`--verify-windows` 默认 256；测试 `tests/test_filter_memmap_db.py`）。
数据库生成侧：`tools/generate_database_100sty.py` 现在**默认不含镜像**，需要时显式 `--mirror`
（`--no-mirror` 保留为兼容别名）。

## 防呆：`Utils/runtime.py::require_unmirrored_db`

`require_unmirrored_db(db)` = 原 `validate_memmap_db_dir` + 拒绝含镜像的库（`segments.json`
里任何 `mirror=true`，或 `meta.mirror_layout` 不是 `none`）。已接入全部训练与预计算入口：
`training/train_basic_vqvae_375to291.py`、`training/train_flowmatch_vqvae_controller.py`、
`training/train_fid_feature_extractor_{camdm,recon,head10}.py`、`training/train_clip_recipe_v2.py`、
`training/train_motion_style_metric.py`、`tools/precompute_{vq_targets,style_code_prior}.py`，以及
lmm / mann / diffusion / mvae 等旧基线 trainer。评估入口不做此限制（冻结输入诊断可能
需要历史库），但其 DB 默认值同样已切到无镜像库。

## 资产命名（代码默认值已按此对齐）

| 用途 | 路径（相对 NewCode） |
|---|---|
| VQ 150 epoch | `checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/` |
| VQ target cache | `cache/vq_targets_vq150ep_nomirror_repair_v2/` |
| style prior | `cache/style_code_prior_vq150ep_nomirror_repair_v2.npz` |
| 240 帧 classifier | `checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth` |
| 逐帧 classifier（style-consistency 用） | `checkpoints/fid_feature_extractor_camdm100_frame_nomirror/100style_position_classifier.pth` |
| 合并 classifier 数据 | `database_100sty_375_memmap_fid_full100_nomirror/` |

## 重建顺序（尚未执行）

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
PY=/home/shinn/miniforge3/envs/mcc/bin/python

# 0) 合并 classifier 数据集（OOD 仅用于 classifier/评估，不进入 generator 训练）
$PY tools/merge_memmap_dbs.py --primary database_100sty_375_memmap_nomirror \
  --secondary database_100sty_375_memmap_zeroshot \
  --out database_100sty_375_memmap_fid_full100_nomirror --secondary-split train

# 1) VQ 150 epoch（默认 DB/OUT 已指向无镜像）
bash scripts/run_train_basic_vqvae_375to291.sh            # EPOCHS=150

# 2) VQ target cache 与 style prior（默认改为短名 vq150ep_nomirror_repair_v2）
bash scripts/run_precompute_vq_targets.sh
bash scripts/run_precompute_style_code_prior.sh
$PY tools/verify_style_code_prior.py --db database_100sty_375_memmap_nomirror \
  --cache cache/vq_targets_vq150ep_nomirror_repair_v2 --prior cache/style_code_prior_vq150ep_nomirror_repair_v2.npz

# 3) 240 帧 classifier（不含镜像）
bash scripts/run_train_fid_feature_extractor_camdm.sh

# 4) controller / Φ（V2 入口；资产默认值来自 Utils/clip_recipe_assets.py）
RUN_ROOT=checkpoints/clip_recipe_v2 bash scripts/run_train_clip_recipe_v2.sh
```

## 本次修订覆盖的代码

- 上述 56 个 `.py` / `.sh` 的默认路径（DB、VQ、cache、prior、classifier、旧 run 的 flow checkpoint）。
- `Utils/runtime.py` 新增 `require_unmirrored_db`，并在 15 个训练/预计算入口替换原校验。
- `Utils/clip_recipe_assets.py`：V2 链路（`training/train_clip_recipe_v2.py`、`training/train_motion_style_metric.py`）的
  `--db/--vqvae-ckpt/--vq-target-cache/--style-code-prior` 默认值。
- `training/train_fid_feature_extractor_camdm.py`：`--include-mirror` 默认关闭；wrapper 的 `INCLUDE_MIRROR` 默认 0。
- `tools/generate_database_100sty.py`：镜像改为显式 `--mirror` 才启用。
- `tools/verify_memmap_db.py`：新增 `unmirrored contract` 一致性检查（`mirror_layout=none` 时不得有 mirror 段）。
- `dataset.py` 文档、`README.md` 数据段、`CHECKPOINT_MAP.md` 顶部说明。

校验：全量测试 `bash scripts/run_tests.sh cpu` / `gpu`；`tools/verify_memmap_db.py --db database_100sty_375_memmap_nomirror`
应只剩抽样相关的 `traj direction unit-norm` 一项（该离群帧来自未镜像的 BentForward/FW，属既有数据属性）。

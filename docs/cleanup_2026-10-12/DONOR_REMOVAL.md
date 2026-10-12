# Donor 训练路径全面移除记录（2026-10-12）

**触发**：10 epoch 对照短训归因——donor 的跨风格合成目标损害闭环运动质量。
同协议封存评估（1000 matched / 5000 cross, 240f, Euler8, std=1）：

| 模型 | MPJPE | jitter | F.S. | RootADE matched/cross | style acc matched/cross |
|---|---:|---:|---:|---:|---:|
| **tf10（donor-free 纯TF）** | **7.97** | **0.568** | **0.717** | **4.78 / 6.10** | 79.4 / 65.7（cross 泛化缺口） |
| phase10（history+donor+rollout） | 10.28 | 2.140 | 2.100 | 7.32 / 9.28 | 95.9 / 90.8 |
| 6tok（旧 donor 训练） | 11.23 | 1.791 | 1.676 | 7.96 / 9.93 | 96.4 / 91.4 |

用户判定：以运动质量为第一优先，全面移除 donor 训练内容。cross-style 泛化缺口作为
后续独立问题处理（不能用 donor 修补）。

## 移除内容

**主训练脚本 `train_flowmatch_vqvae_controller.py`**
- 函数：`matched_root_targets`、`apply_style_decoupling`、`apply_style_decoupling_windows`
- CLI：`--decouple-prob`、`--decouple-latent`、`--cross-donor-source`、`--donor-window`、
  `--donor-windows-per-clip`、`--donor-phase-aware`、`--donor-match-cache-gb`
- `_run_epoch`：donor bank 调用、suffix 映射、`decoupled_fraction`/`gt_target_fraction`/
  `donor_*`/`clean_history`/`mixed_style_history` 指标、donor accumulators
- resume：donor/decouple 字段与 legacy defaults；`--init-ckpt` 不受影响（权重仍可加载）
- objective：统一 `training_objective_version=11`；donor 时代（≤10）ckpt 拒绝 resume

**clip_recipe v2**
- `Utils/clip_recipe_training.py`：`WindowDonorBank`、donor 导入
- `train_clip_recipe_v2.py`：paired-cross donor 请求、`--paired-cross-probability`、
  `run_epoch` 的 bank 参数；`VERSION` → `clip_recipe_v2_window_fm_quantized_style_2`
  （旧 v1 状态无法 resume，防止静默续训）
- `analysis/profile_clip_recipe_v2.py`：donor 构造与 profiler 标签

**Wrapper（全部去掉 donor/decouple 参数）**
- `run_train_flowmatch_6tok_curriculum100.sh`、`run_train_flowmatch_curriculum100.sh`、
  `run_train_flowmatch_motionrecon_ft10.sh`、`run_train_flowmatch_skipcat_notoken.sh`、
  `run_train_flowmatch_transformer_notoken.sh`、`run_train_flowmatch_tf150_staged_noise.sh`、
  `run_train_clip_recipe_revision.sh`、`run_train_flowmatch_history_phase10.sh`、
  `run_train_flowmatch_history_tf10.sh`

**测试**
- 删除 donor 单测/集成测：bank 三连测、matched_root 语义与 sync-free 等价、decoupling
  各模式、donor 指标断言、GPU donor 缓存测等
- 改写：`test_recipe_training`（排除集只含源 clip）、`test_clip_recipe_v2`、objective v11
- 结果：CPU 305 passed / GPU 20 passed（清理前 327/21）

**归档（保留可回溯）**
- `docs/archive/donor_sources_2026-10-12/`：`style_donor_bank.py`、
  `teacher_forced_probe.py`、`verify_equivalence.py`、`profile_training.py`、
  `audit_no_loss_solution.py` + README
- `analysis/training_speed_2026-10-09/before/`（旧版快照）未动
- 历史文档加"donor 已移除"横幅：`NO_EXTRA_LOSS_SOLUTION.md`、`CLIP_RECIPE_V2.md`、
  `V2_50EP_PILOT_PLAN.md`、`VQ_LABEL_TRAIN_PLAN.md`、clip_recipe reports 等

**保留（有意）**
- checkpoints/eval_results 中的 donor 时代模型与结果（历史证据，只读；不能 resume，
  可 `--init-ckpt` 加载权重）
- phase10/tf10/history 双帧条件、TF→rollout 课程、FP32 GPU 缓存、八步反馈、质量选型

## 验证

- CPU 全套 305 passed；GPU 20 passed
- `run_train_flowmatch_history_tf10.sh` / `run_train_flowmatch_history_phase10.sh`
  1-epoch 冒烟通过，objective=11、无任何 donor 键
- benchmark_training.py 去 donor 后复跑通过（60 updates：22.0/28.5/28.7 ups，冷热 RNG 一致）
- 全仓 import 级扫描无残留符号引用；`grep donor/decouple` 仅剩版本注释与历史文档

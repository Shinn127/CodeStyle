# Donor / cross-style 合成监督源码归档（2026-10-12）

这些脚本在"donor（跨风格合成目标）"路线被移除时从活动代码树中归档。保留原件以便回溯
2026-10-12 之前的诊断与基准结论；**不要**在当前的训练流程里引用它们（`Utils/style_donor_bank.py`
已不存在于 `Utils/`）。

移除原因（见 `../../../analysis/interactive_quality_2026-10-12/TF10_DIAGNOSIS.md`）：
纯 TF、无 donor 的 10 epoch 短训（`checkpoints/flowmatch_label_history_tf10`）在同协议封存评估中
MPJPE 7.97cm / jitter 0.568 / F.S. 0.717、RootADE 4.78/6.10cm，全面优于含 donor 的 phase10 与
6tok baseline（MPJPE 10.28/11.23、jitter 2.14/1.79）；虽然 donor 承担了 cross-style 泛化
（tf10 cross 65.7% vs phase10 90.8%），但用户判定 donor 的合成目标损害了闭环运动质量，决定
全面移除 donor 训练内容。

| 文件 | 原位置 | 说明 |
|---|---|---|
| `style_donor_bank.py` | `Utils/style_donor_bank.py` | StyleDonorBank、phase-aware 配对、retarget_motion |
| `teacher_forced_probe.py` | `analysis/clip_recipe_failure_2026-10-09/` | clip recipe 失败分析中的 oracle-hybrid 探针 |
| `verify_equivalence.py` | `analysis/training_speed_2026-10-09/` | 训练提速前后的 donor/recipe 等价性核验 |
| `profile_training.py` | `analysis/profile_training.py` | 旧 target 训练 objective 的 profiler（含 donor 参数） |
| `audit_no_loss_solution.py` | `analysis/motion_quality_diagnosis/` | 2026-10-12 donor 相位错配审计（其速度一致性部分仍可由 trace 复算） |

仍保留的历史快照（有意不动）：`analysis/training_speed_2026-10-09/before/`（当时的旧版本快照）。

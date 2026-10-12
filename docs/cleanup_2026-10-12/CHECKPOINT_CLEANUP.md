# Checkpoint 清理记录（2026-10-12）

**结果：`checkpoints/` 9.9G → 175M。** 旧 controller / 特征提取器 checkpoint 全部删除；
用户指令"保留一个 VQ 就行"，实际保留三份必要资产（见下）。

## 清理前清单

```
84K   flowmatch_label_motionrecon_ft10            (删除)
65M   fid_feature_extractor_camdm100              (删除; 镜像时代旧分类器)
130M  fid_feature_extractor_recon100              (删除; 镜像时代旧 recon 提取器)
259M  fid_feature_extractor_recon100_nomirror     (删除; recon 提取器, 训练入口可重建)
268M  flowmatch_label_motionrecon_ft10_diverged   (删除; 失败实验)
342M  flowmatch_label_history_phase10             (删除; 短训诊断)
342M  flowmatch_label_history_tf10                (删除; 短训诊断)
490M  fid_feature_extractor_camdm100_60fps_240f_nomirror (保留: 240f 风格分类器 .pth)
669M  vqvae_..._nomirror_oldsplit_ep110           (删除; 旧 split VQ)
677M  flowmatch_label_skipcat_notoken             (删除; 消融)
871M  flowmatch_label_transformer_notoken         (删除; 消融)
875M  vqvae_..._nomirror                           (保留: best.pt + norm_stats.npz)
1.4G  flowmatch_label_curriculum100               (删除; 5-token 课程, 含 20 个 epoch 快照)
1.6G  flowmatch_label_6tok_curriculum100          (删除; 96.40/91.42 最终模型, 含 20 个 epoch 快照)
2.2G  flowmatch_label_nomirror_150ep              (删除; 上一代 label 模型, 含 30 个 epoch 快照)
```

## 保留内容与理由

| 保留 | 大小 | 理由 |
|---|---:|---|
| `vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt` | 51M | 冻结 VQ（用户指定保留）；+`config.json`、`norm_stats.npz`、`history.jsonl`（小文件，训练验证指纹需要） |
| `fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth` | 122M | 240f 风格分类器：训练选型（`--tf-val-classifier`）与封存评估（matched/cross 指标）的结构性依赖 |
| `flowmatch_label_nomirror_150ep/latent_stats.npz` | 4K | latent 归一化统计；3 个训练 wrapper 的 `--latent-stats` 默认引用，重算需完整扫描训练集 |

被删目录中的 `epoch_*.pt`/`latest.pt`/`best_loss.pt` 全部移除；保留目录里除上述文件外的
中间快照（VQ 的 15 个 epoch 快照、分类器的 best/final/latest）也已删除。

历史指标与封存报告（`eval_results/`，42M）与训练缓存（`cache/`，224M）未动——它们是证据而非 checkpoint。

## 配套代码修改

删除后引用修补（防止误跑）：

- `run_train_flowmatch_history_{tf10,phase10}.sh`：默认 `--init-ckpt`（原 6tok best）不存在时
  打印警告并从零训练（`INIT_CKPT=<...>` 可显式指定；`FROM_SCRATCH=1` 静默）。
- `run_interactive_label_control.sh`、`run_simulate_realtime_flowmatch_control.sh`、
  `run_evaluate_style_rollout.sh`：`FLOW_CKPT` 缺失时给出明确指引并退出（旧默认模型已删）。
- `run_train_flowmatch_motionrecon_ft10.sh`：`--init-ckpt` 改为仅在显式设置时传入。
- `Visualization/cli/interactive_label_control.py`：缺失 checkpoint 时抛带指引的
  `FileNotFoundError`（替代裸报错）。
- 文档示例中的旧路径改为 `<your_run>` 占位（README、INTERACTIVE_LABEL_CONTROL、
  MOTION_HISTORY_REPAIR、eval/record 脚本 docstring）。

## 验证

- CPU 305 passed / GPU 20 passed
- `FROM_SCRATCH=1` 真实 1-epoch 冒烟通过（从零训练、验证、选型、保存全链路）
- history wrapper 默认行为：检查点缺失→警告→从零，不再引用已删文件
- 全仓仅剩的 `checkpoints/` 引用均解析到保留文件或为输出目录默认值

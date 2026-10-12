> **2026-10-13 注**：本页入口 `run_train_flowmatch_skipcat_notoken.sh` 与 notoken 实验线已在清理中删除，正文保留为历史记录。

# skip_cat_mlp + 无 motion token：50 epoch 短训结果（2026-10-11）

运行：`checkpoints/flowmatch_label_skipcat_notoken/`（`run_train_flowmatch_skipcat_notoken.sh`，
STOP_AFTER_EPOCH=50；150-epoch 预算的 clean 阶段，cosine horizon 未改，可与 transformer 的
同阶段 checkpoint 直接对比）。配置唯一差异：`--field-arch skip_cat_mlp --no-use-motion-token`；
其余（VQ/cache/prior/数据/loss/验资协议）与 label-150 配方完全一致。

## 同期对比（同为 epoch 50，clean-history 阶段结束）

| 指标 | skipcat+无motion | transformer | 差距 |
|---|---:|---:|---:|
| 单步 val flow loss（越低越好） | 0.0487 | **0.0310** | +57% |
| TF val matched | 66.7% | **78.5%** | −11.8pp |
| TF val cross | 67.8% | **76.5%** | −8.7pp |
| TF val RootADE（m/c） | 34.7 / 33.8cm | **16.2 / 17.3cm** | ≈ +18cm |
| 90-clip 同 seed 探针：ADO / 命中 | 30.3cm / 75.6% | **15.2cm / 87.8%** | 明显落后 |
| epoch 时间 | 33.9s | 41.9s | **−19%（更快）** |

验证曲线（每 5 epoch，matched/cross/mADE）：

| ep | skipcat | transformer |
|---|---:|---:|
| 5 | 61.1 / 57.2 / 41.4 | 55.9 / 53.1 / 44.9 |
| 15 | 64.4 / 62.6 / 41.1 | 62.6 / 60.0 / 26.9 |
| 25 | **71.1 / 67.0** / 31.3 | 70.4 / 70.4 / 21.7 |
| 35 | 60.7 / 54.8 / 35.0 | 73.0 / 69.1 / 24.4 |
| 50 | 66.7 / 67.8 / 34.7 | **78.5 / 76.5 / 16.2** |

## 结论

1. **50 epoch 后未追上 transformer**：起点（ep5）略高，中段（ep25）几乎追平，但 ep30 之后
   趋势停滞（66–68% 摆动，ADE 无持续下降），而 transformer 同期仍在稳步上升且 ADE 快速下降。
2. **不只是误差累积**：单步 val flow loss 也差 57%（0.0487 vs 0.0310），说明在当前训练预算下
   5-token skip-MLP 对单步动力学的拟合本身就落后——差距出现在模型容量/归纳偏置层面，不是
   自回归漂移放大。
3. **速度有优势**：训练 −19%、交互单步推理 −35%（此前实测 4.25→2.77ms）；如果后续追上，
   交互体验会更好。
4. **归因未分离**：本轮同时改了架构（transformer→skip_cat_mlp）和输入（去 motion token），
   观测到的差距不能单独归给任一方。selector/prior 正常收敛（prior_KL 0.06→0.04，
   风格准确率接近），不是条件通道问题。
5. **不能据此宣判架构失败**：transformer 在 ep51–150 的噪声阶段还涨了约 7–8 个点
   （78.5/76.5 → 85.9/84.8）；skipcat 也未到自己的收敛点。但按同阶段差距和停滞趋势，
   同等预算下追平需要更多 epoch，且 ADE 差距（18cm）较大。

## 复现与续训

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
# 数据与 checkpoint
checkpoints/flowmatch_label_skipcat_notoken/{best.pt,epoch_0050.pt,history.jsonl}
# 续训到 150（进入噪声阶段；保持原配置，严格 resume）
RESUME=checkpoints/flowmatch_label_skipcat_notoken/latest.pt \
  bash scripts/run_train_flowmatch_skipcat_notoken.sh
```

90-clip 探针复现：

```bash
/home/shinn/miniforge3/envs/mcc/bin/python analysis/controller_compare/render_controller_compare.py \
  --mode rollout --backend none --seed 42 \
  --flow-ckpt checkpoints/flowmatch_label_skipcat_notoken/best.pt
```

## 建议的下一步（未执行，待定）

1. **续训到 150**：走完与 transformer 相同的 150-epoch 预算和噪声课程，再看是否收敛到同一
   水平（最公平但最贵，预计 ~1.5h）。
2. **归因实验（便宜）**：`--field-arch skip_cat_mlp`（带 motion token）或
   transformer+`--no-use-motion-token`，各 50 epoch，把差距归给架构或输入。
3. **交互体验优先**：若短期目标是改善实时窗口，现 transformer 模型仍是最优可用资产
   （test matched 90.3 / cross 86.5；交互单步 4.25ms 已可流畅 60fps）。

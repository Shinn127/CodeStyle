# 50-epoch 三臂归因：架构 vs motion 输入（2026-10-11）

三个 50 epoch run，除标注差异外配方完全一致（同 VQ/cache/prior/数据/loss/验证协议、seed 42、
150-epoch cosine 预算的 clean 阶段，全部在 ep50 停止以便同阶段对比）：

| 代号 | 架构 | motion 输入 | 目录 |
|---|---|---|---|
| **tr6** | transformer | 有（6 token） | `flowmatch_label_nomirror_150ep`（ep50 快照） |
| **tr5** | transformer | 无（5 token） | `checkpoints/flowmatch_label_transformer_notoken` |
| **sk5** | skip_cat_mlp | 无（5 token） | `checkpoints/flowmatch_label_skipcat_notoken` |

## ep50 结果（240 帧 rollout validation：270 matched + 540 cross）

| 指标 | tr6 | tr5 | sk5 |
|---|---:|---:|---:|
| matched / cross acc | **78.5 / 76.5%** | 73.7 / 70.0% | 66.7 / 67.8% |
| RootADE (m/c) | 16.2 / 17.3cm | 16.5 / 16.7cm | 34.7 / 33.8cm |
| 单步 val flow loss | **0.0310** | 0.0334 | 0.0487 |
| epoch 时间 | 42s | 45s | **34s** |
| 90-clip 同 seed 探针（ADE / 命中） | 15.2cm / 79/90 | **14.4cm** / 69/90 | 30.3cm / 68/90 |

## 结论

1. **架构是主导变量**：同为"无 motion"输入，transformer（tr5）在每个验证点的 ADE 都大幅优于
   skip_cat（sk5），ep50 差距 18cm（16.5 vs 34.7）、探针 14.4 vs 30.3cm。此前 sk5 的落后主要是架构。
2. **motion token 主要影响风格保真**：去掉它（tr5 vs tr6）后 RootADE 持平甚至略好（16.5 vs 16.2；
   探针 14.4 vs 15.2），但风格准确率掉 5–9pp（validation 73.7/70.0 vs 78.5/76.5；探针 69/90 vs 79/90）。
   说明 291 维直通帮助的是"风格可判别性"而非根轨迹。
3. 趋势上 sk5 自 ep25 起停滞（66–68% 摆动），tr5 在 ep40 冲到 15.8cm 但 accuracy 波动；两者都还没到
   收敛点，50 epoch 不是终审。transformer 在 ep51–150 的噪声阶段还会涨（tr6: 78.5→85.9）。
4. **速度**：sk5 训练快 ~24%（34 vs 45s），交互单步推理快 ~35%。tr5 与 tr6 速度接近，瓶颈不在
   token 数。

## 推断与下一步

- 若目标是风格指标（论文口径 matched/cross），**运动 token 值 5–9pp，建议保留**；
- 若要继续"无 motion"路线，需要更长训练或从 tr6 best 热启动（6→5 token 重映射，丢弃 motion_proj），
  而不是从头训到收敛；
- skip_cat_mlp 单独立项的价值有限，除非追求交互推理速度（-35%）优先于质量。

原始产物：两个 run 的 `history.jsonl`、`best.pt`（ep50）、日志
`analysis/controller_compare/runs/{skipcat,transformer_notoken}_50ep.log`。

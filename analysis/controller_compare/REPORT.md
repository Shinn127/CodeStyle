# Label controller 生成对比（GenoView）

对 label controller（`flowmatch_label_nomirror_150ep/best.pt`）做与 VQ 排查同样的可视化对比：
左=参考 GT clip，右=模型生成，完整 mesh 并排播放。每次运行随机抽 clip（seed 打印并可复现）。

协议与正式评估一致：autoregressive rollout（每帧用预测 latent 反馈）、Euler8、noise std 1、
clip→quantize→decode；trajectory 通道按参考 clip 的 84 维计划驱动。分类器判定只喂生成帧
（排除 GT 首帧），与 `evaluation/evaluate_minor_revision_generalization.py` 相同。

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode

# matched：目标=源风格；每个 style 随机 240 帧 clip，逐窗打开（ESC 关窗进下一个）
/home/shinn/miniforge3/envs/mcc/bin/python analysis/controller_compare/render_controller_compare.py \
  --styles March Roadrunner Neutral

# cross：March 的轨迹计划 + Robot 风格
/home/shinn/miniforge3/envs/mcc/bin/python analysis/controller_compare/render_controller_compare.py \
  --styles March --target-style Robot

# teacher-forced 上界对照（每步喂 GT 上一帧，不走自回归）
... --mode reconstruct

# 连续 AE 反馈对照（非正式协议）
... --no-requantize

# 固定 seed 复现；--split test 只看 test；--frames 240 默认
... --seed 42 --split test
```

终端输出每个 clip 的 `RootADE`、分类器 top-1（含置信度与是否命中目标 style）、生成延迟；
`analysis/controller_compare/runs/summary_s<seed>.json` 保存全部记录。

已验证（seed 42/7，单 clip 抽查，非正式指标；RootADE 口径与 `Utils.motion_eval` 一致，全帧含首帧）：

| 场景 | 结果 |
|---|---|
| matched Neutral/FW | ADE 11.4cm，pred Neutral 1.00 命中；teacher-forced ADE 2.4cm（上界差距明显） |
| cross March/FW→Robot | ADE 8.7cm，pred Robot 1.00 命中 |
| 多 style cross → Robot（seed 7） | 3 例 1 中（Neutral 12.7cm 命中；March 1.6cm→SpinAntiClock miss；Roadrunner 54.9cm→Tiptoe miss） |

单 clip 方差大（正式 test 是 5000 例 cross 86.48%），此工具用于看具体失败案例的形态：
低 ADE + style miss 通常表示根轨迹跟随正常但肢体风格没跟上；高 ADE 则是轨迹跟随本身崩了。

## 三段分解：表征 / TF 动力学 / 自回归累积（2026-10-11）

同一批 clip（seed 42，每 style 随机 1 个 240 帧窗口，90 个 clip）分别跑：

- `--mode vq`：GT 编码→码本→解码（表征上界）
- `--mode reconstruct`：teacher-forced（每步喂 GT，动力学上界）
- `--mode rollout`：正式自回归协议

| 指标（90 clip 均值） | VQ 上界 | TF | Rollout | VQ→TF | TF→Rollout |
|---|---:|---:|---:|---:|---:|
| RootADE | **1.06 cm** | 3.11 cm | 7.89 cm | +2.05 | +4.78 |
| Jitter | 0.350 | 1.005 | 0.802 | +0.66 | −0.20 |
| F.S. | 0.341 | 0.864 | 0.811 | +0.52 | −0.05 |
| 风格命中 | 88/90 | 88/90 | 85/90 | 0 | −3 |

（median：ADE 0.80 / 2.30 / 5.54；p90：2.07 / 6.74 / 16.04；max：4.47 / 17.53 / 31.95）

结论：

1. **表征层（VQ）只占 1.1cm**，动力学单步预测（TF）再加 2.0cm，自回归累积再加 4.8cm——
   控制器误差的大头在 rollout 的误差累积，不是表征。
2. **ADE 随时间持续增长**（典型病例按 60 帧分段，前 1/4 → 最后 1/4）：
   LeanRight/TR1 0.7→9.3cm、Elated/FR 1.9→40.7cm、ShieldedRight/FW 0.4→33.8cm。
   典型的 compounding drift：开头跟得住，后半段发散。
3. 风格判定 rollout 只掉 3 例（97.8%→94.4%），且最差的 10 个 ADE clip 风格仍全对——
   **当前交互"看起来差"主要是轨迹/姿态漂移，不是风格丢失**。
4. TF 的 jitter/F.S. 反而高于 rollout：TF 每帧独立采样、帧间不连贯；rollout 有反馈自洽性，
   但代价是累积漂移。两者噪声形态不同。
5. 与 VQ 的对照也解释了交互感受：VQ 重建 1.1cm 几乎不可见；rollout 平均 7.9cm、尾部 16–30cm，
   肉眼可见的偏离主要来自这里。

若要从训练侧压缩：误差随时间的增长形态指向"训练未见自己的误差、推理时不断被自己的输出推离"
——可考虑 rollout/unrolled 训练（`--unroll-steps`）、scheduled sampling 或更长的历史噪声课程；
这属训练变更，需要另行确认预算。TF 上界 3.1cm 说明单步动力学本身也还有余量（选型门限 5cm）。

具体改善方案（证据、配置、成本、验证口径）：[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)。


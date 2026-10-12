# VQ 重建排查（2026-10-11）

针对"实时交互的运动效果不理想"的排查：先量化 VQ 重建质量与码本时序稳定性，再与实时 rollout
反馈路径对照，判断问题是否出在 VQ。

模型：`checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt`
（VQ150ep-nomirror，best=ep90；32 heads × 4 dim = 128 维 latent，2048 码本）。
数据：`database_100sty_375_memmap_nomirror` test split，60 帧窗口 / 14,671 条（评估），
30 clips × 240 帧（诊断），seed 42。

## 1. 重建质量：VQ 本身没问题

`evaluation/evaluate_motion_375to291.py`，同一批 14,671 个 test 窗口（60f/60fps）：

| 路径 | MPJPE | Jitter | F.S.(local) | RootADE |
|---|---:|---:|---:|---:|
| quantized（编→码本→解码） | **1.250 cm** | 0.461 | 0.400 | 0.468 |
| quantized + 3-tap 平滑 | 1.236 cm | 0.187 | 0.257 | 0.468 |
| pre_vq（连续 AE 上限） | 1.216 cm | 0.148 | 0.208 | 0.478 |

- 量化相对 AE 上限只多 0.034 cm MPJPE（2.8% 相对差）——**码本信息瓶颈不是问题**。
- 但量化把逐帧抖动放大约 3 倍（0.461 vs 0.148）、F.S. 放大约 2 倍（0.400 vs 0.208）；
  3-tap decode 平滑能以零 MPJPE 代价拿回大部分（0.187 / 0.257）。
- 结论：VQ 重建精度 1.25 cm（约人体尺度的 1.4%），质量合格；量化误差是可见但量级为亚厘米的噪声。

产物：`quantized/`、`quantized_smooth3/`、`pre_vq/` 三个 `motion_eval_summary.json`。

## 2. 码本时序稳定性：每帧都在换码

`diagnose_vq.py`（30 test clips × 240 帧，GT 编码路径）：

- **any-head 换码率 99.0%/帧**，per-head 61.9%；完整 32-head 码向量保持不变的最长游程中位数 = 1 帧，
  99.4% 的"不变段"长度只有 1——**没有任何连续帧落在同一个码组合上**。
- 换码帧的关节逐帧移动 ~0.78 cm；码不变时解码输出完全相同（decode 是逐帧的，没有帧间混合），
  即所有运动都由码切换承载，码序列本身携带台阶噪声。
- latent gap E[(z_e−z_q)²] = 0.0083（归一化空间）。

这正是"量化把抖动放大 3 倍"的来源：相邻帧的 z_e 跨过量化边界 → 解码姿态阶跃 ~0.8cm/帧。
数值上它被控制器的后续动力学吸收了一部分，但反馈进控制器的是带台阶的 motion/latent。

`diagnose_test.json`（含 per-clip）。最差的几个 clip（Morris/FW、PigeonToed/SR）量化-AE 差 0.6–1.1 cm MPJPE。

## 3. 实时 rollout 对照：量化贡献 ~20–25% 抖动

`realtime_switch_probe.py`：真实 label 模型交互会话（240 帧 demo，同 seed42，Euler8/std1）：

| 反馈路径 | any-head 换码率 | 不同码组合数 | 逐帧抖动代理 | 换码跳变 |
|---|---:|---:|---:|---:|
| 量化（默认） | 100% | 240/240 帧 | 1.120 cm | 0.978 cm |
| AE 连续（--no-requantize） | 100%* | 240/240 | **0.899 cm** | 0.897 cm |

\* 连续 latent 每帧移动，最近码当然每帧变化，码本统计在此仅作参考。

- 实时窗口里量化反馈比 AE 反馈多 ~25% 抖动（1.12 vs 0.90 cm），F.S./jitter 的排序与静态评估一致。
- 但注意绝对水平：GT 重建量化抖动 0.46 cm，实时 rollout 量化下 1.12 cm——**控制器采样本身贡献了大部分
  （~0.9 cm），VQ 量化只是叠加项**。

`realtime_switch_quantized.json` / `realtime_switch_ae.json`。

## 4. 结论与建议

1. **VQ 不是实时效果不佳的主因**：重建精度 1.25 cm、AE 差距 0.03 cm；交互中的抖动大部分来自
   控制器采样（std=1/Euler8/自回归反馈），量化叠加约 20–25%。
2. 现有可用的两个杠杆（已验证）：
   - 窗口加 `--no-requantize`：反馈走连续 latent，实测抖动 -20%，显示更平滑（非正式协议）；
   - 显示后处理平滑默认已开启（6Hz 因果平滑），P 键可对照原始姿态——若对照后仍觉得差，问题在模型采样。
3. 若要从表征侧继续改善（可选，非本轮结论）：3-tap decode 平滑（推理侧零训练，F.S. 0.40→0.26，
   jitter 0.46→0.19）；更彻底的做法是时序感知的 VQ 微调/重训（在码序列上加时序一致性目标），
   成本一轮 VQ 训练，属新的训练变更，需另行决定。

## 5. 可视化（ref vs VQ 重建）

`analysis/vq_reconstruction_check/render*/`，matplotlib 骨架并排 GIF（60fps），quantized/continuous 两版：

- `render/Neutral_ID_416.{quantized,continuous}.gif`（Idle 类，180 帧）
- `render_dynamic/March_BW_389.{...}.gif`、`Roadrunner_ID_545.{...}.gif`（行走/跑，240 帧）

数字与动图一致：量化版可见轻微抖动，continuous 版更平滑；姿态整体无漂移、无扭曲。

实时 GenoView 网格窗口（打开桌面窗口，左=参考 GT，右=重建，ESC 关闭并进入下一个 style）：

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode

# 每个 style 随机抽一段 240 帧 clip；每次运行换随机 seed（打印在 [Info] 行）
/home/shinn/miniforge3/envs/mcc/bin/python analysis/vq_reconstruction_check/render_vq_compare.py \
  --backend genoview --styles March Roadrunner --frames 240

# 看连续 AE 路径；--seed 固定后同一 clip 可复现
/home/shinn/miniforge3/envs/mcc/bin/python analysis/vq_reconstruction_check/render_vq_compare.py \
  --backend genoview --styles March --path continuous --seed 42

# 只看 test split 的片段（默认 any 混合三个 split）
... --split test
```

`render_vq_compare.py` 的选段规则：在指定 style（或全部 style）里随机抽 `--clips-per-style` 个 segment，
再在每个 segment 内部随机取 `--frames` 帧窗口；窗口起点与 seed 写入 `summary.json` 及 GIF 文件名（`_s<start>`）。

排查用脚本（保留）：`diagnose_vq.py`（编码稳定性+重建指标）、`render_vq_compare.py`（GIF/GenoView 渲染）、
`realtime_switch_probe.py`（交互会话对照）。

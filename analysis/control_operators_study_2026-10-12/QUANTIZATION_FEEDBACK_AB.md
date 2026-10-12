# 量化反馈 vs 连续反馈：影响量化实测（2026-10-12）

问题：ControlOperators 的反馈回路无量化（连续 latent 直接回喂），我们的回路每帧
clip→quantize→decode。这条差异的影响到底有多大？在 ODE4-100ep/latest.pt 上做 A/B 实测。

方法：同一 checkpoint、同一历史、同一噪声源，仅切换反馈路径
（`--no-requantize` = 连续 AE 反馈；默认 = 正式量化反馈）。
封存 1000 matched sources / 240f；交互探针 300f 脚本化行走。

## 一、单步量化扰动（机理测量）

同一历史、64 个噪声源的单步：`decode(clip(z))` vs `decode(Q(clip(z)))`。

| 指标 | 值 |
|---|---:|
| 单帧量化引起的局部关节位置差 | **0.245 cm**（p95 0.385） |
| 归一化 latent 距离 ‖z−Q(z)‖ | 0.568 |
| 对照：64 噪声源的跨采样离散（姿态 std 范数） | **1.95 cm** |

读出：单步量化扰动只有采样噪声离散度的 **~1/8**，且小于一个典型帧间姿态变化
（行走 0.65–0.85 cm/帧）。量化阶梯本身是小扰动。

## 二、闭环 240f 封存 A/B（MPJPE / jitter / F.S.）

| 配置 | MPJPE | local | RootADE | jitter | F.S. |
|---|---:|---:|---:|---:|---:|
| std=1.0 量化（正式） | 6.43 | 4.71 | 3.18 | 0.791 | 0.845 |
| std=1.0 连续 | 6.56 | 4.86 | 3.16 | **0.683**（−14%） | **0.763**（−10%） |
| std=0.25 量化 | 6.04 | 4.44 | 2.98 | 0.366 | 0.537 |
| std=0.25 连续 | 6.17 | 4.58 | 2.97 | **0.215**（−41%） | **0.487**（−9%） |

交互探针（300f，局部位移二阶差分）：

| 配置 | walk 段 jitter (cm) | 滑行 | 备注 |
|---|---:|---:|---|
| std=1.0 量化 | 0.816 | 0.42% | |
| std=1.0 连续 | 0.861 | 0.42% | 无改善 |
| std=0.25 量化 | 0.475 | 0.84% | |
| std=0.25 连续 | **0.267**（−44%） | 0% | 低噪声下改善显著 |

## 三、结论

1. **量化对轨迹类指标几乎无影响**：MPJPE/RootADE 差异 ≤0.15cm（噪声量级）。
   表示层误差（VQ 重构 2.2cm）与闭环累积（~7cm）都远大于单步量化扰动（0.245cm）。
2. **量化主要影响的是"抖动/足滑"这档次级纹理**：去掉量化后 jitter 降 14%（std=1）
   到 41–44%（std=0.25）、F.S. 降 9–10%。方向明确，幅度中等——不是数量级差异。
3. **因此 ControlOperators 的稳定性不能主要归因于"无量化"**。它的收益组合里，
   我们测到量化只占 jitter 的一小部分；更大头应该在他们另外两个机制：
   **训练时输入噪声增强**（我们完全没有）和 **75% 惯性化 + 模型输出 contact 的 IK**
   （我们用的是保守低通 + 高度/速度启发式）。这个分解与之前"量化约占 20% jitter"的
   旧结论一致。
4. **实操建议**：
   - 正式论文口径继续用量化反馈（不变）。
   - 显示/交互默认可以走连续反馈：`--no-requantize` 在 std=0.25–0.5 档位能白拿
     约 40% 的抖动降低；但注意它偏离正式协议行为（代码里已明确标注为对照路径）。
   - 继续投入的方向仍是学习报告里的方案 A（输入噪声增强）与 B（模型输出 contact），
     量化本身不值得再花精力。

## 复现

```bash
CK=checkpoints/flowmatch_label_history_ode4_100ep/latest.pt
# 控制探针
python analysis/motion_quality_diagnosis/record_control_video.py --flow-ckpt $CK \
  --tag ode4_nq --noise-std 0.25 --no-requantize --render none --out-dir /tmp/fb_diff
# 封存 A/B
python analysis/controller_compare/eval_rollout_mpjpe.py --flow-ckpt $CK --tag ode4_nq \
  --noise-std 0.25 --no-requantize --out-json /tmp/mpjpe_ode4_nq_ns025.json
```

原始数据：`probes/mpjpe_ode4_{q,nq}_ns{1.0,0.25}.json`、
`videos/ode4_{q,nq}_ns{1.0,0.25}/`。

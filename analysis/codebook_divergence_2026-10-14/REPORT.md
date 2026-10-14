# 码本跳变（quantized-code divergence）探针：GT-clip 四 regime 对比

> **后续**：根因分析见 [CAUSE_ANALYSIS.md](CAUSE_ANALYSIS.md)（几何/稳定槽分解/闭环刷新扫描
> 三层定量分解；本文件中"关键读法"第 8、9 条已按该结果修正）。

2026-10-14。原始脚本与结果由辅助分析产生于 `/tmp/flow_code_div/`（脚本自述 "Runs from /tmp;
touches no repository files"，属跨会话临时分析）；本轮原样归档，并作为根因分析
（`CAUSE_ANALYSIS.md`）的基线。

## 目的

量化"生成 latent 的 32 槽量化码（4×32=128 维，每槽 2048 码）与 GT 码的偏离"，把
"码级跳变"分解为三层：**codec 自身稳定性** / **单步模型误差** / **闭环自反馈漂移**；
并区分"近邻码翻转（运动代价小）"与"真实运动分歧"。

## 协议

- clip：test split、未镜像、12 个风格各取 1 条，240 帧 + 1 种子帧；seed=1000。
- 条件：GT 首帧做种子；**轨迹块（84 维）在全部模式中始终用 GT**；条件风格 = clip 自身 label。
- 四种 regime（唯一变量 = "观测/历史"的来源）：

| regime | 观测（motion token） | 历史 latent | prev latent | 含义 |
|---|---|---|---|---|
| roundtrip | —（无控制器） | — | — | `encode(GT)→decode→re-encode`，codec 自身稳定性参照 |
| tf | GT 行 | GT | GT | teacher forcing（训练流形） |
| reconstruct | `decode(z_gt)` | GT | GT | 官方单步推理流形 |
| rollout | **模型自身 requantized 反馈** | 自反馈 | 自反馈 | 官方闭环语义 |

- 评估协议与正式封存一致：FP32/highest、Euler 步数取 checkpoint（均 4）、clip→quantize→decode、
  noise std ∈ {0,1}（tf/reconstruct/rollout）。
- 指标：`match`（槽级码匹配率）、`hamming`（帧均码差槽数/32）、`exact_all`、`near2`、
  `move`（生成 latent 与 GT latent 的逐槽距离，raw 空间）、`L2sig`（归一化 latent 全维偏差）、
  `motion|d|`（生成运动 vs GT 运动平均绝对差 cm）、`code|d|`（**仅码差造成的运动差**：
  `decode(生成码)` vs `decode(GT 码)`）、`switch`（相邻帧码变化槽数/帧，生成 vs GT）、
  flip-by-margin（翻转率 vs GT 码边界余量 d2-d1 分桶）、逐帧 bins、lag 曲线（±6 帧）。
- 两个模型：`tf100_2L`（2 层 100ep 主线）与 `tf10_l6_film`（6 层 FiLM 10ep，当前最优；
  两者共用同一冻结 VQ，故 roundtrip 行完全相同）。

## 总表

`tf100_2L`（同列 = match / hamming / move / L2sig / motion|d| / code|d| / switch g/gt）：

| regime | match | hamming | move | L2sig | motion·d | code·d | switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| roundtrip | 0.378 | 19.9 | 0.134 | 2.91 | 3.10cm | 1.21cm | 20.9/20.2 |
| tf std0 | 0.379 | 19.9 | 0.125 | 2.11 | 4.45cm | 2.62cm | 21.1/20.2 |
| tf std1 | 0.322 | 21.7 | 0.144 | 2.37 | 4.79cm | 3.03cm | 23.9/20.2 |
| reconstruct std0 | 0.375 | 20.0 | 0.126 | 2.12 | 4.47cm | 2.63cm | 21.2/20.2 |
| reconstruct std1 | 0.318 | 21.8 | 0.145 | 2.38 | 4.80cm | 3.03cm | 24.0/20.2 |
| rollout std0 | 0.056 | 30.2 | 0.731 | 15.20 | 12.77cm | 11.82cm | 14.1/20.2 |
| rollout std1 | 0.056 | 30.2 | 0.723 | 15.31 | 13.36cm | 12.39cm | 20.2/20.2 |

`tf10_l6_film`：

| regime | match | hamming | move | L2sig | motion·d | code·d | switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| roundtrip | 0.378 | 19.9 | 0.134 | 2.91 | 3.10cm | 1.21cm | 20.9/20.2 |
| tf std0 | 0.293 | 22.6 | 0.141 | 2.28 | 4.59cm | 2.83cm | 21.0/20.2 |
| tf std1 | 0.237 | 24.4 | 0.165 | 2.60 | 5.19cm | 3.54cm | 25.3/20.2 |
| reconstruct std0 | 0.287 | 22.8 | 0.142 | 2.30 | 4.63cm | 2.84cm | 21.1/20.2 |
| reconstruct std1 | 0.233 | 24.5 | 0.166 | 2.62 | 5.21cm | 3.53cm | 25.4/20.2 |
| rollout std0 | 0.047 | 30.5 | 0.742 | 15.51 | 14.54cm | 13.51cm | 12.0/20.2 |
| rollout std1 | 0.047 | 30.5 | 0.736 | 15.59 | 14.35cm | 13.30cm | 21.8/20.2 |

`exact_all` 与 `near2` 在所有 regime 均为 0.000（没有一帧做到 32 槽全对或仅 ≤2 槽错）。

## 关键读法

1. **codec 自身**：roundtrip `match=0.378` —— 62% 的码在"decode→re-encode"一圈后就变；
   但 `code|d|=1.21cm` —— 翻转码多为近邻码，运动代价很小。
2. **单步模型处于 codec 天花板**：tf/reconstruct（std0）与 roundtrip 基本持平
   （2 层 0.379/0.375 vs 0.378）；FiLM 模型单步 0.293/0.287，略低于天花板。
   → TF 档的"码不符"无法归因于模型：codec 自己重编码也保不住码。
3. **闭环崩塌**：rollout 掉到 0.056/0.047，`move`≈0.73（≈5.5× codec 重编码位移）、
   `L2sig`≈15、`code|d|`≈12-13.5cm —— 生成运动与 GT 运动是"真实分歧"，不是边界噪声。
4. **GT 自身码切换率 = 20.2/32 槽/帧（63%）**：码字逐帧大量更换是常态；生成 tf 档 21.1/帧
   与之相当，rollout 档 12-22/帧。
5. **逐帧 bins**：rollout 第 1 个 bin（0–30 帧）match=0.09、之后稳定在 0.04-0.05
   —— 漂移在第一段就发生并很快饱和，不是缓慢累积型。
6. **lag 曲线**：峰在 lag 0（-1 略高：tf 档 0.40 vs 0.38），未见整体时移；rollout 档
   曲线几乎平坦（全档 ≈0.03-0.06），码流与其说是"错位"不如说是"无关"。
7. **采样噪声不是闭环主因**：std0→std1 在 rollout 档 match 不变（0.056/0.047）；
   TF 档降 5.6-5.7pp。
8. **翻转 vs 边界余量（roundtrip）**：低余量桶（0.01-0.05）翻转 0.54，大余量桶（0.22-3.70）
   仍翻转 0.46 —— 翻转不只在"贴边"码上发生。⚠️ 修正（见 CAUSE_ANALYSIS §2.1）：
   `move` 是量化后的码字-码字距离，非零时必然 ≥ 1 个码本最近邻间距，因此
   "move > margin/2 所以翻转" 是同义反复，不能作为机制；正确的读法是 roundtrip 的
   码身份是"保持或离散跳变"的二值行为，翻转率 = 重编码点越过 Voronoi 边界的概率。
9. **match 的槽级分解已解决**：GT 码"相邻帧不变"的槽占比 = 36.9%。roundtrip
   match = 0.368×0.417 + 0.631×0.355 = 0.377 ✓；tf(2L) = 0.369×0.628 + 0.631×0.235 = 0.380 ✓。
   即 match 完全由"稳定槽保持率"与"过渡槽新码正确率"两个分量决定，见 CAUSE_ANALYSIS §2.2。

## 口径澄清（重要，修正一条此前解读）

- 此前交互录制报告中的 `vq_unchanged=True` **不是"码字没有跳变"**：该字段是
  **VQ 权重指纹未变**（`state_fingerprint(vqvae.state_dict())` 前后一致，即交互推理未篡改
  冻结码本）。本探针表明：码字每帧大量翻转（GT 自身 63%/帧）是**常态**，
  "码字保持"从来就不是交互运行的不变量。
- "码跳变 ≠ 视觉跳变"：开环档翻转码的近邻性质（code|d| 1-3cm）使其视觉代价很小；
  与此前"量化阶梯不是显示抖动主因"的 A/B 结论一致。闭环档（rollout）code|d|≈12cm 的
  "跳变"本质是模型生成运动与 GT 的真实分歧，而非量化边界问题。

## 工件

- `code_divergence.py`：原始探针脚本（未改动）。
- `tf100_2L.json` / `tf10_l6_film.json`：两模型完整结果（含 per-clip）。
- `/tmp/flow_code_div/smoke*.json`：迭代冒烟记录（未归档）。
- 复现（默认 4 模式 × std{0,1}、12 clips × 240 帧）：

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
PY=/home/shinn/miniforge3/envs/mcc/bin/python
$PY analysis/codebook_divergence_2026-10-14/code_divergence.py \
  --flow-ckpt checkpoints/flowmatch_label_history_obsnoise_tf100/best.pt \
  --tag tf100_2L \
  --out analysis/codebook_divergence_2026-10-14/tf100_2L.json
```

## 复现性与口径核对

- 两模型的 roundtrip 行逐位相同（共用冻结 VQ），tf100_2L 的 `tf_std0` match 与本探针
  2026-10-14 首跑一致。
- 任务口径：轨迹块始终为 GT（rollout 档同理）→ 表中 `move/L2sig` 的偏离**不包含任何轨迹
  预测误差**，纯粹来自运动（观测/历史）反馈与生成；若要研究"轨迹也生成"的部署变体，
  需另改脚本（把 `x_norm[k-1, :84]` 换为生成轨迹的投影）。

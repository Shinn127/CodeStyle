# 修订后复查（补充）：K=1024 精确几何首跑 + 三档探针对照

2026-10-14。**权威审计见同目录 [AUDIT.md](AUDIT.md)**（几何修正、初始化/数值接口缺陷、
固定 validation 对照、受控基准、后续边界）。本文件只补充审计明确未执行的部分：
① 精确几何（`nearest_boundary`，vq_stability_version=2）的 30ep 训练结果（K=1024）；
② 三档 VQ 的同探针 v2 对照；③ 与审计一致的效率结论。**不再重复审计已覆盖的正确性核验。**

## 0. 先对齐审计的关键更正（本报告已遵循）

- `recon_cm_mean` 的旧"3.1cm"值（291 维混合 MAE×100）非法；重建一律用 position-only
  local MPJPE（探针 v2 起）。
- 逐维 σ 与槽 L2 不可混用；n-gram/1NN 是有限基线；合并实验不是重训上界；
  "物理不可能/硬上限"类表述已从 CAUSE_ANALYSIS / VQ_LEVEL / CODE_ANALYSIS 撤回（各文首
  已加复审限定）。
- raw 边界距离可被尺度变换放大；**相对余量（r / (0.5×目标码最近不同码间距)）才是
  抗扰指标**。审计固定 validation：baseline .2989 vs pilot(v1几何) .2880——基本未变。
- K=2048 pilot 用的是旧 `nearest_code` 近似（config 无 mode、无 version 字段，已核对）；
  精确几何在本轮之前**未被任何训练消费**。

## 1. K=1024 精确几何首跑（本会话执行）

配置：30ep、bs512、seed42、warm start（encoder/decoder 自 150ep 基线，码本 k-means 重建）、
`latent_margin_weight=1.0`、`roundtrip_weight=1.0`、`latent_margin_mode=nearest_boundary`、
`latent_margin_target=0.05`、K=1024。config 核对：`vq_stability_version=2` ✓。
epoch ≈72s（33–36k samples/s，与 K=2048 同档）。

训练内 90 窗口 eval：ep10 mpjpe 1.287 → ep25 1.253 → ep30 1.233；train recon 0.1790→0.0550；
rt_drift（128 维整体范数口径）1.548→0.977——比 K=2048 档（1.186）更低。

## 2. 三档对照（同一探针 v2，split=test，12 clips×120 帧）

| 指标 | 基线 150ep | pilot K=2048（旧几何） | pilot K=1024（**精确几何**） |
|---|---:|---:|---:|
| 同帧往返翻码率 | 0.672 | 0.596 | **0.592** |
| 精确边界距离 均值 / 中位 | 0.0349 / 0.0110 | **0.0612 / 0.0316** | 0.0827 / 0.0244 |
| 相对余量中位 | 0.299 | 0.296 | 0.342 |
| bdist<0.05 占比 | 0.859 | 0.644 | 0.730 |
| recon（local MPJPE 24j） | 1.343 | **1.172** | 1.274 |
| 越界/未越界翻码 | 0.742 / 0.000 | 0.650 / 0.000 | 0.667 / 0.000 |

**几何不变式（本会话独立验证）**：`nearest_boundary ≤ nearest_code` 对全部 46,464 个
真实行成立（违例 0），反例测试固定于审计；两模式仅是定义差异，不是数值近似误差。

读法（与审计第 4 节一致，不外推）：
- 三档都保持"不越界不翻、越界才翻"的几何决定论；翻码率整体只从 0.672 降到 ~0.59，
  **相对余量仅 K=1024 档略有改善（0.342），K=2048 档与基线持平**——raw bd 的增长主要被
  尺度因素解释，不能读作抗扰能力等比提高。
- K=1024 档重建代价最明显（1.274 vs K=2048 的 1.172），且 30ep 内码本重建未追平；
  粗化收益需更长训练兑现。
- **结论与审计一致：不支持现在把任何新 VQ 推广为新主 VQ**；原 VQ + 原 controller 仍是
  有效配套。准确的下一步是同初始化、同 updates 的单因素对照（margin-only vs RT-only vs
  两者），并同时看相对余量 / RT flip / decoded jitter / contact——本轮数据只支持这一条路。

## 3. 效率（以审计受控基准为准，本会话复核）

审计受控基准（bs512×T2、FP32、保留 STE 图、冻结 EMA、10 次前向/反向、无 optimizer 步）：

| 模式 | mean ms | 相对基线 |
|---|---:|---:|
| baseline | 23.30 | — |
| legacy margin+RT（`nearest_code`） | 27.63 | +18.6% |
| **exact margin+RT**（`nearest_boundary`） | **34.31** | **+47%** |
| allocated 峰值 | ~1.63 GiB | 三档相同 |

本会话更正：此前"净成本被抵消"的表述**不成立**——不同代码状态的 epoch 墙钟不可直接比较；
准确口径是上面的同状态 per-step 基准。真实 epoch 墙钟（事实，不做跨版本对比）：
K=2048 pilot 79.8s、K=1024 pilot ~72s。低维 cdist 核（K=2048/D=4 约 15ms）已由审计换为
分块 broadcast norm；D=4 下精确模式的剩余开销主要是 K×K 分离矩阵（分块至 ≤64MiB）与
逐块全表 masking。

## 4. 回归状态

- 审计新增 `tests/test_vq_stability_audit.py`（12 项）：几何反例、逐平面 oracle 前反向、
  重复/单码、尺度反例、梯度隔离与 EMA 单次更新、缺 buffer 拒绝、K 变化初始化、模式 resume
  保护、CLI 微型跑、温度缓存、CUDA/CPU 一致性。
- 本会话最终全量复跑（原样输出）：**CPU `396 passed, 21 deselected, 5 subtests passed`；
  GPU `21 passed, 396 deselected`**。（审计当时记录 395 passed，因其后工作区又新增若干
  测试项；两处均为同一测试集在不同时点的完整跑。）
- `vq_anatomy.py` 已参数化 `--vq-ckpt`、头数自适应、默认 val split、motion 指标改用
  `Utils.motion_eval`（position-only local/world MPJPE + jitter + 足滑；不再输出非法 cm）。

## 5. 未决与下一步（与审计合并后的清单）

1. **不要**用新 VQ 覆盖旧 controller/缓存；若继续 VQ 线，先做单因素小对照（同 init、
   同 updates），验收同时看：相对余量、RT flip、decoded jitter/contact。
2. K=1024 档若要继续：至少追到 recon 与 K=2048 同档后再评估；或提高 `latent_margin_target`
   与训练长度，观察相对余量是否真正抬升。
3. 控制器侧 code-CE 与本轮 VQ 线的组合实验尚未开始（两者独立、可叠加）。
4. 审计第 6 节边界全部保留：seq_len2/acc_weight0 不监督二阶时间误差；不自动加平滑；
   VQ resume 不声称 RNG 位级可复现。

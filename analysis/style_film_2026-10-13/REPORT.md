# FiLM 风格条件注入：实现、正确性与效率 review（2026-10-13）

在「100ep 纯 TF + 观测噪声」主线配方上实现 per-layer **style FiLM**：由 `e_s` 生成逐层
(scale, shift)，调制 transformer field 的生成 query token（z_t token）后再过每一层 encoder。
保留原 style token、selector、prior/entropy loss 全部不变。开关 `--use-style-film`
（入口 `STYLE_FILM=1`），默认关闭，既有 run 逐位不变。

设计依据（归档计划原文）：`docs/archive/DONOR_FREE_TF_STYLE_PLAN.md` §3.A——"保留
label→selector→style code/prototype→e_s、原 style token 和原 loss；两层 transformer 的
生成 query 再接受小型 style scale/shift，输入统一为 e_s；调制线性层零初始化，scale 有界；
同一次 Euler 的 style 投影缓存一次。不新增 classifier loss 或 label-only 旁路。"

## 1. 实现

| 位置 | 内容 |
|---|---|
| `arch/flowmatch_vqvae_controller.py` | 新增 `StyleFiLM`（style_dim→hidden→2·hidden·layers，SiLU，末层零初始化）；`ConditionalVelocityField(use_style_film=)`；`prepare_film` / `_encode_modulated`（逐层 `q*(1+tanh(scale))+shift`）；`_predict_velocity` / `_integrate` 增加 `film=` 参数并缓存；`FlowMatchVQVAEController.prepare_style_film`；builder 与 from_config 透传 |
| `training/train_flowmatch_vqvae_controller.py` | CLI `--use-style-film`；resume 校验字段（legacy 默认 False）；`--init-ckpt` 从无 film checkpoint 迁移时允许缺失 `field.style_film.*`（零初始化=从恒等开始） |
| `scripts/run_train_flowmatch_history_obsnoise_tf100.sh` | `STYLE_FILM=1` 启用 |
| `tests/test_style_film.py`（新增 11 项）| 见 §2 |
| `tests/test_flowmatch_performance.py` | 打桩 `uncached_euler` 接受并转发 `film` |

关键设计点：

- **scale 有界**：`scale = tanh(raw) ∈ (-1,1)`，shift 无界（零初始化使初始恒等，训练中
  shift 不受限是刻意的——恒等起点 + scale 有界满足计划要求）。
- **零初始化 = 精确恒等**：末层 weight/bias 置零 → `scale=shift=0` → `q*(1+0)+0 = q`。
- **构造顺序**：`style_film` 在所有共享参数（含 `position_embeddings` 的 trunc_normal_）
  之后构造，使同 seed 的 film/非 film 模型共享权重逐位相同（否则 RNG 流错位）。
- **缓存**：同一 Euler solve 内 `film` 只投影一次（每次 `compute_loss`/`sample_latent`/
  `_integrate` 各 1 次），逐层调制复用该系数；rollout 训练逐步走 `compute_loss` 各投影一次。
- **广播形状**：逐层系数显式 `unsqueeze(1)` 为 `[B,1,H]`，避免 `[B,H]` 与 `[B,1,H]`
  广播成 `[B,B,H]`（初版曾出现，已修复并被测试覆盖形状校验）。
- **禁止静默**：非 film field 收到 `film=` 参数 → `ValueError`；`skip_cat_mlp` 声明
  `use_style_film=True` → `ValueError`（field 与 controller 两层均拒绝）。

## 2. 正确性检查（全部实测通过）

单测 `tests/test_style_film.py` 11 项：

| 检查 | 结果 |
|---|---|
| 零初始化：loss / field 输出 / 采样路径 / field 梯度 与无 film 模型**逐位一致** | ✅ `torch.equal` |
| 非零 film 改变预测；film 参数收到 FM 梯度；VQ 冻结（梯度 None + requires_grad False） | ✅ |
| AdaLN-Zero 次序：末层先获梯度（proj.0 经零权重梯度暂为 0），一步更新后 proj.0 获梯度 | ✅ |
| scale 严格有界 `|scale|≤1`（tanh 饱和到界内），shift 不被压缩；零权重时系数精确为 0 | ✅ |
| Euler 缓存：`compute_loss`(solver 8) 与 `sample_latent`(6 步) 各仅 1 次 `prepare_style_film` | ✅ spy 计数 |
| 显式传入的 film 与自动解析结果逐位一致；坏形状（batch/层数不符）→ `ValueError` | ✅ |
| 非 film field 收到 film → `ValueError`；`skip_cat_mlp` + film → field/controller 双层拒绝 | ✅ |
| history conditioning 与 film 组合：共同前向 + 梯度齐备 | ✅ |
| 关闭路径：无 `style_film` 参数、返回 None、loss 与无 film 模型逐位一致 | ✅ |
| 多层同 seed 对：共享权重逐位一致、loss 恒等、系数形状 `[B,layers,H]` | ✅ |
| 训练器接线：CLI 标志、resume 字段、legacy 默认、init 允许列表均在位 | ✅ |

额外集成检查（手工执行）：

| 检查 | 结果 |
|---|---|
| `nn.TransformerEncoder` 与逐层手写循环的数值一致性（CPU + CUDA） | ✅ 逐位一致（调制路径由此安全） |
| 旧 config（无 `use_style_film` 键）→ 构建非 film 模型并正常加载旧 ckpt | ✅ |
| 真实入口 GPU 冒烟：`EPOCHS=1 STYLE_FILM=1` 训练→验证→保存全链路 | ✅ exit 0 |
| 冒烟 ckpt 持久化：config/args 记录 `use_style_film=True`，`field.style_film.*` 落盘 4 键 | ✅ |
| 评估端重建：`Utils.motion_eval.load_motion_model('flowmatch_target', …)` 加载 film ckpt | ✅ 零 missing/unexpected |
| CPU 全量回归 | ✅ **355 passed**（344 旧 + 11 新）, 20 deselected |
| GPU 套件 | ✅ 20 passed（含更新后的 perf 桩） |
| `py_compile` 全部改动文件 | ✅ |

## 3. 效率检查（实测，RTX 5070 Ti，FP32/highest）

配置：6 层 field、hidden 256、style_dim 64、solver 4、bs512（基准用最小化 VQ fixture 与
真实 field 结构；StyleFiLM 参数 = `64·256+256 + (256·2·256·6+2·256·6) = 0.806M`）。

| 指标 | film OFF | film ON | Δ |
|---|---:|---:|---:|
| StyleFiLM 参数量 | — | 0.806M | （真实 6L field 4.936M → **5.742M**） |
| 训练步（bs512、train、含 backward） | 10.77 ms | 11.42 ms | **+6.0%** |
| 推理 `sample_latent`（4 Euler、bs512） | 10.80 ms | 11.12 ms | **+3.0%** |
| 峰值显存（基准） | 0.47 GB | 0.51 GB | +0.04 GB |

效率设计：

- 每层 (scale, shift) 一次线性投影覆盖全部层（非每层一个模块），单次 `Linear` 前向；
- 同一 Euler solve 只投影一次（否则 4 步会多 3 次；实测 spy 验证）；
- 关闭时走原 `self.encoder(tokens)` 融合路径，零额外开销、逐位不变；
- 逐层调制循环与融合 kernel 在 CPU/CUDA 均逐位一致（§2 第 1 条集成检查）。

## 4. 边界与已知取舍

- **FP32 math-attention 分支**：`fp32_math_attention` 的 SDPA-MATH 包装在 film 路径下同样
  生效（`_encode_modulated` 内部逐层调用，语义相同）。
- **resume 语义**：film 与非 film 的 run 不能互相 resume（`resume_fields` 拒绝），与
  `history_conditioning` 的既有先例一致；从非 film 权重迁移只允许经 `--init-ckpt`（显式）。
- **不新增 loss 项**：film 只改条件路径，FM/prior/entropy 全部不变（计划要求）。
- **交互/实时路径**：`sample_latent`/`predict_next_motion` 自动解析 film；交互 blend 改变
  e_s 时会重投影（语义正确；每帧 1 次投影 ≈ +0.4ms@bs512，交互 bs 更小）。

## 5. 复现

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
conda activate mcc
# 单元测试
python -m pytest tests/test_style_film.py -q
# 开关训练（10ep 试点示例）
EPOCHS=10 BATCH_SIZE=512 LR=5e-4 STYLE_FILM=1 bash scripts/run_train_flowmatch_history_obsnoise_tf100.sh
# 冒烟日志/产物（一次性）
analysis/flow_layers_study_2026-10-13/film_smoke.log  # /tmp/film_smoke 为当次输出目录
```

# 解码连续性（decoder-only 校准）修订：实现与正确性检查

2026-10-14。依据 [VQ_DECODE_CONTINUITY_PLAN.md](../../docs/VQ_DECODE_CONTINUITY_PLAN.md)
（GPT-6 推导）。初版为代码修订与实现检查（当时未运行训练）；**其后 A/B 已执行**，并在
独立复核后完成契约修复（§4）。A/B 结果见 [CALIB_AB_RESULT.md](CALIB_AB_RESULT.md)
（纠正版），复核见 [CALIB_AB_AUDIT.md](CALIB_AB_AUDIT.md)。

## 1. 目标（方案的数学核心）

优化对象不是"码 ID 少变"，而是**量化在解码侧额外引入的时间误差**：

```
r_t = p_q(t) - p_c(t)      p_q = P(D(Q(E(x)))),  p_c = P(D(E(x)))
Δ²(p_q - p_GT) = Δ²(p_c - p_GT) + Δ²r
```

新探针 `residual_probe.py` 直接在位置空间（24 关节，local 与世界系）测 `r / Δr / Δ²r`。
基线 150ep VQ（val 12 clips × 120 帧）实测：

| 量（cm，local 24 关节） | mean | p50 | p95 |
|---|---:|---:|---:|
| r（量化残差） | 0.256 | 0.145 | 0.834 |
| Δr | 0.234 | 0.133 | 0.791 |
| **Δ²r** | **0.390** | 0.228 | 1.311 |
| jitter 量化 / 连续 / GT | 0.535 / 0.472 / 0.469 | — | — |
| Δ² 量化 / 连续 / GT | **0.412 / 0.121 / 0.092** | — | — |

读法（与方案一致）：**量化在二阶差分上引入约 3.4× 的额外误差（0.412 vs 0.121）**，
而一阶 jitter 只差 0.06——问题集中在高频/二阶带；且探针明确分列
quantized/continuous/GT 三个量，**不把 `J_q - J_c` 当作 `J(r)`**（方案 Caveat 已落实）。

## 2. 代码修订（三处）

### 2.1 模型层：表示冻结（`arch/basic_vqvae_375to291.py`）

`set_representation_frozen(True)`：
- encoder 全部参数 `requires_grad=False`；
- **`train()` 重写**：每次调用后重新把 encoder/quantizer 置回 `eval()`。这是关键——
  库的 EMA 更新与死码替换守卫是 `self.training`（已核对 `vector_quantize_pytorch`
  源码 780/565 行），**仅冻结参数不足以冻结码本**。
- 新增 `teacher_align_loss = ||D_θ(q) - sg(D0(E0(x)))||²`，方向与既有
  `code_align_loss`（连续靠近离散）**相反**，专门压离散分支的额外时间误差。

### 2.2 训练器：decoder-only 模式（`training/train_basic_vqvae_375to291.py`）

- `--decoder-only`：冻结表示的校验（encoder 无梯度、quantizer 处于 eval、可训练名全为
  `decoder.*`），优化器**只含 decoder 参数**（无冻结参数的 state/weight-decay 槽）；
- `--teacher-ckpt`：冻结 teacher 在 trainer 外部构建，**不进入被训模型的 state_dict**；
  启动时校验 teacher 与 student 的 encoder+quantizer 指纹一致；
- `--teacher-align-weight`：对齐项权重（与 teacher 必须成对出现）；
- 校验矩阵：`--decoder-only` 拒绝 `--latent-margin-weight>0` / `--roundtrip-weight>0`
  （它们训练已冻结的 encoder），自动归零 `vq/code_align/latent_l1/latent_vel`（对冻结部分
  无梯度或属另一方向），并打印被归零项；
- checkpoint 记录 `teacher_fingerprint`；`--resume` 校验
  `decoder_only`/`teacher_align_weight`/teacher 指纹，拒绝静默换目标。

### 2.3 ⚠️ 顺带修复的真实缺陷（影响此前结论）

入口脚本原用 `ru_append_if_enabled cmd "${INIT_CKPT}" --init-ckpt ...` 追加路径参数，但
该 helper 只接受布尔字面量（`1/true/yes/on`），**路径被静默丢弃**：`--init-ckpt` 从未
到达训练器。已核对两个 pilot 的 `config.json`（`init_ckpt` 均为 `null`）——
**K=2048 与 K=1024 pilot 实为从零训练**，"warm start" 表述作废（[REVIEW.md](REVIEW.md) §1
已就地更正）。wrapper 改为显式 `[[ -n ... ]]`，`NORM_STATS` 同款问题一并修复。

意外的好处：两 pilot 因此是**同起点、同预算**对照，与 150ep 基线的 ep30
（recon 0.0554 / mpjpe 1.264）相比，K=2048 新配方（0.0417 / 1.011）同预算显著更优。

## 3. 正确性检查（初版时点，A/B 运行之前）

`tests/test_decoder_only_calibration.py`（初版 9 项；后续修复新增至 14 项）：

| 检查 | 结果 |
|---|---|
| 冻结后 encoder 无梯度、decoder 有梯度 | ✅ |
| `train()` 后 encoder/quantizer 仍为 eval（EMA 真冻结） | ✅ |
| 一次 forward+backward 前后 encoder+quantizer 指纹逐位不变 | ✅ |
| 码索引 q 与码本张量逐位不变 | ✅ |
| teacher 目标 == `D0(E0(x))`（逐位） | ✅ |
| teacher 对齐给出 decoder 梯度、teacher 自身无梯度 | ✅ |
| teacher 形状/缺参校验 | ✅ |
| 表示不匹配的 teacher 被拒绝；码本漂移被指纹捕获 | ✅ |
| decoder-only 与 margin/roundtrip 组合报错；vq/align 自动归零 | ✅ |
| teacher 双参数互斥要求 | ✅ |
| resume 三字段 helper 单元级校验 | ✅ 单元级（⚠️ 入口级组合/顺序缺陷在独立复核中发现并已修复，见 §4） |

回归（初版时点）：**CPU 405 passed / 21 deselected；GPU 21 passed**（最终计数见 §4）。

效率（设计层面，未跑训练）：decoder-only 只反传 decoder（最短反向路径）；teacher 目标走
`decode(encode_pre_vq(x))`（跳过 teacher 的量化前向）；方案允许后续对 q 与 teacher 目标
做 train-only 预计算缓存，本轮未实现（先保证语义正确）。

## 4. 独立复核后的契约修复（2026-10-14 同日，在 A/B 之后）

GPT-6 复核（[CALIB_AB_AUDIT.md](CALIB_AB_AUDIT.md)）指出 A/B 的若干实验契约缺口与代码
隐患；逐项核实**均成立**并已修复：

| 问题 | 修复 |
|---|---|
| teacher 构建消耗全局 RNG，B 的采样顺序与 A 不配对 | `build_frozen_teacher` 包进 `isolated_rng(seed+555)`；新增 `--loader-generator-seed`（>0 时各 split 用独立 generator，默认 0 保持旧行为） |
| teacher 指纹只覆盖 encoder+quantizer，换 decoder 不被发现 | `TEACHER_FINGERPRINT_PREFIXES` 扩为 encoder+quantizer+**decoder**（目标 D0(E0(x)) 的完整路径）；写入 config 与 resume 校验 |
| `--decoder-only` 强制 `--init-ckpt`，与 `--resume` 互斥 → 无法续训 | decoder-only 下允许两者共存（resume 续状态、init 校验表示来源）；普通 run 仍互斥 |
| teacher 指纹在 resume 校验之后才赋值 → B 续训必失败 | 指纹解析提前到 `validate_stability_resume` 之前 |
| norm 未固定（`norm_stats=null`，重算统计改变 0.16% slot 选码） | 入口脚本在 decoder-only 且未显式给 `NORM_STATS` 时自动采用来源 checkpoint 同目录的 `norm_stats.npz`；trainer 在缺失时打印明确警告 |
| 探针口径：混合单位（normalized traj + raw motion）注释错误；`jitter_*` 实为一阶位移 | `frames_of` 注释改为"世界系 globals 由 motion291 自身 root 通道积分，traj 块不参与"；字段更名 `step1_*` 并注明**非**项目正式 Jitter；新增 `r_fixed/dr_fixed/d2r_fixed`（对冻结 teacher 的残差，即 B 真实优化的目标误差），`--teacher-ckpt` 可选启用 |

新增测试 4 项（`tests/test_decoder_only_calibration.py` 共 14 项）：指纹覆盖 decoder、teacher
构建不扰动全局 RNG、loader generator 确定性、decoder-only resume 组合与 norm 警告。
回归：**CPU 410 passed / 21 deselected；GPU 21 passed**。

**旧 A/B 结果保留并注明偏差**（不改配置、不重跑冒充）；结论的正确读法见
[CALIB_AB_RESULT.md](CALIB_AB_RESULT.md) 顶部横幅。

## 5. 边界与未决（沿用方案条款）

1. teacher 有自身误差，不能取代 GT 监督；真实 GT 的 recon/vel/root 仍是主监督。
2. 冻结的是"当前这份" E/C；若之后换 VQ，q 与其下游缓存（`D(q)` 观测、controller 的
   feedback latent）必须整体重算，不能覆盖旧 controller。
3. seq_len=2 / acc_weight=0 不监督二阶；本轮**未**自动开启 acc（方案要求另立受控实验）。
4. 下一阶段（待用户批准）：A（仅 GT 校准）vs B（A + teacher 对齐）各 ≤5ep、同初始化同
   updates，验收同时看 `r/Δr/Δ²r`、jitter、world 足滑、腿部活动/接触覆盖与 IND style
   保真；B 不优于 A 则停止 teacher 路线。

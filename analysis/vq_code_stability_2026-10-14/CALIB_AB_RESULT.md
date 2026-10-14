# Decoder-only 校准 A/B 实验结果（2026-10-14）

> **独立复核（2026-10-14，同日）：见 [CALIB_AB_AUDIT.md](CALIB_AB_AUDIT.md)。**
> 负结果方向成立（位置拟合略好、时间误差未改善；配对 bootstrap 无明确二阶收益），但本报告
> 有三处需要撤正：① A/B 未固定 normalization（`norm_stats=null`，重算统计改变了 0.16% 的
> slot 选码 / 4.7% 帧），"仅 decoder 变化"的对照契约有缺口；② A/B 采样顺序未配对（B 构建
> teacher 消耗全局 RNG）；③ 本报告把（当时探针只能测的）自一致残差当作 B 的优化目标评估，
> 实际 teacher 目标残差（`r_fixed`）需要另行测定。**"decoder 已到能力上限、必须加历史"的
> 归因也一并撤回**——`q−z_e` 固定不等于 `J_D` 固定，本轮没有提供相位信息缺失的证据。
> 代码修复与新增测试见 [DECODER_CALIB_IMPL.md](DECODER_CALIB_IMPL.md) §2.4。
> 原报告正文保留（数字未改），偏差处按上面三条阅读。

依据 [VQ_DECODE_CONTINUITY_PLAN.md](../../docs/VQ_DECODE_CONTINUITY_PLAN.md) §5 执行两组
≤5ep 对照（实现见 [DECODER_CALIB_IMPL.md](DECODER_CALIB_IMPL.md)，规程已复核：EMA 冻结、
码本/encoder 指纹不变、init-ckpt 正确传递）。**结论：decoder-only 校准没有降低量化残差
（本实验的优化目标），teacher 对齐相对 A 无增益——按方案 §5 的预设判定，停止 teacher
路线，且不推广本批 decoder 作为新主 VQ。**

## 1. 实验配置（A/B 唯一变量 = teacher 对齐）

| | 基线 150ep | A：decoder-only | B：A + teacher 对齐 |
|---|---|---|---|
| encoder / EMA 码本 / norm | — | 冻结（自 150ep best 初始化） | 同 A |
| 可训练 | 全部 | 仅 decoder（24 tensors） | 同 A |
| 监督 | recon3/vel15/root3 + vq1 + align2 | recon3/vel15/root3（vq/align/latent 项归零） | 同 A + `‖D(q)−sg(D0(E0(x)))‖²×2` |
| 序列 / batch / lr | 2 / 512 / 2e-4 | 4 / 512 / 1e-4→1e-5 cosine | 同 A |
| epochs / seed / 精度 | 150 / 42 / FP32 | 5 / 42 / FP32 | 同 A |
| 产物 | `…_150ep_nomirror/best.pt`（=ep90） | `vqvae_decoder_calib_A/best.pt`（=ep5） | `vqvae_decoder_calib_B/best.pt`（=ep5） |

**冻结契约核验（执行后）**：三档的 `encoder.*+quantizer.*` 组合指纹**逐位相同**
（`f837eac92a8b476f`），decoder 指纹各不相同；A/B 训练日志 `talign>0`（B）、`dead_reset=0`、
EMA 张量在严格验证中逐位不变。

## 2. 主判据：量化残差 r / Δr / Δ²r（固定 12 val clips × 120 帧，24 关节，cm）

| 指标（mean） | 基线 ep90 | A | B | 读法 |
|---|---:|---:|---:|---|
| r（local） | 0.2564 | 0.2690 | 0.2684 | **未降，+4.9%** |
| Δr | 0.2339 | 0.2373 | 0.2367 | 未降 |
| **Δ²r** | **0.3899** | **0.3944** | **0.3935** | **未降，+1.2%** |
| Δ²q / Δ²c / Δ²gt | 0.4118 / 0.1205 / 0.0918 | 0.4153 / 0.1219 / — | 0.4144 / 0.1217 / — | 量化额外二阶误差仍 ~3.4× |
| r / Δr / Δ²r（world） | 0.5809 / 0.2361 / 0.3909 | 0.6066 / 0.2397 / 0.3956 | 0.6008 / 0.2390 / 0.3947 | 同向 |
| p95 各量 | — | 均 +1~5% | 同 A | — |

**决策依据（方案 §5）："若换码序列保持不变，而 jitter、足滑下降且重建和风格不退化，就说明
改善了换码的运动后果。"** 本实验：翻码序列确实逐位不变（冻结表示），但 **jitter/足滑没有
下降**（见下），残差未降 → 该路线**未兑现其目标**。

## 3. 次判据：训练内固定 90 val 窗口 eval（与基线同一评估程序）

| | 基线 ep90（=其 best.pt） | A | B |
|---|---:|---:|---:|
| mpjpe_cm | 1.2583 | 1.2312（−2.2%） | 1.2240（−2.7%） |
| jitter_cm | 0.4541 | 0.4529（−0.3%） | 0.4529（−0.3%） |
| foot_sliding_cm | **0.1828** | 0.1845（+0.9%） | 0.1842（+0.8%） |
| root_ade_cm | 0.3851 | 0.3583（−7.0%） | 0.3503（−9.0%） |

即：**平均轨迹误差小幅改善（mpjpe/root_ade），jitter 持平、足滑持平略升**——不构成方案
要求的"运动质量联合改善"；且为单 seed、无误差棒，不能作为推广依据。

## 4. B 相对 A：teacher 对齐无增益（预设判定条件）

| | A | B | Δ |
|---|---:|---:|---:|
| Δ²r | 0.3944 | 0.3935 | −0.2%（噪声内） |
| mpjpe | 1.2312 | 1.2240 | −0.6%（噪声内） |
| 其他全部指标 | — | — | 均 ≤1%，方向不一致 |

`talign` 在 5ep 内从 1.01e-3 降到 ~0.90e-3（−11%）后趋于平台，其加权贡献（×2 ≈ 0.0018）
仅占总损失 0.47 的 **0.4%**——teacher 项的梯度量级太小，且其目标本身与主监督高度相关
（teacher = 原 decoder 的连续解码，已由 recon 项覆盖）。**方案 §5 预设："若 B 没优于 A，
停止把 teacher 方案当作解法，不默认延长训练。" → 执行停止。**

## 5. 归因与下一步（⚠️ 本节结论已撤回，以审计为准）

以下推断**不成立或证据不足**（审计 §5）：`q−z_e 固定` 不意味着 `D(q)−D(z_e)` 固定，
decoder 的非线性局部 Jacobian 会变，`J_D(z_e)·(q−z_e)` 只是一阶近似；5ep 失败不能推出单帧
decoder 的函数类或信息论上限；本轮也没有提供"完整 q 缺相位信息"的证据。**§7 的触发条件
未被证实**，不存在"必须加历史"的结论。

正确的下一步判断（合并审计 §6 建议）：
1. 先修实验契约（norm 固定、独立 RNG、teacher 身份、resume 入口、探针口径）——对应代码
   修复见 [DECODER_CALIB_IMPL.md](DECODER_CALIB_IMPL.md) §2.4，这是任何后续有效对照的前提；
2. 若要继续最小训练验证，**优先考虑关闭 teacher、单独开启已有的 GT 二阶差分 `acc` 分支**
   （针对真实运动变化，不把运动压向静止），权重先用真实 train 小 batch 的 decoder 梯度标定；
   同时检查腿部活动、风格与 world 足滑；
3. 不启用 `acc` 时，短历史 decoder 只能作为**独立结构假设**（同参数预算的单帧 vs 因果短
   历史对照），不能当作已被本轮证明的解法；
4. 任何表示层改善最终仍须配套 controller 的原始 rollout 验证。

## 6. 状态与边界

- **不推广**：基线 VQ（ep90）+ 原 controller 仍是唯一有效配套；A/B 仅作实验记录。
  没有重建 cache/latent stats/prior，没有触碰 controller 或任何下游。
- 验收缺口（若将来要推广任何 decoder 变体，须补齐）：腿部活动/接触覆盖、IND style
  保真、world 足滑、以及多 seed 误差棒；本轮因判据为负而不再追加。
- 产物：`checkpoints/vqvae_decoder_calib_{A,B}/`、`calib_{A,B}.log`、
  `residual_{baseline,calib_A,calib_B}.json`、`tests/test_decoder_only_calibration.py`（9 项）。

## 7. 复现

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
BASE=$PWD/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt
# A
EPOCHS=5 BATCH_SIZE=512 SEQ_LEN=4 DECODER_ONLY=1 INIT_CKPT=$BASE \
  LATENT_L1_WEIGHT=0 LATENT_VEL_WEIGHT=0 OUT=checkpoints/vqvae_decoder_calib_A \
  bash scripts/run_train_basic_vqvae_375to291.sh --lr 1e-4 --min-lr 1e-5
# B 追加 TEACHER_CKPT=$BASE TEACHER_ALIGN_WEIGHT=2.0
# 验收
python analysis/vq_code_stability_2026-10-14/residual_probe.py \
  --vq-ckpt checkpoints/vqvae_decoder_calib_A/best.pt --tag calib_A \
  --clips 12 --frames 120 --out analysis/.../residual_calib_A.json
```

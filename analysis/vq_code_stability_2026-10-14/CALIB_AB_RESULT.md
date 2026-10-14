# Decoder-only 校准 A/B：运行记录与结论（纠正版）

2026-10-14。本文档为 decoder-only 校准 A/B 的正式记录。**初版（提交 `ab05397` 附带）包含
三处不成立的结论，本版已删除/改正；原始运行数字与产物路径保留——不隐藏、未重跑篡改。**

> 独立复核（权威）：[CALIB_AB_AUDIT.md](CALIB_AB_AUDIT.md)
> 实现与契约修复：[DECODER_CALIB_IMPL.md](DECODER_CALIB_IMPL.md) §4

## 0. 已撤回的表述（初版 → 本版）

| 初版表述 | 问题（由独立复核认定） | 本版处理 |
|---|---|---|
| "规程已复核：EMA 冻结、指纹不变、init-ckpt 正确传递" | normalization 未固定、采样顺序未配对（见 §2）；init-ckpt 当时确实传入了，但不是"契约已闭合" | 断言删除，改为如实记录偏差 |
| "teacher 只占总 loss 0.4% → 梯度量级太小" | 实测加权梯度为主监督的 1.6–3.0%，且逐帧 MSE 不直接针对高频误差——该推断不成立 | 改为实测陈述 |
| "decoder 已到能力上限，必须加入历史" | `q−z_e` 固定 ≠ `D(q)−D(z_e)` 固定（非线性 Jacobian）；无"相位信息缺失"证据 | 撤回，列为未证假设 |

## 1. 运行配置（唯一设计变量 = teacher 对齐）

| | 基线 150ep | A：decoder-only | B：A + teacher 对齐 |
|---|---|---|---|
| encoder / EMA 码本 | — | 冻结（自 150ep best 初始化） | 同 A |
| 可训练 | 全部 | 仅 decoder（24 tensors） | 同 A |
| 监督 | recon3/vel15/root3 + vq1 + align2 | recon3/vel15/root3（vq/align/latent 项归零） | 同 A + `‖D(q)−sg(D0(E0(x)))‖²×2` |
| 序列 / batch / lr | 2 / 512 / 2e-4 | 4 / 512 / 1e-4→1e-5 cosine5 | 同 A |
| epochs / seed / 精度 | 150 / 42 / FP32 | 5 / 42 / FP32 | 同 A |
| 产物 | `…_150ep_nomirror/best.pt`（=ep90） | `vqvae_decoder_calib_A/best.pt`（=ep5） | `vqvae_decoder_calib_B/best.pt`（=ep5） |

**冻结契约核验（执行后）**：三档的 `encoder.*+quantizer.*` 组合指纹**逐位相同**
（`f837eac92a8b476f`），decoder 指纹各不相同；EMA 张量在严格验证中逐位不变。

## 2. 契约偏差（读以下数字前必看）

1. **Normalization 未固定**：A/B `config.json` 的 `norm_stats=null`，按各自 seq_len4 窗口重算，
   与基线（seq_len2 统计）不同。实际数组差：mean 最大绝对差 0.001219、std 最大 0.014915；
   在同一冻结 E/Q 上换用 A/B norm 后 **0.158% 的 slot 选码改变、4.713% 的帧至少一个 slot
   改变**。→ "权重冻结"不等价于"完整映射冻结"，与基线的比较不是严格单变量。
2. **采样顺序未配对**：B 在迭代前构建 teacher（随机初始化后加载权重）消耗全局 torch RNG，
   DataLoader 无独立 generator；相同 seed42 不保证相同数据顺序，且未保存顺序 manifest。
3. **修正后重测**（审计 §3）：改用各自训练 norm 评估，GT 二阶误差 B=0.40017（基线 norm）
   /0.39937（训练 norm）vs 基线 0.39770——偏差量级不足以解释负结果，**不能把失败归因于 norm**。

（teacher 指纹未覆盖 decoder、decoder-only 无法 resume、resume 校验顺序三处代码缺陷没有
影响本轮 fresh 训练结果；修复见 DECODER_CALIB_IMPL §4。）

## 3. 测量

### 3.1 量化自一致残差 r = P(D(q)) − P(D(z_e))（固定 12 val clips×120 帧，24 关节，cm）

| 指标（mean） | 基线 ep90 | A | B |
|---|---:|---:|---:|
| r（local） | 0.2564 | 0.2690 | 0.2684 |
| Δr | 0.2339 | 0.2373 | 0.2367 |
| **Δ²r** | **0.3899** | **0.3944** | **0.3935** |
| Δ² quantized / continuous / GT | 0.412 / 0.121 / 0.092 | 0.415 / 0.122 / 0.092 | 0.414 / 0.122 / 0.092 |
| world r / Δr / Δ²r | 0.581 / 0.236 / 0.391 | 0.607 / 0.240 / 0.396 | 0.601 / 0.239 / 0.395 |

（`jitter_*` 初名实为一阶位移代理，已更名 `step1_*`；非项目正式 Jitter 指标。）

### 3.2 冻结 teacher 残差 r_fixed = P(D_θ(q)) − P(D0(E0(x)))（审计补测，基线 norm）

| 指标 | 基线 | A | B |
|---|---:|---:|---:|
| 位置残差 | 0.25644 | 0.32464 | 0.31577 |
| 残差二阶范数 | 0.38990 | 0.39380 | 0.39296 |
| GT 位置误差 | 1.06965 | 1.04979 | 1.05236 |
| GT 一阶误差 | 0.26356 | 0.26478 | 0.26437 |
| GT 二阶误差 | 0.39770 | 0.40105 | 0.40017 |
| teacher normalized291 MSE | 0.001201 | 0.001716 | 0.001625 |

配对 bootstrap（按 source_entry_id，1000 次）：B−A 的 GT 二阶误差 = −0.00088cm
[−0.00197, +0.00012]；B−基线 = +0.00247cm [−0.00439, +0.00856]——**支持"未观察到明确时间
收益"，不支持证明严格相等或必然变差**。

### 3.3 训练内固定 90 val 窗口 eval（与基线同一评估程序）

| | 基线 ep90 | A | B |
|---|---:|---:|---:|
| mpjpe_cm | 1.2583 | 1.2312（−2.2%） | 1.2240（−2.7%） |
| jitter_cm | 0.4541 | 0.4529（−0.3%） | 0.4529（−0.3%） |
| foot_sliding_cm | **0.1828** | 0.1845（+0.9%） | 0.1842（+0.8%） |
| root_ade_cm | 0.3851 | 0.3583（−7.0%） | 0.3503（−9.0%） |

（更正：B 相对 A 并非"全部指标 ≤1%"——teacher MSE 低约 5.3%、RootADE 低约 2.2%；
但不足以构成验收要求的"jitter/足滑联合改善"。）

## 4. 结论（纠正版）

1. **负结果成立**：位置拟合略好（mpjpe −2~3%、root_ade −7~9%），**时间连续性无改善**
   （Δ²r +1~5%、jitter 持平、足滑持平略升）。按方案 §5 预设：**停止 teacher 路线、
   不推广本批 decoder**；基线 VQ（ep90）+ 原 controller 仍是唯一有效配套。
2. **不做能力上限声明**：5ep 失败不能推出单帧 decoder 的函数类/信息论上限；"必须加历史"
   不成立。teacher 假设未兑现（其目标与主监督高度相关，且逐帧 MSE 对高频不直接）。
3. **下一步**（先修契约、再做最小验证）：见 [docs/NEXT_TASKS.md](../../docs/NEXT_TASKS.md)——
   优先 `acc`（GT 二阶差分）单变量实验，权重先按真实 train 小 batch 梯度标定；短历史
   decoder 为条件性结构假设，不作为已验证解法。

## 5. 复现（契约修复后的推荐命令）

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
BASE=$PWD/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt
EPOCHS=5 BATCH_SIZE=512 SEQ_LEN=4 DECODER_ONLY=1 INIT_CKPT=$BASE LOADER_SEED=4242 \
  LATENT_L1_WEIGHT=0 LATENT_VEL_WEIGHT=0 OUT=checkpoints/vqvae_decoder_calib_A2 \
  bash scripts/run_train_basic_vqvae_375to291.sh --lr 1e-4 --min-lr 1e-5
# （decoder-only 自动沿用来源 norm_stats；teacher/acc 变体见 NEXT_TASKS.md）
python analysis/vq_code_stability_2026-10-14/residual_probe.py \
  --vq-ckpt checkpoints/vqvae_decoder_calib_A2/best.pt --tag calib_A2 \
  --clips 12 --frames 120 --out analysis/vq_code_stability_2026-10-14/residual_calib_A2.json
```

# 后续任务规划（2026-10-14）

本文是 VQ 码保真/解码连续性方向的正式任务清单。前置结论（均已归档，不再重开）：

- 码本跳变三层分解：codec 往返脆弱（37.8%）+ 单步误差≈半胞元 + 闭环暴露偏差
  （[codebook_divergence_2026-10-14/REPORT.md、CAUSE_ANALYSIS.md](../analysis/codebook_divergence_2026-10-14/)）。
- margin/roundtrip 辅助项：从零 30ep 有改善但不满足运动质量联合验收
  （[vq_code_stability_2026-10-14/AUDIT.md](../analysis/vq_code_stability_2026-10-14/AUDIT.md)）。
- decoder-only 与 teacher 对齐 A/B：负结果，teacher 路线停止
  （[CALIB_AB_RESULT.md](../analysis/vq_code_stability_2026-10-14/CALIB_AB_RESULT.md)、
  [CALIB_AB_AUDIT.md](../analysis/vq_code_stability_2026-10-14/CALIB_AB_AUDIT.md)）。
- 契约修复（norm 固定、独立 RNG、teacher 身份、resume 入口、探针口径）已完成并测试；
  基线 VQ + controller 仍是唯一有效配套，**新 VQ/decoder 未推广**。

## 阶段 0：完成清理（本次，进行中）

- [x] 重写 `CALIB_AB_RESULT.md`：撤回三处错误表述，保留原始数字
- [x] `DECODER_CALIB_IMPL.md`：补 §4 契约修复，标注 §3 为初版时点
- [x] 计划文档横幅同步（`VQ_DECODE_CONTINUITY_PLAN.md`）
- [x] 探针彻底消除混合单位（traj 槽用零块占位，不再拼接 normalized traj）
- [x] 旧报告就地更正：`REPORT.md`/`CAUSE_ANALYSIS.md`/`CODE_ANALYSIS.md`/`VQ_LEVEL.md` 横幅 +
      正文中"物理不可能/硬上限"等表述就地标注撤回；全仓 markdown 断链 = 0
- [x] README 顶部状态行更新为"当前唯一配套 = 基线 VQ(ep90)+原 controller"
- [ ] 提交

## 阶段 1：`acc` 单变量实验 — **已完成，结论 FAIL**（2026-10-14）

> 执行记录（含预注册协议、梯度标定、两臂配置与逐门验收）见
> [`analysis/vq_temporal_source/COMMON.md`](../analysis/vq_temporal_source/COMMON.md) 与
> [`S4/REPORT.md`](../analysis/vq_temporal_source/S4/REPORT.md)；原始 JSON/日志同目录归档。
>
> 概要：从 ep90 完整 VQ 状态冷启两组（B0 acc=0 / B1 acc=4.9842，S2 标定），各 51650 updates、
> 唯一变量 ACC_WEIGHT、固定终点 epoch_0010.pt。Δ²r 与 Δ²e_q 显著下降（-11% / -10%，配对 CI
> 排除 0）、world 足滑改善、风格代理持平；但 **jitter 仅 -9.8%（未达 -15% 预注册阈值）、
> local MPJPE +21.2%（上限 +3%）、活动代理 -5.6%（上限 -5%）**，且连续 AE 路径同向劣化
> （c.local_mpjpe +22%）——变平滑不是量化残差特有收益。**按预注册否决规则判 FAIL，不延长训练。**
>
> 附带实现修复：入口脚本 `--no-decoder-only` 缺陷（非 decoder-only 运行全部受影响），
> 已修 + 回归测试；S0 契约校验现用 trainer 自身 argparse 校验展开命令。

### 原计划文本（保留存档）

**假设**：逐帧 MSE 对高频/二阶误差不敏感；直接监督 GT 二阶差分可压低 quantized 解码的
`Δ²r`，且不把运动压向静止（目标是差分匹配，不是零差分）。

**假设**：逐帧 MSE 对高频/二阶误差不敏感；直接监督 GT 二阶差分可压低 quantized 解码的
`Δ²r`，且不把运动压向静止（目标是差分匹配，不是零差分）。

**设计（单变量 = `--acc-weight`，状态复用已验证的 decoder-only 机制）**：

| 项目 | 值 |
|---|---|
| 初始化 | 150ep 基线 VQ best（=ep90），decoder-only 冻结 E/Q（EMA 已可冻结） |
| 唯一变量 | `ACC_WEIGHT` ∈ {0（对照 A2）, 标定值}，其余与 A2 相同 |
| 序列 | `seq_len=4`（acc 需 ≥3；与 A/B 同） |
| norm | 由入口自动固定为来源 `norm_stats.npz`（契约已修） |
| RNG | `LOADER_SEED=4242`（契约已修；A2/B2 同值） |
| teacher | 关闭（`--teacher-ckpt` 不传） |
| 其他项 | `vq/code_align/latent_l1/latent_vel` 归零（decoder-only 自动）；margin/roundtrip=0 |
| lr | 1e-4→1e-5 cosine |
| epochs | 5（与 A/B 同预算） |
| 种子/精度 | 42 / FP32 |

**权重标定（先做，不训练）**：真实 train 小 batch（≈16×T4，4 个），分别求
`g_GT = ∇(3L_recon+15L_vel+3L_root)` 与 `g_acc = ∇(L_acc)` 的 decoder 梯度范数与余弦；
取 `acc_weight` 使 `‖g_acc·w‖ / ‖g_GT‖ ≈ 0.1–0.3`（**不能**由 loss 数值照搬，审计已证
loss 占比≠梯度占比）。记录到实验报告。

**验收（缺一不可）**：

1. `Δ²r` 与 `Δ² quantized` 相对 A2 下降（探针 `residual_probe.py`，固定 12 val clips；
   p95 同看）；
2. 训练内 90 窗口 eval 的 `jitter`、`foot_sliding` 下降或至少不升；`mpjpe/root_ade` 不退化；
3. 腿部活动代理（移动帧占比/关节行程）与 contact 覆盖不下降——防止"更平滑但不动了"；
4. IND val style fidelity 不回退；world 足滑单独报告；
5. 与 A2 同 norm、同 LOADER_SEED、同 updates；配对 bootstrap 给区间。

**止损**：若 `acc` 不降 `Δ²r` 或触发第 3 条，判定该路线不成立，停止，不延长训练。
（实际结果：`Δ²r` 降了，但运动质量与活动性代价超出上限 → FAIL；见上方执行记录。）

**判读（对阶段 2 的含义）**：acc 在归一化 motion291 上直接监督二阶差分，能压低 Δ²r/jitter，
但代价集中在**逐关节位置精度**上（local MPJPE +21%）。与 S5a 的位置空间 SmoothL1 相比，
本轮的失败模式说明"二阶项位置正确但一级几何漂移"是这条路线的主要风险，而不是权重太小。
若启动阶段 2/5，必须把"local MPJPE 不退化"作为硬约束而非事后观察项，并预先说明二阶项与
recon 在参数空间上的竞争处理（例如按层分组限额或投影，而不是全局加权重）。

## 阶段 2（条件触发）：短历史 decoder 结构实验

**仅当决定继续攻码保真时启动**（阶段 1 已 FAIL，是否继续需用户决定；见 S5d）。定位为独立结构假设，
不是已证解法。

- 预注册：同参数预算的**单帧 vs 因果短历史**（输入 `q_{t-1}, q_t` 或 `Δq`）对照，
  只改 decoder 输入结构；断言"历史携带可泛化的时序信息"需要 val 活动/风格/world 接触
  同时验证；
- 风险：训练历史是 GT 编码（`Q(E(x_{t-1}))`），部署时是 controller 的预测历史，分布不同；
  方案须写明该 gap 的评估方法；
- 依据边界：`q−z_e` 固定 ≠ `D` 的 Jacobian 固定（审计 §5），本轮不预设结论。

## 阶段 3（低优先）：表示层容量/维度再现实验

写清"同预算、同 split、对齐 val 指标"的再现条件后再做；接受"合并实验只是上界估计，
重训结果才能定论"。**不与阶段 1/2 同时进行**（审计要求单变量）。

## 阶段 4（最终验收，任何表示层改动的收口）

- 新 VQ/decoder 必须作为显式版本：重建 `D(q)` 观测缓存、latent stats、style prior；
- 配套 controller 短 TF 适配（如有）后，按封存协议跑 matched/cross/MPJPE/jitter/FS，
  与基线 controller 同协议对照；
- 交互/实时路径仅在上述全绿后才切换；**旧 controller 不得被无记录 override**。

## 效率备忘（审计 §6 第 5 条）

冻结 E/Q 时可在正确 norm 下一次缓存 `q`（或 indices），后续 decoder 训练合批跳过
重复编码/最近邻搜索。属于实现优化，不改变上述实验结论；在阶段 1 的标定脚本里就可先用
（顺带量化缓存收益），但不作为独立实验。

## 与主线的关系

- 本方向是**表示层支线**：目标是把 quantized 解码的运动代价压下去。
- label 主线（controller 的 cross 能力）不受影响；任何 VQ 变更在上线前都必须重跑下游。
- 每阶段结束：更新本文件状态 + 归档报告（含原始 JSON/日志），负结果同样记录。

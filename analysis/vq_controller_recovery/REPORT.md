# VQ / Controller 稳定性修复：执行总报告（G0 → C2a）

规约：[docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md)（已入库）。
分支：`recovery/vq-controller-2026-10-15`（`main` 与现役默认未动）。
执行原则：先诊断真实闭环 → 严格单变量 → 先保姿态/活动/风格再谈 Jitter；VQ 与 Controller 分离验收。

## 1. 范围与授权边界

- 已执行（计划 §9 首轮交付）：**G0 契约 → C0 闭环因果分解 → C1 推理旁路 → V0 基线复核**，以及
  **C2/V1 的代码、单测、dry-run、梯度标定**。
- 训练：按"经批准后 C2/V1 中选择**一个**限额 A/B"，本轮运行 **C2a（观测噪声 3 臂 ×10ep）**；
  V1（decoder-only 物理二阶）已完整就绪但**未启动**，等单独批准。
- 全程 `feedback_probability=0`，无 rollout/DAgger/donor 训练；未改现役 controller、VQ 权重与缓存。

## 2. 各阶段结果

| 阶段 | 结论 | 关键数字 | 归档 |
|---|---|---|---|
| **G0** | **PASS** | 配套指纹四项交叉核验一致；CPU 441 / GPU 22；wrapper DRY_RUN 可被训练器 argparse 解析 | `G0/contract.json,AUDIT.md` |
| **C0** | PASS（诊断） | A≈B（观测流形差 ≤0.02cm）；**B→C 主导缺口** local MPJPE 1.17→4.43cm、strong −44%、style 12/12→10/12；**C vs E 连续反馈并不更好**（5.18 vs 4.43）；单步扰动 15.6/32 槽换码、32 帧残留 0.31cm | `C0/REPORT.md`、`per_source.json`（0 不变量违规） |
| **C1** | **NO_USEFUL_INFERENCE_FIX** | 所有 jitter 改善候选都以 >5% 活动性代价（N0 −43% jitter / moving −15.4%；N05 −31% / −9.8%；S2 −30% / −10.8%）；R0 丢风格；S8 全面变差；H0=去 history 崩坏（style 3/12） | `C1/REPORT.md` |
| **V0** | **PASS** | 固定 val12 探针相对 S1 归档**全部 0.00e+00 差**；逐源 d2r 最大差 0 | `V0/baseline.json,REPORT.md` |
| **V1 准备** | 就绪，待训练授权 | posacc 代码+9 单测全过；标定 decoder 比值 0.105 / cos 0.057 → **λ=1.6278**；效率 **+0.2%（0.1ms）** | `V1/PROTOCOL.md`、`gradient_calibration.json`、`vq_posacc_benchmark.json` |
| **C2a** | **INCONCLUSIVE**（逐臂：A075 有确认增益但质量点估计未全过限；A025 被否决） | 见下表 | `C2/DECISION.md`、`acceptance.json` |

## 3. C2a 主表（三臂唯一变量 = 观测噪声幅度；同 init `26dc40eb`、各 51680 updates、验证用例同集）

| 臂 | matched | cross | cross ADE | FS world | 扰动恢复 root(+32) | 活动性 | 逐臂判定 |
|---|---:|---:|---:|---:|---:|---|---|
| A025 | 0.7704 | 0.0556 | 7.647 | 0.549 | 0.260 | strong −8.3% ✗ | FAIL（否决） |
| A050（对照复算） | 0.7667 | 0.1630 | 7.833 | 0.599 | 0.409 | — | 基准 |
| A075 | **0.8037** | **0.2222** | **7.531** | 0.585 | **0.196** | 全过 ✓ | INCONCLUSIVE（增益确认，限未全过） |

- **cross 对噪声幅度单调敏感**（0.0556→0.1630→0.2222），matched 同时小幅上升；这是自
  C0 诊断（自反馈分布偏移）以来首个正向闭环信号，且完全在纯 TF 框架内。
- A075 未过干净 PASS：闭环 local/world MPJPE/rootADE 点估计 +3.3%/+5.9%/+5.6%（未获 CI 确认）、
  长协议 FS local +1.9%。12 源闭路样本下 CI 宽，需批准的新一轮（加长预算 / C2b 相关性 / 扩大评估集）
  才能定论；**不自动延长、不替换默认 controller**。

## 4. 代码修订后的正确性与效率回查（运行前完成）

正确性：

- 全量回归：**CPU 441 passed / GPU 22 passed**（含新增 `tests/test_v1_posacc.py` 9 项、
  `tests/test_closed_loop_feedback_c0.py` 7 项；V1 改动后 resume 守卫的既有测试同步修复）。
- 四个 wrapper 的 DRY_RUN 展开均由**训练器自身 argparse** 解析通过：VQ 入口、控制器 obsnoise
  TF100、V1-B0/B1（posacc/dedoder-only pin）、C2 三臂（仅噪声不同）。
- 口径交叉验证：C1-REF 与 C0-C 逐位一致；C0/C2 闭路复用同一 `loop_lib`；jitter 恒等断言
  （探针值 vs 官方 `motion_eval` sum/count）通过；V0 与归档 0 差。
- 标定不变量：全程未调用 optimizer；V1 标定后 state fingerprint 不变；C0/C1/C2 验收后
  fingerprint 不变（脚本内断言）。
- posacc 数值性质：常速度/常量偏移二阶差为 0；加速度与解析值一致（4e-7）；时序错位/振荡为正且有序；
  E/Q 在 decoder-only 冻结下逐张量不变（经 `model.train()` 复验）。

效率：

- V1 posacc 开销 **+0.2%**（32.8→32.9ms，B512×T8）；显存无变化。
- C0 全套 12 源 ×5 组+扰动 = 30.3s；C1 7 配置 ×12 源 = 47s；C2 每臂 ≈17min（10ep，94s/epoch）。
- 推理延迟：C1 中 REF 2.17ms/帧、S2 1.54ms；均远低于 16.7ms 实时预算。

## 5. 当前状态与下一步（等待授权）

- **可回滚**：所有输出在独立目录；`main` 未改动；现役 VQ/controller/cache 原样。
- **待批准**（每项都需要单独授权）：
  1. **V1 训练**：`POSACC_WEIGHT=1.6278`，decoder-only、5ep/臂（wrapper 与验收脚本已就绪）。
  2. **C2 追加轮**：沿 `A=0.75` 加长预算，或 C2b（AR(1) 时间相关性，需新 objective version），
     或扩大闭路评估集以收紧 A075 的质量 CI。
  3. 其余按规约：任何 V2/集成/默认替换都需要新的授权与版本化缓存流程。

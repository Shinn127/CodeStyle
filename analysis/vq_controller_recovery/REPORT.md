# VQ / Controller 稳定性修复：执行总报告（G0 → C2a）

规约：[docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md)（已入库）。
分支：`recovery/vq-controller-2026-10-15`（`main` 与现役默认未动）。
执行原则：先诊断真实闭环 → 严格单变量 → 先保姿态/活动/风格再谈 Jitter；VQ 与 Controller 分离验收。

## 1. 范围与授权边界

- 已执行：**G0 契约 → C0 闭环因果分解 → C1 推理旁路 → V0 基线复核**，以及
  **C2a（观测噪声 3 臂 ×10ep）** 与 **V1（decoder-only 物理二阶，5ep/臂）** 两个限额 A/B，
  外加 **30 源扩展评估**（评估侧，零训练成本）。
- 全程 `feedback_probability=0`，无 rollout/DAgger/donor 训练；未改现役 controller、VQ 权重与缓存；
  未超注册预算（C2a 3×10ep、V1 2×5ep）。

## 2. 各阶段结果

| 阶段 | 结论 | 关键数字 | 归档 |
|---|---|---|---|
| **G0** | **PASS** | 配套指纹四项交叉核验一致；CPU 441 / GPU 22；wrapper DRY_RUN 可被训练器 argparse 解析 | `G0/contract.json,AUDIT.md` |
| **C0** | PASS（诊断） | A≈B（观测流形差 ≤0.02cm）；**B→C 主导缺口** local MPJPE 1.17→4.43cm、strong −44%、style 12/12→10/12；**C vs E 连续反馈并不更好**（5.18 vs 4.43）；单步扰动 15.6/32 槽换码、32 帧残留 0.31cm | `C0/REPORT.md`、`per_source.json`（0 不变量违规） |
| **C1** | **NO_USEFUL_INFERENCE_FIX** | 所有 jitter 改善候选都以 >5% 活动性代价（N0 −43% jitter / moving −15.4%；N05 −31% / −9.8%；S2 −30% / −10.8%）；R0 丢风格；S8 全面变差；H0=去 history 崩坏（style 3/12） | `C1/REPORT.md` |
| **V0** | **PASS** | 固定 val12 探针相对 S1 归档**全部 0.00e+00 差**；逐源 d2r 最大差 0 | `V0/baseline.json,REPORT.md` |
| **V1** | **FAIL**（否决：以精度/活动性换 Jitter） | `‖Δ²r‖` −25.7%（CI 确认）、jitter **−23.8%（首次过 15% 门）**、FS −20%；但 local MPJPE **+65.4%**、活动性 −16~43%、风格 −2.22pp；连续路径同幅恶化（+71%） | `V1/DECISION.md`、`acceptance.json`、双探针 |
| **C2a** | **FAIL**（A025 风格否决；A075 经 30 源扩展评估 CI 确认 world MPJPE +14.2% / FS +12.2% 恶化） | 见下表与 §3.1 | `C2/DECISION.md`、`acceptance.json`、`expand_eval.json` |

## 3. C2a 主表（三臂唯一变量 = 观测噪声幅度；同 init `26dc40eb`、各 51680 updates、验证用例同集）

| 臂 | matched | cross | cross ADE | FS world | 扰动恢复 root(+32) | 活动性 | 逐臂判定 |
|---|---:|---:|---:|---:|---:|---|---|
| A025 | 0.7704 | 0.0556 | 7.647 | 0.549 | 0.260 | strong −8.3% ✗ | FAIL（否决） |
| A050（对照复算） | 0.7667 | 0.1630 | 7.833 | 0.599 | 0.409 | — | 基准 |
| A075 | **0.8037** | **0.2222** | **7.531** | 0.585 | **0.196** | 全过 ✓ | INCONCLUSIVE（增益确认，限未全过） |

- **cross 对噪声幅度单调敏感**（0.0556→0.1630→0.2222 @240f；A075 的 cross 优势在 ep5 即建立
  0.2296 vs 0.1222 并保持），matched 同时小幅上升——自 C0 诊断（自反馈分布偏移）以来首个正向
  闭环信号，完全在纯 TF 框架内。
- 但 **30 源扩展评估（评估侧，同协议）确认 A075 的质量代价**：world MPJPE +14.2%
  （CI[+0.044,+2.53]）、world FS +12.2%（CI[+0.004,+0.252]）、moving −3.1%（确认）。
  A025 在 30 源上 root ADE −22.5%（CI 确认改善）却被其 240f cross −10.7pp 的注册否决挡住。
- 结论：观测噪声幅度轴无免费午餐（0.25 伤 cross、0.75 伤 world 质量）。**不执行 30ep 扩展轮**
  ——扩展预算不得用于已被 CI 确认恶化的方向（规约：明显退化即 STOP）。如重开需新机制
  （C2b 时间相关性/选择性噪声）或结构变化，属新授权。

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

## 5. 两条线的共同结论

1. **VQ 侧**（S3/S4 → V1，两次二阶损失实验：混合空间 → 物理位置空间）：`Δ²r`/jitter 可被 decoder
   端显著吸收（V1：−25.7% / −23.8%，首次越过 15% Jitter 门；FS −20%），但**在现架构/预算下
   姿态精度与活动性必然付出远超门槛的代价**（local MPJPE +65.4%、活动 −16~43%、风格 −2.22pp），
   且连续 AE 路径同幅恶化（+71%）说明不是"损失空间选错"。**按决策树 STOP VQ 训练线**；
   V2 前提（V1 PASS）不成立。
2. **Controller 侧**（C0 → C1 → C2a）：闭环缺口主因是**自反馈观测分布偏移**（C0 铁证：1.17→4.43cm）；
   推理期无免费修复（C1：所有 Jitter 改善候选都靠活动性交换）；训练期观测噪声轴存在真实但
   **带代价**的收益——cross +5.9pp（A075，ep5 即建立）伴随 30 源 CI 确认的 world 质量恶化 +14.2%；
   root ADE −22.5%（A025，30 源确认改善）伴随 240f cross 风格 −10.7pp。**10ep 预算内无可推广配置。**
3. 唯一无争议的正向事实：**扰动恢复**（A075 使单步码级扰动的 32 帧 root 残留 −52%，CI 确认）
   ——闭环鲁棒性确实可被训练改善，只是质量代价在当前形式下不可接受。

## 6. 当前状态与下一步（等待授权）

- **可回滚**：所有输出在独立目录；`main` 未改动；现役 VQ/controller/cache 原样。
  两个支线的实验产物均为新身份（未注入现役配套）。
- **未执行的预算**（代码/协议/验收脚本均已就绪）：
  1. **C2 30ep 扩展轮**（`ARM=C2-A050/A075`，各 30ep）：因 30 源确认退化而按 STOP/止损规则搁置；
     若需验证"更长预算能否收敛掉 A075 的质量代价"，这是唯一待跑项（约 1.6h 双卡串行）。
  2. **C2b**（AR(1) 时间相关性）：需新 objective version 与实现，属新立项。
  3. **V3**（因果短历史 decoder）：V1 证明"decoder 端可吸收量化时间误差但代价失控"，
     历史条件结构是唯一未检验的 VQ 侧假设，需独立立项。
- **不可执行**（无干净 PASS）：V2 全 VQ 联合、集成/默认替换、cache 重建。

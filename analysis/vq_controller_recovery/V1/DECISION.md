# V1 决策报告：Decoder-only 物理位置二阶损失（5ep/臂）

依据 [PROTOCOL.md](PROTOCOL.md)（运行前冻结；含 lambda 的 decoder-only-base 更正说明）。
原始数据：[acceptance.json](acceptance.json)、[probe_B0_ep5.json](probe_B0_ep5.json)、
[probe_B1_ep5.json](probe_B1_ep5.json)、各臂 `checkpoints/vq_posacc_V1-B{0,1}_ep90seq8_5ep/`。

## 0. 阶段报告必备字段

- **stage**: V1（首轮 5ep/臂，注册上限内）
- **hypothesis**: mixed normalized-motion291 的二阶项与位置重建竞争；改为**物理 24 关节位置
  （cm）**的 SmoothL1 二阶监督，可在不牺牲位置精度的前提下压低量化附加 jitter。**本轮证伪。**
- **source_commit**: `0431ed2`（pre-flight 修正后）
- **tracked_dirty**: false；**actual_update_counts**: 25825 / 25825（相同）
- **model_ckpt_hashes**: init state=`918b5e07046266f7…`（两臂同）；E/Q 逐张量 bitwise 不变、
  decoder 24/24 张量更新（manifest 契约门 11 项全 PASS）
- **protocol_and_seed**: decoder-only、seq8、bs512、lr 1e-4→1e-5、seed 42、loader_seed 4242、
  beta 0.5cm、B0 λ=0 / B1 λ=1.6295（按训练器实际 base 标定：decoder ratio 0.1046 / cos 0.056）
- **verdict**: **FAIL**（否决规则：以姿态精度/活动性换低 Jitter）
- **next_stage_approval_needed**: VQ 训练线按计划 STOP；如需 V3（短历史 decoder）等新结构，
  另行立项授权。

## 1. 门结果

| 门 | 结果 | 关键数字 |
|---|---|---|
| V1-0 契约 | PASS | 11 项全过（同 init/同 updates/仅 posacc 不同/E/Q bitwise 不变/decoder 有更新） |
| V1-1 `‖Δ²r‖` | **PASS** | mean 0.3939→0.2928（−25.7%，CI[−0.123,−0.077] 确认）；p95 1.2583→0.8388（−33.3%，CI 确认） |
| V1-2 Jitter | **PASS** | 0.4153→0.3165（**−23.8%**，首次达成 ≥15% 预注册门） |
| V1-3 质量 | **FAIL** | local MPJPE 1.0435→1.7260（**+65.4%**）；world MPJPE 1.5973→2.2376（+40.1%）；root ADE +2.8%（限内） |
| V1-4 足滑 | PASS | world −19.9%、local −21.3%（均改善；计数完整） |
| V1-5 活动性 | **FAIL** | moving −15.8%、strong −42.6%、帧均位移 −12.1%、腿行程 −14.1%、趾速 −14.8% |
| V1-6 风格 | **FAIL**（差 0.2pp 过线） | 90 val source 准确率 0.9556→0.9333（−2.22pp > 2pp 限） |
| V1-7 证据 | 报告 | 连续路径同幅恶化（c.local_mpjpe **+71.1%**、c.jitter −9.4%、c.fs_world +31.3%）；codec 翻码率 0.606→0.707 |

## 2. 结论

1. **假设被证伪，且失败模式比 S3/S4 更强**：S3/S4 的 mixed-space acc 是 local MPJPE +21%；
   本轮物理位置空间改进了 −26% Δ²r 与 −23.8% jitter（首次过门），但 local MPJPE +65%、
   活动性 −16~43%。**"换成物理位置空间就能避开 tradeoff"不成立**——两轮一致表明：
   在现 decoder 容量/预算下，二阶平滑目标与逐帧位置精度是**直接竞争**的，与损失空间无关。
2. 连续 AE 路径同幅恶化（+71%），排除"量化残差特有收益"。
3. 唯一在闭环与质量间同时正向的是 controller 侧的**训练期观测噪声**（C2a：A075 的 cross
   0.2222、活动/风格保持、扰动恢复 −52% CI 确认）。VQ 侧二阶损失线（S3/S4 → V1）连续两次
   在注册门内失败，**按决策树 STOP VQ 训练线**。
4. 可留档的正向副产品：`‖Δ²r‖` 与 jitter 可以在冻结 E/Q、只动 decoder 的条件下大幅下降
   （−26%/−24%），说明"量化附加时间误差"确实可被 decoder 端部分吸收；缺口在于代价控制，
   而非可能性。

## 3. 后续建议（均需另行授权）

- **V2（全 VQ 物理损失）不启动**：V1 已证伪前提（frozen-Q 下都无法保住精度，解冻 E/Q 只会
  扩大活动面）。
- **V3（因果短历史 decoder）**：仅当愿意接受"结构改动"作为下一个独立假设；需独立立项。
- **C2 方向**：沿观测噪声轴继续（cross 单调敏感是本轮最强信号）；若继续，建议先做
  扩大评估集以收紧质量 CI（评估侧，零训练成本；脚本 `../C2/c2_expand_eval.py`）。
- 不为 Jitter 单独置换现役 VQ/decoder；不重建 cache。

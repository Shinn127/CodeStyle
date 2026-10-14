# C2a 决策报告：纯 TF 观测噪声三臂 10ep（A ∈ {0.25, 0.5, 0.75}）

依据 [PROTOCOL.md](PROTOCOL.md)（运行前冻结）。原始数据：[acceptance.json](acceptance.json)、
各臂 `checkpoints/c2_obsnoise_A{025,050,075}_10ep/{config.json,history.jsonl,manifest.json}`。

## 0. 阶段报告必备字段

- **stage**: C2a
- **hypothesis**: 训练期把观测替换为病态 GT 观测（U[0,A]·transport_std → clip→Q→D）可缩小
  C0 发现的自反馈分布偏移缺口，且不引入 rollout 训练。
- **source_commit**: `521f60c`（V1/C2 准备提交）；训练期间训练源文件未变（manifest 记录）
- **tracked_dirty**: false
- **model_ckpt_hashes**: init 控制器 state sha=`26dc40eb30bb9370…`（三臂相同）；
  VQ state sha=`918b5e07046266f7…`（冻结）；三臂终态 sha 互不相同（预期）
- **normalization_hash / db_split_fingerprints**: 与 G0 相同（原 norm；无镜像 DB；val 固定用例
  `starts_sha=2cefaade62b6`，**三臂验证用例逐位一致**）
- **protocol_and_seed**: 纯 TF p=0、10ep、bs512、lr 5e-4 cosine、seed 42、solver 4；
  各臂 51680 updates；固定终点 `epoch_0010.pt`
- **actual_update_counts**: 51680 / 51680 / 51680（契约门 PASS）
- **unit_tests_and_regressions**: 训练前 CPU 441 / GPU 22 通过；验收脚本复用 S4 已验证 bootstrap；
  C0-C↔C1-REF 与 C0↔C2 闭路口径交叉一致
- **verdict**: **INCONCLUSIVE**（逐臂：A025=FAIL 触发否决；A075=INCONCLUSIVE——有确认增益、
  质量点估计未全过限；无臂达成干净 PASS。见 §2/§3）
- **next_stage_approval_needed**: 如需沿 A075 方向继续（更长预算或 C2b 时间相关性），需新授权；
  **不自动延长、不替换默认 controller**

## 1. 契约与主结果

契约门 PASS：同一 init（`26dc40eb`）、同 51680 updates、config 差异仅
`{out_dir, observation_noise_std}`、噪声值与臂标签一致、验证用例同集。

| 臂 | A | matched(240f) | cross(240f) | matched ADE | cross ADE | FS world | FS local | jitter(train) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A025 | 0.25 | 0.7704 | **0.0556** | 4.334 | 7.647 | 0.549 | 0.240 | 0.442 |
| A050（对照=现役配方复算） | 0.50 | 0.7667 | 0.1630 | 4.749 | 7.833 | 0.599 | 0.273 | 0.488 |
| A075 | 0.75 | **0.8037** | **0.2222** | 4.750 | **7.531** | 0.585 | 0.278 | 0.453 |

闭路口径（C0-C 协议 12 源，与 C0/C1 同口径）：

| 臂 | local MPJPE | jitter | world MPJPE | root ADE | FS world | 速度方向误差 | moving | strong | 扰动恢复 root(+32) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A025 | 4.202 | 0.427 | 7.02 | 5.05 | 0.696 | 6.49 | 0.510 | 0.092 | 0.260 |
| A050 | 4.042 | 0.459 | 7.52 | 5.75 | 0.697 | 6.51 | 0.528 | 0.101 | 0.409 |
| A075 | 4.175 | 0.453 | 7.97 | 6.07 | 0.794 | 7.11 | 0.504 | 0.100 | **0.196** |

配对 bootstrap（vs A050，seed 42/1000）：

- **A025**：jitter −7.2%（CI[−0.05,−0.01]，确认）；但 strong 帧 −8.3%（>5% 限）、
  **cross style −10.7pp**（远超 2pp 限）→ **否决**。
- **A075**：扰动恢复 root 偏差 −52.0%（CI[−0.44,−0.06]，确认）；matched +3.7pp、
  cross +5.9pp（同集 240f 协议，均为提升）；活动性门通过（moving −4.6%、strong −1.0%、行程 −0.3%）；
  但闭环 local MPJPE +3.3%（点估计超 3% 限，CI[−0.40,+0.85] 未确认）、world MPJPE +5.9%
  （CI 含 0）、FS world +14.1%（CI[−0.02,+0.31] 未确认）；长协议 FS local +1.9%（点估计超 0.5% 限）。
  长协议 ADE/FS 基本中性（matched ADE 持平、cross ADE −3.9%、FS world −2.3%）。

## 2. 判定（按预注册规则，逐臂）

| 臂 | 确认增益 | 质量点估计 | 活动性 | 风格 | 逐臂判定 |
|---|---|---|---|---|---|
| A025 | jitter −7.2%（CI 确认） | closed local MPJPE +4.0%（未确认） | strong −8.3% ✗ | cross −10.7pp ✗ | **FAIL**（否决） |
| A075 | 扰动恢复 root −52%（CI 确认） | closed local +3.3% / world +5.9% / rootADE +5.6%（均未 CI 确认，点估计越 3% 限）；long FS local +1.9% 越 0.5% 限 | 全过 ✓ | matched +3.7pp / cross +5.9pp ✓ | **INCONCLUSIVE**（增益确认，限未全过） |

- 整体 **INCONCLUSIVE**：A025 被否决；A075 是唯一同时保住活动性与风格、
  并在扰动恢复上获得 CI 确认改善的臂，但质量点估计未全过预注册限（也未获反向 CI 确认）。
- 没有出现静止换 Jitter 的作弊路径；未替换默认 controller；未超预算（3 臂 ×10ep = 注册上限）。

## 3. 方向性结论（需要新授权才可继续）

1. **cross 对观测噪声幅度单调敏感**：0.25→0.0556、0.50→0.1630、0.75→0.2222（+16.6pp 于 0.75 vs 0.50），
   且 matched 不降反升。这是从 C0 的"自反馈分布偏移"诊断出发得到的**首个正向闭环信号**，
   与规约禁止的方向（rollout 训练、donor）无关，纯 TF 框架内完成。
2. 但 A075 尚未过"质量不越界"：闭环 local/world MPJPE 与 FS world 的点估计略差（未 CI 确认），
   长协议 FS local +1.9%。12 源闭路样本小、CI 宽，要下结论需要 **批准的新一轮**
   （选项：A) 沿 A=0.75 加长预算（如 20-30ep）；B) C2b 时间相关性 AR(1)；C) 扩大闭路评估集）。
   三者都属新实验，按规约需要单独批准。
3. 防静止检查：三臂 moving/strong 均未崩坏（A025 strong −8.3% 除外，已否决），
   未发现"以冻结换平滑"。

## 4. 失败/异常记录

- `fs_local` 在部分 12 源闭路样本上接触计数为 0（NaN），已在 JSON 保留计数；长协议 90 窗口
  上有完整接触统计（表中 FS 列来自长协议）。
- A025 在 ep5→ep10 出现 cross 从 0.0444 到 0.0556 的低位徘徊（对照 A050 从 0.1222 升到 0.1630），
  与"噪声不足 → 观测分布偏移更严重"一致。
- 训练曲线：三臂 train_flow 均单调下降；A075 的 train_flow 收敛值最高（0.0242 vs A050 0.0150），
  符合"噪声越大任务越难"的预期，非异常。

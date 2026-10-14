# C0 预注册协议：闭环因果分解（只读，不训练）

依据 [docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md) §4 C0。
本文件在 C0 运行前冻结；结果写入 `per_source.json` / `manifest.json` / `REPORT.md`。

## 1. 假设与目标

闭环质量衰退可能来自：(a) 单步 FM/solver 误差；(b) `Q` 的离散映射；(c) 双帧 history/反馈
分布；(d) root/planner 条件。用五组同源对照 + 单次扰动恢复将贡献分离开，且**只诊断、不训练**。

## 2. 固定资产（G0 已核验）

| 项 | 值 |
|---|---|
| VQ | `checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt`（ep90，state sha `918b5e07…`，全冻结） |
| Controller | `checkpoints/flowmatch_label_history_obsnoise_tf100/best.pt`（ep100，objective v12，2L/256dim/heads4，solver 4，obs-noise 0.5，pure TF p=0，`inference_requantize=True`，`inference_latent_clip=True`） |
| norm | VQ run 的 `norm_stats.npz`（sha `bfbe62d5…`），不重算 |
| DB | `database_100sty_375_memmap_nomirror`，只用 val |
| 精度 | FP32 / matmul highest / TF32 off |

## 3. 固定诊断集

- **来源**：`analysis/vq_temporal_source` 的 `fixed_val12`（与 S1/S4 同源，脚本内断言对齐归档 starts）。
- **每 source 帧数**：241（row 0 = GT 种子帧，240 个生成转移），60fps。
- **风格目标**：source 自身 label（matched）。cross 属 C4，本轮不做。
- **噪声计划**：`Utils.reproducibility.case_noise_plan(model, [start], [label], 241, seed=42, noise_std=1.0)`；
  所有组共享同一 z0 序列（同 source 同 step 同噪声），组间差异只能来自状态/条件差异。
- **solver/noise**：官方配置 Euler 4 步、`noise_std=1.0`；`latent_clip=True` 不变。

## 4. 五组对照（同源、同风格、同种子、同噪声）

| ID | 上一帧状态（x_curr.motion / prev_latent / history_latent） | 轨迹条件 | 输出路径 |
|---|---|---|---|
| C0-A | raw GT motion / `Q(E(GT_{t-1}))` / `Q(E(GT_{t-2}))` | GT traj | 官方 clip→Q→D |
| C0-B | `D(Q(E(GT_{t-1})))` / 同上 / 同上 | GT traj | 官方 clip→Q→D |
| C0-C | 自反馈 `D(q_{t-1})` / `q_{t-1}`（生成）/ `q_{t-2}`（生成） | GT traj | 官方 clip→Q→D |
| C0-D | 自反馈同 C | **真实 planner**（脚本化指令） | 官方 clip→Q→D |
| C0-E | 自反馈 `D(clip(z_{t-1}))` / 连续 latent（无码本） | GT traj（与 C 相同） | 连续 clip→D（`feedback_mode=continuous`，仅诊断） |

- A→B 隔离 motion token 的观测流形差；B→C 隔离闭环自反馈；C→E 隔离量化反馈。
- C0-D 的 planner：`Utils.interactive_control.InteractiveTrajectory`（window 60/stride 10，与交互入口一致）；
  指令序列预注册（240 帧，60fps）：
  - 帧 1–60：`forward=1.0, speed=1.2`
  - 帧 61–120：`forward=1.0, turn=+0.3, speed=1.2`
  - 帧 121–180：`forward=0.8, side=0.2, speed=1.0`
  - 帧 181–240：`forward=1.0, turn=-0.2, speed=1.2`
  每步 `planner.advance(上一步生成 motion 的 root 通道)` → `planner.condition(指令)`。
- C0-D 不与 A/B/C/E 混比绝对分数（不同条件语义）；只做行为/质量诊断与指令跟随。

## 5. 单次扰动恢复

- 协议 = C0-C（GT traj、官方量化反馈、同 z0 序列）。
- 注入时刻 `t0=120`：`prev_latent' = prev_latent + 0.5 × transport_std ⊙ u`，
  `u` = 固定单位向量（`torch.Generator(seed=4242)` 采样高斯后归一化，全 source 共用，事前确定），
  即 `‖δ‖₂ = 0.5·‖transport_std‖₂`。随后按状态不变量重建一致观测
  （`q' = Q(clip(z'))`、`m' = D(q')`）→ 正常继续（history 保持真实上一帧）。
- 配对零扰动分支使用相同 z0；比较窗 `[t0+1, t0+8]`、`[t0+1, t0+16]`、`[t0+1, t0+32]`：
  - 与 clean 分支的 root 轨迹偏差（cm）与速度方向误差（deg）；
  - 与 GT 的速度方向误差（deg）；
  - 窗口内 jitter / local FS / 帧均位移（活动性）。

## 6. 时序分桶与指标定义

- 分桶：`[1,16]`、`[17,32]`、`[33,60]`、`[61,120]`、`[121,240]`（帧号，含端点）。
- 指标（全部复用 S1 已验证实现 `vq_temporal_common`）：
  - `local_mpjpe`：24 关节 root-frame 位置 L2（cm），公式 = `Utils.vqvae_eval` 的 local_mpjpe；
  - `world_mpjpe` / `root_ade`：`compute_sequence_metric_sums`（官方口径，>0 均为 cm）；
  - `jitter`：逐 joint-frame 二阶差分 L2（cm），与官方 Jitter sum/count 数值一致；
  - `fs_local` / `fs_world`：事件加权（GT 接触掩码，官方口径）+ contact count；预测接触覆盖另报；
  - `activity`（vt.activity_proxies）：moving/strong frame、pose step、腿行程、趾高、lift、趾速；
  - `vel_dir_err`：生成 root 平面速度 vs GT 的方向误差（deg，|v_gt| ≥ 1 cm/s）；
  - `code_match`：同帧逐槽生成码 == GT 码比例（**仅诊断**，不作成功指标）；
  - 结果按 source 记录 + 全组汇总；后续阶段需要时再上 bootstrap CI（C0 为诊断，不做门槛判定）。
- 状态不变量（每步断言，违反即 `BLOCKED`）：
  1. 传入的 motion token == `decode(携带 latent)`（同一次 decode 输出，逐位相等）；
  2. `history_latent` == 上一步携带的 latent；
  3. 会话结束后模型 state fingerprint 与运行前一致；E/Q/D 无梯度、无 EMA 更新。

## 7. 判定规则

C0 本身不打 PASS/FAIL 门槛，输出诊断结论与后续方向：
- 若 A/B 与 C 的质量差集中体现为闭环反馈退化 → 支持 C2；
- 若 C 与 E（量化开/关）差显著 → 量化反馈是主要来源，指向 V 线；
- 若 C 本身接近 B（闭环稳定）而 D 很差 → 指向 planner/条件端问题；
- 若 A 显著差于 B → GT 观测流形（raw/dcodec）问题。
结论写入 `REPORT.md`，含 `unexpected_behaviors` 与 `next_stage_approval_needed`。

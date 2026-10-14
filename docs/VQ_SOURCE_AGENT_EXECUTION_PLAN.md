# CodeStyle｜VQ 源头时序连续性：Agent 执行规约

> **状态**：提议中的独立实验分支；**不是**已验证修复、不是现役默认配置。  
> **基线仓库**：`Shinn127/CodeStyle`；制定时参考 `main` commit `5cdc6e3450`（2026-10-14）。运行前须记录实际 commit，不可假设分支未变。  
> **拟放置路径**：`docs/VQ_SOURCE_AGENT_EXECUTION_PLAN.md`。  
> **相邻而独立的问题**：Controller 闭环误差见 `docs/CLOSED_LOOP_NO_ROLLOUT_PLAN.md`。**不要合并两个实验分支的变量。**

## 0. 任务声明与不可越界事项

**任务目标**：查明并降低 `Q(E(x)) → D(q)` 相比连续 AE 在物理关节位置上增加的**时间二阶误差 / Jitter / 足滑**，同时保住运动幅度、风格、MPJPE 和 root 质量。允许正常动作产生的码索引切换；不以“尽量不换码”作为优化目标。

**任务不包括**：

1. 不修改主线 Controller/FlowMatch 训练目标、反馈策略或风格体系；**不恢复 rollout / DAgger / scheduled-sampling、多步预测自反馈训练和 donor**。
2. 不在首轮改变 `K=2048`、`slots=32`、`code_dim=4`、网络层数、rotation trick、EMA 配方；**不启用 margin、roundtrip 或 teacher**。
3. 不读 test split 做超参选择；历史 test 探针已用于探索，不能声称其 untouched。使用固定 val 选择，正式 test 只在锁定方案后评估。
4. 不重算归一化统计替代原统计；不混用镜像和非镜像 DB；不重写、不覆盖任何旧 checkpoint、cache、report。
5. 不以对码 ID、per-slot switch rate、raw margin 或混合 291 维 MAE×100 推断视觉质量；不得把 37.8% roundtrip 保持率称为硬上限。
6. **无显式下一阶段批准，不执行**完整 150 epoch VQ 重训、全链缓存刷新、正式 controller 重训、线上默认替换或模型权重删除。
7. 在提供真实数据/权重的环境中，可按本规约完成**只读审计、CPU fixture、干跑、无 optimizer 梯度标定、独立输出目录的至多 10ep 配对 pilot**。任何更大成本或破坏性操作须单独授权。

### 已知证据（不可按不同协议混接）

来源：`analysis/vq_code_stability_2026-10-14/AUDIT.md`、`audit/baseline_val12.json`、`audit/pilot25_val12.json`（同一 val12、FP32、seed42）：

| 指标 | 原 VQ（150ep run，best 实际 ep90） | 旧 margin+RT 30ep pilot，best ep25 |
|---|---:|---:|
| Quantized AE Jitter（cm） | 0.4116 | 0.5065 |
| Continuous AE Jitter（cm） | 0.1205 | 0.1136 |
| World 足滑（cm） | 0.4882 | 0.5216 |
| World MPJPE（cm） | 1.5794 | 1.2060 |
| 同帧 codec 往返翻码率 | 62.60% | 58.64% |
| 相对最近边界余量 median | 0.2989 | 0.2880 |

**解读**：往返换码减少、MPJPE 改善，不等于 Jitter/足滑改善。旧 pilot 同时动了两个辅助项且训练预算不同，不能证明单个辅助项的因果收益。`docs/NEXT_TASKS.md` 原有“冻结 E/Q + decoder-only acc”是另一条已规划的低成本验证路线；本文新增**全 VQ 联合训练实验支线**，不得把两条路线的数据混作严格 A/B 或覆盖原计划。

## 1. 数据、版本和实验契约（S0；未满足则 STOP）

- 模型初始化：`checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt`；**核验 checkpoint 内 epoch 是 90**；原 `norm_stats.npz` 路径必须存在且内容指纹记录。
- DB：`database_100sty_375_memmap_nomirror`，只用其 **train** 训练、**val** 调参；用 `source_entry_id` 配对统计；不用 OOD generator DB 训练。窗口不得越过 segment/source clip 边界。
- 精度：PyTorch FP32、highest、TF32 off，明确记录 torch/CUDA/GPU/第三方 `vector_quantize_pytorch` 版本；EMA 只在训练 forward 更新，评估严格冻结。
- 依赖：原 32×4、K2048 共享 EMA codebook；同 K `--init-ckpt` 应恢复完整 encoder/decoder/quantizer 状态和 EMA buffers；通过张量指纹验证，**不能仅看终端回显**。
- 固定 paired seed=42、DataLoader 独立 `LOADER_SEED=4242`、与 B0/B1 同一实际 batch/source manifest、学习率/优化器/更新次数；原 norm 显式传入。由于 VQ 通用 resume 未完整保存全部 RNG，不宣称跨 resume 位级复现。
- 保留 `seq_len=8`（**新实验起点，不是最优参数**）、`stride=1`；B0/B1 一致。若资源不够，降低两组相同的 batch 或使用等效梯度累积，记录真实有效 batch 与总更新数；不得单独改一组。
- 固定 validation：①审计 val12 的相同 source/start/frames 用于分析 `r, Δr, Δ²r`；②训练器的固定 90-window（60f）表示层评估用于 checkpoint 选择；③按 style/activity 分层记录。不同窗口的指标不互相比绝对阈值。
- 训练实验默认 10epoch、`EVAL_EVERY=1`、`SAVE_EVERY=5`，每个 run 独立新 `OUT`。**避免 `EVAL_EVERY=10` 下 5epoch 不生成 metric-gated best 的陷阱**；无 best 时使用事先约定的最新/periodic checkpoint 方案，不事后挑有利结果。
- 绝不能自动 overwrite 源 run、cache 或原审计报告。每个 run 写 `config.json`、`manifest.json`、`history.jsonl`、测试日志、输出 checkpoint 指纹、代码 commit。

**S0 交付**：`analysis/vq_temporal_source/<run_id>/contract.json` + `S0_AUDIT.md`，写明是否满足全部条件。任一条件不满足：报告 blocker，**不启动训练**。

## 2. S1：只读基线 / 源头诊断

目的：先区分码身份翻转、量化附加时间残差及物理运动质量，避免误优化指标。

定义：`z_e=E(x)`，`q=Q(z_e)`，`p_q=P(D(q))`，`p_c=P(D(z_e))`，`p_gt=P(x)`，其中 `P` 是**反归一化后 24 关节 local position**；world 指标按 `Utils.motion_eval` 的独立管线计算。`r=p_q-p_c`，`e_q=p_q-p_gt`；`Δ²a_t=a_{t+1}-2a_t+a_{t-1}`。

使用/扩展：
- `analysis/vq_code_stability_2026-10-14/residual_probe.py`（残差 r、Δr、Δ²r）和 `audit/*`；
- `analysis/codebook_divergence_2026-10-14/vq_anatomy.py`（真实 Voronoi 边界及相对余量）；
- `Utils/motion_eval.py`（官方 Jitter、local/world MPJPE、root ADE、world 足滑），不能把 `step1_*` 误称 Jitter。

必记：val12 的 per-source mean/p50/p95，`||Δ²r||`、`||Δ²e_q||`、quantized/continuous/GT Jitter、world 足滑及真实/预测 contact、root ADE、local/world MPJPE、slot flip rate、EMA 死码及用量；按 idle/walk/run/turn（实际数据标签和运动阈值）分层。**World 足滑原指标使用 GT 接触掩码：其接触计数不变不代表预测运动正常。必须独立增加预测动作幅度、足部抬起/移动时间与预测接触覆盖诊断。**

S1 DoD：冻结 VQ state 前后 hash 一致；复算审计 val12 与存档数字可比（记录容差及原因）；无“raw291×100=cm”；原始 per-source JSON + 方法文档可追溯。若不一致，先修诊断，不训练。

## 3. S2：验证已有 `acc_loss` 实际梯度（**不**更新优化器）

当前 `arch/basic_vqvae_375to291.py::compute_loss` 已支持：

```text
L_acc = L_type((m_hat[t+1]-2m_hat[t]+m_hat[t-1]), (m_gt[t+1]-2m_gt[t]+m_gt[t-1]))
```

其作用于**归一化 motion291**，并不是物理 position-only 或 world-contact loss。对 4 个独立 train 小批（建议 B16×T8）测 `g_base=∇_{E,D}(L_base)` 与 `g_acc=∇_{E,D}(L_acc)` 的范数、余弦，按 encoder/decoder 分组，记录每组统计；训练模式下量化 EMA 可能发生变化，梯度标定应在隔离模型副本上执行且不污染源 checkpoint。

选单一 `ACC_WEIGHT`，使加权梯度量级相对主监督约 10%–30%（仅**候选量级**，不按 loss 数值或硬编码 3/5 直接设置），若 encoder/decoder 梯度极不均衡、余弦强冲突或统计极不稳，标记风险并停止 pilot 进行分析。

S2 DoD：`gradient_calibration.json` 记录 batch 来源、每组梯度比、余弦、标定理由、hash 不变；未调用 optimizer.step。

## 4. S3：首轮必须是成对、单变量的**全 VQ**试验

**两个 run 都从 ep90 完整 VQ 状态冷启一个新实验**（是 `init`, 不是 `resume` 原优化器），都更新 E+Q(EMA)+D；两者**除 `acc_weight` 外所有设置相同**：

| 字段 | B0 控制 | B1 候选 |
|---|---|---|
| 初始化 | 同一 ep90，含全部 EMA buffers | 相同 |
| `seq_len`,`stride` | 8,1 | 8,1 |
| `K/slots/dim` | 2048/32/4 | 相同 |
| `recon/vq/vel/root/code_align/latent_l1/latent_vel` | 3 / 1 / 15 / 3 / 2 / .01 / .1 | 相同 |
| `margin/roundtrip/teacher` | 0 / 0 / 0 | 相同 |
| `acc_weight` | **0** | **S2 标定值** |
| `epochs/eval_every/loader_seed/seed` | 10 / 1 / 4242 / 42 | 相同 |
| `OUT` | `checkpoints/vq_temporal_B0_<id>` | `checkpoints/vq_temporal_B1_<id>` |

**示意命令（环境中的真实路径由 Agent 先核验，禁止静默补造）**：

```bash
BASE="$PWD/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt"
NORM="$PWD/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz"
# 分别设置 ACC_WEIGHT=0 / <S2测得值>，OUT 互不相同；先 DRY_RUN=1 核实
INIT_CKPT="$BASE" NORM_STATS="$NORM" DECODER_ONLY=0 \
  SEQ_LEN=8 STRIDE=1 BATCH_SIZE=512 EPOCHS=10 EVAL_EVERY=1 \
  LOADER_SEED=4242 ACC_WEIGHT="${ACC_WEIGHT}" \
  RECON_WEIGHT=3 VQ_WEIGHT=1 VEL_WEIGHT=15 ROOT_WEIGHT=3 \
  CODE_ALIGN_WEIGHT=2 LATENT_L1_WEIGHT=0.01 LATENT_VEL_WEIGHT=0.1 \
  LATENT_MARGIN_WEIGHT=0 ROUNDTRIP_WEIGHT=0 \
  OUT="$OUT" bash scripts/run_train_basic_vqvae_375to291.sh \
  --lr 1e-4 --min-lr 1e-5 --seed 42 --save-every 5
```

**训练前**：验证 DRY_RUN 展开确有 `--init-ckpt`、`--norm-stats`、`--acc-weight`；源 `state_dict` 每张量与加载后的初始 hash 一致（训练前，勿触发训练式 EMA）；记录真实每个 batch 的更新数。**训练后**：量化向量发生变化属预期，必须命名为新 VQ 身份，不可借用原 VQ 缓存/旧 controller。

S3 DoD：B0/B1 实际 updates 相同，训练轨迹与 checkpoint 存在，至少初末两个 eval；差异只在 acc。失败/显存/NaN/码本大规模重置立即停止，保存异常日志，不自动增大重试预算。

## 5. S4：联合验收与止损（先 val，后 test）

**主验收为固定 val 上的配对比较 B1-B0**，禁止跨协议拼装分数：

1. `||Δ²r||` mean **及** p95 必须下降；`||Δ²(p_q-p_gt)||` 不增加；使用 source_entry_id 配对 bootstrap 给 95% 区间（在小样本不确定时如实说不确定）。
2. 官方 quantized Jitter 改善；world 足滑不得恶化；MPJPE/root ADE 不得实质恶化；同时提供连续 AE 对照，避免把两个 Decoder 共同变平滑误记为量化残差收益。
3. 活动性防退化：移动帧比例、腿部关节行程、摆腿/接触覆盖和高动态 style 行为不低于对照；GT-contact-count 不可当作此项证明。
4. IND style fidelity 不下降，分活动/风格列出最差退化；数值改善必须不依赖显示 smooth/IK（正式模型指标关闭后处理）。
5. 选型只在 val 进行。可把 Jitter 相对下降≥15%、world 足滑不升、MPJPE/root ADE 相对退化≤3%、活动代理相对下降≤5%设为**预注册工程筛选阈值**；不是已证物理规律，也不是跨不同采样协议的绝对门槛。小样本 CI 未稳定时只做下一批 val 复核，不宣布成功。

**否决规则**：只降低 switch rate、只提高 roundtrip、只降低 MPJPE 或以冻结运动换低 Jitter，均 FAIL；明显退化则 STOP，不自动续到 150ep。`best.pt` 的多指标严格 gate 可能拒绝一切 epoch；不得只挑有利的最新权重比较，严格按实验前声明的 checkpoint 选型规则操作。负结果完整归档。

## 6. S5：条件触发的后续源头实验（每轮先重新取得执行授权）

仅在 S4 有清晰结论后，由人选择一个分支，禁止堆叠：

- **S5a：位置空间二阶损失（优先备选）**：反归一化 `motion[...,3:75]` 到 24×3 local positions；用 `SmoothL1(Δ²p_q, Δ²p_GT)` 对比同预算控制组。注意张量单位、二阶采样帧率、真实 GT 与其指标 P95。新 loss 实现于 `arch/basic_vqvae_375to291.py`，用独立 flag、默认为0；接入训练器/wrapper、resume 目标校验、单测。不是直接沿用 mixed motion291 的 acc 结果。
- **S5b：物理接触约束**：只能在 S5a 的诊断明确足滑未改善后单独追加；以 GT contact supervision 和**可微 Torch world/root 积分**构造损失（`Utils.motion_eval` 的 NumPy 指标不可直接反传）；保证非接触运动活动。不能用静止脚/改 GT 接触阈值刷指标。
- **S5c：几何稳定性**：若物理时间误差已改善但仍发现扰动引发异常跳码，在已有真实 Voronoi 距离上引入 `relative_margin = distance_to_nearest_plane/(0.5*nearest_distinct_code_spacing+eps)`；先 no-optimizer oracle/梯度单测，再做**margin-only / RT-only / combo 的同预算对照**。raw 距离增加不代表鲁棒性；`K=1024`、`code_dim` 变更仍属更晚的独立架构实验。
- **S5d：因果短历史 Decoder**：只有证据说明单帧 `q_t` 表示不足才做，严格比较同参数预算下 `D(q_t)` vs `D(q_{t-1},q_t)`，训练/推理历史口径和模型生成历史分布差必须纳入验证；严禁偷用未来帧。

## 7. 下游推广是另一个明确授权边界

任何 E/Q/D 更新都会更改 VQ 版本或 `D(q)` 观测，因此**不能把新 VQ 权重覆盖至现役 controller**。S4 通过后仍只称“representation-val 候选”。进入推广须用户另行批准：

1. 新 VQ 独立版本 + 全状态 SHA、norm 来源、训练 manifest；保持旧所有 checkpoint 和 cache 可回滚。
2. 按 `docs/NOMIRROR_CHAIN.md` 重建 VQ targets、`D(q)` observations、train-only latent stats、style prior，逐项验证 provenance；涉及 E/Q 更换不能复用旧 indices/latent 统计。
3. 使用与新 VQ 配套的 Controller **纯 TF**（可含既有观测噪声）训练/适配；**不开展 rollout 训练**；保持明确的控制器模型/配置版本。
4. 封存协议 matched/cross 240f、Root ADE、MPJPE、Jitter、local/world 足滑、活动性、风格；旧配套与新配套在同一协议比较。新的 test 仅在 val 选型锁定后一次性评估。
5. 未同时过关前，实时/交互保持旧版本默认；若性能或活动变差即回滚，不绕开指纹校验。

## 8. Agent 文件改动白名单与阶段验收

| 阶段 | 允许新增/修改 | 明确禁止 |
|---|---|---|
| S0/S1 | `analysis/vq_temporal_source/**` 新文件，必要时新增 `tests/test_vq_temporal_*.py` | 改 Controller、数据库、旧报告原始 JSON、生产默认值 |
| S2/S3 | **优先只改新实验配置/脚本**；必要时增独立 `scripts/run_train_vq_temporal_ab.sh` | 修改现有 baseline 配方默认值、清空 checkpoint 目录 |
| S5 | 经授权再改 `arch/basic_vqvae_375to291.py`、`training/train_basic_vqvae_375to291.py`、`scripts/run_train_basic_vqvae_375to291.sh` 和对应 tests | Controller、风格分类器、数据库生成规则；静默改 resume 语义 |
| 下游 | 经单独授权后对 `tools/precompute_*`、控制器适配入口作版本化更新 | 静默覆盖现役资产、关掉 fingerprint 检查 |

阶段报告均须具备：`hypothesis`、`single_variable`、`source_commit`、`dataset/ckpt/norm_hashes`、`config`、`seed/manifest`、`tests`、`per-source_metrics`、`CI`、`unexpected_behaviors`、`decision(PASS/FAIL/INCONCLUSIVE/BLOCKED)`、`next_authorization_needed`。

**测试要求**：每次代码改动必须先运行相关 fixture，再执行 `bash scripts/run_tests.sh cpu`；CUDA 可用时 `bash scripts/run_tests.sh gpu`。没有真实数据或 GPU 时只声明 fixtures/静态审核通过，**不得伪称实际训练/闭环指标已改善**。

## 9. Agent 开始执行时的固定顺序

```text
读取 README / docs/NEXT_TASKS / VQ audit / 相关源码
 → 核验 commit、数据、checkpoint、norm
 → S0 契约与 manifest（不更新模型）
 → S1 固定 val 诊断（不更新模型）
 → S2 四批梯度标定（不更新模型）
 → S3 B0/B1 严格配对、独立 OUT、≤10ep
 → S4 验收/CI/异常与负结果报告
 → 在此停下；等待具体下一阶段授权
```

**终止条件**：数据/权重缺失，单位/指标不一致，初始化/EMA 指纹异常，val 被用作 train，明显质量退化、数值异常或预算到达，均立即 STOP；报告未完成项，不用自动重训替代证据。

## 10. 相关源码与证据

- `arch/basic_vqvae_375to291.py`：E/Q/D、`compute_loss`、acc、EMA、margin、RT。
- `training/train_basic_vqvae_375to291.py`、`scripts/run_train_basic_vqvae_375to291.sh`：训练和包装入口。
- `Utils/vqvae_eval.py`、`Utils/motion_eval.py`、`Utils/motion375.py`：评价口径。
- `tests/test_acc_loss.py`、`tests/test_vq_stability_audit.py`、`tests/test_decoder_only_calibration.py`：回归。
- `analysis/vq_code_stability_2026-10-14/{AUDIT.md,REVIEW.md,CALIB_AB_AUDIT.md}`：审计与撤回结论。
- `docs/{NEXT_TASKS.md,VQ_DECODE_CONTINUITY_PLAN.md,NOMIRROR_CHAIN.md,DATA_PIPELINE.md}`：既有执行计划、版本与数据契约。

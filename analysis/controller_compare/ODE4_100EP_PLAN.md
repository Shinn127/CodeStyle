# ODE4 history 100-epoch 训练配置（2026-10-12 规划）

新训练：donor-free、双帧 latent 历史条件、全链路 Euler4；课程 **50 TF + 20 ramp + 30 rollout**，
共 100 epoch，从零训练。入口脚本 `run_train_flowmatch_history_ode4_100ep.sh`（2026-10-13 清理中删除；
同配置可用 `scripts/run_train_flowmatch_6tok_curriculum100.sh` 加 `FEEDBACK_START_EPOCH=51`、
`FEEDBACK_RAMP_EPOCHS=20`、`FEEDBACK_PROBABILITY=1`、`FEEDBACK_SOLVER_STEPS=4` 复现）。

## 一、命令

```bash
# 历史命令（入口已删除；见上注记的复现方式）
cd /home/shinn/桌面/multimedia/revision/NewCode
bash scripts/run_train_flowmatch_history_ode4_100ep.sh
```

默认参数已是最优配置（见下表），无需额外 flag。预算或显存不足时：

```bash
GPU_DATA_CACHE_GB=6 bash scripts/run_train_flowmatch_history_ode4_100ep.sh     # 收紧 GPU 缓存
OUT=checkpoints/my_ode4 bash scripts/run_train_flowmatch_history_ode4_100ep.sh # 换输出目录
```

若以后拿到兼容的 6-token transformer 权重，可加 `INIT_CKPT=<path>` 做微调初始化（默认从零）。

## 二、训练配置

| 项目 | 值 | 说明 |
|---|---|---|
| 输出目录 | `checkpoints/flowmatch_label_history_ode4_100ep` | `best.pt`（质量选型）、`latest.pt`、每 5 epoch 快照 |
| 初始化 | 从零 | 兼容权重已在 2026-10-12 checkpoint 清理中删除；`INIT_CKPT` 可显式指定 |
| epoch | 100，cosine 100 | `lr 7e-4 → 1e-5`（bs1024 sqrt 缩放），min-lr 1e-5，wd 1e-4，grad-clip 1 |
| 课程 | TF 1–50（p=0）→ ramp 51–70（p 0.05→1.00）→ rollout 71–100（p=1.0） | 按批确定性门控（`--feedback-batch-schedule batch`） |
| 求解器 | `--solver-steps 4` **且** `--feedback-solver-steps 4` | 训练反馈、验证、交互、封存评估全部走 Euler4，无训练/推理步数错配 |
| 模型 | 6-token transformer（`--use-motion-token`）、hidden 256、2 层、FFN1024、dropout 0 | `seq-len 6`（unroll 4 + 2 帧历史）、`unroll-steps 4` |
| 历史条件 | `--history-conditioning` | previous-latent token 加两帧 latent 差分投影（零初始化）；reset 零差分、切换 style 保留历史 |
| 反馈观测 | `--train-feedback-context` | TF 观测 = D(source q)（冻结 VQ 解码的 GT），监督保持 GT |
| 数据 | bs 1024、stride 1、train split；VQ 冻结 nomirror 150ep | GPU 只读缓存 9 GiB（features+indices+teacher motion）、expandable segments |
| 目标/损失 | target 预测、normalized FM；prior CE .02、entropy .003；recon/root/style-consistency = 0 | 无 donor、无跨风格合成监督（objective version 11） |
| 采样 | noise std 1、Euler4、clip→quantize→decode、requantize on | 正式协议口径；显示可加 `NOISE_STD=0.5` |
| 验证 | 每 5 epoch，固定 90 clip × 3 window = 270 matched + 540 cross，240f/60fps rollout | 选型门限：matched≥.85、cross≥.80、ADE≤5/10cm、jitter≤.5、local FS≤.8、world FS≤1 |
| 质量选型 | `--motion-quality-selection` | 未过门限时保存诊断 best（`constraints_passed=false`），按联合分数排序 |

### 调度明细（脚本已核对实际执行值）

| epoch | 反馈比例 p | 阶段 |
|---:|---:|---|
| 1–50 | 0 | 纯 TF（不调用 Euler/decode） |
| 51 | 0.05 | ramp |
| 55 | 0.25 | ramp |
| 60 | 0.50 | ramp |
| 65 | 0.75 | ramp |
| 69 | 0.95 | ramp |
| 70–100 | 1.00 | 全 rollout |

## 三、耗时预估

按本机实测分解（2584 batch/epoch）：纯 TF ≈ 73 s/epoch；rollout 反馈成本 ≈ 27 ms/update 基座
+ 每 4 步 Euler 一次 field 评估；p=1/Euler4 ≈ 180 s/epoch，ramp 平均 ≈ 130 s/epoch。

- 50 TF ≈ 1.0 h + 20 ramp ≈ 0.7 h + 30 rollout ≈ 1.5 h + 20 次验证（Euler4，约 20–30 s 每次）
- **合计约 3.5 GPU 小时**（估算，±30%；数据访问与系统负载会改变实际时间）

## 四、ODE 步数一致性（本次的关键改动）

此前发现"训练反馈 Euler2 vs 推理 Euler8"不一致会恶化闭环保真度。本训练统一 4 步，并且
所有下游入口都跟随 checkpoint 的 `solver_steps`，不再硬编码 8：

| 入口 | 之前的步数 | 现在的行为 |
|---|---|---|
| 训练反馈积分 | 2（wrapper 指定） | 4（`--feedback-solver-steps 4`） |
| 固定 rollout 验证（`controller_validation`） | 硬编码 8 | 跟随 `model.eval_solver_steps` → 4 |
| 交互窗口（`interactive_label_control`） | 默认 8 | `--ode-steps` 缺省继承 checkpoint → 4 |
| 封存评估（`run_evaluate_style_rollout.sh`） | 硬编码 8 | ckpt 默认（`--flow-ode-steps 0`）→ 4 |
| MPJPE 评估（`eval_rollout_mpjpe.py`） | 硬编码 8 | `--ode-steps 0` 跟随 checkpoint → 4 |
| 控制视频录制（`record_control_video.py`） | 硬编码 8 | 跟随 checkpoint → 4 |
| 旧仿真入口（`evaluation/simulate_realtime_flowmatch_control.py`） | 已有 ckpt 跟随逻辑 | 不变（自动 → 4） |

验证记录：冒烟产物的 `rollout_style_validation_cases.json` 中 `ode_steps=4`，即验证清单已正确
记录新步数，历史对照不会混淆。

## 五、训练后评估流程

```bash
# 1) 封存 test（1000 sources × 5 targets，240f，Euler4 + std1）
FLOW_CKPT=checkpoints/flowmatch_label_history_ode4_100ep/best.pt \
  bash scripts/run_evaluate_style_rollout.sh

# 2) 封存 MPJPE（同 manifest 口径，跟随 Euler4）
python analysis/controller_compare/eval_rollout_mpjpe.py \
  --flow-ckpt checkpoints/flowmatch_label_history_ode4_100ep/best.pt --tag ode4 --noise-std 1.0 \
  --out-json analysis/motion_quality_diagnosis/probes/mpjpe_ode4_ns1.0.json

# 3) 交互控制
FLOW_CKPT=checkpoints/flowmatch_label_history_ode4_100ep/best.pt \
  bash scripts/run_interactive_label_control.sh --smooth-plan          # 默认 Euler4
NOISE_STD=0.5 FLOW_CKPT=... bash scripts/run_interactive_label_control.sh --smooth-plan   # 显示降噪档
```

## 六、风险与验收要点

1. **cross 泛化无监督来源**：donor 移除后，训练样本里没有"在别人内容上执行目标风格"的配对，
   cross 准确率预计显著低于 donor 时代的 91.4%（历史参照：donor-free 纯 TF 的 tf10 cross≈65.7%）。
   本训练要回答的问题之一是：**rollout 阶段的闭环训练能否部分补上 cross**。若最终 cross 不达标，
   需要另起 donor-free 的跨风格方案（例如用真实 cross 剪辑配对），而不是恢复 donor。
2. **从零训练**：前 10–15 epoch 指标会很难看（loss 高、验证差），属正常；质量选型在未过门限时
   保存诊断 best，不要只看 `constraints_passed`。
3. **Euler4 是本次实验变量**：步数减半使积分更粗、生成成本减半（训练与评估都更快），但可能改变
   落地质量；若 rollout 阶段发现明显劣化，下一步对照 Euler8 同课程即可归因。
4. 验收顺序建议：先看每 5 epoch 的 rollout 验证曲线（accuracy 与 ADE），再看封存 matched/cross、
   MPJPE/jitter/F.S.，最后用交互窗口做主观检查（滑行、脚步、急转跳变）。

## 七、验证状态（代码侧已就绪）

- wrapper dry-run 用真实 argparse 解析核对：seq-len 6、solver 4、feedback-solver 4、
  start 51 / ramp 20 / p 1.0、history on、无任何 donor/decouple 参数
- 从零 + p=1 + Euler4 冒烟（1 epoch）通过：objective 11、manifest ode_steps=4、验证/选型/保存全链路正常
- CPU 305 passed / GPU 20 passed（含步数跟随改动）

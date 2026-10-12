# 重新训练配置（6-token + TF→rollout 课程，100 epoch）

2026-10-11。汇总本会话全部实验证据，给出下一轮正式训练的唯一推荐配置。
脚本已就绪：`run_train_flowmatch_6tok_curriculum100.sh`（已 DRY_RUN + 2-update GPU 冒烟通过）。

## 证据回顾（每条配置决定的依据）

| 结论 | 证据 | 出处 |
|---|---|---|
| 架构选 transformer | 50ep 归因：skipcat 自 ep25 停滞后 66.7% vs transformer 78.5%（matched），ADE 差 18cm | `ATTRIBUTION_50EP.md` |
| **恢复 motion token（6-token）** | 控制录制：低噪声下无-token 模型坍缩为"冻结姿态平移"（tr5 61.6%、ep86 32.9% 行走帧），有-token 仅 1.7%；腿相位不可从 128 维量化 latent 长期推断 | `CONTROL_VIDEO_ANALYSIS.md` |
| TF→rollout 课程（30+10+60） | curriculum100 平滑过渡（ramp 期 val_flow 稳定 0.030），validation 88.5/85.6 @ep85，滑行随训练从 49%→13% 下降 | `curriculum100/history.jsonl` |
| unroll=4 | 上轮实测收敛好、速度可控（u4+fb2=25.3s/1000up） | `ROLLOUT_SPEED_2026-10-11.md` |
| feedback-solver-steps=2 | 1.31× 提速；FM loss 不受影响 | 同上 |
| batch 1024 + lr 7e-4 | 同样本量 1.45× 提速、GPU 79%→95%；sqrt 缩放 | 同上 |
| 不加 motion-recon | 抖动主因是采样噪声放大 + 无-token 滑行，不是量化台阶（残差/跳码全程平稳）；首版 recon 发散，修正版未验证 | `REPORT.md` |
| 其余保持 label 配方 | prior init、dense prior、.02/.003、decouple .3 matched_root、noise-std 1 | 历次达标 run |

## 配置表

| 项目 | 值 |
|---|---|
| 输出 | `checkpoints/flowmatch_label_6tok_curriculum100` |
| 字段 / 输入 | `--field-arch transformer --use-motion-token`（6 tokens） |
| VQ / cache / prior / latent stats | 无镜像 VQ150ep + `cache/vq_targets_vq150ep_nomirror_repair_v2` + prior + label150 latent_stats（全部冻结，指纹校验） |
| 课程 | ep1–30 纯 TF（p=0，无 Euler/decode）→ ep31–40 反馈比例 10%→100%（逐窗口门控）→ ep41–100 全 rollout |
| rollout | `--seq-len 5 --unroll-steps 4 --feedback-solver-steps 2`（FM loss 单次评估不受影响） |
| batch / lr | `--batch-size 1024 --microbatch-size 0`，lr 7e-4→1e-5 cosine 100 |
| loss | flow=1（normalized）、recon/root/consistency=0；decouple 0.3 matched_root；style prior .02 / entropy .003 |
| 推理配置（训练内） | solver-steps 8、noise-std 1、requantize+clip on |
| 验证 / 选型 | 每 5 ep：90 val clips ×3 windows、240 帧 rollout、绝对门限 85%/5cm/10cm；每 5 ep 存档 |
| 加速（默认开） | shared style table、fast quantize（5×，逐位一致）、反馈 no-grad、matched_root 零同步 |
| seed | 42 |
| 预计耗时 | ≈2.5–3 小时（bs1024：TF ~81s/ep、rollout ~109s/ep + 验证） |

## 启动

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
DEVICE=cuda bash scripts/run_train_flowmatch_6tok_curriculum100.sh
```

监控（p 必须按 0.1 步进爬升、val_flow 平稳）：

```bash
/home/shinn/miniforge3/envs/mcc/bin/python - <<'EOF'
import json
rows=[json.loads(l) for l in open('checkpoints/flowmatch_label_6tok_curriculum100/history.jsonl')]
for r in rows:
    if 'train' in r: print(f"ep{r['epoch']:>3} flow={r['train']['flow_loss']:.4f} p={r['feedback_curriculum']['actual_fraction']:.2f} {r['time_sec']:.0f}s")
    elif 'rollout_style_val' in r:
        m=r['rollout_style_val']['metrics']; print(f"  ^ {m['matched_accuracy']:.3f}/{m['cross_accuracy']:.3f} ADE {m['matched_root_ade_cm']:.1f}/{m['cross_root_ade_cm']:.1f}")
EOF
```

## 验收标准（训练结束后）

1. **指标**（240 帧 rollout validation）：matched ≥85、cross ≥80、matched/cross ADE ≤5/10cm → `constraints_passed=true`；
   然后锁定 checkpoint 跑封存 test（`run_evaluate_style_rollout.sh`，目标未变 matched 85+/cross 80+）。
2. **闭环步态**（关键新增验收，脚本已备）：

```bash
PY=/home/shinn/miniforge3/envs/mcc/bin/python
$PY analysis/motion_quality_diagnosis/record_control_video.py \
  --flow-ckpt checkpoints/flowmatch_label_6tok_curriculum100/best.pt --tag 6tok_std025 \
  --style Neutral --frames 300 --noise-std 0.25 --render dual
$PY analysis/motion_quality_diagnosis/record_control_video.py \
  --flow-ckpt checkpoints/flowmatch_label_6tok_curriculum100/best.pt --tag 6tok_std005 \
  --style Neutral --frames 300 --noise-std 0.05 --render none
```

目标：std=0.25 滑行 ≤2%（tr6 水平）、jitter ≤0.35；std=0.05 滑行 ≤3%（不允许 tr5/ep86 式坍缩）；
90-clip 探针 ADE ≤ ep86 水平（~8cm @std=1）且不劣化 15%。

3. **交互体验**：`NOISE_STD=0.25` 或 `0.5` 作为显示档位（6-token 下两者应同时低抖动+无滑行），
   与 std=1 的正式口径分开记录。

## 明确的排除项（及原因）

- `--no-use-motion-token`：闭环步态不稳的根因（本轮核心修正）。
- motion-recon loss：非主因；修正版剂量（weight 0.25 + anchor 0.1 + 末步监督）已实现但未验证，
  留作基线之后的可选实验。
- 采样噪声随机化训练（每 batch 从 U[0.1,1] 采样）：备选保险，若 6-token 在 std≤0.1 仍有滑行再实现；
  目前 tr6 证据显示不需要。
- batch 2048：1.52× 但每 epoch 优化步数再减半，收益边际，1024 为平衡点。
- 从 ep86 (5-token) 续训：结构无法回填 motion token（仅支持 6→5 重映射），必须从零。
- skip_cat_mlp：已证停滞。

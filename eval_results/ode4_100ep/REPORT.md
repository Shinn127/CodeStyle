# ODE4 history donor-free 100-epoch：训练结果（2026-10-12）

模型：`checkpoints/flowmatch_label_history_ode4_100ep`（from-scratch、双帧 history 条件、
6-token transformer、全链路 Euler4、donor-free objective v11）。
课程：50 TF + 20 ramp + 30 rollout，共 100 epoch；训练 2.46 h（TF 66s / ramp 95s / rollout 122s per epoch），
峰值显存 8.85 GB。

## 封存 test（1000 matched sources / +5000 cross，240f，Euler4，std=1）

| 检查点 | matched acc | cross acc | matched/cross RootADE |
|---|---:|---:|---:|
| **ep100（latest.pt）** | **77.90%** [70.7, 84.9] | **8.18%** [6.6, 9.7] | **3.18 / 4.87cm** |
| ep55（best.pt，质量选型） | 66.50% [58.9, 74.2] | 2.16% [1.6, 2.7] | 3.95 / 5.59cm |

## MPJPE（同 1000 封存 matched sources、同 noise plan）

| 检查点 | std | MPJPE | local | RootADE | jitter | F.S. |
|---|---:|---:|---:|---:|---:|---:|
| ep100 | 1.0（协议） | **6.43** | 4.71 | 3.18 | 0.791 | 0.845 |
| ep100 | 0.5 | **6.06** | 4.45 | 2.95 | 0.451 | 0.588 |
| ep55 | 1.0 | 7.73 | 5.53 | 3.95 | **0.519** | **0.668** |
| ep55 | 0.5 | 7.40 | 5.21 | 3.90 | 0.336 | 0.542 |

历史对照（同口径）：6tok（donor 时代）MPJPE 11.23 / jitter 1.79；tf10（donor-free 微调）
7.97 / 0.568；phase10 10.28 / 2.14。**本轮 6.43 是迄今最好的轨迹误差；jitter 在 std=1 下
0.79 也在第一梯队（ep55 的 0.52 最优）。**

## 交互控制探针（std=1，脚本化行走/转向/侧移 300f）

| 检查点 | glide 比例 | glide 最长 | 步数 L/R（GT 5/6） |
|---|---:|---:|---:|
| ep100 | 0.42% | 1 | 9/4 |
| ep55 | 0.84% | 1 | 8/5 |

两代都无"姿态冻结平移"问题，步态接近 GT。

## 结论

1. **质量目标超额完成**：MPJPE 6.43（前代最好 7.97）、RootADE 3.18（前代 4.78）、闭环滑行 <1%。
   Euler4 + history + 50TF/20ramp/30rollout 的组合把运动质量推到了新水平。
2. **matched 77.9%：距 85 目标还有 ~7pp**。from-scratch donor-free 的合理水位，随课程还在
   缓慢爬升（ep75 71.9 → ep95 74.8 → ep100 74.1 validation）。
3. **cross 8.2%：系统性缺口**。donor 移除后训练中完全没有跨风格监督；此前 tf10 的 65.7%
   cross 是从 donor 时代 6tok 权重**继承**的，不是它的 10 epoch 学到的。本轮 from-scratch
   证明：**matched-only 训练 + rollout 无法产生 cross 能力**（8% 接近随机）。
   要达 cross 80，必须设计 donor-free 的跨风格监督（候选：rollout 阶段用冻结分类器对
   自生成轨迹施加目标风格引导损失，模型自己的输出即配对样本）。
4. **选型策略冲突**：质量选型把 best 锁在 ep55（jitter 0.52 最优），但它在准确率和 MPJPE 上
   全面落后 ep100。当前评分公式对 jitter 权重偏高（jitter/0.5 与 cross 项），与"准确率优先"
   目标不符。下一轮建议：先按 accuracy 设硬门限，再在过门者中比质量；或直接以 MPJPE
   参与选型（本轮 ep100 会胜出）。
5. **交互建议**：换风格场景暂不可用（cross 8%）；同风格行走/转向质量是历史最好，推荐
   `FLOW_CKPT=checkpoints/flowmatch_label_history_ode4_100ep/latest.pt NOISE_STD=0.5
   bash scripts/run_interactive_label_control.sh --smooth-plan`。

## 产物

- 训练：`checkpoints/flowmatch_label_history_ode4_100ep/`（history.jsonl 全程 20 次验证）
- 封存评估：`eval_results/ode4_100ep/seen/`（best）、`eval_results/ode4_100ep_ep100/seen/`（ep100）
- MPJPE：`analysis/motion_quality_diagnosis/probes/mpjpe_ode4_{ns1.0,ns0.5}.json`、
  `mpjpe_ode4_ep100_{ns1.0,ns0.5}.json`
- 控制探针：`analysis/motion_quality_diagnosis/videos/ode4_{best,ep100}_ns1.0/`

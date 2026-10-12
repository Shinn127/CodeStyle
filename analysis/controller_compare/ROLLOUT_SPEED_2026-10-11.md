# Rollout 训练提速优化（2026-10-11）

目标：在保持数学等价/协议不变的前提下压缩 rollout（unroll）训练耗时。本轮两项优化，
均在 `--feedback-solver-steps` 基础上叠加。

## 优化 1：反馈预测跳过 autograd 图（`prediction_no_grad`）

**问题**：unroll 的每个中间步都会跑一次 Euler 积分 + decode 得到反馈 latent，但反馈路径
默认 `detach`——梯度图建好立刻丢弃，纯浪费。

**实现**：`compute_loss(prediction_no_grad=...)`。当该步的预测只用于反馈（非最后一步、
无 aux loss）时，积分+decode 在 `torch.no_grad()` 下执行。末步或启用 recon/root/consistency
时保持梯度（aux loss 会消费它）。等价性：反馈本来就要 detach，数学不变。

## 优化 2：快速量化路径（`quantize_only` 重写）

**问题（本轮最大的发现）**：`requantize_latent` 每步反馈都要调用库的
`vector_quantize_pytorch.forward`，它在 eval 下仍物化 `F.one_hot(indices, 2048)`
（32 heads × 512 batch int64→fp32），单次 **3.49 ms**，占反馈步的 92%；
同次调用里真正的 decode 只要 0.24 ms。

**实现**：`quantize_only` 重写为分块 argmin + gather——逐 head 计算
`|x|² + |e|² - 2x·eᵀ` 后 `argmin`（与库 `argmax(-cdist)` 同一最近码、同一 tie break），
再 `index_select` 取 fp32 码本行。**逐位一致**（5 组随机尺度/批量 + 8k 真实帧
`torch.equal == True`），单次 **0.69 ms（5×）**。`controller.requantize_latent` 已切换该路径，
所有复用点（rollout 评估、交互反馈、VQ2 训练）自动受益。

## 实测阶梯（transformer、batch 512 整批、8 workers、300 updates 稳态、seed42）

| 配置 | s/1000up | 每 epoch(5172up) | 相对最初 | 50ep 预算 |
|---|---:|---:|---:|---:|
| u1（纯 TF 训练） | 11.0 | 57s | — | 0.8h |
| u3，反馈积分=solver8（最初） | 55.8 | 4.8min | 1.00× | ~4.0h |
| u3 + fb2（fast-quantize 前） | 42.5 | 3.7min | 1.31× | ~3.1h |
| **u3 + fb2 + 本轮两项** | **25.3** | **2.2min** | **2.20×** | **~1.8h** |
| u2 + fb2（fast-quantize 前） | 22.5 | 1.9min | 2.48× | ~1.6h |
| **u2 + fb2 + 本轮两项** | **18.0** | **1.6min** | **3.10×** | **~1.3h** |

侧收益：交互窗口单步推理 **4.25 → 3.76 ms**（-12%，同样来自 fast quantize）；
评估路径的每步 requantize 同步提速。

已尝试但无效：`.to()` 快速路径（latent stats/位置嵌入；实测无变化——dispatch 非瓶颈）、
`torch.compile`（环境 inductor bug `duplicate template name`）、worker 数 8→16（数据加载非瓶颈）。

## 第三批（2026-10-11 晚，训练运行中实测，含竞争噪声）

1. **shared style table**（`--shared-style-table`，默认开）：unroll 的 4 步标签相同，全风格
   selector 表（0.3ms/次）提升到每窗口一次。实测 300up：rollout 18.3→17.7s、TF 15.6→14.5s
   （+3~7%）。等价性：forward 一致，梯度仅浮点结合序差（≤4.7e-10），已加测试。
2. **matched_root 零同步**：去掉最后一个 `torch.nonzero(changed)` host 同步——改为固定宽度
   working-set 上用 `torch.where` 全量写回（未命中的行写回原值）。已验证与旧实现逐位一致
   （prob=0/0.3/0.7/1.0 四档）。此改动无法单独 A/B（非开关），收益并入端到端。
3. **batch 放大（最大杠杆）**：同一总样本量（153,600）下，u4+fb2 配置：

   | batch | updates | wall | 相对 |
   |---:|---:|---:|---:|
   | 512 | 300 | 19.3s | 1.00× |
   | 1024 | 150 | 13.3s | **1.45×** |
   | 2048 | 75 | 12.7s | **1.52×** |

   每 update 的发射开销被摊薄，利用率显著提升；bs1024/2048 显存均无压力。
   **注意**：batch_size 是 strict-resume 校验字段，只能用于**新 run**；改 batch 同时应把 lr
   按比例上调（如 5e-4→7e-4，sqrt/linear 缩放二选一），否则每 epoch 的优化步数减半会改变
   训练动力学。待 GPU 空闲时做一次带 lr 缩放的收敛短测确认。

综合推荐（新 run）：`batch 1024 + lr 7e-4 + u4 + fb2 + shared style table`。

## 复现命令

```bash
# 阶梯（u3+fb2 推荐配置）
$PY training/train_flowmatch_vqvae_controller.py ... \
  --seq-len 8 --unroll-steps 3 --feedback-solver-steps 2 --train-feedback-context \
  --batch-size 512 --microbatch-size 0 --num-workers 8 ...
```

验证（307 项全量测试通过，其中含 quantize 等价性与反馈 no-grad 语义的既有覆盖）：

```bash
bash scripts/run_tests.sh cpu
# fast path 逐位一致（随机尺度 + 真实缓存帧）
python - <<'EOF'
... 见 tests/test_clip_recipe_v2_efficiency.py::test_frozen_quantize_only_preserves_codes_and_state
EOF
```

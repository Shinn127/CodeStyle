# Label controller 提速（2026-10-10，FP32 逐位近似）

基线：同一 run（`checkpoints/flowmatch_label_nomirror_150ep`，新无镜像库 5172 updates/epoch、有效 batch512、
Euler8、recon=0、decouple=0.3、prior .02 / entropy .003、FP32/highest/noTF32），
`--no-label-selector-dedup` 关闭全部新优化。热缓存、无其他 GPU 作业。

| 配置 | train s/epoch | batch/s | epoch（含 val）s |
|---|---:|---:|---:|
| 基线（原始代码） | 55.0 | 93.2 | 58.3 |
| 优化后 | **42.0** | **122.1** | **44.8** |
| 提速 | **1.31×** | 1.31× | 1.30× |

## 四项改动（全部保持数学等价）

1. **按 style 去重跑 selector**（`arch/flowmatch_vqvae_controller.py`，`label_selector_dedup`，
   默认 True，`--no-label-selector-dedup` 可回退）。label 模式下 selector 与其正则项只依赖 style，
   原来每步把 `[512, 8, 2048]` 的 softmax/topk/CE/entropy 前反向各算一遍；现在整表 `[90, 8, 2048]`
   只算一次、按 label `index_select` 取行，正则项按 style 出现次数加权（与逐样本均值等价）。
   梯度仍是同一个 `Σ_s w_s ∇L_s / N`。实测该表 0.49 ms → ~0.05 ms/step。
2. **指标累加融合**：26 个标量指标的 `stack + mul + add` 子图换成单个 `torch._foreach_add_`；
   clean/cross/masked 三分组的 6 次 masked-sum 换成一次 `bincount` + 加权和。
3. **suffix → id 映射搬到 GPU**：原路径每步做 `segments.cpu().tolist()` + 512 次 Python 查表；
   现在 epoch 开始建一次 device 侧映射，`matched_root` 只收 `suffix_ids` 张量。
4. **donor 搜索只算请求行**：`matched_root_targets` 原来对全部 512×512 行算配对策（root/yaw/feet），
   多数行并不请求 donor；现在用固定宽度超集（`p + 8σ` 二项上界，prob=.3 时 238 行）只算这些行，
   其余保持原值。RNG 抽取顺序不变，60 组 (seed×prob∈{0,.15,.3,.5,1}) 配置与稠密参考实现
   **逐位一致**（donor 决策、changed mask、fraction、labels 全等）。
   另外消掉了每步 14 次强制同步（nonzero/any/float/unique），只剩 1 次必要的行数同步；
   `decoupled_fraction` 现在以张量形式累加。

## 试过但未采用

- **CUDA Graph 捕获 field+selector**：`make_graphed_callables` 捕获成功，loss/梯度逐位一致
  （残差 1.5e-08），但**速度零收益**（2.686 → 2.690 ms）——说明该段是真实 GPU 计算瓶颈而非
  启动瓶颈，故未启用。
- **worker 内 pin_memory**：fork worker 里调用 CUDA 分配会 `CUDA error: initialization error`，已回退。

## 剩余分布（优化后约 8.1 ms/update GPU 串行口径）

field fwd+bwd ≈2.7 ms（含 selector 表与正则），matched_root ≈1.6 ms（含合成帧重编码 ~1.0 ms），
optimizer 0.6 ms，H2D 0.2 ms，其余 ~3 ms 为 encode_conditions/flow loss/指标/循环与一次必要同步。
GPU 利用率稳定 95–99%，已基本是计算受限；要再上一个台阶需要改数学或协议（例如减少验证频率、
削减 selector 表的 K、或放弃逐位等价用 bf16/TF32），不属于"等价提速"范围。

校验：310 CPU + 21 GPU 测试通过；生产 run 已用优化代码从 epoch 29 续训，
epoch 29/30 实测 45.8 / 44.8 s。

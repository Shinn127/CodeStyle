# S3/S4：acc 单变量全 VQ 配对 pilot 结果

依据 [COMMON.md](COMMON.md)（预注册协议，S3 开跑前冻结）。两臂从 ep90 完整 VQ 状态冷启
（init），都更新 E+Q(EMA)+D；唯一变量 = `ACC_WEIGHT`（B0=0，B1=4.9842，S2 标定）。

- 主分析 checkpoint：各臂 `epoch_0010.pt`（固定终点，零选择）；acceptance JSON：`acceptance_ep10.json`
- 结论：**FAIL**

## 0. 阶段报告必备字段

- **hypothesis**：逐帧重构监督对二阶时间误差不敏感；直接监督 GT 二阶差分可压低量化解码的 `Δ²r`/jitter 而不把运动压向静止。
- **single_variable**：`ACC_WEIGHT`（B0=0 / B1=4.9842），其余全部相同。
- **source_commit**：`ef32703408c3302a18055f42d191b612412e67dd`（dirty=False，S0 于干净树 PASS）
- **init_ckpt**：`/home/shinn/桌面/multimedia/revision/NewCode/checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt` epoch=90，state sha256=918b5e07046266f7…（位级恢复=True）
- **norm**：sha256=bfbe62d548b525de…（显式传入未重算）
- **dataset**：`/home/shinn/桌面/multimedia/revision/NewCode/database_100sty_375_memmap_nomirror`，train/val/test 段数=549/90/90，mirrored=0
- **config**：seq_len=8, stride=1, batch=512, epochs=10, lr=1e-4→1e-5, seed=42, loader_seed=4242；S2 标定 `ACC_WEIGHT=4.9842`（encoder 37.8% / decoder 8.1% of base grad）
- **环境**：{'torch': '2.13.0+cu132', 'cuda': '13.2', 'gpu': 'NVIDIA GeForce RTX 5070 Ti', 'vector_quantize_pytorch': '1.31.4', 'fp32': True, 'tf32': False, 'matmul_precision': 'highest'}
- **B0/B1 更新数**：51650.0 / 51650.0（相同）
- **最终状态指纹**：B0 `376bb1651c71fdc5…` / B1 `ea111ea2ff2c79d7…`（量化器均已改变 → 新 VQ 身份）
- **tests**：`tests/test_vq_temporal_pipeline.py`（CPU 12 项：wrapper 可解析性、A/B pin、S4 门逻辑、val12 选定、探针 schema；GPU 1 项）；全量回归 `bash scripts/run_tests.sh cpu` = 422 passed；GPU 套件见 S4/gpu_tests.log
- **per-source_metrics / CI**：本报告 §1–§4 及 `acceptance_*.json`（12 源配对 bootstrap，seed=42，1000 次）
- **unexpected_behaviors**：① 入口脚本 `--no-decoder-only` 真实缺陷（本实验启动时暴露，已修复 + 回归测试）；② B1 的连续 AE 路径未同步变平滑（c.jitter +2.0%），但两条路径的姿态精度均明显受损（c.local_mpjpe +22.2%、c.fs_world +15.0%）——量化路径的 Δ²r/jitter 下降不能用“解码器整体变平滑”解释，而是新的 E/Q/D 组合改变了量化解码运动的时间特性；③ B1 端点 codec 往返翻码率上升（0.605→0.682），说明 jitter 下降不依赖更稳定的码身份；④ `best.pt` 停在各自的最后一次 gate 接受 epoch（B0 best=ep1（接受 epoch: [1]），B1 best=ep3（接受 epoch: [1, 3]），由 history.jsonl 读出），主分析按预注册使用固定终点 epoch_0010.pt。
- **decision**：**FAIL**（见 §1）；否决规则触发：以运动/精度代价换低 jitter，且 jitter 降幅未达 15% 预注册阈值。**不自动续到 150ep。**
- **next_authorization_needed**：S5 分支选择（S5a 位置空间二阶损失 / S5b 接触约束 / S5c 相对边界几何 / S5d 短历史 decoder）或其它改动，均需用户显式授权；本轮到此停止。

## 1. 预注册门结果（固定 val12，配对 bootstrap）

| 判据 | 结果 | 关键数字 |
|---|---|---|
| B0/B1 contract: manifests + configs identical except ACC_WEIGHT | PASS | 全部 12 项一致；updates=51650.0；acc=[0.0, 4.9842]；config 差异键=['acc_weight', 'out_dir'] |
| d2r mean & p95 both drop (95% CI upper < 0) | PASS | mean 0.4170→0.3703cm (-11.19%); p95 1.3316→1.1729cm (-11.92%) |
| ||d2 e_q|| mean change <= +0.5% | PASS | ‖Δ²e_q‖ 0.4250→0.3812cm (-10.31%) |
| official quantized jitter drop >= 15% | FAIL | jitter 0.4382→0.3952cm (-9.82%; 要求 ≤ -15%) |
| world & local FS change <= +0.5% | PASS | world 0.5124→0.4911cm (-4.16%); local 0.2563→0.2419cm (-5.59%) |
| MPJPE/local/rootADE regression <= 3% | FAIL | mpjpe 1.5060→1.6552 (+9.91%); local_mpjpe 1.0832→1.3128 (+21.20%); root_ade 0.6994→0.6872 (-1.75%) |
| activity proxies drop <= 5% | FAIL | 最差项 activity_q.strong_moving_frame_fraction 0.0708→0.0667 (-5.88%) |
| style proxy accuracy drop <= 2% | PASS | accuracy 0.933→0.922 (-1.11%) |

## 2. 活动性/防退化（门 6 明细）

| 代理 | B0 | B1 | 变化 |
|---|---:|---:|---|
| activity_q.moving_frame_fraction | 0.4993 | 0.4715 | -5.56% |
| activity_q.strong_moving_frame_fraction | 0.0708 | 0.0667 | -5.88% |
| activity_q.pose_step_mean_cm | 0.5430 | 0.5274 | -2.87% |
| activity_q.leg_travel_cm | 166.2 | 164.9 | -0.77% |
| activity_q.toe_height_range_cm | 5.1702 | 5.1706 | +0.01% |
| activity_q.foot_lift_fraction | 0.1364 | 0.1395 | +2.27% |
| activity_q.toe_speed_mean_cm_per_frame | 1.4898 | 1.4850 | -0.32% |
| contact.pred_contact_coverage | 0.1260 | 0.1292 | +2.48% |

## 3. 连续 AE 对照（区分"量化残差收益"与"两条路径共同变平滑"）

读法：连续路径 jitter +2.0%（未变平滑），而连续路径的 local MPJPE +22.2%、world 足滑 +15.0%，
与量化路径的姿态精度损失同量级。因此本轮 Δ²r/jitter 下降**不能**归因为“解码器整体变平滑”，
它是新的 E/Q/D 组合改变量化解码运动时间特性的结果，且代价（关节位置精度）在两条路径上同时存在。

| 连续路径指标 | B0 | B1 | 变化 |
|---|---:|---:|---|
| jitter | 0.1251 | 0.1277 | +2.03% |
| fs_world | 0.2093 | 0.2408 | +15.03% |
| fs_local | 0.1279 | 0.1353 | +5.85% |
| mpjpe | 1.4278 | 1.6409 | +14.92% |
| local_mpjpe | 1.0424 | 1.2739 | +22.21% |
| root_ade | 0.6714 | 0.7094 | +5.67% |

## 4. 分组诊断（小样本，不作选型依据）

| 组 | n | Δ²r (B0→B1) | jitter (B0→B1) |
|---|---:|---|---|
| idle | 1 | 0.0859→0.0725 | 0.0858→0.0713 |
| run | 4 | 0.5267→0.4885 | 0.5617→0.5298 |
| turn | 2 | 0.3798→0.3288 | 0.3886→0.3404 |
| walk | 5 | 0.4103→0.3520 | 0.4298→0.3742 |

## 5. GT 参照

- GT jitter = 0.0917cm；GT world 足滑 = 0.0907cm

## 6. 契约闸门 g0 明细（强制）

- 总判定：全部一致（updates=51650.0；acc=[0.0, 4.9842]）

| 检查项 | 结果 | 详情 |
|---|---|---|
| same_total_updates | PASS | 51650 vs 51650 |
| same_init_state_sha | PASS | 918b5e07046266f7… vs 918b5e07046266f7… |
| same_init_ckpt_file_sha | PASS | 825be6785b3e7fc2… vs 825be6785b3e7fc2… |
| same_dataset_sha | PASS | 6d308e18f05285e6… vs 6d308e18f05285e6… |
| same_split_sha | PASS | c3987af0f1c91568… vs c3987af0f1c91568… |
| same_vq_state_sha | PASS | 918b5e07046266f7… vs 918b5e07046266f7… |
| same_norm_sha | PASS | bfbe62d548b525de… vs bfbe62d548b525de… |
| same_pinned_protocol | PASS | pinned dicts equal |
| acc_is_single_variable | PASS | B0=0.0 B1=4.9842 (each arm must match its own manifest expectation, and they differ) |
| probe_b0_is_from_manifest_run | PASS | /home/shinn/桌面/multimedia/revision/NewCode/checkpoints/vq_temporal_B0_ep90seq8_10ep/epoch_0010.pt under /home/shinn/桌面/multimedia/revision/NewCode/checkpoints/vq_temporal_B0_ep90seq8_10ep |
| probe_b1_is_from_manifest_run | PASS | /home/shinn/桌面/multimedia/revision/NewCode/checkpoints/vq_temporal_B1_ep90seq8_10ep/epoch_0010.pt under /home/shinn/桌面/multimedia/revision/NewCode/checkpoints/vq_temporal_B1_ep90seq8_10ep |
| config_diff_only_allowed | PASS | differing keys: ['acc_weight', 'out_dir']; allowed: ['acc_weight', 'out_dir'] |

- config.json 差异键：['acc_weight', 'out_dir']（允许：['acc_weight', 'out_dir']）

## 7. 中途鲁棒性对照（epoch_0005.pt，非选型依据）

- 结论：**FAIL**

| 判据 | 结果 | 关键数字 |
|---|---|---|
| B0/B1 contract: manifests + configs identical except ACC_WEIGHT | PASS | 全部 12 项一致；updates=51650.0；acc=[0.0, 4.9842]；config 差异键=['acc_weight', 'out_dir'] |
| d2r mean & p95 both drop (95% CI upper < 0) | PASS | mean 0.4080→0.3613cm (-11.45%); p95 1.2925→1.1317cm (-12.44%) |
| ||d2 e_q|| mean change <= +0.5% | PASS | ‖Δ²e_q‖ 0.4160→0.3696cm (-11.15%) |
| official quantized jitter drop >= 15% | FAIL | jitter 0.4298→0.3839cm (-10.67%; 要求 ≤ -15%) |
| world & local FS change <= +0.5% | PASS | world 0.4973→0.4644cm (-6.62%); local 0.2879→0.2514cm (-12.68%) |
| MPJPE/local/rootADE regression <= 3% | FAIL | mpjpe 1.4850→1.6711 (+12.53%); local_mpjpe 1.1035→1.2981 (+17.64%); root_ade 0.6595→0.7426 (+12.60%) |
| activity proxies drop <= 5% | FAIL | 最差项 activity_q.strong_moving_frame_fraction 0.0750→0.0618 (-17.59%) |
| style proxy accuracy drop <= 2% | PASS | accuracy 0.922→0.922 (+0.00%) |

## 边界与后续

- 本结果只覆盖表示层 val 诊断；量化向量已变，属新 VQ 身份，不得覆盖现役 controller 或旧 cache。
- 150ep 重训、cache 重建、controller 训练/替换均需另行授权（COMMON.md §8）。

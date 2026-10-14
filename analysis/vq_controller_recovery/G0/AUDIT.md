# G0 契约核验（只读，不更新任何模型）

- 结论：**PASS**（全部条件满足）
- git：`12861afc4f15527eed3f8c89da501fbfeb2fa8dd`（recovery/vq-controller-2026-10-15，tracked_dirty=False，tracked diff sha=e3b0c44298fc1c14…）
- VQ：`vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt` epoch=90，state sha=918b5e07046266f7…
- Controller：`flowmatch_label_history_obsnoise_tf100/best.pt` epoch=100，objective v12，2L/256dim，solver 4，obs noise 0.5，pure TF（p=0）
- 指纹交叉核验：{'vq_state_matches': True, 'db_matches': True, 'split_matches': True, 'norm_matches': True}
- 辅助资产：latent stats sha=417b174605508826…，VQ target cache indices sha=c93cbd6c07c69787…，style prior sha=5c93fc0e3aace13b…
- 环境：torch 2.13.0+cu132 / cuda 13.2 / NVIDIA GeForce RTX 5070 Ti / vq-lib 1.31.4；FP32 强制、TF32 off
- 测试：CPU 425 passed, 22 deselected, 10 warnings, 5 subtests passed in 34.80s；GPU 22 passed, 425 deselected, 1 warning in 3.97s

## wrapper DRY_RUN 核验

- `vq_entry`：returncode=0，argparse parse ok=True
- `controller_obsnoise_tf100`：returncode=0，argparse parse ok=True

## 已修复（不得重复上报）

- S4 g0 acceptance gate is now mandatory/fail-closed (12861af) — do not re-file as a bug
- S4 best.pt epochs now read from history.jsonl (12861af) — B0 ep1 / B1 ep3 is correct
- S4 continuous-AE causal wording corrected (12861af) — c.jitter +2.0%, not 'smoothed'
- VQ entry wrapper --no-decoder-only argparse bug fixed (28ac8a0) + regression test
- S2/S3/S4 contract fixes: norm pinned, loader generators, teacher fingerprint, resume guards
- codebook_boundary_distance exact Voronoi plane + margin/no_grad search (0a11707 era)

## 未决观察（不阻塞 G0）

- controller wrapper expands duplicated flags (--epochs/--flow-layers twice); argparse last-wins makes it benign, recorded here as cosmetic only
- flowmatch_label_history_obsnoise_tf100 contains an extra best_loss.pt from an older artifact convention; not used by any runner
- historical 'shell tail' anomaly in an old pilot log (AUDIT section 6.1) was never reproduced; wrapper syntax + DRY_RUN verified in G0
- controller trainer resume guard covers objective version; C2 must bump it if semantics change

本文件与 contract.json 在 C0 开跑前冻结。

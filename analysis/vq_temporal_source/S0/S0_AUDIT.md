# S0 契约核验（只读，不更新模型）

- 结论：**BLOCKED**；blockers：['working tree dirty at S0 time (recorded; new commits change the id)']
- git：`5cdc6e345024d3ec5e81022693d167edb2fd063a`（main，dirty=True）
- 源 checkpoint：epoch=90（期望 90），sha256=825be6785b3e7fc2…，recipe_ok=True
- init-ckpt 全量位级恢复：True（恢复后 state sha256=918b5e07046266f7…）
- norm：`norm_stats.npz` sha256=bfbe62d548b525de…（显式传入，不重算）
- DB：729 segments，train/val/test=549/90/90，mirrored=0，不含 OOD 库
- seq_len=8 真实窗口：train=2644949（每 epoch 5165 updates，10ep 共 51650）
- DRY_RUN 展开：B0=通过，B1=通过（--init-ckpt / --norm-stats / --acc-weight / --loader-generator-seed 均已核验）
- val12：与归档 audit starts 一致=True，archived sha256=cf0e19422a6c3d1e…
- 环境：{'torch': '2.13.0+cu132', 'cuda': '13.2', 'gpu': 'NVIDIA GeForce RTX 5070 Ti', 'vector_quantize_pytorch': '1.31.4', 'fp32': True, 'tf32': False, 'matmul_precision': 'highest'}

唯一变量 = ACC_WEIGHT（B0=0，B1 由 S2 梯度标定决定）；其余设置见 COMMON.md §6。
本文件与 contract.json 在 S3 开跑前冻结。

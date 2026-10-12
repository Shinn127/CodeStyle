> 2026-10-10：本表所列在含镜像库上训练的 checkpoint 与 cache 已全部删除（镜像段导致 mesh 扭曲）。
> 无镜像链路的重建顺序见 [NOMIRROR_CHAIN.md](../NOMIRROR_CHAIN.md)，下表仅作历史记录。

# 当前checkpoint与依赖

| 用途 | 文件 |
|---|---|
| 固定VQ150 | `checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep/best.pt` |
| 固定normalization | 上述目录的 `norm_stats.npz` |
| 240f/60fps、100类classifier | `checkpoints/fid_feature_extractor_camdm100_mirror_60fps_240f/100style_position_classifier.pth` |
| Label150基线 | `checkpoints/flowmatch_tf150_staged_noise/best.pt` |
| Clip/recipe150基线 | `checkpoints/clip_recipe_scratch150/best.pt` |
| 当前失败的history50模型 | `checkpoints/clip_recipe_gt_history50/best.pt`（epoch35，旧TF选型） |
| 旧纯recipe150诊断 | `checkpoints/flowmatch_recipe_tf150/best.pt` |
| Target cache / prior | `cache/vq_targets_vq150ep_repair_v2` / `cache/style_code_prior_vq150ep_repair_v2.npz` |

各run的best/latest、配置和统计保留，label epoch50对照也保留。被删除中间epoch见[manifest](../cleanup_2026-10-09/manifest.json)。其他历史run仍保留best/latest，按其config核验VQ/normalization与来源。

外部VQ必须匹配controller内嵌权重，不能通过override更换latent manifold。部分很早的run保存迁移前绝对路径，不能当作当前默认。当前说明见[README](README.md)。

# head10 产物清理记录（2026-10-05）

按用户要求删除 head10 相关产物。依据 checkpoint 的训练 `config.db`、数据库名称及评估/可视化/配方的 metadata 判断，不单凭文件日期或名称。删除清单及旧 checkpoint 配置见 [JSON manifest](CLEANUP_HEAD10_2026-10-05.json)。

共删除 18 个 checkpoint 目录、54 个评估目录、1 个 head10 数据库，以及关联的配方和可视化文件；文件逻辑大小合计 3.365 GiB。

90-style/full100/zero-shot 数据库、非 head10 权重与 VQ/style prior 缓存保留。历史文字报告保留；其中引用的已删除产物不能再直接读取，旧 head10 FID 数字不代表现行评估口径。

模型、训练和评估源代码保留，便于重训 baseline。`training/train_fid_feature_extractor_head10.py` 是共享训练器，仍供 full100 wrapper 调用。旧 baseline 的默认 checkpoint 路径可能已经不存在，需要显式提供新训练的 checkpoint；不能自动替换成其他实验权重。

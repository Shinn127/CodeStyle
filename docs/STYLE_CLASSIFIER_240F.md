> **2026-10-10 状态**：镜像段已确认导致 mesh 扭曲，训练数据一律不使用镜像；含镜像库上训练的
> VQ/controller/classifier 与派生 cache 已全部删除，代码默认路径与防呆见 [NOMIRROR_CHAIN.md](NOMIRROR_CHAIN.md)。
> 本文下列内容为修订前的记录，其中涉及 `--mirror`/`--no-mirror`、`--include-mirror`、`--unmirrored-sources`
> 等开关与已被删除的 checkpoint 的说明不再适用。

# Style classifier：60fps、240帧

当前报告分类器为 `checkpoints/fid_feature_extractor_camdm100_60fps_240f_nomirror/100style_position_classifier.pth`（**待重建**），100类，输入CAMDM的15个关节/root-frame position45维，240个连续帧/60fps，4秒。旧60帧分类器保留用于历史口径，不作当前默认。

Classifier独立于generator；OOD10类可用于报告classifier训练，不能用于generator或下一版独立风格度量Φ训练。主实验不优化报告classifier。

重训前先合并数据：

```bash
python tools/merge_memmap_dbs.py --primary database_100sty_375_memmap \
  --secondary database_100sty_375_memmap_zeroshot \
  --out database_100sty_375_memmap_fid_full100 --secondary-split train

bash scripts/run_train_fid_feature_extractor_camdm.sh
```

Classifier已训练好，日常评估不需要已删除的合并副本。原IND/OOD与重建metadata保留，见[清理记录](cleanup_2026-10-09/README.md)。全100类通用classifier与reconstruction extractor仍有独立wrapper；`training/train_fid_feature_extractor_head10.py`是full100 wrapper仍使用的通用旧文件名，不能按名称删掉。

评估只分类生成的240帧，排除source GT初始帧；不足240帧不能用padding冒充4秒。对应回归包含真实240f保存加载、关节切片、window长度与防覆盖检查。

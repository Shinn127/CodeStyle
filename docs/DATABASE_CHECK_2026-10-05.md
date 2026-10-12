# 数据库检查（2026-10-05）

检查未修改数据库或训练代码。现存 4 份库，均为 motion375_memmap schema v4，float32 特征（84 trajectory + 291 motion）和 int32 标签。head10 数据库已不存在。

| 数据库 | style | 帧数 | 段数 | 文件大小 GiB | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| database_100sty_375_memmap | 90 | 7,068,516 | 1,458 | 9.901 | 当前 VQ / FlowMatch 默认训练库，含镜像 |
| database_100sty_375_memmap_unmirrored | 90 | 3,534,258 | 729 | 4.951 | 同一 IND 数据的无镜像版本 |
| database_100sty_375_memmap_zeroshot | 10 | 423,710 | 81 | 0.594 | OOD / recipe exemplar，无 split |
| database_100sty_375_memmap_fid_full100 | 100 | 7,492,226 | 1,539 | 10.495 | IND+OOD 合并的 FID / style classifier 专用库 |

主训练库 train / val / test 帧数为 5,714,320 / 746,384 / 607,812（80.8% / 10.6% / 8.6%），90 个 style 均覆盖三个 split；无镜像库为 2,857,160 / 373,192 / 303,906。两库未发现跨 split 帧重叠，镜像变体与原 clip 使用同一 split。IND 与 OOD 的 style 和 source clip 范围均无交集。

## 验证范围与发现

使用现有 `tools/verify_memmap_db.py`，对四库检查全部标签/segment 和文件大小，每库随机抽样 20,000 帧（seed=0）检查有限值、trajectory direction、rotation basis 和数值范围。原始结果见 [JSON](DATABASE_CHECK_2026-10-05.json)。

1. 主训练库 35 项抽样检查通过；zero-shot 库 30 项通过。抽样通过不代表每帧均无异常。
2. 无镜像库 direction unit-norm 检查失败：抽样 20,000 帧中 1 帧超过 0.002 阈值，最大偏差 0.002211951。定位到 FairySteps_SR、source_entry_id=215、test split、无镜像库 frame=1,041,286、trajectory sample=2。主训练库对应 frame=2,077,563 的 direction 数值完全相同，因此是原数据中已有的小幅数值偏差；主库随机抽样未命中。不能宣称主库全量数值检查通过。
3. FID 库的 10 个 OOD style 全部被设为 train，val/test 仍仅含 90 IND style。通用检查器要求所有 style 覆盖三个 split，因而报 FAIL；该布局符合 `tools/merge_memmap_dbs.py` 默认 secondary_split=train 的专用设计。此库不应作为 motion generator 的 OOD 隔离训练库；分类器 val/test 也不覆盖那 10 个 style。CAMDM classifier 训练器另采用无 split、全部窗口训练的协议。
4. FID 库缺少顶层 frame_cuts_sha256，通用检查器给出 WARN；其 derived_from 内记录了来源 metadata 哈希，两份源库当前哈希均匹配，溯源可核实。每份源库抽样 2,000 帧（seed=42）与合并库逐值一致。
5. 无镜像库全部 729 个 clip 的长度、标签、split、source_start/stop 与主库原始非镜像段一致；每段抽取首/中/末帧，共 2,187 帧，特征逐值一致。这不是全文件逐字节比较。

数据合计约 25.94 GiB。本轮不删除库、不重新生成数据、不重算 cache、不启动训练。

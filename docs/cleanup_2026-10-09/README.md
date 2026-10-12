# 2026-10-09清理与测试整理

已完成清理，NewCode约42GiB→17GiB。物理删除198个文件，共**25.10GiB**；详细文件、字节数、原因及保护清单见[manifest](manifest.json)。

## 删除与保留

- 删除full100合并数据库与未镜像IND副本：先逐字节检查全部features、全部labels及segment对应关系。原始IND90/OOD10未改，派生metadata在[derived_metadata](derived_metadata)保留。Full100可用现有merge_memmap_dbs.py重建；unmirrored评估可直接按原始IND的mirror=false segments读取，不需要另存5GiB副本。
- 删除被repair_v2替代的旧target cache/prior；保留当前432MiB repair cache与train-only prior。
- 删除没有当前代码或run依赖的中间epoch文件；每个历史run的best/latest、normalization、config/history仍保留，label epoch50及代码明确引用的final150对照保留。当前VQ、classifier、label、clip/recipe核心文件清理前后SHA256一致。
- 旧规划、sweep、recipe TF50/150 wrapper、head10专用wrapper、LMM转发入口和旧recipe样例集中为[一个zip](../archive/legacy_sources_2026-10-09.zip)。不整体覆盖当前代码恢复。Baseline架构、真正被full100使用的通用旧名trainer和唯一best/latest未当成“无用”删除。
- 保留eval_results：它只有38MiB，包含失败分析、固定cases与逐样本研究记录。删除大的profiler traces，摘要/profile表与最新性能来源保留。

## 当前入口与文档

[README](../../README.md)是当前入口，checkpoint与cache见[索引](../archive/CHECKPOINT_MAP_2026-10-10.md)。正式评估默认rollout；OOD/recipe默认当前clip/recipe模型，precompute默认固定VQ150/repair_v2，避免指向删除的旧产物。

最新diagnosis、下一版方案与性能核验分别保留在analysis的两个2026-10-09目录；训练smoke统一为`analysis/profile_training.py`。原profile多分支脚本与陈旧源码快照已归档。新增smoke默认100预热+100计时，可只运行2+2作入口检查；临时随机controller，不写正式权重。

## 测试

```bash
bash scripts/run_tests.sh         # CPU全套：236项，GPU用例排除
bash scripts/run_tests.sh core    # 核心回归
bash scripts/run_tests.sh gpu     # 6项GPU-only，不重复CPU套件
bash scripts/run_tests.sh all     # CPU+GPU完整回归，要求CUDA可见
```

GPU参数化用例自动标记，独立CUDA用例显式标记；`gpu/all`在GPU不可见时明确报错，避免将skip当成完成。pytest限定tests目录，严格检查markers。小fixture与子进程限制CPU线程，已过时的wrapper测试移除，真实训练/恢复、240帧协议、GT-context隔离、冻结VQ、梯度与recipe接口覆盖保留。

首次整理后的CPU全套236 passed/6 deselected/5 subtests，27.93秒；减少重复validation并控制子进程线程后，同一覆盖为**21.35秒**。GPU-only **6 passed、4.06秒**，包含真实GPU epoch边界恢复的model/optimizer/CPU-CUDA RNG一致性。结果受机器负载影响，不能把测试速度当作训练吞吐。完整日志见[CPU](cpu_tests.log)、[GPU](gpu_tests.log)、[核心](core_tests.log)。

本次未启动完整训练或大规模评估，也不证明85/80/70达标。下一版temporal Es及输出风格监督仍需按保留方案实施。

# C1 报告：推理旁路扫描（不训练；权重全冻结）

依据 [PROTOCOL.md](PROTOCOL.md)（运行前冻结）与
[docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md](../../../docs/VQ_CONTROLLER_RECOVERY_AGENT_PLAN.md) §4 C1。
原始数据：[per_source.json](per_source.json)、[manifest.json](manifest.json)。
交叉校验：REF 与 C0-C 在同源同种子下逐位一致（LeanLeft local MPJPE 5.52 ↔ C0-C 5.52）。

## 0. 阶段报告必备字段

- **stage**: C1
- **hypothesis**: 存在无需训练的推理期设置（关闭 requantize / 改 solver / 改噪声 / 去历史），
  能在不损害风格与活动性的前提下减小闭环缺口。
- **source_commit**: `58b8870`（C0 提交；本阶段无 tracked 代码改动，见 manifest）
- **model_ckpt_hashes**: Controller state=`26dc40eb30bb9370…`；VQ state=`918b5e07046266f7…`
- **protocol_and_seed**: 12 源 ×241 帧、seed 42、共享 z0、仅改单项（见 §1 表）
- **actual_update_counts**: 0（只读）
- **verdict**: **NO_USEFUL_INFERENCE_FIX**（见 §2/§3）
- **next_stage_approval_needed**: C2a（纯 TF 观测噪声 {0.25,0.5,0.75} 三臂 ×10ep）训练授权

## 1. 扫描矩阵与 12 源汇总

| 配置 | 唯一变量 | local MPJPE | Jitter | world MPJPE | rootADE | FS world | 速度方向误差 | moving | strong | pose step | style | 延迟 ms/帧 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF | =C0-C | 4.431 | 0.460 | 8.145 | 5.927 | 0.888 | 6.70 | 0.508 | 0.107 | 0.566 | 10/12 | 2.17 |
| R0 | requantize off | 5.184 | 0.425 | 9.243 | 6.528 | 0.748 | 7.78 | 0.549 | 0.108 | 0.587 | 9/12 | 2.01 |
| S2 | Euler 2 | 4.469 | 0.322 | 7.518 | 5.225 | 0.756 | 6.92 | 0.453 | 0.070 | 0.513 | 11/12 | 1.54 |
| S8 | Euler 8 | 4.746 | 0.631 | 8.299 | 5.843 | 1.073 | 5.55 | 0.604 | 0.141 | 0.634 | 10/12 | 3.37 |
| N0 | noise std 0 | 4.591 | 0.263 | 6.802 | 4.467 | 0.699 | 4.65 | 0.430 | 0.062 | 0.484 | 11/12 | 2.16 |
| N05 | noise std 0.5 | 4.283 | 0.318 | 7.569 | 5.612 | 0.702 | 5.75 | 0.459 | 0.076 | 0.513 | 11/12 | 2.16 |
| H0 | history=0（OOD） | 7.154 | 1.785 | 29.03 | 26.22 | 2.095 | 24.5 | 0.761 | 0.446 | 1.180 | 3/12 | 2.18 |

（FS local 在 11/12 源有效、1 源零接触已标记；逐源均值见 JSON。style 列为 240f 冻结分类器
matched 正确数。）

## 2. 逐配置判定（PROTOCOL §3 停止条件）

- **R0（关闭 requantize）**：style 10/12→9/12，world MPJPE +1.10cm，root ADE +0.60cm；
  仅 jitter/FS 略降。反馈语义变更且质量不升 → **剔除**（且属正式协议外旁路）。
- **S2（Euler 2）**：jitter 12/12 改善（−30%）、世界指标小幅改善、延迟 −29%；但
  **moving −10.8%、strong −34.6%、帧均位移 −9.4%** → 触发"冻结比例上升"停止条件。
  与 GT 的活动性参照（C0 记录 GT moving 0.584）相比进一步偏离 → **不推荐**。
- **S8（Euler 8）**：全部指标 0/12 改善（jitter +37%、FS world +21%、延迟 +55%）→ **剔除**。
- **N0（噪声 0）**：jitter 12/12 改善（−43%）、world MPJPE/rootADE/FS 明显改善、速度方向误差
  6.70→4.65；但 **moving −15.4%、strong −41.6%、帧均位移 −14.5%** → 冻结/滑行风险显著，
  正是规约禁止的"以静止换 Jitter" → **剔除**。
- **N05（噪声 0.5）**：jitter 12/12 改善（−31%）、world MPJPE −0.58、rootADE −0.32、
  FS world −0.19；moving −9.8%、strong −29.1%、pose step −9.4%、local MPJPE −0.15（5/12 更优）。
  **活动性代价依然超过 5% 工程闸门** → 单凭推理改动不可采用；但它是所有候选中质量/活动权衡
  最接近可接受的一个，且与 C2 训练路线的方向一致。
- **H0（去 history）**：灾难性（local 7.15、world 29.0、style 3/12）→ **确认双帧 history 是
  现役结构的必要部件**，同时也说明 C1 的 history ablation 仅作分布外诊断。

## 3. 结论

**NO_USEFUL_INFERENCE_FIX**：所有能改善 Jitter/FS 的推理期候选（S2/N0/N05）都以超过 5% 的
活动性下降为代价（moving 帧占比 −10% ~ −15%、strong 帧 −29% ~ −42%），违反停止条件；
R0 与 S8 在质量上不升反降；H0 证明结构本身不能拆。结合 C0 的"自反馈分布偏移是主因"，
推理期无免费午餐 → **请求 C2a 纯 TF 观测噪声三臂 10ep 限额训练授权**（方向与 N05 的噪声
敏感性证据一致：更低推理噪声改善 Jitter 但活动性受损，训练期观测噪声正是把该权衡纳入
参数学习的手段）。

## 4. 效率

7 配置 × 12 源 ≈ 47s 墙钟（含分类器）。每帧延迟：REF 2.17ms、S2 1.54ms、S8 3.37ms、
噪声候选项 ≈2.16ms——均远低于 16.7ms 实时预算。

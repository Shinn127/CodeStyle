# 新 donor 离线数据构建：执行 agent 指导书

日期：2026-10-12。**本文件是当前 donor 任务的执行范围。只完成离线构建、小规模质量验证和交付，不接入训练器，不启动真实训练。**

目标：生成少量可审查的 `真实A前缀 → 连续过渡 → 适配后的B后缀`，证明数据在原始运动和冻结VQ重建两条路径上都合格。不是恢复已删除的单步donor替换，也不是追求命中率。拒绝不合理样本是正确结果。

用户当前指令和项目AGENTS约定优先。旧方案中的CFG、history dropout、style调制、四帧联合FM均不属于本任务。不要自行扩大工作范围；遇到局部不支持的样本记录拒绝并继续其余样本，只有全局阻塞才停止交付原因。

## 0. 可直接复制给执行 agent 的任务

> 请完整阅读 `/home/shinn/桌面/multimedia/revision/NewCode/docs/DONOR_AGENT_EXECUTION_GUIDE.md`，按本文完成阶段0–4。实现并验证保守的离线donor过渡数据小样本流程，保留现有用户改动。遵守文件白名单、预算、原始/VQ双重验收、train-only和停止条件。不得修改训练/模型/评估主流程，不得训练，不得降低门槛凑样本。交付代码、针对性测试、通过/拒绝记录、未经显示平滑的对照视频及报告。明确区分机器合格、人工待审、可供未来训练；零合格样本也要如实报告并解释。

## 1. 固定边界

### 可以做

- 新建少量离线构建代码、只读质量分析和针对性测试。
- 读取train原始BVH、现有数据库、VQ和normalization；使用原骨架/FK/特征提取约定。
- 使用小幅关节旋转过渡、有限下肢IK，重新提取完整375维特征、重新VQ编码。
- 执行小规模CPU测试、冻结VQ编码/解码、缓存读取测试和有界可视化。

### 不可以做

- 启动任意真实controller/VQ/classifier训练，包括本任务之外的A/B或50epoch；fixture测试中的极小梯度检查不等于真实训练。
- 恢复旧`matched_root_targets`或“原source history→直接donor下一帧”的训练分支。
- 修改VQ/codebook、normalization、split、原始DB、旧cache、checkpoint或旧结果。
- 增加loss、CFG、rollout、history遮蔽、style注入、latent slot替换，或改变交互/正式评估协议。
- 用最终报告classifier优化、筛选或调参合成样本；本轮不需要加载该classifier。
- 只修改root/position而复用旧velocity、旧trajectory或旧latent；直接线性混合375维或6D rotation。
- 用插值latent代替骨架空间构造，或靠显示平滑/显示IK掩盖数据问题。
- 读取val/test/OOD作为候选、校准或补充来源；越过train clip原有裁剪边界取上下文。
- 大规模全对全搜索、全90-style建库、自动sweep、自动删除/清理旧资产或安装新框架。

### 本版刻意缩小的能力范围

本版先做 **4个上肢姿态风格、同suffix走路**：`Neutral`、`ArmsAboveHead`、`ArmsFolded`、`ArmsBehindBack`；suffix只允许`FW/BW/SW`且两端必须相同。不宣称支持全风格或跑跳转圈。

本版**完整root平移XYZ和root旋转逐帧取source A**，不改root、不加骨盆高度补偿；合成非root关节旋转，有限IK只动下肢旋转。这样能保留原root控制计划，并减少坐标/特征错误。代价是不能表达需要改变root上下起伏/倾斜的style：若保留source root使B后缀失真，拒绝pair，不能称为完整B。

这是从更大过渡方案中抽出的保守pilot。大幅gait迁移、root适配、time warp、骨架retarget、跳跃及特殊滑步留待后续单独授权。**本版时间倍率固定1，不实现任意时间扭曲。**

## 2. 文件与预算

工作目录：`/home/shinn/桌面/multimedia/revision/NewCode`。

允许新建的实现文件，优先控制在以下5个：

1. `build_donor_transition_dataset.py`：CLI、调度、产物写入。
2. `Utils/donor_transition.py`：候选、骨架过渡、必要的小型IK函数。
3. `Utils/donor_transition_quality.py`：参考校准、数据质量检查。
4. `tests/test_donor_transition.py`：有针对性的fixtures。
5. `run_build_donor_transition_pilot.sh`：固定有界入口和dry-run。

可以少于5个，避免通用框架。旧文件默认只读；确需改已有公用函数时，先交付准确阻塞点，不能自行修改其行为绕过检查。不要改`train_*`、`arch/*`、`dataset.py`、`run_tests.py`或UI来让新数据“能用”。本轮用独立reader验证接口。

独立产物目录：`analysis/donor_transition_pilot_v1/<run_id>/`；run_id取时间+seed。已有目录不覆盖。代码之外允许在这里保存报告、manifest、缓存及视频；不要修改历史analysis记录。

硬上限：

| 项目 | 上限/默认 |
|---|---|
| seed / 精度 | 42；运动数组、VQ输入/计算均FP32、highest、TF32=False |
| 真实参考窗口 | 最多128个train窗口，每个240个有效motion帧 |
| 不同原始BVH | 总共最多24个，载入后复用；候选BVH必须属于上述4个style |
| 几何候选 | 最多128个不同的有向A/B窗口对 |
| 过渡时长尝试 | 每对固定尝试24/36/48个帧间隔，最多384次；不能无限调参 |
| 接受样本 | 最多32条；没有最低配额 |
| 单条工作clip | 最多512 raw帧，包含特征提取保护上下文 |
| 可视化 | 最多12条对照，优先6通过+6拒绝；不足按实际数量 |
| 输出总量 | 最多2 GiB，原始BVH不要完整复制到产物 |
| GPU | 累计不超过120秒冻结VQ计算；不占用/停止其他训练任务 |
| 总执行 | 不超过30分钟实际数据处理；超限保存进度和原因，不自动扩大预算 |

预算不含正常编辑代码时间。第一次先跑2–4个fixture/真实pair，确认路径后才运行pilot。每个pair尝试短时长失败可继续下一个预定时长，所有时长失败就记录拒绝。缺少GPU时可完成CPU fixtures和至多2个真实pair，不用CPU硬跑全量来规避资源限制。

CLI只需提供`--db/--vq-ckpt/--norm-stats/--out/--device/--seed/--max-pairs/--max-accepted/--max-videos/--dry-run`，其余pilot常量写入配置记录，不建通用配置框架。max-pairs等参数不能超过上表硬上限。dry-run只核对元数据/输入路径并打印计划，不创建输出、加载GPU或开始生成。wrapper将默认值显式传入，支持`OUT`指定新目录；输出目录已非空时拒绝覆盖。

先跑`--max-pairs 4 --max-accepted 4 --max-videos 2`；通过后换全新run目录再跑完整pilot。机器合格、等待人工检查不需要在这两个有界阶段之间反复请示；不得因此自行进入训练。统计汇总可用Python数值，运动数组与网络计算不得改成FP16/BF16/TF32。

## 3. 必读代码与输入核验

只读以下最小路径：

- `tools/generate_database_100sty.py::prepare_motion/extract_features_375`
- `Utils/bvh.py`、`Utils/quat.py`：BVH与FK/旋转约定。
- `Utils/motion375.py`：375布局、24个joint顺序、root积分。
- `Utils/vq_target_cache.py`、`Utils/reproducibility.py`：指纹与完成标记思路。
- `Utils/model_loading.py`：正确加载现有VQ。
- `Visualization/pose_postprocess.py::two_bone_target`：仅可参考/复用纯几何函数；不可直接套用有状态render后处理器、显示坐标转换或默认足底高度。
- 设计背景：`DONOR_TRANSITION_STRATEGY.md`。归档旧donor只能用于理解问题，不能整体复制恢复。

固定输入：

```text
database_100sty_375_memmap_nomirror/
checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/best.pt
checkpoints/vqvae_traj_motion_skip_cat_mlp_375to291_32_4_2048_150ep_nomirror/norm_stats.npz
```

原BVH路径从`segments.json.source_path`读取；只读，不写回。两端`split`都必须是train。名称/label使用`StyleNames.json`及segment当前label；不要用meta中历史100类列表或`source_global_label`代替90类顺序。

已核对：Neutral和ArmsAboveHead当前train没有FW片段，但有BW/SW。**不能为了凑FW配对取它们的validation FW。** 用共同train suffix；不足就报告。

本库：60fps，prepare_motion将cm转m，`coordinate_convert=False`，window60、context_stride10、selected24索引在meta中。运行时仍要核对实际metadata，不凭文字覆盖。`bvh.load()`当前不返回frametime：需读取BVH头部`Frame Time`验证约1/60（绝对误差≤1e-6秒），不能假设字典里存在frametime，也不要为此改公用loader。

要求两端joint names、parents和rest offsets兼容；root index与selected24名称核对。发现骨架不同、非root存在不支持的动态translation或帧率不同，拒绝该pair；本版不做通用retarget/resample。prepare_motion只调用一次，避免重复单位转换/翻轴。

## 4. 阶段0：identity基线先通过

先对少量原train clip复用原prepare/extract流程，重算安全内区的特征并与原DB对应行比较。由源码可知：

```text
DB segment内row r对应的trajectory时刻：source_start + window + r
该row中的motion时刻：上述时刻 - 1
X_k = [trajectory_k, motion_(k-1)]
```

必须显式测试source_start/source_stop是原raw裁剪范围，DB frame_start不是raw帧号。先按完整裁剪范围准备原clip，再选择安全窗口，避免改变Savgol边缘行为。

identity校验初始使用normalized features `atol=2e-4, rtol=1e-5`；记录每块最大/均值误差。失败先查索引/坐标/滤波边缘，不能自动放宽容差。A=D同一序列的零改动过渡也应复现原运动。

当前特征提取的trajectory speed包含root速度向量的范数，不应自行改成另一种定义；root yaw通道也按原提取函数计算，不自行改成heading差分。**因此本版复制完整source root变换，避免只复制XZ却改变其他相关特征。**

identity与指纹校验没通过，不进入实际构建。

## 5. 阶段1：校准、候选与配对

### 5.1 先固定质量阈值

校准只读train真实运动及其冻结VQ重建，最多128个240帧窗口；包含其他train style的正常手臂/腿部活动，不能只用4种静态上肢姿态校准关节运动上限，否则会把合理的举手过渡也全判坏。采样按style/suffix分层并记录，最多24个BVH。

在构建合成样本前保存并冻结`quality_thresholds.json`及哈希。每个动作组少于8个参考窗口时使用已记录的全train参考集合回退，不从val/test借数据。

对“越小越好”的连续性指标M，分别在raw和quantized参考上用：

`limit_M = max(P99(reference window maxima of M), numerical_floor_M)`。

数值floor固定为：position步差`1e-3 cm`、线速度`1e-4 m/s`、线加速度`1e-2 m/s²`、线jerk`1 m/s³`、rotation步差`1e-3 deg`、角速度`1e-2 deg/s`、角加速度`1 deg/s²`、角jerk`60 deg/s³`。这些只是数值下限，不是可随意增加的质量余量。

原始和codec两套阈值不得混用。下面注明raw的硬几何/接触门槛只约束raw，和raw参考上限取严格者；不要把raw骨长或root逐值相等的规则套到有重建误差的codec输出。codec的两种root口径分别建立自己的真实参考。不得看过合成通过率后反复抬高阈值。若超过20%的真实参考在某项适用硬门槛上失败，或identity不通过，停止生成并交付指标/参考分布问题，不能假装阈值合理。

指标定义也需冻结：位置用选定24关节的世界轨迹及去root后的局部轨迹分别计算，一阶/二阶/三阶有限差分除以dt/dt²/dt³；旋转用相邻quaternion最短弧相对旋转的log，先转到一致坐标再差分，禁止Euler角相减。每个window分别记录joint最大值和joint均值，先在window内取峰值，再跨reference windows取P99。边界检查覆盖接点前后各至少3帧，不只验证alpha端点值。速度/加速度/jerk的差分保护区不计入评分，但必须有足够原始上下文。

### 5.2 确定候选后再拟合

- 只从4个指定style、相同FW/BW/SW suffix、不同entry/style选A/B。
- 本pilot只接受稳定moving窗口：120帧上下文的root平面平均速度≥0.20m/s，且每个完整60帧子窗的平面路程≥0.15m。平均速度≤0.05m/s为idle，其余为ambiguous；idle/ambiguous均不进入本版候选。两端同样检查，不仅看文件名。起步/停止留待后续扩展。
- root局部平面速度差≤0.10m/s、yaw rate差≤10deg/s，检查整段候选，不只检查均值。这只是预筛，不能替代最终足滑检查。
- 再匹配支撑脚、接触事件与phase方向；边界及过渡中有明确左右支撑冲突就拒绝。最多允许接触事件时间误差2帧，不重新扭曲时间修补。
- 每个train候选clip在能容纳最长工作clip的安全区等间隔取最多16个anchor。通过硬筛后按以下tuple排序：接触事件最大时间误差、phase方向距离、root速度RMS差、yaw-rate RMS差、全身局部rotation边界RMS差、entry/start作为确定性tie-break。phase比较左右脚相对位置/实际速度的归一化方向；没有可靠动态方向时不能凭0向量伪造完美匹配。候选额度按有向(A style,B style,suffix)组轮转分配，组内seed42取source anchor并查最近donor；记录不可用组，不总挑最容易的一类。
- 同一过渡固定一个donor和offset，不逐帧换donor。candidate清单先写盘并哈希，固定seed重复执行一致。

## 6. 阶段2：保守的连续骨架过渡

### 6.1 时间轴

工作clip包括：前保护区至少60帧、可见A前缀60帧、T+1个过渡采样点、B后缀240帧、尾保护区至少62帧。T取24/36/48。所有保护区也必须在原train裁剪范围内，禁止重复末帧补长度。

完整root变换全程来自A对应时间轴；D的局部关节旋转按固定offset对齐。不改变骨长或非root rest offsets。先确保“D装在A的root上”的B后缀没有明显失真；不合格就拒绝。

### 6.2 固定实现方式

第一版用局部关节旋转的最短弧SLERP，权重为：

`alpha(u) = 6u^5 - 15u^4 + 10u^3, u in [0,1]`。

非root关节：过渡前取A，过渡中`SLERP(R_A(t),R_D(t),alpha)`，过渡后取D。quaternion先统一符号并unroll、再归一化；极小角度用稳定分支。不要对Euler角、6D表示或global joint positions直接线性混合。root取A，FK生成全骨架position/rotation。

alpha端点一、二阶导数为零，有助于接上两条本身平滑的序列，但**不保证中段自然，也不保证接触正确**；必须继续验收。不要在第一版另造复杂SO(3)优化器或学习型过渡网络。

### 6.3 接触与有限IK

- contact计划来自两端**原始真实运动**及预先冻结的phase对齐，不从合成结果重新检测后删除坏contact。
- 使用joint marker而非凭空假设mesh足底；floor proxy固定为train参考中该Foot/Toe marker在水平速度≤0.10m/s时世界高度的P01，参考不足则标记不可核验。接触候选：marker高度≤proxy+2cm、水平速度≤0.10m/s、连续≥3帧；退出高度>4cm或速度>0.20m/s。规则/完整预定mask写入产物，不能看到合成结果后缩短区间。
- 过渡开始前正在接触的脚保留source世界锚点。只约束可靠的稳定marker区间；摆动脚不锁。原两端接触计划明显矛盾就拒绝，不强行合并为双脚常驻接触。
- 可以做有限的两骨IK调整，**root和骨盆translation不动**。单次marker所需修正>3cm、任一腿关节相对未IK结果旋转改变>10deg、腿不可达或骨长破坏时拒绝，不放宽。
- 接触进入/退出的权重要连续，但最终质量判定使用完整预定接触区间，不只统计“完全锁定成功”的帧。IK后重新FK，并复查前后边界、中段速度/加速度和B后缀偏差。
- 无可靠contact参考的pair本版拒绝为`contact_unverifiable`；不能报0滑足通过。本版不覆盖SlideFeet等特殊接触模式。

### 6.4 重提取与命令标签

使用原`extract_features_375`对最终完整序列重新提取全部375维。所有velocity、rotation、trajectory由最终骨架导出；严禁复制拼接两端的这些feature块。

保留raw帧到feature row映射。假设发出B命令的motion时刻为k_switch，则预测motion_k时的命令为：k<k_switch取A，否则取B。X_k中的motion仍是k-1；目标在X_(k+1)中。不要把整个A前缀标B，也不要按“当前视觉上像谁”反复改变请求label。

对于未来history模型的6-row TF窗口：q[0]为older、q[1]为current、q[2:6]为四个target。取样需保证四个预测时刻的**命令label相同**，但允许更早history来自切换前A。不能因旧dataset要求整窗同label，就把prefix label伪改为B。本任务独立reader验证这一约定，暂不改训练dataset。

## 7. 双路径验收：缺一不可

固定逐样本和总体报告单位：position cm，velocity m/s，acceleration m/s²，angular velocity deg/s，duration frame与second同时给出。

| 检查 | 通过规则 |
|---|---|
| 来源 | 两端、上下文、阈值校准全部train；style/entry/cut可追溯 |
| 基础 | 全部finite；fps/shape/joint order正确；四元数范数误差≤1e-5 |
| raw骨长 | 全骨架每条bone长度相对rest偏差≤1e-4m；不可用selected24跳级骨替代完整骨架 |
| root/trajectory | 与对应A原轨迹/特征一致，按identity容差验证；不靠后处理覆盖特征 |
| 过渡边界/内部 | position/rotation步差、实际速度/加速度、jerk均不超过冻结对应参考上限；同时报告均值/P95/max |
| raw预定contact | 水平实际速度P95≤0.10m/s、max≤0.20m/s；每个stance最大锚点漂移≤2cm，且受参考上限约束 |
| raw地面proxy | 预定marker不得低于其floor proxy超过1cm；说明是marker proxy，不宣称mesh穿地为零 |
| IK幅度 | 修正≤3cm、单腿关节变化≤10deg；全部区间通过，不只查最后一帧 |
| B后缀保真 | 去除全局XZ/yaw刚体差后，对比原donor：selected24平均pose偏差≤3cm且joint最大≤8cm；root tilt差≤5deg、root height差≤2cm；同时报告完整旋转差 |
| 活动性 | moving时，腿部活动/支撑推进不得塌为静态；使用train同动作/相近速度的P01活动下界，并输出冻结proxy。失败不能靠降noise修复 |
| VQ路径 | 用原norm、冻结VQ重新encode→quantize→decode；连续性/contact/活动性在独立codec参考上限内；不得用raw阈值宣称所有codec误差为零 |

position/rotation的速度一致性需独立从最终世界运动检查；原提取器fk_vel与中心差分会有离散误差，按真实参考误差包络验收，不武断要求两种算法逐bit相同。

活动性最少用两项：①去root后四个Foot/Toe marker的一阶位移RMS；②60帧滑窗内双脚相对前后/左右位置的变化幅度。按suffix和每0.25m/s速度桶取reference P01下界，桶内少于8窗回退该suffix，再不足回退全参考并记录。不用body整体位移代替腿部活动。支撑交替次数另报，不把固定的一秒步数门槛强加给所有人类步态。

B后缀保真比较只能做单次刚体XZ/yaw对齐，不允许逐帧重新配准或伸缩骨架掩盖差异；root height以各自原始地面proxy为基准。原source root检查针对raw合成结果，codec root漂移按下述独立口径评分。

VQ检查包括两种轨迹口径：①decoded局部pose叠在规定source root路径上，隔离pose/接触重建；②按decoded root通道重新积分的原始输出路径，报告累计root漂移及world足滑。两者均需有真实GT经VQ的对应参考，不能只展示较好的一种。

这些硬门槛是保守pilot定义，不是论文标准或自然度证明。若导致大量真实参考/identity失败，说明当前规则/实现不适配，交付校准问题，**不能自动宣布合格或改阈值凑数**。有接受样本也不代表全90类覆盖。

## 8. 缓存与完成状态

每条machine-pass片段至少保存：FP32未归一化X[rows,375]、command_labels、raw_frame_ids、transition区间、valid_window_starts、VQ code indices；可保存精简全骨架local rotations供复查。使用独立NPZ/JSON即可，不冒充原始DB或复用其cache指纹。

manifest必须含：

```text
schema = donor_transition_pilot_v1
complete / quality_status / review_status / train_eligible
seed, source/donor entry/style/label/suffix/raw cut/window
input BVH、DB meta/split/StyleNames、VQ、normalization 内容指纹
root_policy = copy_source_full_root
time_rate = 1, transition_frames, blend/IK/contact参数
raw→feature映射, command_schedule, 窗口eligibility规则
threshold_file_sha256, raw/codec指标, 接受/拒绝原因
device、FP32/TF32、chunk、依赖版本、生成代码版本/哈希
产物文件列表/shape/dtype/hash、用时、大小
```

开始写入`complete=false`，写完校验全部文件后原子更新为true。格式完整与质量合格是两回事；即使生成任务complete，也可能零合格样本。默认`review_status=pending`、`train_eligible=false`：机器检查后仍需用户看自然度。不得替用户将其改成已人工审查。

独立reader拒绝incomplete、文件替换/损坏、指纹不匹配、越过片段边界、未通过机器质量的数据；audit模式可以读取待人工审查片段，但必须显式标明不能自动用于训练。不要修改原`VQTargetCache`来接纳不兼容格式。

## 9. 阶段3：测试与最小实际验证

至少覆盖这些有意义的失败/正确性场景，不写只检查字段存在的表面测试：

1. A=D、alpha端点、单位/坐标/DB-row与raw帧映射正确。
2. 固定落地脚在root平移/转向后实际world速度检查正确；“只替root并修velocity”的旧式样本必须被拒绝。
3. 相反支撑脚、moving对idle、不可达IK、关节大改动、越界上下文各自拒绝且reason明确。
4. IK/插值后所有features重提取；伪造/复用旧velocity不能通过独立一致性检查。
5. command label从预测时刻算；两帧history允许跨命令切换，四个target命令保持一致；无一帧偏移。
6. quantized motion与对应code decode匹配，VQ无梯度且前后权重指纹不变；原normalization不变。
7. zero-contact是不可验收而非零足滑；只有成功锁脚帧合格不能掩盖整个预定stance失败。
8. 拒绝不完整/同路径换内容的cache，固定seed的candidate清单可复现。
9. raw通过但codec失败的fixture正确拒绝；所有pair失败仍生成完整报告，不输出伪造accepted样本。

先执行新增测试，再运行已有相关运动/FK/缓存轻量回归；不要因历史不相关失败而展开重构。真实试跑先2–4对，再按预算完成pilot。没有更新模型参数的真实训练smoke。

## 10. 阶段4：交付与停止条件

交付报告应让用户不用读代码就能判断数据是否值得训练：

- 修改/新增文件清单、准确运行命令、测试结果。
- 候选数、逐阶段拒绝数、machine-pass数、style pair/suffix覆盖；区分pair与时长尝试，不能重复统计。
- 各指标raw/codec/reference对照，全部失败原因，而非只报告最好case。
- 对照视频：A、最终过渡、原B、量化重建，统一相机/时间轴，标明命令切换和过渡区间；显示原始结果，不额外smooth/IK。已有离线构造IK应明确标注。
- 必须包括最差machine-pass和典型reject；视频不足12条按实交付，不能用重复片段充数。渲染不可用时保存可播放BVH+关键帧/轨迹图，明确人工自然度仍待检查。
- 预处理CPU/GPU时间、峰值内存/显存、缓存大小和候选吞吐；不声称训练效率提升或accuracy改善。

全局停止条件：输入/指纹或identity无法验证，无法对应完整骨架，contact/feature坐标约定不清，全部参考校准失败，资源/时间/大小超限。保存已做工作和具体原因后交付，不反复扩大搜索、调高门槛或恢复旧target拼接。

**本任务终点：数据构建代码和小规模可审查产物交付。不是“模型已经work”。** 只有用户确认自然度与覆盖后，才另立任务接入纯TF训练：先小比例采样、同初始化A/B，再决定正式预算。CFG属于独立控制实验，不在本任务内实现。

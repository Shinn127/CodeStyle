> **2026-10-12 checkpoint 清理**：旧 controller checkpoint（含下方提到的 `flowmatch_label_nomirror_150ep`、
> `flowmatch_label_6tok_curriculum100`、`flowmatch_label_history_*`）已删除；入口需 `FLOW_CKPT=<新训练模型>`。
> 下文保留当时的验证记录。
>
> **2026-10-11 状态**：入口已接入无镜像链路（`flowmatch_label_nomirror_150ep/best.pt` +
> `database_100sty_375_memmap_nomirror`），并完成 120 帧 headless 与 Xvfb 窗口复测，见
> [验证记录](../analysis/interactive_label_nomirror/verification.json)。镜像修订前的旧记录仍保留在
> [analysis/interactive_label/](../analysis/interactive_label)，其中引用的旧 checkpoint/DB 已删除。

# Label模型实时交互控制

入口默认使用`checkpoints/flowmatch_label_nomirror_150ep/best.pt`（validation 选型=ep150）及其固定 VQ150ep-nomirror/normalization。逐帧autoregressive生成；原有`evaluation/simulate_realtime_flowmatch_control.py`的离线生成与播放入口保留（默认路径同步切到 nomirror 资产）。

2026-10-12检查：默认不会自动加载新6-token curriculum100模型。新模型虽然style accuracy更高，但噪声抖动/低噪声滑行仍未解决；模型、noise和后处理需分别核对。原因、短探针和训练日志修复见[质量诊断](../analysis/interactive_quality_2026-10-12/REPORT.md)。

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
bash scripts/run_interactive_label_control.sh

# 改初始目标风格；风格名必须在当前90类label集合中
STYLE=Robot bash scripts/run_interactive_label_control.sh
# 自定义兼容label checkpoint；VQ/normalization默认按checkpoint解析并校验
FLOW_CKPT=/absolute/path/to/best.pt bash scripts/run_interactive_label_control.sh
# 立即切换条件，关闭默认0.25秒过渡
bash scripts/run_interactive_label_control.sh --style-blend 0
# 反馈不做量化：连续latent直接decode（默认仍是clip→quantize→decode）
bash scripts/run_interactive_label_control.sh --no-requantize
REQUANTIZE=0 bash scripts/run_interactive_label_control.sh   # 等价写法
```

在本机桌面打开GenoView/Raylib窗口，需要mcc环境已有的`numpy/torch/raylib`及`revision/GenoViewPython-main/resources`。默认CUDA；没有GPU时可设置`DEVICE=cpu`，速度需实测。远程无显示环境使用headless检查，不会启动窗口。

| 操作 | 功能 |
|---|---|
| W / S | 沿人物期望朝向前进 / 后退 |
| A / D | 左 / 右移；斜向输入规范化，避免速度变为√2倍 |
| Q / E | 期望朝向向左 / 右转，默认90deg/s |
| Shift | 1.7倍速度，上限3m/s |
| 右侧Search与列表 | 搜索、点击切换90类seen style；编辑搜索框时禁用运动快捷键 |
| `[` / `]` | 上一个 / 下一个style，循环切换 |
| `-` / `=` 或速度滑块 | 调整移动速度，默认1.2m/s |
| Space | 暂停 / 继续生成 |
| R | 保留当前目标style，重置姿态、根位置、轨迹历史和noise seed |
| 鼠标右键拖动 / 滚轮 | 相机环绕 / 缩放；相机跟随人物根位置 |
| T / F1 | 轨迹 / 骨架叠加开关 |
| P / 点击Post-processing按钮 | 后处理总开关；绿色ON、红色OFF |
| I / 点击Foot IK按钮 | 单独切换脚部IK；仅总开关ON时可操作，切换时清除旧接触状态 |
| ESC | 关闭窗口 |

风格切换保持已有motion、latent和world root状态，默认将旧条件平滑插值到新label条件（0.25秒）；不读取新style的GT、不重置人物或改变VQ。最终条件严格到达目标label feature，中间插值只用于交互过渡，不代表模型学过这些连续style，也不作为正式style评估设置。

右侧状态区持续显示`Post-processing ON/OFF`、`Foot IK ON/OFF/INACTIVE`、`Pose smooth`及当前`Feet locked: 0/2..2/2`，暂停时也可查看。总开关OFF时显示RAW、平滑OFF，IK置灰并标INACTIVE；记住原IK设置，重新开启总开关后恢复。IK ON表示处理分支启用，锁定脚数表示当前检测到的实际接触；0/2不代表IK开关关闭。2026-10-10已修正Geno角色朝+Z时的输入符号：A/D与Q/E反转，W/S保持；数据库坐标转换仍单独处理。

此方向/UI修订的28项针对性测试及三种20-frame窗口状态检查通过，见[验证记录](../analysis/interactive_label_ui/verification.json)、[开启预览](../analysis/interactive_label_ui/on.png)、[关闭预览](../analysis/interactive_label_ui/off.png)。代码更新后需要关闭旧窗口再启动，运行中的Python进程不热加载修改。

## 状态与坐标

初始化仅取Neutral（或`--source-style`）的一帧未镜像train ID/FW motion375。编码得到q，并以D(q)建立匹配的motion/latent；之后不再读取任何数据库motion。Target默认Neutral，可用`--style`或界面选择。

每个60Hz模型步先按上一帧生成的root vx/yaw/vz积分world root，然后构造当前轨迹条件：过去60帧来自实际生成root历史，中心速度使用独立平滑的目标速度，未来50帧由键盘意图平滑外推。中心位置为实际root、中心朝向为root-frame正前方；目标速度不会因实际root暂时静止而始终归零，从而允许起步。采样间隔10帧，12点，按数据库布局拼成`px,pz,dx,dz,vx,vz,speed`七块，共84维；位置/朝向/速度转换到当前生成root frame。Root yaw用deg/s转rad/s积分，归一化使用冻结VQ统计。

Controller接收`当前84维trajectory + 上一帧生成motion291 + 上一帧latent + label condition`，FP32/highest/noTF32、std1；Euler步数默认跟随checkpoint的`solver_steps`（ODE4模型为4，旧模型为8），可用`--ode-steps`/`ODE_STEPS=`显式覆盖。根速度和yaw来自模型；键盘只提供期望轨迹，没有直接覆盖root输出或挪动角色来掩盖控制误差。

反馈有两种模式，窗口右下角与`[Live]`日志的`feedback`字段会标明当前模式：

- **默认`clip→quantize→decode`**（`feedback=clip_quantize_decode`）：每步latent先clip到数据范围再snap到VQ码本，反馈motion严格等于量化latent的decode结果。与正式rollout评估协议一致，用于观察论文指标对应的实际行为（含量化抖动）。
- **`--no-requantize`（`feedback=clip_continuous_decode`）**：跳过码本lookup，clip后的连续latent直接decode，反馈motion与反馈latent仍严格逐个对应（测试断言atol=0）。去掉了每帧量化误差，姿态更平滑，但偏离正式评估协议，仅用于对照观察量化对交互质量的影响；不用于指标或选型。

橙色线是实际root heading，灰色点是生成历史，绿色曲线是期望future trajectory。相机位置不参与控制器条件。数据库coordinate conversion沿用现有GenoView转换，左右/转向输入按显示坐标校正。

模拟步长固定1/60秒，渲染目标60FPS；慢帧最多补两个模型步，并限制积压，避免长时间回放过时输入。实际FPS受设备、显示和模型影响，不能保证每台机器60FPS。模型对停走、横移、急转及某些style的响应质量仍受训练分布与rollout稳定性限制；该界面用于观察这些问题，不修改指标或模型权重。

## 显示后处理（默认启用）

只在生成新模型帧时更新显示滤波，暂停时保持姿态；低渲染FPS下按实际跨过的模型步数计算dt。整个处理位于GenoView坐标转换之后，不写回motion291、latent、planner root/history或noise generator，训练/正式评估也不使用它。

1. **因果自适应平滑**：在local joint quaternion上做最短弧插值，并滤波显示Hips位置。默认基础cutoff6Hz，先对带符号角速度/位移速度作2Hz低通，再根据运动速度提高cutoff；减少量化抖动被误认为快速动作。无未来帧buffer，平滑仍会带来少量视觉滞后；增大cutoff可减少滞后。相机跟随单独作6Hz平滑。
2. **固定骨长FK**：沿Geno模板的固定local offsets重建global pose，避免独立平滑global关节坐标拉长腿或扭曲皮肤。该约束也属于显示修正，会略改变原始decoded joint positions。
3. **接触滞回与脚部IK**：低脚底代理高度、低竖直/水平速度持续约两帧后，锁定heel/ToeBase的world XZ；约83ms逐步启用、约63ms释放。抬脚、跳跃、超过20cm拉扯会释放，快速摆腿不会新建接触。脚部roll沿用生成姿态，解析two-bone IK保持大腿/小腿长度，夹住不可达目标；直腿退化时用脚前方作为膝盖pole，避免膝盖向侧面翻折。
4. **地面与骨盆修正**：用ankle中心8cm、ToeBase中心2cm的近似sole高度减少穿地，并对显示Hips高度作最多±8cm的平滑补偿。不覆盖模型的root速度、yaw或world XZ。
5. **场景**：增强背面填充光/轮廓光，添加轻量脚下blob shadow，增强落地感。P只对照姿态处理，新光照和阴影保留。

这些接触/sole高度是当前Geno平地角色的启发式代理，不是物理碰撞求解；新骨架、斜坡或复杂脚部roll需要重新校准。过强平滑可能削弱快速style细节，接触误判也可能短暂约束脚部，可按P/I对照。

```bash
bash scripts/run_interactive_label_control.sh                       # 默认6Hz + foot IK
bash scripts/run_interactive_label_control.sh --smooth-cutoff 8     # 响应更快、滤波更弱
bash scripts/run_interactive_label_control.sh --no-foot-ik          # 保留平滑/FK
bash scripts/run_interactive_label_control.sh --no-postprocess      # 原始显示姿态
```

`--smooth-cutoff 0`只关闭时间滤波，骨长约束和IK仍按开关生效。Headless模型入口不渲染，显示诊断使用下面的专用短测。

同一seed42、120帧demo中，world joint二阶差分代理5.45→3.08cm，下降43.5%；相同85个完全锁定support pairs（占35.7%）的XZ滑动代理2.30→0.030cm/帧。仅平滑/FK时可增加穿地，所以默认联合IK。脚底高度代理的最大穿地0.75cm降至数值误差范围；不代表整个mesh精确零穿地。这些都是显示诊断，不是论文F.S./Jitter或style accuracy。

后处理平均约2.2—2.3ms/帧，120帧虚拟窗口检查显示60 render FPS；原始model motion、noise状态逐帧不变，world root与修订前demo逐值一致，VQ不变。17项针对性测试通过（交互7项+后处理10项）。记录见[后处理验证](../analysis/interactive_label_postprocess/verification.json)和[预览](../analysis/interactive_label_postprocess/preview.png)。

```bash
python -m pytest tests/test_pose_postprocess.py tests/test_interactive_control.py -q
# 120帧真实模型的显示诊断，不训练；指标定义独立于论文评估。
python analysis/interactive_label_postprocess/check.py
```

## 验证与复测

7项针对性CPU测试通过：轨迹布局、速度/方向一致性、过去与未来采样、静止起步的独立目标速度、WASD/QE符号、root积分与旧motion375实现一致、label切换保持历史、噪声reset可复现、冻结VQ、motion/latent匹配和搜索focus屏蔽输入。

2026-10-11 无镜像链路复测（`flowmatch_label_nomirror_150ep/best.pt`，120-frame headless demo 含移动/转向及两次style切换）：量化路径平均推理4.25ms，P95 4.53ms；AE连续路径（`--no-requantize`）4.19ms，P95 4.75ms，两路径反馈motion均与反馈latent严格一致。Xvfb窗口120-frame渲染通过（60 render FPS，量化路径平均推理5.22ms、AE路径4.56ms；显示后处理约2.24ms/帧），VQ状态不变，checkpoint指纹与nomirror DB `segments.json`一致。针对性测试`tests/test_interactive_control.py tests/test_pose_postprocess.py` 29项通过（含AE路径新测试）。检查记录见[复测记录](../analysis/interactive_label_nomirror/verification.json)、[量化路径截图](../analysis/interactive_label_nomirror/gui.png)、[AE路径截图](../analysis/interactive_label_nomirror/gui_ae.png)。镜像修订前的旧记录（旧checkpoint）保留在[analysis/interactive_label/](../analysis/interactive_label)。均为当前设备的短测，不能作为本机长期/所有设备FPS保证；没有训练或完整benchmark。

```bash
DRY_RUN=1 bash scripts/run_interactive_label_control.sh
# 有界真实模型检查，不开窗口、不训练
bash scripts/run_interactive_label_control.sh --headless --demo --max-frames 120 \
  --report-json /tmp/interactive_label_check.json
# 有界窗口检查并保存截图到指定绝对路径
bash scripts/run_interactive_label_control.sh --demo --max-frames 120 \
  --screenshot /tmp/interactive_label.png --report-json /tmp/interactive_label_gui.json
python -m pytest tests/test_interactive_control.py tests/test_pose_postprocess.py -q
```

`--demo`是脚本化输入检查，正常交互运行不要设置它。这个入口仅支持label模型的seen styles；clip/recipe与OOD条件尚未接入此交互窗口。

2026-10-12新增history模型支持：自动读取checkpoint的两帧latent历史配置，切换style保留真实反馈history。`--smooth-plan`显式开启按键目标速度/转向平滑，默认关闭以保留旧对照；后处理/IK状态还显示contact与full blend数量，开关ON不代表当前有锁定脚。IK接触速度经过滤波，同时保留原始速度防滑行误锁。新模型训练后用 `FLOW_CKPT=checkpoints/<your_run>/best.pt bash scripts/run_interactive_label_control.sh --smooth-plan` 切换（原示例指向的 history 短训 checkpoint 已在 2026-10-12 清理中删除）。实现、测试与短训计划见 [修订说明](MOTION_HISTORY_REPAIR.md)。

# VQ码本跳变：独立审计、修订与实测

2026-10-14。**代码存在确定缺陷，已修复；当前K2048 pilot不能视为运动跳变问题已经解决。** 它的重建和同帧往返码身份有所改善，但固定validation样本的量化AE jitter/world足滑退化。不能用raw margin上升、换码率下降替代运动质量验收。

本报告与同目录另行更新的REVIEW独立，以下数字均对应本报告链接的JSON和日志。其他split、seed或epoch的结果不能拼接比较。本轮未启动新正式训练；仅执行fixture测试、冻结模型探针及无optimizer更新的微基准。

## 1. 先区分三个问题

1. 训练中EMA码本向量变化：本库32个slot共享2048×4的EMA表。没有Parameter不等于冻结；train forward会更新EMA或替换死码，eval才冻结。
2. 同帧codec往返换码：Q(E([同一trajectory,D(q)]))是否回到q，用于表征codec一致性。
3. 相邻时间或controller生成码变化：真实运动本来就频繁换码，需检查解码运动与GT时间变化，而不是仅压换码率。

## 2. 已修复的问题

### 最近边界几何

到c1/ck二等分平面的距离为 `r_k=(||z-ck||²-||z-c1||²)/(2||ck-c1||)`；真正的胞元安全半径为所有不同位置竞争码的min r_k。**欧氏次近码未必给出最近平面。**

独立反例：c1=(0,0)、c2=(.2,0)、c3=(0,1)、z=(.09,.495)。c2为次近码，其平面距离.010；c3平面仅.005。旧实现只取c2，高估安全半径。

新增`codebook_boundary_distance`，默认`nearest_boundary`；no_grad搜索全部竞争平面，梯度只经两个选中向量的距离。重复位置码不定义边界；单一码/全重复码返回valid=false、有限零损失，不伪造可实现的margin。

保留`nearest_code`显式回放旧近似。trainer/wrapper传模式，新exact配置记录vq_stability_version=2；旧checkpoint缺失mode时按nearest_code解释。resume拒绝静默更换几何/目标/辅助权重，切换目标需init另开run。

注意：min-plane hinge等于所有plane hinge的最大值。如果该hinge为0，所有不同码的平面都满足余量，并不是“只约束一个平面所以漏掉其他面”。是否额外使用top-k hinge是另一种实验目标，不能据此直接宣布必要。

### EMA与roundtrip

库源码与测试确认：本次quantized值先按pre-EMA表gather，然后更新EMA。新margin以当前post-EMA表监督encoder；RT目标仍为实际解码的pre-EMA quantized向量并detach，二者时序已明确。

roundtrip正确训练encoder/decoder，且不额外调用quantizer更新第二次EMA。但这是**同帧cycle**。现有controller直接携带feedback q作为prev_latent，不在每步重新E(D(q))；RT保持率不是controller下一帧码准确率的上限。相关代码/CLI注释已改正。

### 初始化、快速量化和CE

- 同K init现在严格要求完整quantizer buffer，旧实现无条件忽略quantizer missing，可能静默留下不完整EMA状态。仅实际tensor的K改变才丢弃整套EMA状态；不再因args缺失K而误重置。
- init/resume互斥，非法辅助权重/target拒绝；新run不能覆盖已有config/latest。
- 分head码本的快速量化fallback原把[B,H,D]直接reshape成[H,B,D]，已改成permute并恢复batch顺序。现役shared-table不走该分支。
- nearest-code/margin/CE搜索显式禁止外层autocast，保持FP32；`.float()`本身不能阻止autocast降低matmul精度。
- CE温度缓存使用storage pointer/version/shape/device，不依赖临时detach tensor的Python id；重复码不再将auto temperature压到1e-12。非法shape/chunk和NaN温度拒绝。
- code CE监督随机FM时刻的target endpoint，不是单独Euler积分的最终输出。测试文案已更正，独立Euler输出反例已验证。
- wrapper新增真正DRY_RUN，避免误开训练；辅助权重0保留旧损失惰性。

CE温度仍是启发式：码对距离l、T=(l/2)²时，中心的两类logit gap为4，中点为0。CE可能把预测推出中心，不保证回归误差下降。

## 3. 需要撤回的分析推断

更新探针`vq_anatomy.py`：默认val，显式DB/norm/seed，真实/旧近似边界并报，相对余量、指纹与实际帧数落盘；motion质量复用现有motion_eval。

- 原recon_cm_mean把291维混合MAE统一乘100，其中含rotation与velocity，**不是合法cm**。现仅positions算local MPJPE，另报world MPJPE/root ADE/jitter/足滑。旧3.11等混合值不能作MPJPE或质量地板。
- σ=.077是逐维Gaussian标准差；4维槽噪声RMS L2为sqrt(4)*σ=.154，不能等同于实测槽L2中位数.077。即使匹配误差尺度，各向同性注噪曲线也不是任意controller的硬上限。
- n-gram/1NN是有限基线，不是信息论oracle；RT保持率不能证明GT码复现“物理不可能”。
- post-hoc合并结果不是重训结果保证上界；减少K不是唯一杠杆，少换码还可能带来更大单次解码跳幅。
- quantized AE是当前编码路线的参考，不是任何controller误差的数学下界。
- “运动层面良性”“已排除量化影响”等判断需要实际运动对照；平滑衰减曲线也不能排除所有代码bug。

此前test片段已用于方法诊断，不能再称完全未触碰的封存集。本轮新比较仅用固定validation，不重划split、不用test选模型。

## 4. 固定validation对照

seed42，12个source entries，每clip121帧/120区间，总1452帧。FP32/highest/无TF32；两模型case与norm一致，VQ状态前后不变。原150epoch run的best实际为epoch90，pilot best为epoch25、latest为30。

| 指标 | 原VQ best | margin+RT pilot best |
|---|---:|---:|
| 真实最近边界median，raw latent | .0103 | .0298 |
| 相对余量median：r(z)/(0.5×目标码最近不同码间距) | .2989 | .2880 |
| 旧近似高估边界的槽比例 | 12.64% | 11.33% |
| 同帧codec往返换码率 | 62.60% | 58.64% |
| GT相邻帧每槽自然换码率 | 56.08% | 57.58% |
| Quantized AE world MPJPE | 1.579cm | 1.206cm |
| Quantized AE jitter | .412cm | .507cm |
| Quantized AE world足滑 | .488cm | .522cm |
| Continuous AE jitter | .120cm | .114cm |
| Continuous AE world足滑 | .207cm | .168cm |
| 同一GT jitter / world足滑 | .092 / .091cm | 同左 |

结果：[baseline_val12.json](audit/baseline_val12.json)、[pilot25_val12.json](audit/pilot25_val12.json)。local/world定义不可混用；这里的codec cycle也不是AR rollout。

raw距离可被尺度变换改变：E'=aE、C'=aC、D'(z)=D(z/a)不改变码身份/运动，却放大raw margin。raw RT MSE也有尺度依赖。因此absolute bd增长不等于相对抗扰提高；当前数据提示尺度因素，不能断言pilot只是纯缩放。

pilot重建/往返身份有改善，但量化jitter与world足滑变差；continuous路径仍更平滑。**不支持现在就推广pilot作为新主VQ。** 两aux同时启用且额外训练25epoch，没有同预算继续训练基线，不能把变化分别因果归给margin或RT。样本小、无多seed，尚非完整style/controller评估。

此次pilot沿用旧nearest-code几何，本轮修复没有参与这30epoch；不能拿它评价新exact目标的训练收益。

## 5. 测试和开销

独立oracle验证值与梯度；包含反例、重复/单一码、scale反例、margin-only encoder梯度、RT双网络梯度、EMA一次及pre/post时序、缺buffer拒绝、K变化初始化、模式resume、真实微型VQ CLI、温度缓存、外层autocast和CUDA/CPU一致性。

最终回归：**395 CPU passed、1 skipped；21 CUDA passed**。日志见`audit/cpu_tests.log`与`audit/gpu_tests.log`。这是代码/接口验证，不是修订后训练效果证明。

RTX5070Ti、B512×T2，FP32，保留training STE图但冻结EMA，10次计时前向/反向，无optimizer更新：baseline23.30ms、旧margin+RT27.63ms、新exact margin+RT34.31ms；allocated峰值约1.63GiB。**不是epoch耗时**，不能沿用“净成本被抵消”的说法。

发现低维direct-cdist核在K2048/D4约15ms，已换为分块broadcast norm，保留零距离别名与精度；pair差临时张量限制约64MiB。搜索no_grad，反向仅两次gather。基准见`audit/benchmark_auxiliaries.py`和`audit/performance.json`。

一次真实batch32的encoder梯度范数：baseline7.12、margin .48、RT .19；RT decoder .27。只代表这一个batch，说明不能用loss数值占比代替梯度占比。

## 6. 后续建议与边界

1. 原VQ+原controller仍是有效配套基线。即使K仍2048，新encoder/decoder/码向量已变，不能用override偷换进旧controller；新训练前须重建targets、train-only latent stats与style prior，但当前不要先花费全链重建成本。
2. 先做同初始化、同updates的单因素小对照；同时看相对余量、RT flip、decoded jitter/contact。相对margin或尺度归一化RT属于新实验，本轮未自动启用。
3. 当前seq_len2/acc_weight0不直接监督二阶时间误差。不能为降换码率自动加平滑、acc或persistent noise，避免静止滑行；需要时另立受控实验。
4. VQ通用resume主要恢复model/optimizer/scheduler，未像controller入口完整保存恢复Python/NumPy/CPU/CUDA RNG。本轮新增的是目标兼容检查，**不声称任意VQ epoch续训位级可复现**。

后续新geometry run用`LATENT_MARGIN_MODE=nearest_boundary`及新OUT。旧pilot同目标续训需显式`LATENT_MARGIN_MODE=nearest_code`并保持原aux权重；改目标用INIT_CKPT另开run。本轮未执行这些训练。

### 原pilot的shell尾部异常

本轮在其bash等待Python时改了wrapper，训练完成后出现`ick: command not found`。这是审计操作导致的shell读取偏移，并非Python训练失败。latest30/best25均已加载验证，未覆写或重跑；当前wrapper语法和dry-run通过。原日志保留异常，不能把外层shell exit称为干净成功。

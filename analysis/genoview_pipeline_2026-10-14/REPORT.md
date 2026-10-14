# 交互视口接入完整 GenoView 渲染管线（含共享模块抽取、正确性检查与效率提升）

2026-10-14。目标：把交互式 label 视口从"轻量内联 shader 渲染"换成 **GenoView 完整延迟管线**
（shadow map → G-Buffer → SSAO → 可分离 blur → lighting → FXAA，mesh 地面 + 真实阴影），
并在此过程中消除 `genoview_backend.py` 里"同一渲染链复制两遍"的重复实现。

## 1. 实现结构

| 文件 | 变更 |
|---|---|
| `Visualization/backends/genoview_pipeline.py`（新增） | `GenoViewPipeline`：拥有完整链的全部资源（8 个 shader、mesh 地面、1024² shadow map、G-Buffer、3 张全屏 RT）、缓存全部 cdata/Vector/Rectangle 对象、一次性上传恒定量 uniform；`render(camera, models, shadow_focus, overlay)` 画完整帧（含 FXAA），`close()` 释放资源 |
| `Visualization/backends/genoview_backend.py` | 双人对比循环与单人预览循环**都改为调用 `GenoViewPipeline`**；删除约 300 行重复的 shader/uniform/资源代码；保留原对外 API（`visualize_dual_motion375_genoview_from_375`、`main`、`_load_genoview_module`、`GENOVIEW_RESOURCES_DIR` 等导入路径不变） |
| `Visualization/cli/interactive_label_control.py` | 新增 `--full-pipeline`（**默认开**，`--no-full-pipeline` 回退轻量渲染）；pipeline 模式下不再编译/上传内联 shader；overlay（骨架/轨迹/朝向线）通过回调进入 lighting 的 3D 段，深度测试+FAA 与 GenoView 一致；诊断 JSON 增加 `render.full_pipeline` |
| `tests/test_interactive_control.py` | 新增 `--full-pipeline` 默认值与 `--no-full-pipeline` 解析测试 |

设计要点：

- **overlay 语义不变**：骨架/轨迹/朝向线在 lighting 之后、FXAA 之前以 3D 方式绘制
  （与 genoview_backend 原做法一致），因此被深度正确遮挡并接受 FXAA 抗锯齿；
- **地面中心跟随角色**（`shadow_focus`），阴影相机每帧对准人物 root（沿用原 GenoView 做法）；
- **交互特有内容保留**：显示后处理（平滑/IK）、轨迹、脚下软圆柱阴影在 pipeline 模式下
  仍由后处理/overlay 提供；pipeline 自带真实投影阴影叠加在地面上。

## 2. 正确性检查（代码检查）

### 2.1 分路径运行验证（隐藏窗口 + 截图/录制，全部通过）

| 路径 | 验证方式 | 结果 |
|---|---|---|
| 交互视口 · 全管线 | `--hidden --reel --max-frames 90/720 --record` | ✅ exit 0，画面=棋盘地面+SSAO 明暗+角色软阴影（与原版 GenoView 参考帧逐项对照一致） |
| 交互视口 · 轻量回退 | `--hidden --reel --no-full-pipeline --max-frames 40` | ✅ exit 0，画面与旧版一致（无阴影、扁平地面） |
| 双人对比循环 | 隐藏窗口 31 帧 + `screenshots` 截图 | ✅ exit 0，画面=蓝色参考/橙色预测双角色、骨架与轨迹叠加、阴影正常 |
| 单人预览循环 | 隐藏窗口 21 帧 | ✅ exit 0 |

### 2.2 逻辑不变量

- 全管线模式下模型材质不再绑定内联 shader（由 pipeline 按 pass 设置）→ 无"半新半旧"状态；
- 常量 uniform（specularity/glossiness、clip 距离、太阳/天空/地面/环境强度、曝光、阴影分辨率倒数）
  在 `__init__` 上传一次并在每帧生效验证（画面光照与旧实现一致）；
- 帧内矩阵（view/proj/inv/lvp）与相机位置每帧上传，验证了相机联动正常（视频中相机跟随流畅）；
- `close()` 按 shader→RT→GBuffer→shadow map→ground 顺序释放，退出无 GL 报错。

### 2.3 发现并修复的既有缺陷

1. **双人对比截图路径失效**：`TakeScreenshot` 在 `os.chdir(GENOVIEW_DIR)` 后把绝对路径塌缩为
   基名，截图被写到 `GenoViewPython-main/` 目录而非调用者指定路径。已改为
   `LoadImageFromScreen` + `ExportImage`（与交互录制同一做法），修复后截图落在指定路径。
2. **交互默认渲染与 GenoView 不一致**（本任务本体）：现已统一。
3. 交互管线初始化时 `SetShaderValue` 对 `Vector2` 传值报 ctype 错误（抽取过程中引入并当场修复：
   统一用 `ffi.addressof`）。

### 2.4 回归

- `bash scripts/run_tests.sh cpu`：**368 passed**（含新增解析测试），20 deselected；
- 三条渲染路径独立冒烟（见 2.1）；`py_compile` 全部改动文件。

## 3. 效率检查与提升

### 3.1 已实施的优化（相对被替换的原始实现）

| 项 | 原实现（每帧） | 新实现 |
|---|---|---|
| cdata 分配 | ~6 个 `ffi.new("float*")` + 多个 `Vector2/Rectangle`/ffi 指针（lighting 内又新建 5 个） | **0**（全部预建缓存） |
| 恒定量 uniform 上传 | 每帧每 pass 重传 specularity/glossiness/clip 距离/太阳天空强度/曝光等 | **一次性**（仅 frame 内矩阵/相机位置每帧上传） |
| 重复 uniform 设置 | 部分 uniform 同帧内重复设置 2–3 次 | 每次渲染仅必要的 ~25 次调用 |
| 代码重复 | 完整链在 backend 复制 2 份（±交互将是 3 份） | **1 份共享实现** |

### 3.2 实测（RTX 5070 Ti，1280×800，隐藏窗口）

| 指标 | 轻量单 pass | 完整管线 |
|---|---:|---:|
| 持续绘制吞吐（无同步，500 帧） | 0.047 ms/帧（~21k FPS 上限） | 0.099 ms/帧（~10k FPS 上限） |
| CPU 编码（无同步，120 帧） | — | 0.101 ms/帧 |
| 含 GPU 同步的单帧 | 0.072 ms | 0.246 ms |

读法：完整管线的 CPU 编码开销 **0.10 ms/帧（60FPS 帧预算的 0.6%）**，GPU 侧多出
5 个全屏 pass（shadow/G-Buffer/SSAO/两个 blur/lighting/FXAA）共约 +0.17 ms/帧。
**端到端实测交互+录制的持续渲染为 59–89 FPS（录制 12 秒视频全程 ≥59 FPS，目标 60）**，
说明完整管线引入的成本在交互场景完全可接受；旧实现的每帧分配模式实测固定成本约
0.013 ms/帧 + 重复上传，已全部消除。

### 3.3 结论

交互视口渲染性能瓶颈仍是**模型推理（5.2–6.2 ms/帧）+ 显示后处理（2.2–2.6 ms/帧）**，
渲染链不是瓶颈；完整管线在 60 FPS 目标下留有余量。视觉质量升级为 GenoView 同款
（软阴影、SSAO、FXAA、mesh 地面），且三处调用统一到一个模块，后续维护/优化单点生效。

## 4. 验证产物

- 视频：`analysis/genoview_pipeline_2026-10-14/reel_full_pipeline.mp4`（12s/720 帧，
  全管线+录制，6 层 FiLM 模型驱动，风格序列 Robot→Old→HighKnees→ArmsAboveHead→Neutral）
- 报告 JSON：同目录 `reel_full_pipeline.json`（含 `render.full_pipeline=true`、录制帧数、渲染诊断）
- 日志：同目录 `reel_full_pipeline.log`（59 FPS 持续记录）

## 5. 使用

```bash
cd /home/shinn/桌面/multimedia/revision/NewCode
# 交互（默认完整管线）
FLOW_CKPT=$PWD/checkpoints/flowmatch_label_history_obsnoise_tf10_l6_film/best.pt \
  bash scripts/run_interactive_label_control.sh
# 回退轻量渲染
FLOW_CKPT=... bash scripts/run_interactive_label_control.sh --no-full-pipeline
# 离屏录制（全管线）
DISPLAY=:0 FLOW_CKPT=... bash scripts/run_interactive_label_control.sh \
  --hidden --reel --max-frames 720 --record out.mp4
```

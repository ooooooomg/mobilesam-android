# 实时分割 · MobileSAM Android App

在 Android 手机上用 MobileSAM / YOLO-seg 实现实时物体分割：调用摄像头，自动检测画面中的物体并用半透明蒙版标注，支持**多模型切换**、**前后摄像头**、**相册分割**与**分割画面保存**。

推理引擎为 **ONNX Runtime（CPU）**，界面为**靛蓝浅色 / 深色双主题**，跟随系统深浅自动切换。

## 功能

- 📷 **相机实时分割**：CameraX 实时预览，分割蒙版 + 检测框叠加（浅色取景器，圆角取景框）
- 🔄 **多模型切换**：点击顶部模型名，5 个模型一键切换（见下方模型表）
- 📦 **前后摄像头**：一键翻转
- 💾 **保存分割画面**：底部保存按钮，当前分割帧存到系统相册
- 🖼️ **相册分割**：选一张照片 → 自动检测 → 分割 → 显示蒙版
- 🏷️ **弹幕式类别图例**：识别到的物体类别以"● 中文名"弹幕胶囊形式显示，带滞留与上飘动画
- ⚙️ **设置 · 模型详情**：全部模型的分组卡片 + 独立详情页（架构 / 参数量 / 精度 / 计算量 / 适用场景 / 论文 / 源码链接）
- 🌗 **双主题**：浅色纯白 + 深靛蓝强调色；深色近黑 + 亮靛蓝，跟随系统

## 模型表

| 名称 | 副标 | 类型 | 精度 | 速度 | 说明 |
|---|---|---|---|---|---|
| 智能检测 | YOLO11n / 384 | 纯检测 | FP32 | 轻量/快速 | 仅方框，无蒙版，最快 |
| 快速分割 | YOLO11n / 384 | 端到端 | FP32 | 轻量/快速 | 单模型一步分割，均衡 |
| 快速分割/节能 | YOLO11n / 384 / INT8 | 端到端 | INT8 | 最省电 | INT8 量化，更省电 |
| 高清分割 | YOLO11n / 640 | 端到端 | FP32 | 高精度 | 640 高分辨率，蒙版精细 |
| 极致分割 | SAM / 384 | 两段式 | FP32 | 高精度 | YOLO 找框 + MobileSAM 蒙版 |

## 环境要求

- Android Studio（Koala+，含 Android SDK）
- Android SDK：`platforms;android-35`、`build-tools;35.0.0`、`platform-tools`
- 真机（API 26+）或模拟器

## 快速开始

1. 克隆仓库，用 Android Studio 打开
2. 等待 Gradle 同步（首次下载 onnxruntime-android 等依赖，需几分钟）
3. 连真机（开 USB 调试）或启动模拟器
4. Run → 安装到设备 → 打开即进入相机实时分割

> 首次启动会请求相机权限；点击顶部模型名可切换模型，底部三个按钮分别是相册 / 保存 / 设置。

## 模型文件（已包含在 assets/）

| 文件 | 大小 | 说明 |
|---|---|---|
| `mobile_sam_encoder.onnx` | 27MB | MobileSAM TinyViT 图像编码器 |
| `mobile_sam_decoder.onnx` | 16MB | MobileSAM mask decoder |
| `yolov8n.onnx` | 13MB | YOLOv8n 检测器（两段式 prompt 用） |
| `yolo11n-384.onnx` | 11MB | YOLO11n 纯检测（384） |
| `yolo11n-seg-384.onnx` | 12MB | YOLO11n-seg 分割（384） |
| `yolo11n-seg-384-int8.onnx` | 3.4MB | YOLO11n-seg 分割（384，INT8 量化） |
| `yolo11n-seg.onnx` | 12MB | YOLO11n-seg 分割（640） |

## 代码结构

```
app/src/main/java/com/example/mobilesam/
├── CameraActivity.kt      # 相机实时分割（CameraX + 叠加渲染 + 弹幕图例）
├── MainActivity.kt        # 相册图片分割
├── InferencePipeline.kt   # 可插拔 Segmenter 编排 + Result
├── Segmenter.kt           # Segmenter 接口（分段流 / 单张）
├── SegmenterFactory.kt    # 按 ModelInfo 构建各类型 Segmenter
├── TwoStageSegmenter.kt   # MobileSAM 两段式（YOLO→encoder→decoder，encoder 复用）
├── YoloSegSegmenter.kt    # 端到端 YOLO-seg（box-local 蒙版重建）
├── YoloDetectSegmenter.kt # 纯检测（仅方框）
├── YoloDetector.kt        # YOLO ONNX 检测 + NMS
├── DetPostprocess.kt      # letterbox / IoU / NMS 共享工具
├── SamImageEncoder.kt     # MobileSAM encoder ONNX
├── SamMaskDecoder.kt      # MobileSAM decoder ONNX（box prompt → 掩码）
├── MaskComposer.kt        # 掩码/框叠加渲染
├── ModelRegistry.kt       # 模型注册表（名称/路径/参数/详情/链接）
├── ModelPickerDialog.kt   # 居中 70% 模型选择弹窗
├── ModelDetailActivity.kt # 模型详情独立页（含论文/源码链接）
├── SettingsActivity.kt    # 设置页（分组卡片列表）
├── ImageUtils.kt          # YUV→RGB（整数运算）
├── CocoLabels.kt          # COCO 80 类 + 中文映射
└── RoundedFrameLayout.kt  # 圆角取景框
```

## 性能说明

- 相机分析流 480×360，`STRATEGY_KEEP_ONLY_LATEST` 单线程
- 两段式（极致分割）：encoder 每 6 帧重算一次并缓存 embedding，YOLO 每帧跟踪
- 端到端（快速/高清分割）：单模型单次推理
- 纯检测（智能检测）：仅检测无分割，帧率最高
- 优化点：YUV→RGB 整数运算、mask 重建 anchor 哈希查找、buffer 复用
- ONNX Runtime：`setIntraOpNumThreads(4)` / `setInterOpNumThreads(1)` / `OptLevel.ALL_OPT`

## 许可证

- **本 app 代码**：Apache-2.0
- **MobileSAM**：Apache-2.0
- **YOLOv8n / YOLO11n**：AGPL-3.0 —— 若用于商用需注意 AGPL 传染性，或替换为其他检测器（如 YOLO-Fastest, Apache-2.0）

模型文件版权归各自上游项目所有。

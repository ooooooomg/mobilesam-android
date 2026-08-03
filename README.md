# MobileSAM Android App

在 Android 手机上用 MobileSAM 实现实时物体分割：调用摄像头，自动检测画面中的物体并用半透明蒙版标注。

基于 **MobileSAM**（TinyViT 图像编码器）做分割，配合 **YOLOv8n** 检测器自动生成物体框作为 prompt，推理引擎为 **ONNX Runtime**。

## 功能

- 📷 **相机实时分割**：摄像头画面实时叠加物体蒙版（CameraX）
- 🖼️ **相册分割**：选一张照片 → 自动检测 → 分割 → 显示蒙版
- ⚡ **多分辨率**：512px 编码器（快）+ 可选 384px 编码器（更快）

## 环境要求

- Android Studio（Koala+，含 Android SDK）
- Android SDK：`platforms;android-35`、`build-tools;35.0.0`、`platform-tools`
- 真机（API 26+）或模拟器

## 快速开始

1. 克隆仓库，用 Android Studio 打开
2. 等待 Gradle 同步（首次下载 onnxruntime-android 等依赖，需几分钟）
3. 连真机（开 USB 调试）或启动模拟器
4. Run → 安装到设备 → 打开即进入相机实时分割

> 首次启动会请求相机权限；点「选择图片」可切换相册模式。

## 模型文件（已包含在 assets/）

| 文件 | 说明 | 来源 |
|---|---|---|
| `mobile_sam_encoder.onnx` (27MB) | MobileSAM TinyViT 图像编码器 | MobileSAM 官方权重导出 |
| `mobile_sam_decoder.onnx` (16MB) | MobileSAM mask decoder | MobileSAM 官方权重导出 |
| `yolov8n.onnx` (13MB) | YOLOv8n 物体检测器 | Ultralytics 导出 |

重新生成模型：
```bash
# MobileSAM 编码器（512 或 384）
PYTHONPATH=. python scripts/export_encoder.py --checkpoint mobile_sam.pt \
  --output mobile_sam_encoder.onnx --image-size 512
# YOLOv8n
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=17)"
```

## 代码结构

```
app/src/main/java/com/example/mobilesam/
├── CameraActivity.kt     # 相机实时分割（CameraX + 叠加渲染）
├── MainActivity.kt       # 相册图片分割
├── InferencePipeline.kt  # YOLO→encoder→decoder→overlay 编排（含 encoder 复用）
├── YoloDetector.kt       # YOLOv8n ONNX 检测 + NMS
├── SamImageEncoder.kt    # MobileSAM encoder ONNX（ResizeLongestSide 预处理）
├── SamMaskDecoder.kt     # MobileSAM decoder ONNX（box prompt → 掩码）
└── MaskComposer.kt       # 掩码/框叠加渲染
```

预处理链：
- YOLO：letterbox + NMS（conf 0.35, iou 0.4）
- Encoder：ResizeLongestSide + normalize + pad
- Decoder：box prompt，`mask_input` 全零

## 性能说明

- 相机模式默认用 **512px 编码器** + encoder 复用（每 6 帧重算），手机 CPU 上约 3-6 fps
- 帧率瓶颈在 MobileSAM 编码器（手机 CPU 算力有限）；更高帧率需 GPU/NPU 或更轻模型
- 已测试：NNAPI（`addNnapi()`）在部分设备初始化极慢，默认回退 CPU

## 许可证

- **本 app 代码**：Apache-2.0
- **MobileSAM**：Apache-2.0
- **YOLOv8n**：AGPL-3.0 —— 若用于商用需注意 AGPL 传染性，或替换为其他检测器（如 YOLO-Fastest, Apache-2.0）

模型文件版权归各自上游项目所有。

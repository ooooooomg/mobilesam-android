# MobileSAM Android App

Real-time object segmentation on Android phones using MobileSAM: capture with the camera, automatically detect objects in the frame, and mark them with semi-transparent masks.

Built on **MobileSAM** (TinyViT image encoder) for segmentation, with a **YOLOv8n** detector that generates object boxes as prompts. Inference engine: **ONNX Runtime**.

## Features

- 📷 **Real-time camera segmentation**: camera feed with object masks overlaid live (CameraX)
- 🖼️ **Gallery segmentation**: pick a photo → auto-detect → segment → show masks
- ⚡ **Multiple resolutions**: 512px encoder (fast) + optional 384px encoder (faster)

## Requirements

- Android Studio (Koala+, with Android SDK)
- Android SDK: `platforms;android-35`, `build-tools;35.0.0`, `platform-tools`
- A physical device (API 26+) or an emulator

## Quick Start

1. Clone the repo and open it in Android Studio
2. Wait for Gradle sync (first run downloads onnxruntime-android and other dependencies, takes a few minutes)
3. Connect a physical device (USB debugging enabled) or start an emulator
4. Run → install to device → open, and you are in real-time camera segmentation

> The first launch requests camera permission; tap "Select Image" to switch to gallery mode.

## Model Files (included in assets/)

| File | Description | Source |
|---|---|---|
| `mobile_sam_encoder.onnx` (27MB) | MobileSAM TinyViT image encoder | Exported from official MobileSAM weights |
| `mobile_sam_decoder.onnx` (16MB) | MobileSAM mask decoder | Exported from official MobileSAM weights |
| `yolov8n.onnx` (13MB) | YOLOv8n object detector | Exported via Ultralytics |

Regenerate the models:
```bash
# MobileSAM encoder (512 or 384)
PYTHONPATH=. python scripts/export_encoder.py --checkpoint mobile_sam.pt \
  --output mobile_sam_encoder.onnx --image-size 512
# YOLOv8n
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=17)"
```

## Code Structure

```
app/src/main/java/com/example/mobilesam/
├── CameraActivity.kt     # Real-time camera segmentation (CameraX + overlay rendering)
├── MainActivity.kt       # Gallery image segmentation
├── InferencePipeline.kt  # YOLO→encoder→decoder→overlay orchestration (with encoder reuse)
├── YoloDetector.kt       # YOLOv8n ONNX detection + NMS
├── SamImageEncoder.kt    # MobileSAM encoder ONNX (ResizeLongestSide preprocessing)
├── SamMaskDecoder.kt     # MobileSAM decoder ONNX (box prompt → mask)
└── MaskComposer.kt       # Mask/box overlay rendering
```

Preprocessing chain:
- YOLO: letterbox + NMS (conf 0.35, iou 0.4)
- Encoder: ResizeLongestSide + normalize + pad
- Decoder: box prompt, zero `mask_input`

## Performance Notes

- Camera mode defaults to the **512px encoder** with encoder reuse (recomputed every 6 frames); roughly 3-6 fps on phone CPU
- The frame-rate bottleneck is the MobileSAM encoder (limited phone CPU throughput); higher FPS needs GPU/NPU or a lighter model
- Tested: NNAPI (`addNnapi()`) initializes very slowly on some devices; defaults to CPU fallback

## Limitations & Known Issues

- **Limited frame rate**: ~3-6 fps on phone CPU. The bottleneck is the MobileSAM encoder (computational cost of 512px input); higher FPS requires GPU/NPU acceleration or a lighter segmentation model.
- **Accuracy trade-off**: the 512px encoder is chosen for real-time use; segmentation detail is below that of 1024px input.
- **Detection depends on YOLOv8n**: recall is limited for small objects and crowded scenes; a higher-accuracy detector or a lower confidence threshold helps, but at the cost of frame rate.
- **Platform adaptation**: tested on MediaTek (Mali GPU) devices; NNAPI behavior on other SoCs (Qualcomm/Kirin) may differ, and CPU fallback is used by default for stability.

## Contact

This project is under active improvement. For suggestions, bug reports, or collaboration, contact the author:

**AshMe** — <AshMe37@outlook.com>

## License

- **App code**: Apache-2.0
- **MobileSAM**: Apache-2.0
- **YOLOv8n**: AGPL-3.0 — note the AGPL copyleft for commercial use, or replace with another detector (e.g., YOLO-Fastest, Apache-2.0)

Model files remain copyrighted by their respective upstream projects.

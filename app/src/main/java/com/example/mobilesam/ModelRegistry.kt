package com.example.mobilesam

/**
 * Data-driven registry of switchable segmentation models. Adding a model is:
 * 1) drop the ONNX file(s) under assets/, 2) add one entry here.
 */
object ModelRegistry {

    enum class Type { TWOSTAGE, ENDTOEND, DETECT_ONLY, PICODET }

    data class ModelInfo(
        val id: String,
        val friendlyName: String,   // 通俗主名,如 "快速分割"
        val subLabel: String,       // 技术副标,如 "YOLO11n / 384"
        val type: Type,
        val typeLabel: String,      // "端到端" / "两段式" / "纯检测"
        val precision: String,      // "FP32" / "INT8"
        val speed: String,          // "轻量/快速" / "最省电" / "高精度"
        val desc: String = "",      // 一句话介绍
        val architecture: String = "", // 架构概述
        val params: String = "",    // 参数量(近似)
        val map: String = "",       // 精度指标(近似)
        val flops: String = "",     // 计算量(近似)
        val scenarios: String = "", // 适用场景
        val paperUrl: String = "",  // 原始论文链接
        val githubUrl: String = "", // GitHub 项目链接
        val modelAssetPath: String? = null,      // ENDTOEND: single model
        val detectorAssetPath: String? = null,   // TWOSTAGE: YOLO
        val encoderAssetPath: String? = null,    // TWOSTAGE: image encoder
        val decoderAssetPath: String? = null,    // TWOSTAGE: mask decoder
        val inputSize: Int,
        val conf: Float = 0.35f,
        val nms: Float = 0.4f,
    ) {
        /** One-line path summary used by the settings cards. */
        fun pathSummary(): String = when (type) {
            Type.ENDTOEND, Type.DETECT_ONLY, Type.PICODET -> "模型 / ${modelAssetPath ?: "—"}"
            Type.TWOSTAGE -> listOfNotNull(
                detectorAssetPath?.let { "检测 / $it" },
                encoderAssetPath?.let { "编码 / $it" },
                decoderAssetPath?.let { "解码 / $it" },
            ).joinToString("\n")
        }
    }

    val models = listOf(
        ModelInfo(
            id = "picodet-s-320",
            friendlyName = "轻量检测",
            subLabel = "PicoDet-S / 320",
            type = Type.PICODET,
            typeLabel = "纯检测",
            precision = "FP32",
            speed = "最省电",
            desc = "PP-PicoDet-S 锚点无关轻量检测器，比 YOLO 更轻。",
            architecture = "PicoDet-S + ESNet 主干,anchor-free + DFL 回归,80 类。",
            params = "≈1.2M 参数(近似)",
            map = "COCO mAP50-95 ≈ 30%(近似)",
            flops = "≈1.7G FLOPs @320(近似)",
            scenarios = "极低功耗设备上的实时物体检测。",
            paperUrl = "https://arxiv.org/abs/2111.00902",
            githubUrl = "https://github.com/PaddlePaddle/PaddleDetection",
            modelAssetPath = "picodet_s_320_coco.onnx",
            inputSize = 320,
            conf = 0.4f,
            nms = 0.5f,
        ),
        ModelInfo(
            id = "yolo11n-384",
            friendlyName = "智能检测",
            subLabel = "YOLO11n / 384",
            type = Type.DETECT_ONLY,
            typeLabel = "纯检测",
            precision = "FP32",
            speed = "轻量/快速",
            desc = "仅输出目标方框,不生成像素级蒙版,速度最快。",
            architecture = "YOLO11n 检测头:1 个检测输出(方框+类别置信度)。",
            params = "≈2.6M 参数(近似)",
            map = "COCO mAP50-95 ≈ 39%(近似)",
            flops = "≈6.5G FLOPs @384(近似)",
            scenarios = "对蒙版精度无要求、追求最高帧率的场景;物体定位与计数。",
            paperUrl = "https://docs.ultralytics.com/models/yolo11/",
            githubUrl = "https://github.com/ultralytics/ultralytics",
            modelAssetPath = "yolo11n-384.onnx",
            inputSize = 384,
        ),
        ModelInfo(
            id = "yolo11n-seg-384",
            friendlyName = "快速分割",
            subLabel = "YOLO11n / 384",
            type = Type.ENDTOEND,
            typeLabel = "端到端",
            precision = "FP32",
            speed = "轻量/快速",
            desc = "单模型一步完成检测与分割,速度与精度均衡。",
            architecture = "YOLO11n-seg 分割头:检测输出 + Proto(掩码原型)输出。",
            params = "≈3.2M 参数(近似)",
            map = "COCO mask mAP50-95 ≈ 34%(近似)",
            flops = "≈8.5G FLOPs @384(近似)",
            scenarios = "实时视频分割,人/车/物等常见类别的像素级蒙版。",
            paperUrl = "https://docs.ultralytics.com/models/yolo11/",
            githubUrl = "https://github.com/ultralytics/ultralytics",
            modelAssetPath = "yolo11n-seg-384.onnx",
            inputSize = 384,
        ),
        ModelInfo(
            id = "yolo11n-seg-384-int8",
            friendlyName = "快速分割/节能",
            subLabel = "YOLO11n / 384 / INT8",
            type = Type.ENDTOEND,
            typeLabel = "端到端",
            precision = "INT8",
            speed = "最省电",
            desc = "INT8 量化版本,体积更小、更快、更省电,精度略有损失。",
            architecture = "YOLO11n-seg 分割头 + 动态量化(INT8 QDQ)。",
            params = "≈3.2M 参数,INT8 量化(近似)",
            map = "COCO mask mAP50-95 ≈ 33%(近似,较 FP32 略降)",
            flops = "≈8.5G FLOPs @384(近似)",
            scenarios = "低功耗设备、发热敏感场景;能接受轻微精度损失的实时分割。",
            paperUrl = "https://docs.ultralytics.com/models/yolo11/",
            githubUrl = "https://github.com/ultralytics/ultralytics",
            modelAssetPath = "yolo11n-seg-384-int8.onnx",
            inputSize = 384,
        ),
        ModelInfo(
            id = "yolo11n-seg",
            friendlyName = "高清分割",
            subLabel = "YOLO11n / 640",
            type = Type.ENDTOEND,
            typeLabel = "端到端",
            precision = "FP32",
            speed = "高精度",
            desc = "640 高分辨率输入,蒙版细节更精细,适合近距离观察。",
            architecture = "YOLO11n-seg 分割头,640×640 输入。",
            params = "≈3.2M 参数(近似)",
            map = "COCO mask mAP50-95 ≈ 38%(近似,输入更大)",
            flops = "≈23G FLOPs @640(近似)",
            scenarios = "对蒙版边界要求高、可接受较低帧率的场景。",
            paperUrl = "https://docs.ultralytics.com/models/yolo11/",
            githubUrl = "https://github.com/ultralytics/ultralytics",
            modelAssetPath = "yolo11n-seg.onnx",
            inputSize = 640,
        ),
        ModelInfo(
            id = "mobilesam",
            friendlyName = "极致分割",
            subLabel = "SAM / 384",
            type = Type.TWOSTAGE,
            typeLabel = "两段式",
            precision = "FP32",
            speed = "高精度",
            desc = "MobileSAM + YOLO 两段式:检测器找框,SAM 生成高质量蒙版。",
            architecture = "YOLOv8n 检测 + MobileSAM 图像编码器/掩码解码器。",
            params = "≈10M(编码器 5M + 解码器 + YOLOv8n)(近似)",
            map = "COCO mask mAP ≈ 31%(近似)",
            flops = "检测 8.7G + 编码器 1.6G FLOPs(近似)",
            scenarios = "对蒙版质量要求最高、不苛求帧率的场景。",
            paperUrl = "https://arxiv.org/abs/2306.14289",
            githubUrl = "https://github.com/ChaoningZhang/MobileSAM",
            detectorAssetPath = "yolov8n.onnx",
            encoderAssetPath = "mobile_sam_encoder.onnx",
            decoderAssetPath = "mobile_sam_decoder.onnx",
            inputSize = 384,
        ),
    )

    fun byId(id: String): ModelInfo? = models.firstOrNull { it.id == id }

    fun default(): ModelInfo = models.first()
}

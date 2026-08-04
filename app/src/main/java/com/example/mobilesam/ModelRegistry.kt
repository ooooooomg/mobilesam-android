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
        val authors: String = "",   // 作者
        val year: String = "",      // 提出时间
        val org: String = "",       // 提出机构
        val structure: String = "", // 主要结构
        val features: String = "",  // 特点
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
            Type.ENDTOEND, Type.DETECT_ONLY, Type.PICODET -> "模型 · ${modelAssetPath ?: "—"}"
            Type.TWOSTAGE -> listOfNotNull(
                detectorAssetPath?.let { "检测 · $it" },
                encoderAssetPath?.let { "编码 · $it" },
                decoderAssetPath?.let { "解码 · $it" },
            ).joinToString("\n")
        }
    }

    val models = listOf(
        ModelInfo(
            id = "picodet-s-320",
            friendlyName = "轻量检测",
            subLabel = "PicoDet-S · 320",
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
            authors = "Guanghua Yu, Qiyang Chang, Wenyu Lv, 等(百度)",
            year = "2021",
            org = "百度 PaddlePaddle 团队",
            structure = "以 ESNet 为骨干,配合 CSP-PAN 特征金字塔与轻量 ESHead 检测头,采用锚点无关(anchor-free)设计,回归头用 DFL(Distribution Focal Loss)预测框的分布。",
            features = "参数量仅约 1.2M,是当前最轻的实时检测器之一;支持 320 低分辨率输入,在低功耗移动设备上帧率高;Apache-2.0 宽松许可,适合嵌入式与端侧部署。",
            modelAssetPath = "picodet_s_320_coco.onnx",
            inputSize = 320,
            conf = 0.4f,
            nms = 0.5f,
        ),
        ModelInfo(
            id = "yolo11n-384",
            friendlyName = "智能检测",
            subLabel = "YOLO11n · 384",
            type = Type.DETECT_ONLY,
            typeLabel = "纯检测",
            precision = "FP32",
            speed = "轻量快速",
            desc = "仅输出目标方框,不生成像素级蒙版,速度最快。",
            architecture = "YOLO11n 检测头:1 个检测输出(方框+类别置信度)。",
            params = "≈2.6M 参数(近似)",
            map = "COCO mAP50-95 ≈ 39%(近似)",
            flops = "≈6.5G FLOPs @384(近似)",
            scenarios = "对蒙版精度无要求、追求最高帧率的场景;物体定位与计数。",
            paperUrl = "https://docs.ultralytics.com/models/yolo11/",
            githubUrl = "https://github.com/ultralytics/ultralytics",
            authors = "Ultralytics 团队",
            year = "2024",
            org = "Ultralytics(美国)",
            structure = "基于 CSPDarknet 骨干与 C2PSA 注意力模块,沿用 YOLO 单阶段检测框架,输出单张多尺度特征图上的方框与类别置信度。",
            features = "YOLO 系列最新轻量版,在精度与速度间取得优秀平衡;单模型单次前向即可完成检测,CPU 上延迟低;社区生态成熟。",
            modelAssetPath = "yolo11n-384.onnx",
            inputSize = 384,
        ),
        ModelInfo(
            id = "yolo11n-seg-384",
            friendlyName = "快速分割",
            subLabel = "YOLO11n-seg · 384",
            type = Type.ENDTOEND,
            typeLabel = "端到端",
            precision = "FP32",
            speed = "轻量快速",
            desc = "单模型一步完成检测与分割,速度与精度均衡。",
            architecture = "YOLO11n-seg 分割头:检测输出 + Proto(掩码原型)输出。",
            params = "≈3.2M 参数(近似)",
            map = "COCO mask mAP50-95 ≈ 34%(近似)",
            flops = "≈8.5G FLOPs @384(近似)",
            scenarios = "实时视频分割,人、车、物等常见类别的像素级蒙版。",
            paperUrl = "https://docs.ultralytics.com/models/yolo11/",
            githubUrl = "https://github.com/ultralytics/ultralytics",
            authors = "Ultralytics 团队",
            year = "2024",
            org = "Ultralytics(美国)",
            structure = "在 YOLO11n 检测骨架上并联 ProtoNet 掩码原型分支,输出每个实例的掩码系数与共享原型图,组合成逐实例像素蒙版。",
            features = "检测与分割共享同一个骨干,单次前向同时得到方框和蒙版;掩码质量与速度均衡;无需两阶段流水线,部署简单。",
            modelAssetPath = "yolo11n-seg-384.onnx",
            inputSize = 384,
        ),
        ModelInfo(
            id = "yolo11n-seg-384-int8",
            friendlyName = "快速分割节能",
            subLabel = "YOLO11n-seg · INT8",
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
            authors = "Ultralytics 团队",
            year = "2024",
            org = "Ultralytics(美国)",
            structure = "与快速分割相同的 YOLO11n-seg 网络,通过 QDQ 节点将权重与激活量化为 INT8,推理时反量化回浮点计算。",
            features = "模型体积降至约 3.4MB,INT8 计算在 CPU 上更快更省电;量化引入的精度损失通常小于 1-2 mAP,适合移动端长期运行。",
            modelAssetPath = "yolo11n-seg-384-int8.onnx",
            inputSize = 384,
        ),
        ModelInfo(
            id = "yolo11n-seg",
            friendlyName = "高清分割",
            subLabel = "YOLO11n-seg · 640",
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
            authors = "Ultralytics 团队",
            year = "2024",
            org = "Ultralytics(美国)",
            structure = "与快速分割同源但以 640×640 分辨率输入,同样的 CSPDarknet 骨干 + ProtoNet 掩码分支,更高分辨率带来更精细的空间细节。",
            features = "在相同模型参数下,640 输入相比 384 显著提升小物体与边缘的分割精度;适合对蒙版边界要求高的场景,代价是计算量约为 384 的三倍。",
            modelAssetPath = "yolo11n-seg.onnx",
            inputSize = 640,
        ),
        ModelInfo(
            id = "mobilesam",
            friendlyName = "极致分割",
            subLabel = "SAM · 384",
            type = Type.TWOSTAGE,
            typeLabel = "两段式",
            precision = "FP32",
            speed = "高精度",
            desc = "MobileSAM + YOLO 两段式:检测器找框,SAM 生成高质量蒙版。",
            architecture = "YOLOv8n 检测 + MobileSAM 图像编码器与掩码解码器。",
            params = "≈10M(编码器 5M + 解码器 + YOLOv8n)(近似)",
            map = "COCO mask mAP ≈ 31%(近似)",
            flops = "检测 8.7G + 编码器 1.6G FLOPs(近似)",
            scenarios = "对蒙版质量要求最高、不苛求帧率的场景。",
            paperUrl = "https://arxiv.org/abs/2306.14289",
            githubUrl = "https://github.com/ChaoningZhang/MobileSAM",
            authors = "Chaoning Zhang, Dongshen Han, Yu Qiao, 等",
            year = "2023",
            org = "韩国庆熙大学等",
            structure = "将 SAM(Segment Anything Model)的 ViT-H 编码器蒸馏为 TinyViT 轻量编码器,保留原掩码解码器,构成两段式:检测器定框 + SAM 生成蒙版。",
            features = "在几乎不损失分割质量的前提下把 SAM 编码器压缩到约 5M 参数,使 SAM 级分割能力可在手机 CPU 上运行;泛化能力强,能分割任意物体。",
            detectorAssetPath = "yolov8n.onnx",
            encoderAssetPath = "mobile_sam_encoder.onnx",
            decoderAssetPath = "mobile_sam_decoder.onnx",
            inputSize = 384,
        ),
    )

    fun byId(id: String): ModelInfo? = models.firstOrNull { it.id == id }

    fun default(): ModelInfo = models.first()
}

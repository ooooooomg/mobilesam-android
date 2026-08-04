package com.example.mobilesam

import ai.onnxruntime.OrtEnvironment
import android.content.Context

/**
 * Builds a [Segmenter] from a [ModelRegistry.ModelInfo] by reading the ONNX
 * assets and creating the ONNX sessions. Throws on missing/corrupt files so the
 * caller can abort a switch without touching the currently active segmenter.
 */
object SegmenterFactory {

    fun create(context: Context, info: ModelRegistry.ModelInfo): Segmenter {
        val env = OrtEnvironment.getEnvironment()
        return when (info.type) {
            ModelRegistry.Type.TWOSTAGE -> {
                val detector = YoloDetector(env, readAsset(context, info.detectorAssetPath!!))
                val encoder = SamImageEncoder(env, readAsset(context, info.encoderAssetPath!!))
                val decoder = SamMaskDecoder(env, readAsset(context, info.decoderAssetPath!!))
                TwoStageSegmenter(detector, encoder, decoder)
            }
            ModelRegistry.Type.ENDTOEND -> {
                YoloSegSegmenter(
                    env,
                    readAsset(context, info.modelAssetPath!!),
                    inputSize = info.inputSize,
                    confThreshold = info.conf,
                    nmsThreshold = info.nms,
                )
            }
            ModelRegistry.Type.DETECT_ONLY -> {
                YoloDetectSegmenter(
                    env,
                    readAsset(context, info.modelAssetPath!!),
                    inputSize = info.inputSize,
                    confThreshold = info.conf,
                    nmsThreshold = info.nms,
                )
            }
            ModelRegistry.Type.PICODET -> {
                PicoDetSegmenter(
                    env,
                    readAsset(context, info.modelAssetPath!!),
                    inputSize = info.inputSize,
                    confThreshold = info.conf,
                    nmsThreshold = info.nms,
                )
            }
        }
    }

    private fun readAsset(context: Context, path: String): ByteArray =
        context.assets.open(path).use { it.readBytes() }
}

package com.example.mobilesam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * MobileSAM mask decoder on ONNX Runtime.
 *
 * Takes the image embedding + per-object boxes (in 1024-padded
 * coordinates), runs the decoder, and produces a mask at the original
 * image resolution. Mirrors scripts/mobile_e2e.py.
 */
class SamMaskDecoder(
    private val environment: OrtEnvironment,
    modelBytes: ByteArray,
) {
    private val session: OrtSession

    init {
        // CPU 多线程（NNAPI 初始化慢且混合执行，弃用）
        val opts = OrtSession.SessionOptions()
        val cores = Runtime.getRuntime().availableProcessors()
        opts.setIntraOpNumThreads(cores)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = environment.createSession(modelBytes, opts)
        android.util.Log.d("MobileSAM", "decoder CPU threads=$cores")
    }

    data class SegmentedMask(
        val mask: FloatArray,     // HxW logits (positive => foreground)
        val width: Int,
        val height: Int,
        val score: Float,
    )

    companion object {
        private const val EMBED_C = 256
        private const val EMBED_HW = 24
        private const val IMG_SIZE = 384f
        private const val MASK_INPUT_HW = 96  // 4 * EMBED_HW
    }

    /**
     * Transform a box from original-image coords to 512-padded coords.
     * Uses the same scale the encoder applied (ResizeLongestSide).
     */
    private fun to1024(box: YoloDetector.DetBox, scale: Float): FloatArray {
        val x1 = box.x1 * scale
        val y1 = box.y1 * scale
        val x2 = box.x2 * scale
        val y2 = box.y2 * scale
        return floatArrayOf(
            max(0f, min(x1, IMG_SIZE)), max(0f, min(y1, IMG_SIZE)),
            max(0f, min(x2, IMG_SIZE)), max(0f, min(y2, IMG_SIZE)),
        )
    }

    /**
     * Decode one box into a mask.
     *
     * @param embedding flat (1,256,64,64) from SamImageEncoder
     * @param box box in original-image coordinates
     * @param scale encoder scale (original -> 1024 padded)
     * @param origW original image width
     * @param origH original image height
     */
    fun decode(
        embedding: FloatArray,
        box: YoloDetector.DetBox,
        scale: Float,
        origW: Int,
        origH: Int,
    ): SegmentedMask {
        val b = to1024(box, scale)
        val coords = floatArrayOf(b[0], b[1], b[2], b[3])
        val labels = floatArrayOf(2f, 3f)  // box prompt: start, end
        val maskInput = FloatArray(MASK_INPUT_HW * MASK_INPUT_HW)  // zeros
        val hasMask = floatArrayOf(0f)
        val origSize = floatArrayOf(origH.toFloat(), origW.toFloat())

        val tensors = mutableMapOf(
            "image_embeddings" to OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(embedding), longArrayOf(1, EMBED_C.toLong(), EMBED_HW.toLong(), EMBED_HW.toLong())),
            "point_coords" to OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(coords), longArrayOf(1, 2, 2)),
            "point_labels" to OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(labels), longArrayOf(1, 2)),
            "mask_input" to OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(maskInput), longArrayOf(1, 1, MASK_INPUT_HW.toLong(), MASK_INPUT_HW.toLong())),
            "has_mask_input" to OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(hasMask), longArrayOf(1)),
            "orig_im_size" to OnnxTensor.createTensor(environment, java.nio.FloatBuffer.wrap(origSize), longArrayOf(2)),
        )
        val results = session.run(tensors)
        tensors.values.forEach { it.close() }

        // Outputs: masks (1,1,H,W), iou_predictions (1,1), low_res_masks.
        // Result.get(int) returns OnnxValue; cast to OnnxTensor (not .value).
        val masks = results.get(0) as OnnxTensor
        val scores = results.get(1) as OnnxTensor
        val maskFloat = masks.floatBuffer ?: throw IllegalStateException("bad mask output")
        val scoreFloat = scores.floatBuffer ?: throw IllegalStateException("bad score output")
        val maskArr = FloatArray(maskFloat.capacity())
        maskFloat.rewind(); maskFloat.get(maskArr)
        val score = scoreFloat.get(0)
        masks.close(); scores.close()
        results.close()

        return SegmentedMask(maskArr, origW, origH, score)
    }

    fun close() {
        session.close()
    }
}

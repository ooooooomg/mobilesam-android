package com.example.mobilesam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8n object detector running on ONNX Runtime.
 *
 * Input:  RGB bitmap (any size). Output: list of [DetBox] in original
 * image coordinates. Mirrors the Python reference in
 * scripts/mobile_e2e.py (640 letterbox, NMS).
 */
class YoloDetector(
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
        android.util.Log.d("MobileSAM", "yolo CPU threads=$cores")
    }

    data class DetBox(
        val x1: Float, val y1: Float,
        val x2: Float, val y2: Float,
        val score: Float,
        val classId: Int,
    )

    companion object {
        private const val INPUT_SIZE = 480
        private const val CONF_THRESHOLD = 0.35f
        private const val NMS_THRESHOLD = 0.4f
    }

    /** Resize keeping aspect ratio, pad to 480x480 with gray 114. */
    private fun letterbox(bitmap: Bitmap): Pair<FloatArray, Float> {
        val w = bitmap.width
        val h = bitmap.height
        val scale = INPUT_SIZE.toFloat() / max(w, h)
        val newW = (w * scale).toInt().coerceAtLeast(1)
        val newH = (h * scale).toInt().coerceAtLeast(1)
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val canvas = IntArray(INPUT_SIZE * INPUT_SIZE)
        // Fill gray 114
        val gray = 114
        for (i in canvas.indices) canvas[i] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
        val pixels = IntArray(newW * newH)
        resized.getPixels(pixels, 0, newW, 0, 0, newW, newH)
        for (y in 0 until newH) {
            val srcRow = y * newW
            val dstRow = y * INPUT_SIZE
            for (x in 0 until newW) {
                canvas[dstRow + x] = pixels[srcRow + x]
            }
        }
        resized.recycle()
        // Convert ARGB to NCHW float (RGB / 255.0)
        val input = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val px = canvas[y * INPUT_SIZE + x]
                val r = ((px shr 16) and 0xFF) / 255.0f
                val g = ((px shr 8) and 0xFF) / 255.0f
                val b = (px and 0xFF) / 255.0f
                val i = y * INPUT_SIZE + x
                input[i] = r
                input[INPUT_SIZE * INPUT_SIZE + i] = g
                input[2 * INPUT_SIZE * INPUT_SIZE + i] = b
            }
        }
        return input to scale
    }

    /** Run YOLO, NMS, return boxes in original-image coordinates. */
    fun detect(bitmap: Bitmap): List<DetBox> {
        val (input, scale) = letterbox(bitmap)
        val tensor = OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            java.nio.FloatBuffer.wrap(input),
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
        val results = session.run(mapOf("images" to tensor))
        tensor.close()
        // YOLOv8n output shape (1, 84, N). Result.get(int) returns an
        // OnnxValue; the tensor itself is an OnnxTensor (NOT .value, which
        // would be the raw float[][][] data).
        val outTensor = results.get(0) as OnnxTensor
        val buf = outTensor.floatBuffer ?: throw IllegalStateException("bad yolo output")
        val numRows = 84
        val total = buf.capacity()
        if (total < numRows) {
            throw IllegalStateException("unexpected yolo output size ${total}")
        }
        val numPreds = total / numRows
        val raw = FloatArray(total)
        buf.rewind()
        buf.get(raw)
        outTensor.close()
        results.close()

        val boxes = ArrayList<DetBox>()
        for (j in 0 until numPreds) {
            var bestClass = -1
            var bestScore = CONF_THRESHOLD
            for (c in 4 until 84) {
                if (raw[c * numPreds + j] > bestScore) {
                    bestScore = raw[c * numPreds + j]
                    bestClass = c - 4
                }
            }
            if (bestClass < 0) continue
            val cx = raw[0 * numPreds + j] / scale
            val cy = raw[1 * numPreds + j] / scale
            val w = raw[2 * numPreds + j] / scale
            val h = raw[3 * numPreds + j] / scale
            boxes.add(
                DetBox(
                    x1 = cx - w / 2, y1 = cy - h / 2,
                    x2 = cx + w / 2, y2 = cy + h / 2,
                    score = bestScore, classId = bestClass,
                )
            )
        }
        return nms(boxes)
    }

    private fun iou(a: DetBox, b: DetBox): Float {
        val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = iw * ih
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (union > 0f) inter / union else 0f
    }

    private fun nms(boxes: List<DetBox>): List<DetBox> {
        val sorted = boxes.sortedByDescending { it.score }
        val keep = ArrayList<DetBox>()
        val remaining = sorted.toMutableList()
        while (remaining.isNotEmpty()) {
            val first = remaining.removeAt(0)
            keep.add(first)
            remaining.removeAll { iou(first, it) >= NMS_THRESHOLD }
        }
        return keep
    }

    fun close() {
        session.close()
    }
}

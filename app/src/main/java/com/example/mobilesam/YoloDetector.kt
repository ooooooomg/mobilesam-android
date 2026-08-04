package com.example.mobilesam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap

/**
 * YOLOv8n object detector running on ONNX Runtime.
 *
 * Input:  RGB bitmap (any size). Output: list of [DetBox] in original
 * image coordinates (480 letterbox, NMS).
 */
class YoloDetector(
    private val environment: OrtEnvironment,
    modelBytes: ByteArray,
    private val inputSize: Int = 480,
    private val confThreshold: Float = 0.35f,
    private val nmsThreshold: Float = 0.4f,
) {
    private val session: OrtSession
    private val post = DetPostprocess(inputSize)

    init {
        // CPU 多线程（NNAPI 初始化慢且混合执行，弃用）
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        opts.setInterOpNumThreads(1)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = environment.createSession(modelBytes, opts)
        android.util.Log.d("MobileSAM", "yolo input=$inputSize threads=4/1")
    }

    data class DetBox(
        val x1: Float, val y1: Float,
        val x2: Float, val y2: Float,
        val score: Float,
        val classId: Int,
    )

    /** Run YOLO, NMS, return boxes in original-image coordinates. */
    fun detect(bitmap: Bitmap): List<DetBox> {
        val (input, scale) = post.letterbox(bitmap)
        val tensor = OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            java.nio.FloatBuffer.wrap(input),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
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
            var bestScore = confThreshold
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
        return post.nms(boxes, nmsThreshold)
    }

    fun close() {
        session.close()
    }
}

package com.example.mobilesam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap

/**
 * Lightweight detect-only segmenter: a YOLO detection model (e.g. yolo11n)
 * outputs boxes + classes but NO masks. Returns boxes only; the composer draws
 * just the bounding boxes (no mask overlay). Fastest of all models.
 *
 * Output: det (1, 4+80, N) at inputSize (e.g. (1, 84, 3024) at 384).
 */
class YoloDetectSegmenter(
    environment: OrtEnvironment,
    modelBytes: ByteArray,
    private val inputSize: Int,
    private val confThreshold: Float,
    private val nmsThreshold: Float,
) : Segmenter {

    private val session: OrtSession
    private val post = DetPostprocess(inputSize)

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        opts.setInterOpNumThreads(1)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = environment.createSession(modelBytes, opts)
        android.util.Log.d("MobileSAM", "yolo-detect input=$inputSize threads=4/1")
    }

    override fun segmentStreaming(source: Bitmap): SegmentationOutput =
        segmentFull(source)

    override fun segmentFull(source: Bitmap): SegmentationOutput {
        val t0 = System.currentTimeMillis()

        val (input, scale) = post.letterbox(source)
        val tensor = OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            java.nio.FloatBuffer.wrap(input),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )
        val results = session.run(mapOf("images" to tensor))
        tensor.close()

        val outTensor = results.get(0) as? OnnxTensor
            ?: throw IllegalStateException("bad yolo-detect output")
        val shape = outTensor.info.shape
        val nAnchors = if (shape.size == 3) shape[2].toInt() else {
            outTensor.close(); results.close()
            throw IllegalStateException("unexpected yolo-detect output: ${shape.joinToString("x")}")
        }
        val buf = outTensor.floatBuffer
        val raw = FloatArray(buf.capacity())
        buf.rewind(); buf.get(raw)
        outTensor.close()
        results.close()

        val boxes = ArrayList<YoloDetector.DetBox>()
        for (j in 0 until nAnchors) {
            var bestClass = -1
            var bestScore = confThreshold
            for (c in 4 until 84) {
                val s = raw[c * nAnchors + j]
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c - 4
                }
            }
            if (bestClass < 0) continue
            val cx = raw[0 * nAnchors + j] / scale
            val cy = raw[1 * nAnchors + j] / scale
            val w = raw[2 * nAnchors + j] / scale
            val h = raw[3 * nAnchors + j] / scale
            boxes.add(
                YoloDetector.DetBox(
                    x1 = cx - w / 2, y1 = cy - h / 2,
                    x2 = cx + w / 2, y2 = cy + h / 2,
                    score = bestScore, classId = bestClass,
                )
            )
        }
        val nmsBoxes = post.nms(boxes, nmsThreshold)
        val elapsed = System.currentTimeMillis() - t0

        return SegmentationOutput(
            masks = emptyList(),
            boxes = nmsBoxes,
            yoloMs = elapsed,
            encMs = 0L,
            decMs = 0L,
            encodedThisFrame = true,
            embeddingReady = true,
        )
    }

    override fun close() {
        session.close()
    }
}

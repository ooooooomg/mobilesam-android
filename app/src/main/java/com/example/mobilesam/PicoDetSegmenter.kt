package com.example.mobilesam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import kotlin.math.exp

/**
 * PP-PicoDet (anchor-free + DFL box) pure-detection segmenter on ONNX Runtime.
 *
 * Model export (paddle2onnx, nms=False) emits per-stride outputs:
 *  - 4 x cls tensors  (1, grid, 80)     COCO 80 classes
 *  - 4 x box tensors  (1, grid, 4*(reg_max+1))   DFL distributions, reg_max=7
 *
 * App-side decode: per anchor, softmax each 8-bin DFL distribution, take the
 * expectation as the box offset, add the anchor stride offset, then NMS.
 */
class PicoDetSegmenter(
    environment: OrtEnvironment,
    modelBytes: ByteArray,
    private val inputSize: Int = 320,
    private val confThreshold: Float = 0.4f,
    private val nmsThreshold: Float = 0.5f,
) : Segmenter {

    companion object {
        private const val NUM_CLASSES = 80
        private const val REG_MAX = 7
        private const val DFL_BINS = REG_MAX + 1 // 8
        private val STRIDES = intArrayOf(8, 16, 32, 64)
    }

    private val session: OrtSession
    private val post = DetPostprocess(inputSize)

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        opts.setInterOpNumThreads(1)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = environment.createSession(modelBytes, opts)
        android.util.Log.d("MobileSAM", "picodet input=$inputSize threads=4/1")
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
        val results = session.run(mapOf("image" to tensor))
        tensor.close()

        // Outputs are ordered cls0..cls3, box0..box3 per stride.
        val clsTensors = ArrayList<FloatArray>(4)
        val boxTensors = ArrayList<FloatArray>(4)
        val grids = IntArray(4)
        for (s in 0 until 4) {
            val cls = results.get(s) as OnnxTensor
            val box = results.get(s + 4) as OnnxTensor
            val grid = cls.info.shape[1].toInt()
            grids[s] = grid
            val cBuf = FloatArray(grid * NUM_CLASSES)
            cls.floatBuffer.rewind(); cls.floatBuffer.get(cBuf)
            clsTensors.add(cBuf)
            val bBuf = FloatArray(grid * 4 * DFL_BINS)
            box.floatBuffer.rewind(); box.floatBuffer.get(bBuf)
            boxTensors.add(bBuf)
            cls.close(); box.close()
        }
        results.close()

        // DFL integral -> box offsets in input-space.
        val weights = FloatArray(DFL_BINS)
        for (i in 0 until DFL_BINS) weights[i] = i.toFloat()

        val raw = ArrayList<YoloDetector.DetBox>()
        for (s in 0 until 4) {
            val stride = STRIDES[s]
            val grid = grids[s]
            val cls = clsTensors[s]
            val box = boxTensors[s]
            val feats = inputSize / stride
            for (i in 0 until grid) {
                val cy = i / feats
                val cx = i % feats
                var bestClass = -1
                var bestScore = confThreshold
                for (c in 0 until NUM_CLASSES) {
                    val sc = cls[i * NUM_CLASSES + c]
                    if (sc > bestScore) {
                        bestScore = sc
                        bestClass = c
                    }
                }
                if (bestClass < 0) continue

                // Recover 4 box offsets from 8-bin DFL distributions (softmax
                // over each 8-bin window, then weighted expectation).
                val base = i * 4 * DFL_BINS
                val dl = expDist(box, base)
                val dt = expDist(box, base + DFL_BINS)
                val dr = expDist(box, base + 2 * DFL_BINS)
                val db = expDist(box, base + 3 * DFL_BINS)
                var l = 0f; var t = 0f; var r = 0f; var b = 0f
                for (k in 0 until DFL_BINS) {
                    l += weights[k] * dl[k]
                    t += weights[k] * dt[k]
                    r += weights[k] * dr[k]
                    b += weights[k] * db[k]
                }

                val x1 = (cx + 0.5f - l) * stride
                val y1 = (cy + 0.5f - t) * stride
                val x2 = (cx + 0.5f + r) * stride
                val y2 = (cy + 0.5f + b) * stride

                raw.add(
                    YoloDetector.DetBox(
                        x1 = x1 / scale, y1 = y1 / scale,
                        x2 = x2 / scale, y2 = y2 / scale,
                        score = bestScore, classId = bestClass,
                    )
                )
            }
        }

        val boxes = post.nms(raw, nmsThreshold)
        val tDet = System.currentTimeMillis() - t0

        return SegmentationOutput(
            masks = emptyList(),
            boxes = boxes,
            yoloMs = tDet,
            encMs = 0L,
            decMs = 0L,
            encodedThisFrame = true,
            embeddingReady = true,
        )
    }

    override fun close() {
        session.close()
    }

    /** Softmax of one 8-bin DFL window -> probability weights. */
    private fun expDist(box: FloatArray, off: Int): FloatArray {
        val out = FloatArray(DFL_BINS)
        var sum = 0f
        for (k in 0 until DFL_BINS) {
            val e = exp(box[off + k])
            out[k] = e
            sum += e
        }
        if (sum > 0f) {
            for (k in 0 until DFL_BINS) out[k] /= sum
        }
        return out
    }
}

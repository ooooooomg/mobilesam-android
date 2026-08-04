package com.example.mobilesam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.RectF
import kotlin.math.exp

/**
 * End-to-end YOLO-seg (e.g. YOLO11n-seg / YOLOv8n-seg) instance segmenter.
 *
 * A single ONNX model outputs per-anchor detection rows (4 box + 80 classes +
 * 32 mask coefficients) and a 32x160x160 prototype mask tensor. Boxes are
 * decoded from the detection tensor, then each box's mask is rebuilt from its
 * coefficients + prototypes and returned as a box-local [SamMaskDecoder.SegmentedMask]
 * so the composer only draws within the box.
 *
 * Expected outputs (export with nms=False):
 *  - det:   (1, 4 + 80 + 32, N)   e.g. (1, 116, 8400) at imgsz=640
 *  - proto: (1, 32, 160, 160)
 */
class YoloSegSegmenter(
    environment: OrtEnvironment,
    modelBytes: ByteArray,
    private val inputSize: Int,
    private val confThreshold: Float,
    private val nmsThreshold: Float,
) : Segmenter {

    private val session: OrtSession
    private val post = DetPostprocess(inputSize)

    // Reused decode buffers (single-threaded).
    private var detArr: FloatArray? = null
    private var protoArr: FloatArray? = null
    private var coeffArr: FloatArray? = null

    init {
        val opts = OrtSession.SessionOptions()
        // availableProcessors() includes little cores; 4 big cores + single
        // inter-op thread is more predictable on mobile (avoids cache thrash
        // and sync overhead; XNNPACK ops are single-threaded anyway).
        opts.setIntraOpNumThreads(4)
        opts.setInterOpNumThreads(1)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = environment.createSession(modelBytes, opts)
        android.util.Log.d("MobileSAM", "yolo-seg input=$inputSize threads=4/1")
    }

    /** A decoded box plus the anchor index that produced it (for mask coeffs). */
    private class DetWithAnchor(
        val box: YoloDetector.DetBox,
        val anchor: Int,
    )

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

        // Identify det (1,C,N) and proto (1,32,160,160) by shape.
        var det: OnnxTensor? = null
        var proto: OnnxTensor? = null
        for (i in 0 until results.size()) {
            val t = results.get(i) as? OnnxTensor ?: continue
            val shape = t.info.shape
            if (shape.size == 3 && shape[0] == 1L && shape[1] > 100L) det = t
            else if (shape.size == 4 && shape[0] == 1L && shape[1] == 32L) proto = t
        }
        if (det == null || proto == null) {
            val shapes = (0 until results.size()).map {
                (results.get(it) as? OnnxTensor)?.info?.shape?.joinToString("x") ?: "?"
            }
            results.close()
            throw IllegalStateException("unexpected yolo-seg outputs: $shapes (need nms=False export)")
        }

        val detShape = det!!.info.shape
        val nAnchors = detShape[2].toInt()
        val nCoeff = detShape[1].toInt() - 84   // 4 box + 80 classes
        val protoShape = proto!!.info.shape
        val protoH = protoShape[2].toInt()
        val protoW = protoShape[3].toInt()

        var dArr = detArr
        if (dArr == null || dArr.size < det!!.floatBuffer.capacity()) {
            dArr = FloatArray(det!!.floatBuffer.capacity())
            detArr = dArr
        }
        det!!.floatBuffer.rewind(); det!!.floatBuffer.get(dArr)
        val dA = dArr
        var pArr = protoArr
        if (pArr == null || pArr.size < proto!!.floatBuffer.capacity()) {
            pArr = FloatArray(proto!!.floatBuffer.capacity())
            protoArr = pArr
        }
        proto!!.floatBuffer.rewind(); proto!!.floatBuffer.get(pArr)
        val pA = pArr
        det!!.close(); proto!!.close()
        results.close()

        // Decode boxes (keep anchor index for mask coefficients).
        val raw = ArrayList<DetWithAnchor>()
        for (j in 0 until nAnchors) {
            var bestClass = -1
            var bestScore = confThreshold
            for (c in 4 until 84) {
                val s = dA[c * nAnchors + j]
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c - 4
                }
            }
            if (bestClass < 0) continue
            val cx = dA[0 * nAnchors + j] / scale
            val cy = dA[1 * nAnchors + j] / scale
            val w = dA[2 * nAnchors + j] / scale
            val h = dA[3 * nAnchors + j] / scale
            raw.add(
                DetWithAnchor(
                    YoloDetector.DetBox(
                        x1 = cx - w / 2, y1 = cy - h / 2,
                        x2 = cx + w / 2, y2 = cy + h / 2,
                        score = bestScore, classId = bestClass,
                    ),
                    j,
                )
            )
        }
        val boxes = post.nms(raw.map { it.box }, nmsThreshold)
        val tDet = System.currentTimeMillis() - t0

        // Index raw anchors by exact input-space top-left (float bit pattern)
        // so surviving boxes find their mask coefficients in O(1). Using raw
        // float bits (not rounding) avoids two anchors colliding on the same key.
        val anchorByKey = HashMap<Long, Int>(raw.size)
        for (item in raw) {
            val kx = (item.box.x1 * scale).toRawBits()
            val ky = (item.box.y1 * scale).toRawBits()
            anchorByKey[(ky.toLong() shl 32) or (kx.toLong() and 0xFFFFFFFFL)] = item.anchor
        }

        // Rebuild a box-local mask per surviving box.
        val masks = ArrayList<SamMaskDecoder.SegmentedMask>(boxes.size)
        for (box in boxes) {
            if (box.x2 - box.x1 < 5f || box.y2 - box.y1 < 5f) continue
            val kx = (box.x1 * scale).toRawBits()
            val ky = (box.y1 * scale).toRawBits()
            val anchor = anchorByKey[(ky.toLong() shl 32) or (kx.toLong() and 0xFFFFFFFFL)] ?: continue
            masks.add(rebuildMask(box, anchor, dA, pA, nAnchors, nCoeff, protoH, protoW, scale))
        }
        val tMask = System.currentTimeMillis() - t0

        return SegmentationOutput(
            masks = masks,
            boxes = boxes,
            yoloMs = 0L,
            encMs = tMask,
            decMs = 0L,
            encodedThisFrame = true,
            embeddingReady = true,
        )
    }

    /**
     * Build a box-local mask: the mask is sampled from the 160-space prototype
     * region corresponding to the box, in the box's aspect (not stretched to the
     * whole image). Returned with the original-coordinate box rect so the
     * composer draws it only inside the box.
     */
    private fun rebuildMask(
        box: YoloDetector.DetBox,
        anchor: Int,
        detArr: FloatArray,
        protoArr: FloatArray,
        nAnchors: Int,
        nCoeff: Int,
        protoH: Int,
        protoW: Int,
        scale: Float,
    ): SamMaskDecoder.SegmentedMask {
        // Mask coefficients for this anchor (reused buffer).
        var coeff = coeffArr
        if (coeff == null || coeff.size < nCoeff) {
            coeff = FloatArray(nCoeff)
            coeffArr = coeff
        }
        val c = coeff
        for (i in 0 until nCoeff) {
            c[i] = detArr[(84 + i) * nAnchors + anchor]
        }

        // Box in input-space -> proto-space (inputSize/protoW maps input px to
        // proto px, e.g. 640->160 or 384->96).
        val s160 = inputSize.toFloat() / protoW
        val bx1 = (box.x1 * scale / s160).toInt().coerceIn(0, protoW - 1)
        val by1 = (box.y1 * scale / s160).toInt().coerceIn(0, protoH - 1)
        val bx2 = (box.x2 * scale / s160).toInt().coerceIn(bx1 + 1, protoW)
        val by2 = (box.y2 * scale / s160).toInt().coerceIn(by1 + 1, protoH)
        val bw = bx2 - bx1
        val bh = by2 - by1

        val mask = FloatArray(bw * bh)
        var idx = 0
        for (y in by1 until by2) {
            for (x in bx1 until bx2) {
                var acc = 0f
                for (ci in 0 until nCoeff) {
                    acc += c[ci] * protoArr[ci * protoH * protoW + y * protoW + x]
                }
                val prob = 1f / (1f + exp(-acc))
                mask[idx++] = if (prob > 0.5f) 1f else 0f
            }
        }
        return SamMaskDecoder.SegmentedMask(
            mask = mask,
            width = bw,
            height = bh,
            score = box.score,
            boxRect = RectF(box.x1, box.y1, box.x2, box.y2),
        )
    }

    override fun close() {
        session.close()
    }
}

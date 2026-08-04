package com.example.mobilesam

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min

/**
 * Shared detection post-processing reused by YoloDetector and YoloSegSegmenter:
 * letterbox resize to a square input, IoU, and NMS. Reusable buffers are kept
 * internally; callers run on a single thread.
 */
class DetPostprocess(private val inputSize: Int) {

    private var canvas: IntArray? = null
    private var pixels: IntArray? = null
    private var input: FloatArray? = null

    /**
     * Resize keeping aspect ratio, pad with gray 114, and produce an NCHW float
     * array (RGB / 255.0). Returns the input array plus the resize scale so box
     * coordinates can be mapped back to the original image.
     */
    fun letterbox(bitmap: Bitmap): Pair<FloatArray, Float> {
        val w = bitmap.width
        val h = bitmap.height
        val scale = inputSize.toFloat() / max(w, h)
        val newW = (w * scale).toInt().coerceAtLeast(1)
        val newH = (h * scale).toInt().coerceAtLeast(1)
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)

        val c = canvas ?: IntArray(inputSize * inputSize).also { canvas = it }
        // Fill gray 114
        val gray = 114
        for (i in c.indices) c[i] = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray

        var px = pixels
        if (px == null || px.size < newW * newH) {
            px = IntArray(newW * newH)
            pixels = px
        }
        resized.getPixels(px, 0, newW, 0, 0, newW, newH)
        for (y in 0 until newH) {
            val srcRow = y * newW
            val dstRow = y * inputSize
            for (x in 0 until newW) {
                c[dstRow + x] = px[srcRow + x]
            }
        }
        // createScaledBitmap returns the ORIGINAL bitmap when the size is
        // unchanged; recycling it would destroy the caller's frame.
        if (resized !== bitmap) resized.recycle()

        // Convert ARGB to NCHW float (RGB / 255.0)
        val input = input ?: FloatArray(3 * inputSize * inputSize).also { this.input = it }
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val cpx = c[y * inputSize + x]
                val r = ((cpx shr 16) and 0xFF) / 255.0f
                val g = ((cpx shr 8) and 0xFF) / 255.0f
                val b = (cpx and 0xFF) / 255.0f
                val i = y * inputSize + x
                input[i] = r
                input[inputSize * inputSize + i] = g
                input[2 * inputSize * inputSize + i] = b
            }
        }
        return input to scale
    }

    fun iou(a: YoloDetector.DetBox, b: YoloDetector.DetBox): Float {
        val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = iw * ih
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (union > 0f) inter / union else 0f
    }

    fun nms(boxes: List<YoloDetector.DetBox>, nmsThreshold: Float): List<YoloDetector.DetBox> {
        val sorted = boxes.sortedByDescending { it.score }
        val keep = ArrayList<YoloDetector.DetBox>()
        val remaining = sorted.toMutableList()
        while (remaining.isNotEmpty()) {
            val first = remaining.removeAt(0)
            keep.add(first)
            remaining.removeAll { iou(first, it) >= nmsThreshold }
        }
        return keep
    }
}

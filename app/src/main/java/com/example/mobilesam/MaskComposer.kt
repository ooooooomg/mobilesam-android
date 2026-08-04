package com.example.mobilesam

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF

/**
 * Composes segmentation masks and detection boxes into overlay bitmaps.
 *
 * Uses Canvas drawBitmap for mask coloring (fast, single native draw per
 * mask) instead of per-pixel setPixel. Masks arrive at original-image
 * resolution as float logits (positive => foreground); we downscale each mask
 * to a small colored overlay and stretch it over the canvas.
 *
 * [composeOpaque] copies the source and bakes masks + boxes into the copy, so
 * the result always aligns with the underlying image.
 */
object MaskComposer {

    private val MASK_COLORS = intArrayOf(
        0xFF00FF41.toInt(), 0xFF00E5FF.toInt(), 0xFFFFB454.toInt(), 0xFFFF6A00.toInt(),
        0xFF8AB4F8.toInt(), 0xFF00C853.toInt(), 0xFFE040FB.toInt(), 0xFFFFFF00.toInt(),
    )

    /** Color for a class index, matching the mask overlay colors. */
    fun colorForClass(classId: Int): Int =
        MASK_COLORS[classId % MASK_COLORS.size]

    /** Copy source and bake masks/boxes into the copy. */
    fun composeOpaque(
        source: Bitmap,
        masks: List<SamMaskDecoder.SegmentedMask>,
        boxes: List<YoloDetector.DetBox>,
    ): Bitmap {
        val w = source.width
        val h = source.height
        // No detections: skip the full-frame copy entirely.
        if (masks.isEmpty() && boxes.isEmpty()) {
            return source.copy(Bitmap.Config.ARGB_8888, true)
        }
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        drawMasks(canvas, masks, w, h)
        drawBoxes(canvas, boxes, w, h)
        return result
    }

    private fun drawMasks(
        canvas: Canvas,
        masks: List<SamMaskDecoder.SegmentedMask>,
        dstW: Int,
        dstH: Int,
    ) {
        masks.forEachIndexed { idx, seg ->
            if (seg.width == 0 || seg.height == 0) return@forEachIndexed
            val color = MASK_COLORS[idx % MASK_COLORS.size]
            if (seg.boxRect != null) {
                // Box-local mask: draw a small colored bitmap directly into the
                // box region of the target canvas (no full-size overlay).
                drawBoxMask(canvas, seg, color)
            } else {
                // Full-image mask (MobileSAM): existing path.
                val overlay = buildMaskOverlay(seg, color)
                if (overlay != null) {
                    canvas.drawBitmap(
                        overlay,
                        Rect(0, 0, overlay.width, overlay.height),
                        Rect(0, 0, dstW, dstH),
                        null,
                    )
                    overlay.recycle()
                }
            }
        }
    }

    /**
     * Sample the box-local mask (0/1 floats at [seg.width]x[seg.height]) into a
     * small colored bitmap and draw it stretched into the box rectangle on the
     * target canvas. Avoids allocating a full-frame overlay per mask.
     */
    private fun drawBoxMask(
        canvas: Canvas,
        seg: SamMaskDecoder.SegmentedMask,
        color: Int,
    ) {
        val srcW = seg.width
        val srcH = seg.height
        val mask = seg.mask
        val box = seg.boxRect
        if (srcW <= 0 || srcH <= 0 || mask.size < srcW * srcH || box == null) return

        val r = (color shr 16) and 0xFF
        val g = (color shr 8) and 0xFF
        val b = color and 0xFF
        val alpha = MASK_ALPHA
        val px = IntArray(srcW * srcH)
        var i = 0
        for (y in 0 until srcH) {
            val rowBase = y * srcW
            for (x in 0 until srcW) {
                px[i++] = if (mask[rowBase + x] > 0f) {
                    (alpha shl 24) or (r shl 16) or (g shl 8) or b
                } else {
                    0x00000000
                }
            }
        }
        val maskBmp = Bitmap.createBitmap(px, srcW, srcH, Bitmap.Config.ARGB_8888)
        canvas.drawBitmap(
            maskBmp,
            Rect(0, 0, srcW, srcH),
            RectF(box.left, box.top, box.right, box.bottom),
            null,
        )
        maskBmp.recycle()
    }

    private fun drawBoxes(
        canvas: Canvas,
        boxes: List<YoloDetector.DetBox>,
        dstW: Int,
        dstH: Int,
    ) {
        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = (dstH * 0.004f).coerceAtLeast(2f)
        }
        val corner = (dstH * 0.006f).coerceAtLeast(2f)
        boxes.forEachIndexed { idx, b ->
            boxPaint.color = MASK_COLORS[idx % MASK_COLORS.size]
            canvas.drawRoundRect(
                RectF(b.x1, b.y1, b.x2, b.y2), corner, corner, boxPaint
            )
        }
    }

    /**
     * Build a colored overlay bitmap (at most 512x512) from the mask logits.
     * Foreground pixels (logit > 0) become the mask color with alpha [MASK_ALPHA];
     * background is transparent. The overlay is stretched by drawBitmap.
     */
    private fun buildMaskOverlay(
        seg: SamMaskDecoder.SegmentedMask,
        color: Int,
    ): Bitmap? {
        val srcW = seg.width
        val srcH = seg.height
        val mask = seg.mask
        if (srcW <= 0 || srcH <= 0 || mask.size < srcW * srcH) return null

        // Downscale to at most 512x512 for the overlay (fast, and the final
        // stretch by the canvas gives a smooth result).
        val scale = 512.0 / maxOf(srcW, srcH)
        val ovW = (srcW * scale).toInt().coerceAtLeast(1)
        val ovH = (srcH * scale).toInt().coerceAtLeast(1)

        val px = ovW * ovH
        val colors = IntArray(px)
        val r = (color shr 16) and 0xFF
        val g = (color shr 8) and 0xFF
        val b = color and 0xFF
        val stepX = srcW / ovW.toDouble()
        val stepY = srcH / ovH.toDouble()
        for (y in 0 until ovH) {
            val sy = (y * stepY).toInt()
            for (x in 0 until ovW) {
                val sx = (x * stepX).toInt()
                val mi = sy * srcW + sx
                if (mask[mi] > 0f) {
                    colors[y * ovW + x] = (MASK_ALPHA shl 24) or (r shl 16) or (g shl 8) or b
                } else {
                    colors[y * ovW + x] = 0x00000000  // transparent
                }
            }
        }
        return Bitmap.createBitmap(colors, ovW, ovH, Bitmap.Config.ARGB_8888)
    }

    /** Semi-transparent mask overlay alpha (matches the original look). */
    private const val MASK_ALPHA = 120
}

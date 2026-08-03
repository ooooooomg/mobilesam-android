package com.example.mobilesam

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF

/**
 * Composes segmentation masks and detection boxes into an overlay bitmap.
 *
 * Uses Canvas drawBitmap for mask coloring (fast, single native draw per
 * mask) instead of per-pixel setPixel (which is extremely slow on large
 * images). Masks arrive at original-image resolution as float logits
 * (positive => foreground); we downscale the mask to a small overlay and
 * stretch it over the canvas so a 3000x3000 image never loops per-pixel.
 */
object MaskComposer {

    private val MASK_COLORS = intArrayOf(
        0xFFFF4081.toInt(), 0xFF7C4DFF.toInt(), 0xFF00E676.toInt(), 0xFFFFAB40.toInt(),
        0xFF40C4FF.toInt(), 0xFFFF5252.toInt(), 0xFF69F0AE.toInt(), 0xFFD500F9.toInt(),
    )

    /**
     * Draw each mask (HxW logits, thresholded at 0) as a colored,
     * semi-transparent overlay on a copy of the original bitmap.
     */
    fun compose(
        source: Bitmap,
        masks: List<SamMaskDecoder.SegmentedMask>,
        boxes: List<YoloDetector.DetBox>,
    ): Bitmap {
        val w = source.width
        val h = source.height
        val result = source.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)

        // 1. Draw masks using low-res alpha bitmaps stretched to canvas size.
        masks.forEachIndexed { idx, seg ->
            if (seg.width == 0 || seg.height == 0) return@forEachIndexed
            val color = MASK_COLORS[idx % MASK_COLORS.size]
            val overlay = buildMaskOverlay(seg, color)
            if (overlay != null) {
                val srcRect = android.graphics.Rect(
                    0, 0, overlay.width, overlay.height
                )
                val dstRect = android.graphics.Rect(0, 0, w, h)
                canvas.drawBitmap(overlay, srcRect, dstRect, null)
                overlay.recycle()
            }
        }

        // 2. Draw boxes.
        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = (h * 0.006f).coerceAtLeast(3f)
        }
        boxes.forEachIndexed { idx, b ->
            boxPaint.color = MASK_COLORS[idx % MASK_COLORS.size]
            canvas.drawRect(RectF(b.x1, b.y1, b.x2, b.y2), boxPaint)
        }
        return result
    }

    /**
     * Build a colored overlay bitmap (at most 512x512) from the mask logits.
     * Foreground pixels (logit > 0) become the mask color with alpha 120;
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

        // Downscale to at most 512x512 for the overlay (fast enough, and
        // final stretch by the canvas gives a smooth result).
        val scale = 512.0 / maxOf(srcW, srcH)
        val ovW = (srcW * scale).toInt().coerceAtLeast(1)
        val ovH = (srcH * scale).toInt().coerceAtLeast(1)

        val px = ovW * ovH
        val colors = IntArray(px)
        val r = (color shr 16) and 0xFF
        val g = (color shr 8) and 0xFF
        val b = color and 0xFF
        val alpha = 120
        val stepX = srcW / ovW.toDouble()
        val stepY = srcH / ovH.toDouble()
        for (y in 0 until ovH) {
            val sy = (y * stepY).toInt()
            for (x in 0 until ovW) {
                val sx = (x * stepX).toInt()
                val mi = sy * srcW + sx
                if (mask[mi] > 0f) {
                    colors[y * ovW + x] = (alpha shl 24) or (r shl 16) or (g shl 8) or b
                } else {
                    colors[y * ovW + x] = 0x00000000  // transparent
                }
            }
        }
        return Bitmap.createBitmap(colors, ovW, ovH, Bitmap.Config.ARGB_8888)
    }
}

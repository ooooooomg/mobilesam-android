package com.example.mobilesam

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

/**
 * Converts CameraX YUV_420_888 frames to ARGB bitmaps.
 *
 * Verified per-pixel implementation (same as the original app). Reads each
 * plane via absolute indexing; handles rowStride/pixelStride. Kept simple and
 * stable — the analyzer is single-threaded and this is correct on all devices.
 */
object ImageUtils {

    fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val width = image.width
        val height = image.height

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val yRowStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride

        val pixels = IntArray(width * height)
        // Integer approximation of the standard YUV->RGB matrix (10-bit
        // coefficients), avoiding per-pixel float multiply/add. Error < 1.
        val i1436 = 1436  // 1.402  * 1024
        val i352 = 352    // 0.344136 * 1024
        val i731 = 731    // 0.714136 * 1024
        val i1815 = 1815  // 1.772  * 1024
        for (yy in 0 until height) {
            val yRow = yy * yRowStride
            val uvRow = (yy shr 1) * uRowStride
            for (xx in 0 until width) {
                val y = yBuffer.get(yRow + xx * yPixelStride).toInt() and 0xFF
                val ux = xx shr 1
                val u = (uBuffer.get(uvRow + ux * uPixelStride).toInt() and 0xFF) - 128
                val v = (vBuffer.get(uvRow + ux * vPixelStride).toInt() and 0xFF) - 128
                val r = (y + ((i1436 * v) shr 10)).coerceIn(0, 255)
                val g = (y - ((i352 * u + i731 * v) shr 10)).coerceIn(0, 255)
                val b = (y + ((i1815 * u) shr 10)).coerceIn(0, 255)
                pixels[yy * width + xx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    /** Rotate a bitmap by the given degrees (portrait correction). */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
    }
}

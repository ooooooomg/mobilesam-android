package com.example.mobilesam

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max

/**
 * MobileSAM (vit_t) image encoder on ONNX Runtime.
 *
 * Input: RGB bitmap. Output: (1, 256, 24, 24) embedding as FloatArray.
 * Preprocessing matches the Python reference:
 * ResizeLongestSide(384) -> normalize -> pad to 384x384.
 */
class SamImageEncoder(
    private val environment: OrtEnvironment,
    modelBytes: ByteArray,
) {
    private val session: OrtSession

    init {
        // NNAPI 在部分设备上初始化极慢且算子混合执行，性能反而更差。
        // 统一用 CPU 多线程，稳定且可预期。
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        opts.setInterOpNumThreads(1)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = environment.createSession(modelBytes, opts)
        android.util.Log.d("MobileSAM", "encoder CPU threads=4/1")
    }

    companion object {
        private const val IMG_SIZE = 384
        // SAM pixel mean/std
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
        // Output dims (384 -> 24x24)
        private const val OUT_CHANNELS = 256
        private const val OUT_HW = 24
    }

    data class Preprocessed(val input: FloatArray, val scale: Float, val newH: Int, val newW: Int)

    /** ResizeLongestSide + normalize + pad, returns NCHW float. */
    fun preprocess(bitmap: Bitmap): Preprocessed {
        val w = bitmap.width
        val h = bitmap.height
        val scale = IMG_SIZE.toFloat() / max(w, h)
        val newW = (w * scale).toInt().coerceAtLeast(1)
        val newH = (h * scale).toInt().coerceAtLeast(1)
        val resized = Bitmap.createScaledBitmap(bitmap, newW, newH, true)

        // Read pixels into a padded 384x384 ARGB buffer (zeros = black)
        val padded = IntArray(IMG_SIZE * IMG_SIZE)
        val px = IntArray(newW * newH)
        resized.getPixels(px, 0, newW, 0, 0, newW, newH)
        for (y in 0 until newH) {
            System.arraycopy(px, y * newW, padded, y * IMG_SIZE, newW)
        }
        resized.recycle()

        // Normalize -> NCHW
        val input = FloatArray(3 * IMG_SIZE * IMG_SIZE)
        val total = IMG_SIZE * IMG_SIZE
        for (y in 0 until IMG_SIZE) {
            for (x in 0 until IMG_SIZE) {
                val c = padded[y * IMG_SIZE + x]
                val r = Color.red(c)
                val g = Color.green(c)
                val b = Color.blue(c)
                val i = y * IMG_SIZE + x
                input[i] = (r - MEAN[0]) / STD[0]
                input[total + i] = (g - MEAN[1]) / STD[1]
                input[2 * total + i] = (b - MEAN[2]) / STD[2]
            }
        }
        return Preprocessed(input, scale, newH, newW)
    }

    /** Encode a bitmap to (1, 256, 24, 24) embedding. */
    fun encode(bitmap: Bitmap): Pair<FloatArray, Preprocessed> {
        val prep = preprocess(bitmap)
        return encodePreprocessed(prep) to prep
    }

    /**
     * Run the encoder session for an already-preprocessed input.
     * Called on the background encoder thread in camera mode.
     */
    fun encodePreprocessed(prep: Preprocessed): FloatArray {
        val tensor = OnnxTensor.createTensor(
            environment,
            java.nio.FloatBuffer.wrap(prep.input),
            longArrayOf(1, 3, IMG_SIZE.toLong(), IMG_SIZE.toLong())
        )
        val results = session.run(mapOf("image" to tensor))
        tensor.close()
        // onnx output shape (1, 256, 24, 24), flattened. Result.get(int)
        // returns OnnxValue; cast to OnnxTensor (not .value).
        val outTensor = results.get(0) as OnnxTensor
        val embedding = outTensor.floatBuffer ?: throw IllegalStateException("bad output")
        results.close()
        val out = FloatArray(embedding.capacity())
        embedding.rewind()
        embedding.get(out)
        return out
    }

    val outputSize: Int
        get() = OUT_CHANNELS * OUT_HW * OUT_HW

    fun close() {
        session.close()
    }
}

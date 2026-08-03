package com.example.mobilesam

import ai.onnxruntime.OrtEnvironment
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Real-time camera segmentation: CameraX -> YOLO detect -> MobileSAM 512
 * encoder -> decoder masks -> overlay on PreviewView.
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusText: TextView

    private var pipeline: InferencePipeline? = null
    private val busy = AtomicBoolean(false)
    private val inferenceJob = Job()
    private val scope = CoroutineScope(Dispatchers.Default + inferenceJob)
    private val inferenceMutex = Mutex()
    private var cameraExecutor = Executors.newSingleThreadExecutor()

    private val ENCODER_ASSET = "mobile_sam_encoder.onnx"
    private val DECODER_ASSET = "mobile_sam_decoder.onnx"
    private val YOLO_ASSET = "yolov8n.onnx"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusText = findViewById(R.id.statusText)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private val requestCameraPermission =
        registerForActivityResult(androidx.activity.result.contract.ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else statusText.text = "需要相机权限"
        }

    private fun startCamera() {
        statusText.text = "加载模型..."
        scope.launch {
            val env = OrtEnvironment.getEnvironment()
            val encoderBytes = assets.open(ENCODER_ASSET).use { it.readBytes() }
            val decoderBytes = assets.open(DECODER_ASSET).use { it.readBytes() }
            val yoloBytes = assets.open(YOLO_ASSET).use { it.readBytes() }
            pipeline = InferencePipeline(
                SamImageEncoder(env, encoderBytes),
                SamMaskDecoder(env, decoderBytes),
                YoloDetector(env, yoloBytes),
            )
            bindCamera()
        }
    }

    private fun bindCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                // Low resolution for real-time 30fps (preview stays full-res).
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(android.util.Size(640, 480))
                    .build()
                analysis.setAnalyzer(cameraExecutor) { image ->
                    analyzeFrame(image)
                }
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
                runOnUiThread { statusText.text = "实时分割中" }
            } catch (e: Exception) {
                runOnUiThread { statusText.text = "相机启动失败: ${e.message}" }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        if (busy.get()) {
            image.close()
            return
        }
        busy.set(true)
        try {
            var t0 = System.currentTimeMillis()
            val raw = ImageUtils.imageProxyToBitmap(image)
            if (raw == null) return
            val tConvert = System.currentTimeMillis() - t0

            // Rotate according to the frame's actual rotation metadata.
            val rotation = image.imageInfo.rotationDegrees
            t0 = System.currentTimeMillis()
            val bitmap = if (rotation != 0) {
                ImageUtils.rotateBitmap(raw, rotation.toFloat())
            } else {
                raw
            }
            if (bitmap !== raw) raw.recycle()
            val tRotate = System.currentTimeMillis() - t0

            val pipe = pipeline
            if (pipe != null) {
                val tInfer0 = System.currentTimeMillis()
                val result = pipe.run(bitmap)
                val tInfer = System.currentTimeMillis() - tInfer0
                val total = tConvert + tRotate + tInfer
                val fps = 1000.0 / total.coerceAtLeast(1)
                val encMark = if (result.encodedThisFrame) "ENC" else "trk"
                runOnUiThread {
                    overlayView.setResult(result.overlay)
                    statusText.text = "${result.objectCount}物体 · ${fps.toInt()}fps · $encMark · " +
                        "conv${tConvert}ms rot${tRotate}ms inf${tInfer}ms"
                }
                bitmap.recycle()
            } else {
                bitmap.recycle()
            }
        } catch (e: Exception) {
            runOnUiThread { statusText.text = "处理失败: ${e.message?.take(40)}" }
        } finally {
            image.close()
            busy.set(false)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceJob.cancel()
        cameraExecutor.shutdown()
    }
}

/** Custom view drawing the segmentation overlay above the preview. */
class OverlayView @JvmOverloads constructor(
    context: android.content.Context,
    attrs: android.util.AttributeSet? = null,
) : View(context, attrs) {

    private var overlay: Bitmap? = null
    private val paint = Paint().apply { isFilterBitmap = true }

    fun setResult(bmp: Bitmap) {
        val old = overlay
        overlay = bmp
        old?.recycle()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        overlay?.let { bmp ->
            // Fit-center: keep aspect ratio, center within the view.
            // Mirrors PreviewView's default scale type so the overlay aligns.
            val vw = width.toFloat()
            val vh = height.toFloat()
            val bmpW = bmp.width.toFloat()
            val bmpH = bmp.height.toFloat()
            if (bmpW <= 0f || bmpH <= 0f) return
            val scale = minOf(vw / bmpW, vh / bmpH)
            val dw = bmpW * scale
            val dh = bmpH * scale
            val left = (vw - dw) / 2f
            val top = (vh - dh) / 2f
            canvas.drawBitmap(
                bmp,
                null,
                RectF(left, top, left + dw, top + dh),
                paint,
            )
        }
    }
}

/** Converts a CameraX ImageProxy (YUV_420_888) to an RGB Bitmap. */
object ImageUtils {
    fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        // Direct YUV_420_888 -> RGB, no JPEG round-trip (fast path for
        // real-time). Handles rowStride/pixelStride per plane.
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
        var yRow = 0
        for (yy in 0 until height) {
            // Position each plane's buffer at this row's start.
            yBuffer.position(yy * yRowStride)
            uBuffer.position((yy / 2) * uRowStride)
            vBuffer.position((yy / 2) * vRowStride)
            for (xx in 0 until width) {
                val y = (yBuffer.get(yy * yRowStride + xx * yPixelStride).toInt() and 0xFF)
                val ux = xx / 2
                val u = (uBuffer.get((yy / 2) * uRowStride + ux * uPixelStride).toInt() and 0xFF) - 128
                val v = (vBuffer.get((yy / 2) * vRowStride + ux * vPixelStride).toInt() and 0xFF) - 128
                val r = (y + 1.402f * v).toInt().coerceIn(0, 255)
                val g = (y - 0.344136f * u - 0.714136f * v).toInt().coerceIn(0, 255)
                val b = (y + 1.772f * u).toInt().coerceIn(0, 255)
                pixels[yy * width + xx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    /** Rotate a bitmap by the given degrees (portrait correction). */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        if (degrees == 0f) return bitmap
        val matrix = android.graphics.Matrix().apply { postRotate(degrees) }
        val rotated = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
        return rotated
    }
}

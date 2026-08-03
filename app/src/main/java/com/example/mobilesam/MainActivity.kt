package com.example.mobilesam

import ai.onnxruntime.OrtEnvironment
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private var pipeline: InferencePipeline? = null
    private val scope = CoroutineScope(Dispatchers.Default)

    // 模型资源（放 assets/ 下）
    private val ENCODER_ASSET = "mobile_sam_encoder.onnx"
    private val DECODER_ASSET = "mobile_sam_decoder.onnx"
    private val YOLO_ASSET = "yolov8n.onnx"

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { loadAndSegment(it) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        statusText = findViewById(R.id.statusText)
        val pickBtn = findViewById<Button>(R.id.pickButton)
        val demoBtn = findViewById<Button>(R.id.demoButton)

        pickBtn.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                != PackageManager.PERMISSION_GRANTED
            ) {
                requestPermission.launch(Manifest.permission.READ_MEDIA_IMAGES)
            } else {
                pickImage.launch("image/*")
            }
        }
        demoBtn.setOnClickListener { runDemoImage() }

        loadModels()
    }

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) pickImage.launch("image/*")
            else statusText.text = "需要图片读取权限"
        }

    private fun loadModels() {
        statusText.text = "加载模型..."
        scope.launch {
            val env = OrtEnvironment.getEnvironment()
            val encoderBytes = assets.open(ENCODER_ASSET).use { it.readBytes() }
            val decoderBytes = assets.open(DECODER_ASSET).use { it.readBytes() }
            val yoloBytes = assets.open(YOLO_ASSET).use { it.readBytes() }
            val pipe = InferencePipeline(
                SamImageEncoder(env, encoderBytes),
                SamMaskDecoder(env, decoderBytes),
                YoloDetector(env, yoloBytes),
            )
            pipeline = pipe
            withContext(Dispatchers.Main) { statusText.text = "模型就绪，选择图片开始" }
        }
    }

    private fun loadAndSegment(uri: Uri) {
        statusText.text = "处理中..."
        scope.launch {
            try {
                val bmp = withContext(Dispatchers.IO) {
                    decodeSampledBitmap(uri, 2048)
                }
                val pipe = pipeline
                if (bmp == null || pipe == null) {
                    withContext(Dispatchers.Main) { statusText.text = "加载失败" }
                    return@launch
                }
                val result = pipe.run(bmp)
                bmp.recycle()
                withContext(Dispatchers.Main) {
                    imageView.setImageBitmap(result.overlay)
                    statusText.text = "分割 ${result.objectCount} 个物体，耗时 ${result.inferenceMs}ms"
                }
            } catch (e: OutOfMemoryError) {
                withContext(Dispatchers.Main) { statusText.text = "内存不足，图片过大" }
                System.gc()
            } catch (e: Exception) {
                val detail = buildString {
                    append(e.javaClass.simpleName)
                    e.message?.let { append(": ").append(it) }
                    append("\n")
                    // 找 ONNX Runtime 具体错误
                    var t: Throwable? = e
                    var depth = 0
                    while (t != null && depth < 6) {
                        t.stackTrace?.firstOrNull()?.let {
                            append("at ").append(it.className).append(".").append(it.methodName)
                            append("(").append(it.fileName ?: "?").append(":")
                            append(it.lineNumber).append(")\n")
                        }
                        t = t.cause
                        depth++
                    }
                }
                withContext(Dispatchers.Main) {
                    statusText.text = "处理失败: $detail"
                    android.util.Log.e("MobileSAM", "segment failed", e)
                }
            }
        }
    }

    /** 降采样解码，限制最大边不超过 maxDimension，避免大图 OOM。 */
    private fun decodeSampledBitmap(uri: Uri, maxDimension: Int): Bitmap? {
        contentResolver.openInputStream(uri)?.use { input ->
            // 第一次只读尺寸
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeStream(input, null, opts)
            val w = opts.outWidth
            val h = opts.outHeight
            if (w <= 0 || h <= 0) return null
            // 计算采样率（2 的幂）
            var sample = 1
            while (w / sample > maxDimension || h / sample > maxDimension) {
                sample *= 2
            }
            // 第二次真正解码
            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sample
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            contentResolver.openInputStream(uri)?.use { input2 ->
                val bmp = BitmapFactory.decodeStream(input2, null, decodeOpts) ?: return null
                return applyExifOrientation(bmp, uri)
            }
        }
        return null
    }

    /** 应用 EXIF 旋转，避免手机照片横竖颠倒导致分割错位。 */
    private fun applyExifOrientation(bitmap: Bitmap, uri: Uri): Bitmap {
        return try {
            val exif = androidx.exifinterface.media.ExifInterface(
                contentResolver.openInputStream(uri)
                    ?: return bitmap
            )
            val orientation = exif.getAttributeInt(
                androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,
                androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL
            )
            val matrix = android.graphics.Matrix()
            when (orientation) {
                androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90 ->
                    matrix.postRotate(90f)
                androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180 ->
                    matrix.postRotate(180f)
                androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270 ->
                    matrix.postRotate(270f)
                else -> return bitmap
            }
            val rotated = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )
            if (rotated != bitmap) bitmap.recycle()
            rotated
        } catch (e: Exception) {
            bitmap
        }
    }

    /** 内置测试图（res/drawable/demo.png 若存在），无需选图即可验证。 */
    private fun runDemoImage() {
        val id = resources.getIdentifier("demo", "drawable", packageName)
        if (id == 0) {
            statusText.text = "无内置测试图，请点'选择图片'"
            return
        }
        val bmp = BitmapFactory.decodeResource(resources, id)
        statusText.text = "处理中..."
        scope.launch {
            val pipe = pipeline ?: return@launch
            val result = pipe.run(bmp)
            withContext(Dispatchers.Main) {
                imageView.setImageBitmap(result.overlay)
                statusText.text = "分割 ${result.objectCount} 个物体，耗时 ${result.inferenceMs}ms"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
    }
}

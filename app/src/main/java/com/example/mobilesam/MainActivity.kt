package com.example.mobilesam

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.progressindicator.CircularProgressIndicator
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var progressBar: CircularProgressIndicator
    private var pipeline: InferencePipeline? = null
    private val scope = CoroutineScope(Dispatchers.Default)

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { loadAndSegment(it) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val root = findViewById<android.widget.LinearLayout>(R.id.mainRoot)
        val basePad = resources.getDimensionPixelSize(R.dimen.spacing_l)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(
                basePad + bars.left,
                basePad + bars.top,
                basePad + bars.right,
                basePad + bars.bottom,
            )
            WindowInsetsCompat.CONSUMED
        }

        imageView = findViewById(R.id.imageView)
        statusText = findViewById(R.id.statusText)
        progressBar = findViewById(R.id.progressBar)
        val pickBtn = findViewById<View>(R.id.pickButton)
        val backToCameraBtn = findViewById<View>(R.id.backToCameraButton)

        findViewById<View>(R.id.backButton).setOnClickListener { finish() }
        backToCameraBtn.setOnClickListener { finish() }

        pickBtn.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                != PackageManager.PERMISSION_GRANTED
            ) {
                requestPermission.launch(Manifest.permission.READ_MEDIA_IMAGES)
            } else {
                pickImage.launch("image/*")
            }
        }

        loadModels()
    }

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) pickImage.launch("image/*")
            else statusText.text = getString(R.string.need_image_permission)
        }

    private fun loadModels() {
        statusText.text = getString(R.string.loading_models)
        scope.launch {
            try {
                // Same model the camera page last selected.
                val prefs = getSharedPreferences("mobilesam_prefs", MODE_PRIVATE)
                val info = ModelRegistry.byId(prefs.getString("model_id", null) ?: "")
                    ?: ModelRegistry.default()
                pipeline = InferencePipeline(SegmenterFactory.create(this@MainActivity, info))
                withContext(Dispatchers.Main) {
                    statusText.text = getString(R.string.models_ready)
                }
            } catch (e: Exception) {
                android.util.Log.e("MobileSAM", "model load failed", e)
                withContext(Dispatchers.Main) {
                    statusText.text = getString(R.string.model_load_failed) + ": ${e.message?.take(60)}"
                }
            }
        }
    }

    private fun loadAndSegment(uri: Uri) {
        statusText.text = getString(R.string.processing)
        progressBar.visibility = View.VISIBLE
        scope.launch {
            try {
                val bmp = withContext(Dispatchers.IO) {
                    decodeSampledBitmap(uri, 2048)
                }
                val pipe = pipeline
                if (bmp == null || pipe == null) {
                    withContext(Dispatchers.Main) {
                        statusText.text = getString(R.string.load_failed)
                        progressBar.visibility = View.GONE
                    }
                    return@launch
                }
                val result = pipe.runSingle(bmp)
                bmp.recycle()
                withContext(Dispatchers.Main) {
                    imageView.setImageBitmap(result.overlay)
                    statusText.text = getString(
                        R.string.segmented_format, result.objectCount, result.totalMs
                    )
                    progressBar.visibility = View.GONE
                }
            } catch (e: OutOfMemoryError) {
                withContext(Dispatchers.Main) {
                    statusText.text = getString(R.string.memory_insufficient)
                    progressBar.visibility = View.GONE
                }
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
                    statusText.text = getString(R.string.processing_failed) + ": $detail"
                    progressBar.visibility = View.GONE
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

    override fun onDestroy() {
        super.onDestroy()
        pipeline?.close()
    }
}

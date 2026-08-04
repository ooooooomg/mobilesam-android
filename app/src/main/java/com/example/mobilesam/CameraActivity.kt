package com.example.mobilesam

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Real-time camera segmentation: CameraX -> YOLO detect -> MobileSAM encoder
 * (async) -> decoder masks -> transparent overlay on PreviewView.
 */
class CameraActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var topBar: View
    private lateinit var modelPickerButton: View
    private lateinit var modelTitleText: TextView
    private lateinit var hudFps: TextView
    private lateinit var legendText: TextView
    private lateinit var legendPanel: View
    private lateinit var loadingChip: View

    private var pipeline: InferencePipeline? = null
    private val busy = AtomicBoolean(false)
    private val inferenceJob = Job()
    private val scope = CoroutineScope(Dispatchers.Default + inferenceJob)
    private var cameraExecutor = Executors.newSingleThreadExecutor()
    private var useFrontCamera = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        topBar = findViewById(R.id.topBar)
        modelPickerButton = findViewById(R.id.modelPickerButton)
        modelTitleText = findViewById(R.id.modelTitleText)
        hudFps = findViewById(R.id.hudFps)
        legendText = findViewById(R.id.legendText)
        legendPanel = findViewById(R.id.legendPanel)
        loadingChip = findViewById(R.id.loadingChip)

        overlayView.isClickable = false

        setupModelPicker()
        setupTabs()

        // Frost the regions behind the top/bottom bars once laid out.
        findViewById<View>(R.id.cameraRoot).post {
            updateFrostRegions()
        }

        findViewById<View>(R.id.flipButton).setOnClickListener {
            useFrontCamera = !useFrontCamera
            bindCamera()
        }

        // Apply system-bar insets so the top bar and tab bar don't sit under
        // the status/navigation bars (Android 15 enforces edge-to-edge).
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.cameraRoot)) { _, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val topLp = topBar.layoutParams as ViewGroup.MarginLayoutParams
            topLp.topMargin = bars.top + dp(12)
            topBar.layoutParams = topLp
            val tabParams = findViewById<View>(R.id.tabBar).layoutParams as ViewGroup.MarginLayoutParams
            tabParams.bottomMargin = bars.bottom + dp(20)
            findViewById<View>(R.id.tabBar).layoutParams = tabParams
            val legendParams = legendPanel.layoutParams as ViewGroup.MarginLayoutParams
            legendParams.bottomMargin = bars.bottom + resources.getDimensionPixelSize(R.dimen.legend_margin_bottom)
            legendPanel.layoutParams = legendParams
            WindowInsetsCompat.CONSUMED
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    /** Compute the on-screen rects of the top/bottom bars for frost blur. */
    private fun updateFrostRegions() {
        val regions = ArrayList<RectF>()
        val root = findViewById<View>(R.id.cameraRoot)
        val rootLoc = IntArray(2)
        root.getLocationOnScreen(rootLoc)
        fun barRect(v: View): RectF? {
            if (v.width <= 0 || v.height <= 0) return null
            val loc = IntArray(2)
            v.getLocationOnScreen(loc)
            // Convert to root-local coords (the overlay fills the root), so the
            // regions are immune to status-bar / window offsets.
            return RectF(
                (loc[0] - rootLoc[0]).toFloat(),
                (loc[1] - rootLoc[1]).toFloat(),
                (loc[0] - rootLoc[0] + v.width).toFloat(),
                (loc[1] - rootLoc[1] + v.height).toFloat(),
            )
        }
        barRect(topBar)?.let { regions.add(it) }
        barRect(findViewById<View>(R.id.tabBar))?.let { regions.add(it) }
        overlayView.setFrostRegions(regions)
    }

    /** Bottom icons: album -> MainActivity; seg saves frame; settings page. */
    private fun setupTabs() {
        findViewById<View>(R.id.tabAlbum).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }
        findViewById<View>(R.id.tabSeg).setOnClickListener {
            saveCurrentFrame()
        }
        findViewById<View>(R.id.tabSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
    }

    /** Save the current segmented overlay to the system gallery via MediaStore. */
    private fun saveCurrentFrame() {
        val bmp = overlayView.currentOverlay() ?: run {
            android.widget.Toast.makeText(this, R.string.save_no_frame, android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        scope.launch {
            try {
                val saved = withContext(kotlinx.coroutines.Dispatchers.IO) {
                    saveToGallery(bmp)
                }
                runOnUiThread {
                    android.widget.Toast.makeText(
                        this@CameraActivity,
                        if (saved) R.string.save_ok else R.string.save_failed,
                        android.widget.Toast.LENGTH_SHORT,
                    ).show()
                }
            } catch (e: Exception) {
                android.util.Log.e("MobileSAM", "save failed", e)
                runOnUiThread {
                    android.widget.Toast.makeText(
                        this@CameraActivity, R.string.save_failed, android.widget.Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    private fun saveToGallery(bmp: android.graphics.Bitmap): Boolean {
        val resolver = contentResolver
        val name = "mobilesam_" + System.currentTimeMillis() + ".png"
        val values = android.content.ContentValues().apply {
            put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, name)
            put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, android.os.Environment.DIRECTORY_PICTURES + "/MobileSAM")
        }
        val uri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: return false
        return try {
            resolver.openOutputStream(uri)?.use { out ->
                bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
            } != null
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            false
        }
    }

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                loadingChip.visibility = View.GONE
                cancelLegendHide()
                legendText.text = getString(R.string.need_camera_permission)
                legendPanel.visibility = View.VISIBLE
            }
        }

    private fun startCamera() {
        scope.launch {
            try {
                val info = selectedModel()
                pipeline = InferencePipeline(SegmenterFactory.create(this@CameraActivity, info))
                runOnUiThread {
                    loadingChip.visibility = View.GONE
                    modelTitleText.text = info.friendlyName
                }
            } catch (e: Exception) {
                android.util.Log.e("MobileSAM", "model load failed", e)
                runOnUiThread {
                    loadingChip.visibility = View.GONE
                    cancelLegendHide()
                    legendText.text = getString(R.string.model_load_failed) + ": ${e.message?.take(60)}"
                    legendPanel.visibility = View.VISIBLE
                }
            }
            bindCamera()
        }
    }

    /** The model id persisted by the picker, or the registry default. */
    private fun selectedModel(): ModelRegistry.ModelInfo {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val id = prefs.getString(KEY_MODEL_ID, null)
        return ModelRegistry.byId(id ?: "") ?: ModelRegistry.default()
    }

    private fun setupModelPicker() {
        val info = selectedModel()
        modelTitleText.text = info.friendlyName
        modelPickerButton.setOnClickListener {
            ModelPickerDialog(
                this,
                ModelRegistry.models,
                selectedModel().id,
                { info -> switchToModel(info) },
            ).show()
        }
    }

    private fun switchToModel(info: ModelRegistry.ModelInfo) {
        if (info.id == selectedModel().id) return
        // Persist selection immediately so a relaunch uses it.
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            .edit().putString(KEY_MODEL_ID, info.id).apply()

        loadingChip.visibility = View.VISIBLE
        legendPanel.visibility = View.GONE
        scope.launch {
            try {
                val newSegmenter = SegmenterFactory.create(this@CameraActivity, info)
                val pipe = pipeline
                if (pipe == null) {
                    // Camera not started yet: just record the choice; startCamera
                    // will read it.
                    newSegmenter.close()
                    return@launch
                }
                pipe.switchSegmenter(newSegmenter)
                runOnUiThread {
                    loadingChip.visibility = View.GONE
                    modelTitleText.text = info.friendlyName
                }
            } catch (e: Exception) {
                android.util.Log.e("MobileSAM", "model switch failed", e)
                runOnUiThread {
                    loadingChip.visibility = View.GONE
                    cancelLegendHide()
                    legendText.text = getString(R.string.model_load_failed) + ": ${e.message?.take(60)}"
                    legendPanel.visibility = View.VISIBLE
                }
            }
        }
    }

    /** Rebuild the danmaku-style legend chips only when the class set changes. */
    private var legendSignature: String? = null
    private val legendHide = android.os.Handler(android.os.Looper.getMainLooper())
    private var legendHidePending = false
    private val legendHideRunnable = Runnable {
        legendPanel.visibility = View.GONE
        legendSignature = null
        legendHidePending = false
    }

    /** Cancel any pending danmaku-hide so status/error text stays visible. */
    private fun cancelLegendHide() {
        legendHide.removeCallbacks(legendHideRunnable)
        legendHidePending = false
    }

    private fun updateLegend(classes: List<Int>) {
        val sig = classes.distinct().sorted().take(4).joinToString(",")
        if (sig.isEmpty()) {
            // Hold the chips for a moment after the object disappears so they
            // stay readable; error/status messages bypass this delay.
            if (!legendHidePending) {
                legendHidePending = true
                legendHide.postDelayed(legendHideRunnable, 3000)
            }
            return
        }
        legendHide.removeCallbacks(legendHideRunnable)
        legendHidePending = false
        if (sig == legendSignature) return
        legendSignature = sig

        // Keep the legendText (error-message) child; drop any rebuilt chips.
        val panel = legendPanel as android.view.ViewGroup
        panel.removeViews(1, (panel.childCount - 1).coerceAtLeast(0))
        legendText.text = ""
        val shown = classes.distinct().take(3)
        shown.forEachIndexed { i, cls ->
            val chip = android.widget.TextView(this).apply {
                text = "● ${CocoLabels.chineseFor(cls)}"
                textSize = 16f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setTextColor(context.getColor(R.color.text_primary))
                setBackgroundResource(R.drawable.bg_legend_item)
                setPadding(dp(12), dp(8), dp(12), dp(8))
            }
            panel.addView(chip, android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
            ))
            if (i < shown.size - 1) {
                val spacer = View(this)
                spacer.layoutParams = android.view.ViewGroup.LayoutParams(dp(8), 1)
                panel.addView(spacer)
            }
        }
        if (classes.distinct().size > 3) {
            val more = android.widget.TextView(this).apply {
                text = "＋${classes.distinct().size - 3}"
                textSize = 16f
                setTextColor(context.getColor(R.color.text_secondary))
                setBackgroundResource(R.drawable.bg_legend_item)
                setPadding(dp(12), dp(8), dp(12), dp(8))
            }
            panel.addView(more)
        }
        legendPanel.visibility = View.VISIBLE
        // Danmaku float-up: each chip rises from below with a staggered fade-in.
        for (i in 0 until panel.childCount) {
            val child = panel.getChildAt(i)
            child.alpha = 0f
            child.translationY = dp(18).toFloat()
            child.animate()
                .alpha(1f)
                .translationY(0f)
                .setStartDelay((i * 60).toLong())
                .setDuration(280)
                .start()
        }
    }

    private fun bindCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()

                // Use the transparent TextureView implementation so the
                // overlay's mask shows through cleanly.
                previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                // FILL_CENTER crop-fills the screen (no letterbox bands); the
                // overlay applies the same crop transform below so masks align.
                previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

                val preview = Preview.Builder()
                    .setTargetResolution(android.util.Size(480, 360))
                    .build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(android.util.Size(480, 360))
                    .build()
                analysis.setAnalyzer(cameraExecutor) { image ->
                    analyzeFrame(image)
                }
                provider.unbindAll()
                val cameraSelector = if (useFrontCamera) {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
                provider.bindToLifecycle(this, cameraSelector, preview, analysis)
                runOnUiThread { hudFps.text = getString(R.string.initializing) }
            } catch (e: Exception) {
                android.util.Log.e("MobileSAM", "camera bind failed", e)
                runOnUiThread {
                    cancelLegendHide()
                    legendText.text = getString(R.string.camera_start_failed) + ": ${e.message ?: e.javaClass.simpleName}"
                    legendPanel.visibility = View.VISIBLE
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        if (busy.get()) {
            image.close()
            return
        }
        busy.set(true)
        var bitmap: Bitmap? = null
        try {
            val raw = ImageUtils.imageProxyToBitmap(image)
            if (raw == null) return

            // Rotate according to the frame's actual rotation metadata.
            val rotation = image.imageInfo.rotationDegrees
            bitmap = if (rotation != 0) {
                ImageUtils.rotateBitmap(raw, rotation.toFloat())
            } else {
                raw
            }
            if (bitmap !== raw) raw.recycle()

            val pipe = pipeline
            if (pipe != null) {
                val result = pipe.onFrame(bitmap)
                val fps = 1000.0 / result.totalMs.coerceAtLeast(1)
                runOnUiThread {
                    overlayView.setResult(result.overlay)
                    hudFps.text = fps.toInt().toString() + " fps"
                    updateLegend(result.detectedClasses)
                }
            }
        } catch (e: Exception) {
            android.util.Log.e("MobileSAM", "analyzeFrame failed", e)
            runOnUiThread {
                cancelLegendHide()
                legendText.text = getString(R.string.processing_failed) + ": ${e.message ?: e.javaClass.simpleName}"
                legendPanel.visibility = View.VISIBLE
            }
        } catch (e: OutOfMemoryError) {
            // Never crash the app on a heavy frame; skip it and let GC recover.
            android.util.Log.e("MobileSAM", "analyzeFrame OOM", e)
            System.gc()
        } finally {
            bitmap?.recycle()
            image.close()
            busy.set(false)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        legendHide.removeCallbacks(legendHideRunnable)
        inferenceJob.cancel()
        cameraExecutor.shutdown()
        pipeline?.close()
    }

    companion object {
        private const val PREFS_NAME = "mobilesam_prefs"
        private const val KEY_MODEL_ID = "model_id"

        private fun dp(v: Int): Int =
            (v * android.content.res.Resources.getSystem().displayMetrics.density).toInt()
    }
}

/** Custom view drawing the segmentation overlay above the preview. */
class OverlayView @JvmOverloads constructor(
    context: android.content.Context,
    attrs: android.util.AttributeSet? = null,
) : View(context, attrs) {

    private var overlay: Bitmap? = null
    private val paint = Paint().apply { isFilterBitmap = true }
    private val frostPaint = Paint().apply { isFilterBitmap = true }
    private val appCornerPx = resources.getDimension(R.dimen.app_corner)

    /** Screen-space regions (top bar / bottom bar) that get a frosted-glass blur. */
    private var frostRegions: List<android.graphics.RectF> = emptyList()

    /** The previous overlay bitmap is recycled; the caller creates fresh ones. */
    fun setResult(bmp: Bitmap) {
        val old = overlay
        overlay = bmp
        old?.recycle()
        invalidate()
    }

    /** Copy of the current segmented frame, or null if none yet. */
    fun currentOverlay(): Bitmap? = overlay?.copy(Bitmap.Config.ARGB_8888, true)

    /** Regions (in this view's coords) where the camera feed is frosted. */
    fun setFrostRegions(regions: List<android.graphics.RectF>) {
        frostRegions = regions
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        overlay?.let { bmp ->
            // Center-crop: scale so the 4:3 analysis frame fills the screen,
            // cropping the longer dimension. Mirrors PreviewView's FILL_CENTER.
            val vw = width.toFloat()
            val vh = height.toFloat()
            val bmpW = bmp.width.toFloat()
            val bmpH = bmp.height.toFloat()
            if (bmpW <= 0f || bmpH <= 0f) return
            val scale = maxOf(vw / bmpW, vh / bmpH)
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

            // Frosted-glass: down-sample each bar region to a tiny size, then
            // stretch back -> strong blur of the feed behind the bars.
            if (frostRegions.isNotEmpty()) {
                for (r in frostRegions) {
                    // Intersect with the drawn image rect.
                    val ir = RectF(left, top, left + dw, top + dh)
                    if (!ir.intersect(r)) continue
                    // Map screen region -> source (bitmap) coordinates.
                    val sx = (ir.left - left) / scale
                    val sy = (ir.top - top) / scale
                    val sw = ir.width() / scale
                    val sh = ir.height() / scale
                    if (sw < 2f || sh < 2f) continue
                    val si = android.graphics.Rect(
                        sx.toInt(), sy.toInt(),
                        (sx + sw).toInt().coerceAtMost(bmp.width),
                        (sy + sh).toInt().coerceAtMost(bmp.height),
                    )
                    if (si.width() <= 0 || si.height() <= 0) continue
                    // Tiny down-sampled version (about 1/24 of the region).
                    val tw = (si.width() / 24).coerceAtLeast(1)
                    val th = (si.height() / 24).coerceAtLeast(1)
                    val small = Bitmap.createScaledBitmap(
                        Bitmap.createBitmap(bmp, si.left, si.top, si.width(), si.height()),
                        tw, th, true,
                    )
                    // Clip to the capsule's rounded corners so the frosted
                    // region never shows a hard square edge around the bar.
                    val save = canvas.save()
                    val clipPath = android.graphics.Path().apply {
                        addRoundRect(
                            RectF(ir.left, ir.top, ir.right, ir.bottom),
                            appCornerPx, appCornerPx, android.graphics.Path.Direction.CW,
                        )
                    }
                    canvas.clipPath(clipPath)
                    canvas.drawBitmap(
                        small,
                        Rect(0, 0, tw, th),
                        RectF(ir.left, ir.top, ir.right, ir.bottom),
                        frostPaint,
                    )
                    canvas.restoreToCount(save)
                    small.recycle()
                }
            }
        }
    }
}

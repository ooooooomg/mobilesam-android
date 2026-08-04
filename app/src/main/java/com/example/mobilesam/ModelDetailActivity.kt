package com.example.mobilesam

import android.content.Intent
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Dedicated model-detail page (readable opaque background), opened from the
 * settings "查看详情" buttons.
 */
class ModelDetailActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_detail)

        val root = findViewById<LinearLayout>(R.id.detailRoot)
        val basePad = resources.getDimensionPixelSize(R.dimen.spacing_l)
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(androidx.core.view.WindowInsetsCompat.Type.systemBars())
            v.setPadding(
                basePad + bars.left,
                basePad + bars.top,
                basePad + bars.right,
                basePad + bars.bottom,
            )
            androidx.core.view.WindowInsetsCompat.CONSUMED
        }

        val info = ModelRegistry.byId(
            intent.getStringExtra(EXTRA_MODEL_ID) ?: ""
        ) ?: ModelRegistry.default()

        findViewById<TextView>(R.id.detailName).text = info.friendlyName
        findViewById<TextView>(R.id.detailSub).text = "${info.typeLabel} · ${info.subLabel}"
        findViewById<TextView>(R.id.detailDesc).text = info.desc
        findViewById<TextView>(R.id.detailMeta).text = buildString {
            append("作者 · ${info.authors}\n")
            append("提出 · ${info.year} · ${info.org}\n")
            append("参数量 · ${info.params}\n")
            append("精度 · ${info.map}\n")
            append("计算量 · ${info.flops}")
        }
        findViewById<TextView>(R.id.detailStructure).text = "结构 · ${info.structure}"
        findViewById<TextView>(R.id.detailFeatures).text = "特点 · ${info.features}"
        findViewById<TextView>(R.id.detailArch).text = "架构 · ${info.architecture}"
        findViewById<TextView>(R.id.detailParams).text = "参数量 · ${info.params}"
        findViewById<TextView>(R.id.detailMap).text = "精度 · ${info.map}"
        findViewById<TextView>(R.id.detailFlops).text = "计算量 · ${info.flops}"
        findViewById<TextView>(R.id.detailScenarios).text = info.scenarios

        val paper = findViewById<TextView>(R.id.detailPaper)
        val github = findViewById<TextView>(R.id.detailGithub)
        paper.text = "论文 · ${info.paperUrl}"
        github.text = "源码 · ${info.githubUrl}"
        paper.setOnClickListener { openUrl(info.paperUrl) }
        github.setOnClickListener { openUrl(info.githubUrl) }

        findViewById<android.view.View>(R.id.detailBack).setOnClickListener { finish() }
    }

    private fun openUrl(url: String) {
        if (url.isEmpty()) return
        try {
            startActivity(
                android.content.Intent(
                    android.content.Intent.ACTION_VIEW,
                    android.net.Uri.parse(url)
                )
            )
        } catch (e: Exception) {
            android.util.Log.e("MobileSAM", "open url failed", e)
        }
    }

    companion object {
        const val EXTRA_MODEL_ID = "model_id"

        fun start(context: android.content.Context, info: ModelRegistry.ModelInfo) {
            context.startActivity(
                Intent(context, ModelDetailActivity::class.java)
                    .putExtra(EXTRA_MODEL_ID, info.id)
            )
        }
    }
}

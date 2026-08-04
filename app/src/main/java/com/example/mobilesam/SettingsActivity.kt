package com.example.mobilesam

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Settings page: grouped cards — current model (with runtime params), all
 * models, and open-source notes. Consistent serif typography throughout.
 */
class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val root = findViewById<android.widget.LinearLayout>(R.id.settingsRoot)
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

        val prefs = getSharedPreferences("mobilesam_prefs", MODE_PRIVATE)
        val current = ModelRegistry.byId(prefs.getString("model_id", null) ?: "")
            ?: ModelRegistry.default()

        val currentCard = findViewById<FrameLayout>(R.id.currentModelCard)
        currentCard.addView(buildModelCard(current, current = true))

        val container = findViewById<LinearLayout>(R.id.modelsContainer)
        ModelRegistry.models.forEachIndexed { index, info ->
            if (index > 0) {
                container.addView(buildDivider())
            }
            container.addView(buildModelCard(info, current = false))
        }

        findViewById<View>(R.id.settingsBack).setOnClickListener { finish() }
    }

    private fun buildDivider(): View {
        val v = View(this)
        v.layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(2)
        )
        v.setBackgroundColor(
            androidx.core.content.ContextCompat.getColor(this, R.color.accent_glow)
        )
        return v
    }

    private fun buildModelCard(info: ModelRegistry.ModelInfo, current: Boolean): View {
        val inflater = LayoutInflater.from(this)
        val card = inflater.inflate(
            R.layout.item_model_card, null, false
        ) as LinearLayout

        card.findViewById<TextView>(R.id.modelCardName).text = info.friendlyName
        card.findViewById<TextView>(R.id.modelCardSub).text = info.subLabel
        card.findViewById<TextView>(R.id.modelCardDesc).text = info.desc

        val detailBtn = card.findViewById<TextView>(R.id.modelCardDetail)
        detailBtn.text = "查看详情 ▸"
        detailBtn.setOnClickListener { ModelDetailActivity.start(this, info) }

        if (current) {
            val runtime = inflater.inflate(
                R.layout.item_runtime_line, card, false
            ) as TextView
            runtime.text = getString(
                R.string.current_runtime_format,
                info.inputSize, info.inputSize, 4, 1,
                Runtime.getRuntime().availableProcessors(),
            )
            card.addView(runtime)
        }
        return card
    }

    private fun dp(v: Int): Int =
        (v * resources.displayMetrics.density).toInt()
}

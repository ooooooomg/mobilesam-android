package com.example.mobilesam

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Settings page: all models as one rounded group, and open-source notes.
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

        val container = findViewById<LinearLayout>(R.id.modelsContainer)
        ModelRegistry.models.forEachIndexed { index, info ->
            if (index > 0) {
                container.addView(buildDivider())
            }
            container.addView(buildModelCard(info))
        }

        findViewById<View>(R.id.settingsBack).setOnClickListener { finish() }
    }

    private fun buildDivider(): View {
        val v = View(this)
        v.layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 1
        )
        v.setBackgroundColor(
            androidx.core.content.ContextCompat.getColor(this, R.color.outline_variant)
        )
        return v
    }

    private fun buildModelCard(info: ModelRegistry.ModelInfo): View {
        val inflater = LayoutInflater.from(this)
        val card = inflater.inflate(
            R.layout.item_model_card, null, false
        ) as LinearLayout

        card.findViewById<TextView>(R.id.modelCardName).text = info.friendlyName

        val detailBtn = card.findViewById<TextView>(R.id.modelCardDetail)
        detailBtn.text = "查看详情 ▸"
        detailBtn.setOnClickListener { ModelDetailActivity.start(this, info) }
        return card
    }
}

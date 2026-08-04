package com.example.mobilesam

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.BaseAdapter
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Centered 70%-width model picker dialog. Each row shows the friendly name
 * (large) plus a technical sub-label (small); the current model is highlighted
 * with an indigo tint and a check mark.
 */
class ModelPickerDialog(
    context: Context,
    private val models: List<ModelRegistry.ModelInfo>,
    private val currentId: String,
    private val onSelect: (ModelRegistry.ModelInfo) -> Unit,
) : Dialog(context) {

    init {
        setContentView(R.layout.dialog_model_picker)
        window?.apply {
            // Make the dialog window fully transparent so the layout's own
            // rounded bg_dialog is the ONLY background (no double-frame).
            setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            val lp = attributes
            lp.width = (context.resources.displayMetrics.widthPixels * 0.7f).toInt()
            lp.height = WindowManager.LayoutParams.WRAP_CONTENT
            attributes = lp
        }

        val list = findViewById<android.widget.ListView>(R.id.modelList)
        list.adapter = ModelAdapter()
        list.setOnItemClickListener { _, _, position, _ ->
            if (position in models.indices) {
                dismiss()
                onSelect(models[position])
            }
        }
    }

    private inner class ModelAdapter : BaseAdapter() {
        override fun getCount() = models.size
        override fun getItem(p: Int) = models[p]
        override fun getItemId(p: Int) = p.toLong()

        override fun getView(p: Int, convert: View?, parent: ViewGroup): View {
            val row = convert ?: android.view.LayoutInflater.from(context)
                .inflate(R.layout.item_model_picker, parent, false)
            val info = models[p]
            val selected = info.id == currentId

            row.findViewById<TextView>(R.id.pickerName).text = info.friendlyName
            row.findViewById<TextView>(R.id.pickerSub).text = info.subLabel
            row.findViewById<TextView>(R.id.pickerCheck).visibility =
                if (selected) View.VISIBLE else View.GONE
            row.background = if (selected) {
                context.getDrawable(R.drawable.bg_model_selected)
            } else {
                null
            }
            return row
        }
    }
}

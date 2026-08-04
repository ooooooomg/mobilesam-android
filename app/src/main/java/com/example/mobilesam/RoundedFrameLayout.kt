package com.example.mobilesam

import android.content.Context
import android.graphics.Canvas
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Wraps the camera preview + overlay and clips them to rounded corners,
 * matching the app's @dimen/app_corner radius (20dp).
 */
class RoundedFrameLayout @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : android.widget.FrameLayout(context, attrs) {

    private val clipPath = Path()
    private val cornerPx = resources.getDimension(R.dimen.app_corner)

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        clipPath.reset()
        clipPath.addRoundRect(
            RectF(0f, 0f, w.toFloat(), h.toFloat()),
            cornerPx, cornerPx, Path.Direction.CW
        )
    }

    override fun dispatchDraw(canvas: Canvas) {
        canvas.save()
        canvas.clipPath(clipPath)
        super.dispatchDraw(canvas)
        canvas.restore()
    }
}

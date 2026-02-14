package com.onyxvo.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class KeypointOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00FF00.toInt()  // green
        style = Paint.Style.FILL
    }

    // Keypoint coords as flat array: [x0, y0, x1, y1, ...]
    private var keypoints: FloatArray? = null
    private var modelWidth = 640
    private var modelHeight = 480

    // Radius in dp, converted to px
    private val radiusPx = 3f * context.resources.displayMetrics.density

    fun updateKeypoints(kpts: FloatArray, modelW: Int, modelH: Int) {
        keypoints = kpts
        modelWidth = modelW
        modelHeight = modelH
        postInvalidate()
    }

    fun clear() {
        keypoints = null
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val kpts = keypoints ?: return
        if (kpts.isEmpty()) return

        val scaleX = width.toFloat() / modelWidth
        val scaleY = height.toFloat() / modelHeight

        var i = 0
        while (i + 1 < kpts.size) {
            val x = kpts[i] * scaleX
            val y = kpts[i + 1] * scaleY
            canvas.drawCircle(x, y, radiusPx, paint)
            i += 2
        }
    }
}

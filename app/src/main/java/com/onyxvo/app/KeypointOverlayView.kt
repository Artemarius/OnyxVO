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

    private val keypointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00FF00.toInt()  // green
        style = Paint.Style.FILL
    }

    private val matchPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00FFFF.toInt()  // cyan
        style = Paint.Style.STROKE
        strokeWidth = 2f * context.resources.displayMetrics.density
    }

    // Keypoint coords as flat array: [x0, y0, x1, y1, ...]
    private var keypoints: FloatArray? = null
    // Match lines as flat array: [prev_x, prev_y, curr_x, curr_y, ...]
    private var matchLines: FloatArray? = null
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

    fun updateMatches(lines: FloatArray) {
        matchLines = lines
        postInvalidate()
    }

    fun clear() {
        keypoints = null
        matchLines = null
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scaleX = width.toFloat() / modelWidth
        val scaleY = height.toFloat() / modelHeight

        // Draw match lines first (underneath keypoints)
        val lines = matchLines
        if (lines != null && lines.size >= 4) {
            var i = 0
            while (i + 3 < lines.size) {
                val x1 = lines[i] * scaleX
                val y1 = lines[i + 1] * scaleY
                val x2 = lines[i + 2] * scaleX
                val y2 = lines[i + 3] * scaleY
                canvas.drawLine(x1, y1, x2, y2, matchPaint)
                i += 4
            }
        }

        // Draw keypoints on top
        val kpts = keypoints ?: return
        if (kpts.isEmpty()) return

        var i = 0
        while (i + 1 < kpts.size) {
            val x = kpts[i] * scaleX
            val y = kpts[i + 1] * scaleY
            canvas.drawCircle(x, y, radiusPx, keypointPaint)
            i += 2
        }
    }
}

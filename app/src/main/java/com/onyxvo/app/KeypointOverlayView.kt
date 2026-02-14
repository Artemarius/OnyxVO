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
    private var rotationDegrees = 0

    // Radius in dp, converted to px
    private val radiusPx = 3f * context.resources.displayMetrics.density

    fun updateKeypoints(kpts: FloatArray, modelW: Int, modelH: Int, rotation: Int = 0) {
        keypoints = kpts
        modelWidth = modelW
        modelHeight = modelH
        rotationDegrees = rotation
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

    // Transform model-space (x, y) to view-space accounting for sensor rotation.
    // CameraX ImageProxy reports rotation needed to match display orientation.
    // Model processes the raw sensor image (landscape), but PreviewView auto-rotates.
    private fun transformX(mx: Float, my: Float): Float {
        return when (rotationDegrees) {
            90 -> (modelHeight - my) / modelHeight * width
            180 -> (modelWidth - mx) / modelWidth * width
            270 -> my / modelHeight * width
            else -> mx / modelWidth * width
        }
    }

    private fun transformY(mx: Float, my: Float): Float {
        return when (rotationDegrees) {
            90 -> mx / modelWidth * height
            180 -> (modelHeight - my) / modelHeight * height
            270 -> (modelWidth - mx) / modelWidth * height
            else -> my / modelHeight * height
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw match lines first (underneath keypoints)
        val lines = matchLines
        if (lines != null && lines.size >= 4) {
            var i = 0
            while (i + 3 < lines.size) {
                val x1 = transformX(lines[i], lines[i + 1])
                val y1 = transformY(lines[i], lines[i + 1])
                val x2 = transformX(lines[i + 2], lines[i + 3])
                val y2 = transformY(lines[i + 2], lines[i + 3])
                canvas.drawLine(x1, y1, x2, y2, matchPaint)
                i += 4
            }
        }

        // Draw keypoints on top
        val kpts = keypoints ?: return
        if (kpts.isEmpty()) return

        var i = 0
        while (i + 1 < kpts.size) {
            val x = transformX(kpts[i], kpts[i + 1])
            val y = transformY(kpts[i], kpts[i + 1])
            canvas.drawCircle(x, y, radiusPx, keypointPaint)
            i += 2
        }
    }
}

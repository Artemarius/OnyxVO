package com.onyxvo.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

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

    private var keypoints: FloatArray? = null
    private var matchLines: FloatArray? = null
    private var modelWidth = 640
    private var modelHeight = 480
    private var cameraWidth = 640
    private var cameraHeight = 480
    private var rotationDegrees = 0

    private val radiusPx = 3f * context.resources.displayMetrics.density

    fun updateKeypoints(
        kpts: FloatArray, modelW: Int, modelH: Int,
        camW: Int, camH: Int, rotation: Int
    ) {
        keypoints = kpts
        modelWidth = modelW
        modelHeight = modelH
        cameraWidth = camW
        cameraHeight = camH
        rotationDegrees = rotation
        invalidate()
    }

    fun updateMatches(lines: FloatArray) {
        matchLines = lines
        invalidate()
    }

    fun clear() {
        keypoints = null
        matchLines = null
        invalidate()
    }

    // Transform model-space (mx, my) to view-space, matching PreviewView FILL_CENTER.
    //
    // Pipeline:
    // 1. Model coords -> normalized camera coords (undo the non-uniform resize)
    //    norm_x = mx / modelWidth, norm_y = my / modelHeight   (both in 0..1)
    // 2. Apply sensor rotation (90/180/270) to get display-oriented coords
    // 3. Apply PreviewView FILL_CENTER scaling (center-crop) to get view pixels
    private fun transformX(mx: Float, my: Float): Float {
        // Step 1: normalize to camera space (0..1)
        val normCamX = mx / modelWidth
        val normCamY = my / modelHeight

        // Step 2: apply rotation to get display-oriented normalized coords
        val imgNormX = when (rotationDegrees) {
            90 -> 1f - normCamY
            180 -> 1f - normCamX
            270 -> normCamY
            else -> normCamX
        }

        // Step 3: FILL_CENTER mapping
        // Rotated image dimensions
        val imgW: Float
        val imgH: Float
        if (rotationDegrees == 90 || rotationDegrees == 270) {
            imgW = cameraHeight.toFloat()
            imgH = cameraWidth.toFloat()
        } else {
            imgW = cameraWidth.toFloat()
            imgH = cameraHeight.toFloat()
        }

        val scale = max(width / imgW, height / imgH)
        val offsetX = (width - imgW * scale) / 2f

        return imgNormX * imgW * scale + offsetX
    }

    private fun transformY(mx: Float, my: Float): Float {
        val normCamX = mx / modelWidth
        val normCamY = my / modelHeight

        val imgNormY = when (rotationDegrees) {
            90 -> normCamX
            180 -> 1f - normCamY
            270 -> 1f - normCamX
            else -> normCamY
        }

        val imgW: Float
        val imgH: Float
        if (rotationDegrees == 90 || rotationDegrees == 270) {
            imgW = cameraHeight.toFloat()
            imgH = cameraWidth.toFloat()
        } else {
            imgW = cameraWidth.toFloat()
            imgH = cameraHeight.toFloat()
        }

        val scale = max(width / imgW, height / imgH)
        val offsetY = (height - imgH * scale) / 2f

        return imgNormY * imgH * scale + offsetY
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

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

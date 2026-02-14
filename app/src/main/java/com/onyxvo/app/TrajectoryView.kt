package com.onyxvo.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.max

class TrajectoryView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var xzCoords = FloatArray(0)

    private val pathPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    private val startPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }

    private val currentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(60, 255, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 1f
    }

    private val path = Path()

    fun updateTrajectory(coords: FloatArray) {
        xzCoords = coords
        postInvalidate()
    }

    fun clear() {
        xzCoords = FloatArray(0)
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        // Draw crosshair axes
        val cx = w / 2f
        val cy = h / 2f
        canvas.drawLine(cx, 0f, cx, h, axisPaint)
        canvas.drawLine(0f, cy, w, cy, axisPaint)

        val count = xzCoords.size / 2
        if (count < 2) return

        // Find bounding box
        var minX = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var minZ = Float.MAX_VALUE
        var maxZ = -Float.MAX_VALUE
        for (i in 0 until count) {
            val x = xzCoords[i * 2]
            val z = xzCoords[i * 2 + 1]
            if (x < minX) minX = x
            if (x > maxX) maxX = x
            if (z < minZ) minZ = z
            if (z > maxZ) maxZ = z
        }

        // Uniform scale with padding
        val padding = 20f
        val rangeX = max(abs(maxX - minX), 0.01f)
        val rangeZ = max(abs(maxZ - minZ), 0.01f)
        val range = max(rangeX, rangeZ)
        val scale = (min(w, h) - 2 * padding) / range

        val centerX = (minX + maxX) / 2f
        val centerZ = (minZ + maxZ) / 2f

        // Transform: trajectory coords -> view coords
        // X maps to horizontal, Z maps to vertical (inverted: Z+ = forward = up on screen)
        fun toViewX(tx: Float) = cx + (tx - centerX) * scale
        fun toViewY(tz: Float) = cy - (tz - centerZ) * scale

        // Draw path
        path.reset()
        val x0 = toViewX(xzCoords[0])
        val y0 = toViewY(xzCoords[1])
        path.moveTo(x0, y0)
        for (i in 1 until count) {
            path.lineTo(toViewX(xzCoords[i * 2]), toViewY(xzCoords[i * 2 + 1]))
        }
        canvas.drawPath(path, pathPaint)

        // Start dot (green)
        canvas.drawCircle(x0, y0, 5f, startPaint)

        // Current position dot (red)
        val lastIdx = count - 1
        val xLast = toViewX(xzCoords[lastIdx * 2])
        val yLast = toViewY(xzCoords[lastIdx * 2 + 1])
        canvas.drawCircle(xLast, yLast, 5f, currentPaint)
    }

    private fun min(a: Float, b: Float): Float = if (a < b) a else b
}

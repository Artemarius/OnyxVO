package com.onyxvo.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

class TrajectoryView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var xzCoords = FloatArray(0)
    private var headings = FloatArray(0)
    private var inlierRatios = FloatArray(0)

    private val segmentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 1f
        strokeCap = Paint.Cap.ROUND
    }

    private val startPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }

    private val arrowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(38, 255, 255, 255)  // 15% alpha
        style = Paint.Style.STROKE
        strokeWidth = 1f
    }

    private val arrowPath = Path()

    fun updateTrajectory(coords: FloatArray, hdg: FloatArray, ratios: FloatArray) {
        xzCoords = coords
        headings = hdg
        inlierRatios = ratios
        postInvalidate()
    }

    fun clear() {
        xzCoords = FloatArray(0)
        headings = FloatArray(0)
        inlierRatios = FloatArray(0)
        postInvalidate()
    }

    // Interpolate green -> red based on inlier ratio (1.0=green, 0.0=red)
    private fun ratioToColor(ratio: Float): Int {
        val r = ratio.coerceIn(0f, 1f)
        val red = ((1f - r) * 255).toInt()
        val green = (r * 255).toInt()
        return Color.argb(220, red, green, 60)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        val cx = w / 2f
        val cy = h / 2f

        // Draw subtle grid (5 lines each axis)
        val gridSpacing = min(w, h) / 6f
        for (i in 1..5) {
            val offset = gridSpacing * i
            // Vertical lines
            if (cx - offset >= 0) canvas.drawLine(cx - offset, 0f, cx - offset, h, gridPaint)
            if (cx + offset <= w) canvas.drawLine(cx + offset, 0f, cx + offset, h, gridPaint)
            // Horizontal lines
            if (cy - offset >= 0) canvas.drawLine(0f, cy - offset, w, cy - offset, gridPaint)
            if (cy + offset <= h) canvas.drawLine(0f, cy + offset, w, cy + offset, gridPaint)
        }

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

        fun toViewX(tx: Float) = cx + (tx - centerX) * scale
        fun toViewY(tz: Float) = cy - (tz - centerZ) * scale

        // Draw path segments individually, colored by inlier ratio
        for (i in 1 until count) {
            val x1 = toViewX(xzCoords[(i - 1) * 2])
            val y1 = toViewY(xzCoords[(i - 1) * 2 + 1])
            val x2 = toViewX(xzCoords[i * 2])
            val y2 = toViewY(xzCoords[i * 2 + 1])

            val ratio = if (i < inlierRatios.size) inlierRatios[i] else 0.5f
            segmentPaint.color = ratioToColor(ratio)
            canvas.drawLine(x1, y1, x2, y2, segmentPaint)
        }

        // Start dot (green)
        val x0 = toViewX(xzCoords[0])
        val y0 = toViewY(xzCoords[1])
        canvas.drawCircle(x0, y0, 2.5f, startPaint)

        // Heading arrow at current position
        val lastIdx = count - 1
        val xLast = toViewX(xzCoords[lastIdx * 2])
        val yLast = toViewY(xzCoords[lastIdx * 2 + 1])

        if (lastIdx < headings.size) {
            val heading = headings[lastIdx]
            val arrowSize = 4f

            // Arrow points in heading direction (heading = atan2(forward.x, forward.z))
            // In view space: X maps to horizontal, Z maps to vertical (inverted)
            val dx = sin(heading) * arrowSize
            val dy = -cos(heading) * arrowSize  // negated because Z+ is up in view

            arrowPath.reset()
            // Tip
            arrowPath.moveTo(xLast + dx, yLast + dy)
            // Left wing
            arrowPath.lineTo(xLast - dy * 0.5f - dx * 0.3f, yLast + dx * 0.5f - dy * 0.3f)
            // Right wing
            arrowPath.lineTo(xLast + dy * 0.5f - dx * 0.3f, yLast - dx * 0.5f - dy * 0.3f)
            arrowPath.close()
            canvas.drawPath(arrowPath, arrowPaint)
        } else {
            // Fallback: white dot
            canvas.drawCircle(xLast, yLast, 2.5f, arrowPaint)
        }
    }
}

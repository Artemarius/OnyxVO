package com.onyxvo.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

data class DashboardData(
    val fps: Float = 0f,
    val preprocessUs: Float = 0f,
    val inferenceUs: Float = 0f,
    val matchingUs: Float = 0f,
    val poseUs: Float = 0f,
    val totalUs: Float = 0f,
    val keypointCount: Int = 0,
    val matchCount: Int = 0,
    val inlierCount: Int = 0,
    val keyframeCount: Int = 0,
    val budgetExceeded: Boolean = false,
    val modelType: String = "FP32",
    val matcherType: String = "GPU",
    val useNeon: Boolean = true
)

class PerformanceDashboardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val density = context.resources.displayMetrics.density
    private val scaledDensity = context.resources.displayMetrics.scaledDensity

    // Dimensions in pixels (converted from dp/sp)
    private val viewWidthPx = (280 * density).toInt()
    private val viewHeightPx = (200 * density).toInt()
    private val paddingHPx = 12 * density
    private val paddingVPx = 10 * density
    private val cornerRadiusPx = 8 * density
    private val barHeightPx = 16 * density
    private val dotRadiusPx = 4 * density
    private val sectionGapPx = 8 * density

    // Stage bar colors
    private val colorPreprocess = Color.rgb(66, 133, 244)    // blue
    private val colorInference = Color.rgb(255, 152, 0)      // orange
    private val colorMatching = Color.rgb(76, 175, 80)        // green
    private val colorPose = Color.rgb(244, 67, 54)            // red

    // FPS threshold colors
    private val colorFpsGood = Color.rgb(76, 175, 80)         // green >= 25
    private val colorFpsWarn = Color.rgb(255, 235, 59)        // yellow 15-24
    private val colorFpsBad = Color.rgb(244, 67, 54)          // red < 15

    // Mode indicator colors
    private val colorModeActive = Color.rgb(76, 175, 80)      // green
    private val colorModeInactive = Color.rgb(128, 128, 128)  // gray

    // Budget line color
    private val colorBudgetLine = Color.argb(200, 255, 255, 255)

    // Pre-allocated paint objects
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val fpsTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 18 * scaledDensity
        typeface = Typeface.MONOSPACE
        isFakeBoldText = true
    }

    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 11 * scaledDensity
        typeface = Typeface.MONOSPACE
    }

    private val smallTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 255, 255, 255)
        textSize = 9 * scaledDensity
        typeface = Typeface.MONOSPACE
    }

    private val barPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val budgetLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = colorBudgetLine
        style = Paint.Style.STROKE
        strokeWidth = 1.5f * density
        pathEffect = android.graphics.DashPathEffect(
            floatArrayOf(4 * density, 3 * density), 0f
        )
    }

    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val modeTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 10 * scaledDensity
        typeface = Typeface.MONOSPACE
    }

    // Reusable drawing objects
    private val bgRect = RectF()
    private val barRect = RectF()

    // Current data
    private var data = DashboardData()

    fun updateData(data: DashboardData) {
        this.data = data
        invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val desiredWidth = viewWidthPx
        val desiredHeight = viewHeightPx

        val width = resolveSize(desiredWidth, widthMeasureSpec)
        val height = resolveSize(desiredHeight, heightMeasureSpec)
        setMeasuredDimension(width, height)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0 || h <= 0) return

        // Draw rounded rect background
        bgRect.set(0f, 0f, w, h)
        canvas.drawRoundRect(bgRect, cornerRadiusPx, cornerRadiusPx, bgPaint)

        val contentLeft = paddingHPx
        val contentRight = w - paddingHPx
        val contentWidth = contentRight - contentLeft
        var curY = paddingVPx

        // --- Section 1: FPS Counter ---
        curY = drawFpsSection(canvas, contentLeft, curY)

        curY += sectionGapPx

        // --- Section 2: Stage Timing Bars ---
        curY = drawTimingBars(canvas, contentLeft, contentRight, contentWidth, curY)

        curY += sectionGapPx

        // --- Section 3: Counts Row ---
        curY = drawCountsRow(canvas, contentLeft, contentWidth, curY)

        curY += sectionGapPx

        // --- Section 4: Mode Indicators ---
        drawModeIndicators(canvas, contentLeft, curY)
    }

    private fun drawFpsSection(canvas: Canvas, left: Float, startY: Float): Float {
        val fps = data.fps
        val fpsColor = when {
            fps >= 25f -> colorFpsGood
            fps >= 15f -> colorFpsWarn
            else -> colorFpsBad
        }

        fpsTextPaint.color = fpsColor
        val fpsStr = if (fps > 0f) "%.0f FPS".format(fps) else "-- FPS"
        val textHeight = fpsTextPaint.descent() - fpsTextPaint.ascent()
        val baseline = startY - fpsTextPaint.ascent()
        canvas.drawText(fpsStr, left, baseline, fpsTextPaint)

        // Draw budget indicator on the right side
        if (data.budgetExceeded && data.totalUs > 0f) {
            smallTextPaint.color = colorFpsBad
            val budgetStr = "OVER BUDGET"
            val budgetWidth = smallTextPaint.measureText(budgetStr)
            val rightEdge = width - paddingHPx
            canvas.drawText(budgetStr, rightEdge - budgetWidth, baseline, smallTextPaint)
            smallTextPaint.color = Color.argb(200, 255, 255, 255) // reset
        }

        return startY + textHeight
    }

    private fun drawTimingBars(
        canvas: Canvas,
        left: Float,
        right: Float,
        contentWidth: Float,
        startY: Float
    ): Float {
        var curY = startY

        // Convert microseconds to milliseconds for display
        val preprocessMs = data.preprocessUs / 1000f
        val inferenceMs = data.inferenceUs / 1000f
        val matchingMs = data.matchingUs / 1000f
        val poseMs = data.poseUs / 1000f
        val totalMs = data.totalUs / 1000f

        // The bar represents time from 0 to max(totalMs, 33ms) to show budget context
        val budgetMs = 33.33f
        val maxMs = max(totalMs, budgetMs) * 1.1f // 10% extra for visual breathing room

        if (maxMs <= 0f) {
            // No data yet, draw empty bar
            barPaint.color = Color.argb(40, 255, 255, 255)
            barRect.set(left, curY, right, curY + barHeightPx)
            canvas.drawRoundRect(barRect, 3 * density, 3 * density, barPaint)
            curY += barHeightPx

            // "No data" label
            curY += 3 * density
            canvas.drawText("-- ms", left, curY - smallTextPaint.ascent(), smallTextPaint)
            curY += smallTextPaint.descent() - smallTextPaint.ascent()
            return curY
        }

        val pxPerMs = contentWidth / maxMs

        // Draw stage segments
        var barX = left

        // Preprocess (blue)
        if (preprocessMs > 0f) {
            barPaint.color = colorPreprocess
            val segWidth = preprocessMs * pxPerMs
            barRect.set(barX, curY, barX + segWidth, curY + barHeightPx)
            canvas.drawRect(barRect, barPaint)
            barX += segWidth
        }

        // Inference (orange)
        if (inferenceMs > 0f) {
            barPaint.color = colorInference
            val segWidth = inferenceMs * pxPerMs
            barRect.set(barX, curY, barX + segWidth, curY + barHeightPx)
            canvas.drawRect(barRect, barPaint)
            barX += segWidth
        }

        // Matching (green)
        if (matchingMs > 0f) {
            barPaint.color = colorMatching
            val segWidth = matchingMs * pxPerMs
            barRect.set(barX, curY, barX + segWidth, curY + barHeightPx)
            canvas.drawRect(barRect, barPaint)
            barX += segWidth
        }

        // Pose (red)
        if (poseMs > 0f) {
            barPaint.color = colorPose
            val segWidth = poseMs * pxPerMs
            barRect.set(barX, curY, barX + segWidth, curY + barHeightPx)
            canvas.drawRect(barRect, barPaint)
            barX += segWidth
        }

        // Draw 33ms budget line
        val budgetX = left + budgetMs * pxPerMs
        if (budgetX >= left && budgetX <= right) {
            canvas.drawLine(budgetX, curY - 2 * density, budgetX, curY + barHeightPx + 2 * density, budgetLinePaint)
        }

        curY += barHeightPx

        // Labels below the bar
        curY += 3 * density
        val labelBaseline = curY - smallTextPaint.ascent()

        // Total time on left
        val totalStr = "%.1f ms".format(totalMs)
        canvas.drawText(totalStr, left, labelBaseline, smallTextPaint)

        // Legend: colored squares with labels on the right
        val legendY = labelBaseline
        val legendItems = listOf(
            Pair(colorPreprocess, "Pre"),
            Pair(colorInference, "Inf"),
            Pair(colorMatching, "Mat"),
            Pair(colorPose, "Pos")
        )
        val sqSize = 7 * density
        val legendGap = 4 * density
        // Measure total legend width to right-align
        var totalLegendWidth = 0f
        for ((_, label) in legendItems) {
            totalLegendWidth += sqSize + 2 * density + smallTextPaint.measureText(label) + legendGap
        }
        totalLegendWidth -= legendGap // no trailing gap

        var lx = right - totalLegendWidth
        for ((color, label) in legendItems) {
            barPaint.color = color
            val sqTop = legendY + smallTextPaint.ascent() + (smallTextPaint.descent() - smallTextPaint.ascent() - sqSize) / 2f
            barRect.set(lx, sqTop, lx + sqSize, sqTop + sqSize)
            canvas.drawRect(barRect, barPaint)
            lx += sqSize + 2 * density
            canvas.drawText(label, lx, legendY, smallTextPaint)
            lx += smallTextPaint.measureText(label) + legendGap
        }

        curY += smallTextPaint.descent() - smallTextPaint.ascent()
        return curY
    }

    private fun drawCountsRow(
        canvas: Canvas,
        left: Float,
        contentWidth: Float,
        startY: Float
    ): Float {
        val baseline = startY - labelTextPaint.ascent()

        val kpStr = "KP:${data.keypointCount}"
        val mStr = "M:${data.matchCount}"
        val inStr = "IN:${data.inlierCount}"
        val kfStr = "KF:${data.keyframeCount}"

        // Distribute evenly across the content width
        val items = listOf(kpStr, mStr, inStr, kfStr)
        val spacing = contentWidth / items.size

        for ((index, item) in items.withIndex()) {
            val x = left + spacing * index
            canvas.drawText(item, x, baseline, labelTextPaint)
        }

        return startY + (labelTextPaint.descent() - labelTextPaint.ascent())
    }

    private fun drawModeIndicators(canvas: Canvas, left: Float, startY: Float) {
        val baseline = startY - modeTextPaint.ascent()
        var x = left

        // Model type: FP32 or INT8
        val isInt8 = data.modelType == "INT8"
        drawModeChip(canvas, x, baseline, data.modelType, if (isInt8) colorModeActive else colorModeActive)
        x += modeTextPaint.measureText(data.modelType) + dotRadiusPx * 2 + 12 * density

        // Matcher type: GPU or CPU
        val isGpu = data.matcherType == "GPU"
        drawModeChip(canvas, x, baseline, data.matcherType, if (isGpu) colorModeActive else colorModeInactive)
        x += modeTextPaint.measureText(data.matcherType) + dotRadiusPx * 2 + 12 * density

        // NEON or Scalar
        val neonLabel = if (data.useNeon) "NEON" else "Scalar"
        drawModeChip(canvas, x, baseline, neonLabel, if (data.useNeon) colorModeActive else colorModeInactive)
    }

    private fun drawModeChip(canvas: Canvas, x: Float, baseline: Float, label: String, dotColor: Int) {
        // Draw colored dot
        val dotCenterY = baseline + modeTextPaint.ascent() / 2f
        dotPaint.color = dotColor
        canvas.drawCircle(x + dotRadiusPx, dotCenterY, dotRadiusPx, dotPaint)

        // Draw label text
        val textX = x + dotRadiusPx * 2 + 3 * density
        canvas.drawText(label, textX, baseline, modeTextPaint)
    }
}

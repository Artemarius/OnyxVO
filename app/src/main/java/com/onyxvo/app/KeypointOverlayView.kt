package com.onyxvo.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class KeypointOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val dp = context.resources.displayMetrics.density

    // Pre-allocated paints (no allocation in onDraw)
    private val unmatchedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x59888888.toInt()  // gray, 35% alpha
        style = Paint.Style.FILL
    }

    private val outlierFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x73777777.toInt()  // gray, 45% alpha
        style = Paint.Style.FILL
    }

    private val outlierRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x59AA5555.toInt()  // muted red, 35% alpha
        style = Paint.Style.STROKE
        strokeWidth = 0.75f * dp
    }

    private val inlierFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val inlierRingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 0.75f * dp
    }

    private val inlierLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 0.75f * dp
    }

    private val outlierLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x33888888.toInt()  // gray, 20% alpha
        style = Paint.Style.STROKE
        strokeWidth = 0.375f * dp
    }

    private val qualityBarPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val kfAgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 5.5f * dp
        typeface = Typeface.MONOSPACE
    }

    // Data arrays
    private var keypoints: FloatArray? = null
    private var scores: FloatArray? = null
    private var matchInfo: FloatArray? = null
    private var matchLines: FloatArray? = null
    private var ratioQualities: FloatArray? = null
    private var inlierFlags: FloatArray? = null
    private var qualityScore = 0f
    private var keyframeAge = 0

    private var modelWidth = 640
    private var modelHeight = 480
    private var cameraWidth = 640
    private var cameraHeight = 480
    private var rotationDegrees = 0

    private val unmatchedRadius = 1f * dp
    private val outlierRadius = 1.25f * dp
    private val inlierRadius = 1.75f * dp
    private val qualityBarHeight = 1.5f * dp

    fun updateKeypoints(
        kpts: FloatArray, kpScores: FloatArray, kpMatchInfo: FloatArray,
        modelW: Int, modelH: Int, camW: Int, camH: Int, rotation: Int
    ) {
        keypoints = kpts
        scores = kpScores
        matchInfo = kpMatchInfo
        modelWidth = modelW
        modelHeight = modelH
        cameraWidth = camW
        cameraHeight = camH
        rotationDegrees = rotation
        invalidate()
    }

    fun updateMatches(lines: FloatArray, rQualities: FloatArray, iFlags: FloatArray) {
        matchLines = lines
        ratioQualities = rQualities
        inlierFlags = iFlags
        invalidate()
    }

    fun updateFrameIndicators(quality: Float, kfAge: Int) {
        qualityScore = quality
        keyframeAge = kfAge
        invalidate()
    }

    fun clear() {
        keypoints = null
        scores = null
        matchInfo = null
        matchLines = null
        ratioQualities = null
        inlierFlags = null
        qualityScore = 0f
        keyframeAge = 0
        invalidate()
    }

    // Color interpolation: [0,1] -> yellow(#FFD54F) -> teal(#00BCD4) -> cyan(#00E5FF)
    private fun qualityToColor(q: Float): Int {
        val cq = q.coerceIn(0f, 1f)
        return if (cq < 0.5f) {
            val t = cq * 2f
            interpolateColor(0xFFFFD54F.toInt(), 0xFF00BCD4.toInt(), t)
        } else {
            val t = (cq - 0.5f) * 2f
            interpolateColor(0xFF00BCD4.toInt(), 0xFF00E5FF.toInt(), t)
        }
    }

    // Score to ring color: [0,1] -> dim gray(#666) -> white(#EEF)
    private fun scoreToRingColor(s: Float): Int {
        val cs = s.coerceIn(0f, 1f)
        return interpolateColor(0xFF666666.toInt(), 0xFFEEEEFF.toInt(), cs)
    }

    private fun interpolateColor(c1: Int, c2: Int, t: Float): Int {
        val a = ((c1 shr 24 and 0xFF) + t * ((c2 shr 24 and 0xFF) - (c1 shr 24 and 0xFF))).toInt()
        val r = ((c1 shr 16 and 0xFF) + t * ((c2 shr 16 and 0xFF) - (c1 shr 16 and 0xFF))).toInt()
        val g = ((c1 shr 8 and 0xFF) + t * ((c2 shr 8 and 0xFF) - (c1 shr 8 and 0xFF))).toInt()
        val b = ((c1 and 0xFF) + t * ((c2 and 0xFF) - (c1 and 0xFF))).toInt()
        return (a shl 24) or (r shl 16) or (g shl 8) or b
    }

    // Transform model-space coords to view-space (matching PreviewView FILL_CENTER)
    private fun transformX(mx: Float, my: Float): Float {
        val normCamX = mx / modelWidth
        val normCamY = my / modelHeight

        val imgNormX = when (rotationDegrees) {
            90 -> 1f - normCamY
            180 -> 1f - normCamX
            270 -> normCamY
            else -> normCamX
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

        val w = width.toFloat()
        if (w <= 0) return

        // 1. Frame quality bar (top edge)
        if (qualityScore > 0f) {
            val barWidth = w * qualityScore.coerceIn(0f, 1f)
            qualityBarPaint.color = qualityToColor(qualityScore)
            canvas.drawRect(0f, 0f, barWidth, qualityBarHeight, qualityBarPaint)
        }

        // 2. KF age text (top-left, below bar)
        if (keyframeAge > 0) {
            val ageColor = when {
                keyframeAge < 10 -> Color.WHITE
                keyframeAge < 30 -> 0xFFFFD54F.toInt()  // yellow
                else -> 0xFFFF5252.toInt()               // red
            }
            kfAgePaint.color = ageColor
            canvas.drawText("KF+$keyframeAge", 2f * dp, qualityBarHeight + 7f * dp, kfAgePaint)
        }

        // 3. Outlier match lines (thin, gray)
        val lines = matchLines
        val rq = ratioQualities
        val iflags = inlierFlags
        if (lines != null && iflags != null) {
            val matchCount = iflags.size
            for (i in 0 until matchCount) {
                val off = i * 4
                if (off + 3 >= lines.size) break
                if (iflags[i] < 0.5f) {
                    val x1 = transformX(lines[off], lines[off + 1])
                    val y1 = transformY(lines[off], lines[off + 1])
                    val x2 = transformX(lines[off + 2], lines[off + 3])
                    val y2 = transformY(lines[off + 2], lines[off + 3])
                    canvas.drawLine(x1, y1, x2, y2, outlierLinePaint)
                }
            }

            // 4. Inlier match lines (colored by ratio quality)
            if (rq != null) {
                for (i in 0 until matchCount) {
                    val off = i * 4
                    if (off + 3 >= lines.size) break
                    if (iflags[i] >= 0.5f) {
                        val x1 = transformX(lines[off], lines[off + 1])
                        val y1 = transformY(lines[off], lines[off + 1])
                        val x2 = transformX(lines[off + 2], lines[off + 3])
                        val y2 = transformY(lines[off + 2], lines[off + 3])
                        inlierLinePaint.color = qualityToColor(rq[i])
                        canvas.drawLine(x1, y1, x2, y2, inlierLinePaint)
                    }
                }
            }
        }

        // Draw keypoints
        val kpts = keypoints ?: return
        if (kpts.isEmpty()) return
        val mi = matchInfo
        val sc = scores
        val kpCount = kpts.size / 2

        for (i in 0 until kpCount) {
            val off = i * 2
            if (off + 1 >= kpts.size) break
            val x = transformX(kpts[off], kpts[off + 1])
            val y = transformY(kpts[off], kpts[off + 1])

            val info = if (mi != null && i < mi.size) mi[i] else 0f
            val score = if (sc != null && i < sc.size) sc[i] else 0.5f

            when {
                // 5. Unmatched keypoints (gray dots)
                info == 0f -> {
                    canvas.drawCircle(x, y, unmatchedRadius, unmatchedPaint)
                }
                // 6. Outlier keypoints (gray + muted red ring)
                info < 0f -> {
                    canvas.drawCircle(x, y, outlierRadius, outlierFillPaint)
                    canvas.drawCircle(x, y, outlierRadius, outlierRingPaint)
                }
                // 7. Inlier keypoints (colored by ratio quality, ring by score)
                else -> {
                    inlierFillPaint.color = qualityToColor(info)
                    canvas.drawCircle(x, y, inlierRadius, inlierFillPaint)
                    inlierRingPaint.color = scoreToRingColor(score)
                    canvas.drawCircle(x, y, inlierRadius, inlierRingPaint)
                }
            }
        }
    }
}

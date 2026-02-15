package com.onyxvo.app

import android.util.Log
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class CameraManager(
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val nativeBridge: NativeBridge,
    private val onFrameProcessed: (FrameResult) -> Unit
) {
    companion object {
        private const val TAG = "OnyxVO.Camera"
    }

    data class FrameResult(
        val width: Int,
        val height: Int,
        val format: Int,
        val resizeTimeUs: Float,
        val normalizeTimeUs: Float,
        val totalTimeUs: Float,
        val useNeon: Boolean,
        // Phase 3 additions
        val inferenceTimeUs: Float = 0f,
        val keypointCount: Int = 0,
        val keypointCoords: FloatArray = FloatArray(0),
        // Phase 4 additions
        val matchingTimeUs: Float = 0f,
        val matchCount: Int = 0,
        val matchLines: FloatArray = FloatArray(0),
        // Phase 5 additions
        val poseTimeUs: Float = 0f,
        val inlierCount: Int = 0,
        val keyframeCount: Int = 0,
        val trajectoryXZ: FloatArray = FloatArray(0),
        // Phase 6 additions
        val budgetExceeded: Boolean = false,
        // Sensor rotation (degrees CW to match display)
        val rotationDegrees: Int = 0,
        // Phase 7 additions
        val keypointScores: FloatArray = FloatArray(0),
        val keypointMatchInfo: FloatArray = FloatArray(0),
        val matchRatioQualities: FloatArray = FloatArray(0),
        val matchInlierFlags: FloatArray = FloatArray(0),
        val trajectoryHeadings: FloatArray = FloatArray(0),
        val trajectoryInlierRatios: FloatArray = FloatArray(0),
        val framesSinceKeyframe: Int = 0,
        val frameQualityScore: Float = 0f
    )

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val pipelineExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "OnyxVO-Pipeline")
    }
    private val pipelineBusy = AtomicBoolean(false)
    private var stagingBuffer: ByteBuffer? = null

    @Volatile
    var useNeon: Boolean = true

    @Volatile
    var modelReady: Boolean = false

    @Volatile
    var paused: Boolean = false

    @Volatile
    var benchmarkRequested: Boolean = false

    var onBenchmarkResult: ((FloatArray) -> Unit)? = null

    fun start() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(previewView.context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindUseCases(cameraProvider)
        }, ContextCompat.getMainExecutor(previewView.context))
    }

    fun shutdown() {
        analysisExecutor.shutdown()
        pipelineExecutor.shutdown()
    }

    fun pause() {
        paused = true
        // Wait for in-flight pipeline work to drain
        while (pipelineBusy.get()) {
            Thread.sleep(5)
        }
    }

    fun resume() {
        paused = false
    }

    private fun bindUseCases(cameraProvider: ProcessCameraProvider) {
        // Both use cases must share the same aspect ratio so CameraX uses the
        // same sensor crop region.  4:3 matches the 640x480 model input.
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            preview,
            imageAnalysis
        )

        Log.i(TAG, "Camera use cases bound")
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (paused) {
            imageProxy.close()
            return
        }

        val yPlane = imageProxy.planes[0]
        val yBuffer = yPlane.buffer
        val rowStride = yPlane.rowStride

        yBuffer.rewind()

        val width = imageProxy.width
        val height = imageProxy.height

        // Run preprocessing benchmark if requested (one-shot, stays on camera thread)
        if (benchmarkRequested) {
            benchmarkRequested = false
            val result = nativeBridge.nativeBenchmarkPreprocessing(
                yBuffer, width, height, rowStride, 100
            )
            if (result != null) {
                onBenchmarkResult?.invoke(result)
            }
            yBuffer.rewind()
        }

        if (modelReady) {
            // Full pipeline: copy Y-plane and offload to pipeline thread
            if (pipelineBusy.get()) {
                // Pipeline still processing previous frame — drop this one
                imageProxy.close()
                return
            }

            // Copy Y-plane data to staging buffer (~0.3ms for 1080p)
            val ySize = yBuffer.remaining()
            if (stagingBuffer == null || stagingBuffer!!.capacity() < ySize) {
                stagingBuffer = ByteBuffer.allocateDirect(ySize)
                    .order(ByteOrder.nativeOrder())
            }
            val staging = stagingBuffer!!
            staging.clear()
            staging.put(yBuffer)
            staging.flip()

            // Capture frame metadata before closing ImageProxy
            val format = imageProxy.format
            val rotation = imageProxy.imageInfo.rotationDegrees
            val capturedUseNeon = useNeon

            // Release ImageProxy immediately — camera thread is now free
            imageProxy.close()

            pipelineBusy.set(true)
            pipelineExecutor.execute {
                try {
                    processPipelineFrame(
                        staging, width, height, rowStride,
                        capturedUseNeon, format, rotation
                    )
                } finally {
                    pipelineBusy.set(false)
                }
            }
        } else {
            // Fallback: preprocessing only (fast, stays on camera thread)
            try {
                val timing = nativeBridge.nativePreprocessFrame(
                    yBuffer, width, height, rowStride, useNeon
                )

                if (timing != null) {
                    onFrameProcessed(
                        FrameResult(
                            width = width,
                            height = height,
                            format = imageProxy.format,
                            resizeTimeUs = timing[0],
                            normalizeTimeUs = timing[1],
                            totalTimeUs = timing[2],
                            useNeon = useNeon,
                            rotationDegrees = imageProxy.imageInfo.rotationDegrees
                        )
                    )
                }
            } finally {
                imageProxy.close()
            }
        }
    }

    /**
     * Runs the full pipeline on the dedicated pipeline thread.
     * Called with staging buffer containing a copy of the Y-plane data.
     */
    private fun processPipelineFrame(
        yBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        useNeon: Boolean,
        format: Int,
        rotationDegrees: Int
    ) {
        val result = nativeBridge.nativeProcessFrame(
            yBuffer, width, height, rowStride, useNeon
        )

        if (result != null && result.size >= 12) {
            val preprocessUs = result[0]
            val inferenceUs = result[1]
            val matchingUs = result[2]
            val poseUs = result[3]
            val kpCount = result[4].toInt()
            val matchCount = result[5].toInt()
            val inlierCount = result[6].toInt()
            val keyframeCount = result[7].toInt()
            val trajectoryCount = result[8].toInt()
            val budgetExceeded = result[9] > 0.5f
            val framesSinceKeyframe = result[10].toInt()
            val frameQualityScore = result[11]

            // Extract per-keypoint data (x, y, score, match_info per kp)
            val kpStart = 12
            val kpEnd = kpStart + kpCount * 4
            val kpCoords: FloatArray
            val kpScores: FloatArray
            val kpMatchInfo: FloatArray
            if (kpCount > 0 && result.size >= kpEnd) {
                kpCoords = FloatArray(kpCount * 2)
                kpScores = FloatArray(kpCount)
                kpMatchInfo = FloatArray(kpCount)
                for (i in 0 until kpCount) {
                    val off = kpStart + i * 4
                    kpCoords[i * 2] = result[off]
                    kpCoords[i * 2 + 1] = result[off + 1]
                    kpScores[i] = result[off + 2]
                    kpMatchInfo[i] = result[off + 3]
                }
            } else {
                kpCoords = FloatArray(0)
                kpScores = FloatArray(0)
                kpMatchInfo = FloatArray(0)
            }

            // Extract per-match data (prev_x, prev_y, curr_x, curr_y, ratio_quality, is_inlier)
            val matchStart = kpEnd
            val matchEnd = matchStart + matchCount * 6
            val matchLines: FloatArray
            val matchRatioQualities: FloatArray
            val matchInlierFlags: FloatArray
            if (matchCount > 0 && result.size >= matchEnd) {
                matchLines = FloatArray(matchCount * 4)
                matchRatioQualities = FloatArray(matchCount)
                matchInlierFlags = FloatArray(matchCount)
                for (i in 0 until matchCount) {
                    val off = matchStart + i * 6
                    matchLines[i * 4] = result[off]
                    matchLines[i * 4 + 1] = result[off + 1]
                    matchLines[i * 4 + 2] = result[off + 2]
                    matchLines[i * 4 + 3] = result[off + 3]
                    matchRatioQualities[i] = result[off + 4]
                    matchInlierFlags[i] = result[off + 5]
                }
            } else {
                matchLines = FloatArray(0)
                matchRatioQualities = FloatArray(0)
                matchInlierFlags = FloatArray(0)
            }

            // Extract per-trajectory data (x, z, heading_rad, inlier_ratio)
            val trajStart = matchEnd
            val trajEnd = trajStart + trajectoryCount * 4
            val trajectoryXZ: FloatArray
            val trajectoryHeadings: FloatArray
            val trajectoryInlierRatios: FloatArray
            if (trajectoryCount > 0 && result.size >= trajEnd) {
                trajectoryXZ = FloatArray(trajectoryCount * 2)
                trajectoryHeadings = FloatArray(trajectoryCount)
                trajectoryInlierRatios = FloatArray(trajectoryCount)
                for (i in 0 until trajectoryCount) {
                    val off = trajStart + i * 4
                    trajectoryXZ[i * 2] = result[off]
                    trajectoryXZ[i * 2 + 1] = result[off + 1]
                    trajectoryHeadings[i] = result[off + 2]
                    trajectoryInlierRatios[i] = result[off + 3]
                }
            } else {
                trajectoryXZ = FloatArray(0)
                trajectoryHeadings = FloatArray(0)
                trajectoryInlierRatios = FloatArray(0)
            }

            onFrameProcessed(
                FrameResult(
                    width = width,
                    height = height,
                    format = format,
                    resizeTimeUs = preprocessUs,
                    normalizeTimeUs = 0f,
                    totalTimeUs = preprocessUs + inferenceUs + matchingUs + poseUs,
                    useNeon = useNeon,
                    inferenceTimeUs = inferenceUs,
                    keypointCount = kpCount,
                    keypointCoords = kpCoords,
                    matchingTimeUs = matchingUs,
                    matchCount = matchCount,
                    matchLines = matchLines,
                    poseTimeUs = poseUs,
                    inlierCount = inlierCount,
                    keyframeCount = keyframeCount,
                    trajectoryXZ = trajectoryXZ,
                    budgetExceeded = budgetExceeded,
                    rotationDegrees = rotationDegrees,
                    keypointScores = kpScores,
                    keypointMatchInfo = kpMatchInfo,
                    matchRatioQualities = matchRatioQualities,
                    matchInlierFlags = matchInlierFlags,
                    trajectoryHeadings = trajectoryHeadings,
                    trajectoryInlierRatios = trajectoryInlierRatios,
                    framesSinceKeyframe = framesSinceKeyframe,
                    frameQualityScore = frameQualityScore
                )
            )
        }
    }
}

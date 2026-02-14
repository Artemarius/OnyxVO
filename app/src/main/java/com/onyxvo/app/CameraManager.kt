package com.onyxvo.app

import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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
        val trajectoryXZ: FloatArray = FloatArray(0)
    )

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    @Volatile
    var useNeon: Boolean = true

    @Volatile
    var modelReady: Boolean = false

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
    }

    private fun bindUseCases(cameraProvider: ProcessCameraProvider) {
        val preview = Preview.Builder()
            .build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(1920, 1080))
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
        try {
            val yPlane = imageProxy.planes[0]
            val yBuffer = yPlane.buffer
            val rowStride = yPlane.rowStride

            yBuffer.rewind()

            val width = imageProxy.width
            val height = imageProxy.height

            // Run preprocessing benchmark if requested (one-shot)
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
                // Phase 3+4: Full pipeline (preprocess + feature extraction + matching)
                val result = nativeBridge.nativeProcessFrame(
                    yBuffer, width, height, rowStride, useNeon
                )

                if (result != null && result.size >= 9) {
                    val preprocessUs = result[0]
                    val inferenceUs = result[1]
                    val matchingUs = result[2]
                    val poseUs = result[3]
                    val kpCount = result[4].toInt()
                    val matchCount = result[5].toInt()
                    val inlierCount = result[6].toInt()
                    val keyframeCount = result[7].toInt()
                    val trajectoryCount = result[8].toInt()

                    // Extract keypoint coordinates (packed as x0,y0,x1,y1,...)
                    val kpStart = 9
                    val kpEnd = kpStart + kpCount * 2
                    val kpCoords = if (kpCount > 0 && result.size >= kpEnd) {
                        result.copyOfRange(kpStart, kpEnd)
                    } else {
                        FloatArray(0)
                    }

                    // Extract match lines (packed as prev_x,prev_y,curr_x,curr_y,...)
                    val matchStart = kpEnd
                    val matchEnd = matchStart + matchCount * 4
                    val matchLines = if (matchCount > 0 && result.size >= matchEnd) {
                        result.copyOfRange(matchStart, matchEnd)
                    } else {
                        FloatArray(0)
                    }

                    // Extract trajectory XZ positions
                    val trajStart = matchEnd
                    val trajEnd = trajStart + trajectoryCount * 2
                    val trajectoryXZ = if (trajectoryCount > 0 && result.size >= trajEnd) {
                        result.copyOfRange(trajStart, trajEnd)
                    } else {
                        FloatArray(0)
                    }

                    onFrameProcessed(
                        FrameResult(
                            width = width,
                            height = height,
                            format = imageProxy.format,
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
                            trajectoryXZ = trajectoryXZ
                        )
                    )
                }
            } else {
                // Fallback: preprocessing only (Phase 2 behavior)
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
                            useNeon = useNeon
                        )
                    )
                }
            }
        } finally {
            imageProxy.close()
        }
    }
}

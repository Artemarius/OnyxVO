package com.onyxvo.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.onyxvo.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "OnyxVO.Main"
        private const val SMOOTHING = 0.1f
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var nativeBridge: NativeBridge
    private var cameraManager: CameraManager? = null

    private var frameCount = 0L
    private var lastUiUpdate = 0L
    private var avgTotalUs = 0f
    private var avgInferenceUs = 0f
    private var avgMatchingUs = 0f
    private var useInt8 = false
    private var modelLoaded = false
    private var matcherReady = false
    private var gpuMatcherAvailable = false

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            binding.debugOverlay.text = "Camera permission denied"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        nativeBridge = NativeBridge()
        nativeBridge.nativeInit()

        val version = nativeBridge.nativeGetVersion()
        Log.i(TAG, "Native version: $version")
        binding.debugOverlay.text = "OnyxVO v$version\nLoading model..."

        // Initialize FP32 model, then matcher
        initModel(useInt8 = false)

        binding.toggleNeonButton.setOnClickListener {
            cameraManager?.let { cam ->
                cam.useNeon = !cam.useNeon
                binding.toggleNeonButton.text = if (cam.useNeon) "NEON" else "Scalar"
            }
        }

        binding.toggleModelButton.setOnClickListener {
            useInt8 = !useInt8
            binding.toggleModelButton.text = if (useInt8) "INT8" else "FP32"
            binding.debugOverlay.text = "Switching model..."

            Thread {
                val success = nativeBridge.nativeSwitchModel(assets, useInt8)
                runOnUiThread {
                    if (success) {
                        Log.i(TAG, "Switched to ${if (useInt8) "INT8" else "FP32"}")
                    } else {
                        Log.e(TAG, "Model switch failed, reverting")
                        useInt8 = !useInt8
                        binding.toggleModelButton.text = if (useInt8) "INT8" else "FP32"
                    }
                }
            }.start()
        }

        binding.toggleMatcherButton.setOnClickListener {
            if (!matcherReady) return@setOnClickListener
            val currentlyGpu = binding.toggleMatcherButton.text == "GPU"
            val newUseGpu = !currentlyGpu
            // Only allow GPU if it's available
            val effectiveGpu = newUseGpu && gpuMatcherAvailable
            nativeBridge.nativeSetMatcherUseGpu(effectiveGpu)
            binding.toggleMatcherButton.text = if (effectiveGpu) "GPU" else "CPU"
        }

        binding.benchmarkButton.setOnClickListener {
            cameraManager?.let { cam ->
                binding.benchmarkResult.text = "Running benchmark..."
                binding.benchmarkResult.visibility = View.VISIBLE
                cam.onBenchmarkResult = { result ->
                    val text = String.format(
                        "Preprocess (100 iters):\nNEON: %.0f us  Scalar: %.0f us\nSpeedup: %.2fx",
                        result[0], result[1], result[2]
                    )
                    Log.i(TAG, text.replace('\n', ' '))
                    runOnUiThread {
                        binding.benchmarkResult.text = text
                    }
                }
                cam.benchmarkRequested = true
            }

            // Also run inference benchmark if model is loaded
            if (modelLoaded) {
                Thread {
                    val result = nativeBridge.nativeBenchmarkInference(assets, 10)
                    if (result != null) {
                        val text = String.format(
                            "Inference (10 iters):\nFP32: %.1f ms (%d kp)\n" +
                            "INT8: %.1f ms (%d kp)\nSpeedup: %.2fx",
                            result[0] / 1000, result[3].toInt(),
                            result[1] / 1000, result[4].toInt(),
                            result[2]
                        )
                        Log.i(TAG, text.replace('\n', ' '))
                        runOnUiThread {
                            binding.benchmarkResult.text = "${binding.benchmarkResult.text}\n\n$text"
                        }
                    }

                    // Run matching benchmark if matcher is ready
                    if (matcherReady) {
                        val matchResult = nativeBridge.nativeBenchmarkMatching(20)
                        if (matchResult != null) {
                            val matchText = String.format(
                                "Matching (20 iters, 500 desc):\nGPU: %.1f ms  CPU: %.1f ms\n" +
                                "Speedup: %.2fx  Matches: %d",
                                matchResult[0] / 1000, matchResult[1] / 1000,
                                matchResult[2], matchResult[3].toInt()
                            )
                            Log.i(TAG, matchText.replace('\n', ' '))
                            runOnUiThread {
                                binding.benchmarkResult.text =
                                    "${binding.benchmarkResult.text}\n\n$matchText"
                            }
                        }
                    }
                }.start()
            }
        }

        if (hasCameraPermission()) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager?.shutdown()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun initModel(useInt8: Boolean) {
        Thread {
            val success = nativeBridge.nativeInitModel(assets, useInt8)
            modelLoaded = success
            runOnUiThread {
                if (success) {
                    cameraManager?.modelReady = true
                    Log.i(TAG, "Model loaded: ${if (useInt8) "INT8" else "FP32"}")
                } else {
                    Log.e(TAG, "Model loading failed")
                    binding.debugOverlay.text = "OnyxVO v${nativeBridge.nativeGetVersion()}\nModel load failed"
                }
            }
            // Initialize matcher after model
            if (success) {
                initMatcher()
            }
        }.start()
    }

    private fun initMatcher() {
        try {
            gpuMatcherAvailable = nativeBridge.nativeInitMatcher()
            matcherReady = true
            Log.i(TAG, "Matcher initialized: GPU=${if (gpuMatcherAvailable) "yes" else "no"}")
            runOnUiThread {
                binding.toggleMatcherButton.text = if (gpuMatcherAvailable) "GPU" else "CPU"
                binding.toggleMatcherButton.visibility = View.VISIBLE
            }
        } catch (e: Exception) {
            Log.e(TAG, "Matcher init failed: ${e.message}")
            matcherReady = true  // CPU fallback always works
            runOnUiThread {
                binding.toggleMatcherButton.text = "CPU"
                binding.toggleMatcherButton.visibility = View.VISIBLE
            }
        }
    }

    private fun startCamera() {
        cameraManager = CameraManager(
            lifecycleOwner = this,
            previewView = binding.cameraPreview,
            nativeBridge = nativeBridge,
            onFrameProcessed = ::onFrameProcessed
        )
        cameraManager?.modelReady = modelLoaded
        cameraManager?.start()
    }

    private fun onFrameProcessed(result: CameraManager.FrameResult) {
        frameCount++

        // Exponential moving average for smooth display
        avgTotalUs = if (frameCount == 1L) result.totalTimeUs
                     else avgTotalUs * (1 - SMOOTHING) + result.totalTimeUs * SMOOTHING

        if (result.inferenceTimeUs > 0) {
            avgInferenceUs = if (frameCount == 1L) result.inferenceTimeUs
                             else avgInferenceUs * (1 - SMOOTHING) + result.inferenceTimeUs * SMOOTHING
        }

        if (result.matchingTimeUs > 0) {
            avgMatchingUs = if (frameCount == 1L) result.matchingTimeUs
                            else avgMatchingUs * (1 - SMOOTHING) + result.matchingTimeUs * SMOOTHING
        }

        // Update keypoint overlay
        if (result.keypointCoords.isNotEmpty()) {
            binding.keypointOverlay.updateKeypoints(result.keypointCoords, 640, 480)
        }

        // Update match lines overlay
        if (result.matchLines.isNotEmpty()) {
            binding.keypointOverlay.updateMatches(result.matchLines)
        } else {
            binding.keypointOverlay.updateMatches(FloatArray(0))
        }

        val now = System.currentTimeMillis()
        if (now - lastUiUpdate < 200) return // Update UI ~5 times per second
        lastUiUpdate = now

        val formatName = when (result.format) {
            ImageFormat.YUV_420_888 -> "YUV_420_888"
            else -> "fmt=${result.format}"
        }
        val mode = if (result.useNeon) "NEON" else "Scalar"
        val modelName = if (useInt8) "INT8" else "FP32"
        val matcherMode = if (gpuMatcherAvailable && binding.toggleMatcherButton.text == "GPU") "GPU" else "CPU"

        val text = buildString {
            append("OnyxVO v${nativeBridge.nativeGetVersion()}\n")
            append("${result.width}x${result.height} $formatName -> 640x480\n")
            append("Mode: $mode | Model: $modelName | Match: $matcherMode\n")

            if (result.inferenceTimeUs > 0) {
                append(String.format("Preproc: %.1f ms | Infer: %.1f ms\n",
                    result.resizeTimeUs / 1000, result.inferenceTimeUs / 1000))
                if (result.matchingTimeUs > 0) {
                    append(String.format("Match: %.1f ms | Total: %.1f ms\n",
                        result.matchingTimeUs / 1000, result.totalTimeUs / 1000))
                } else {
                    append(String.format("Total: %.1f ms (avg %.1f ms)\n",
                        result.totalTimeUs / 1000, avgTotalUs / 1000))
                }
                append("Keypoints: ${result.keypointCount}")
                if (result.matchCount > 0) {
                    append(" | Matches: ${result.matchCount}")
                }
            } else {
                append(String.format("Resize: %.0f us | Norm: %.0f us\n",
                    result.resizeTimeUs, result.normalizeTimeUs))
                append(String.format("Total: %.0f us (avg %.1f ms)",
                    result.totalTimeUs, avgTotalUs / 1000))
            }
        }

        runOnUiThread {
            binding.debugOverlay.text = text
        }
    }
}

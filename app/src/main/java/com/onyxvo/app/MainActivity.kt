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
    private var useInt8 = false
    private var modelLoaded = false

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

        // Initialize FP32 model
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
        }.start()
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

        // Update keypoint overlay
        if (result.keypointCoords.isNotEmpty()) {
            binding.keypointOverlay.updateKeypoints(result.keypointCoords, 640, 480)
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

        val text = buildString {
            append("OnyxVO v${nativeBridge.nativeGetVersion()}\n")
            append("${result.width}x${result.height} $formatName -> 640x480\n")
            append("Mode: $mode | Model: $modelName\n")

            if (result.inferenceTimeUs > 0) {
                append(String.format("Preproc: %.1f ms | Infer: %.1f ms\n",
                    result.resizeTimeUs / 1000, result.inferenceTimeUs / 1000))
                append(String.format("Total: %.1f ms (avg %.1f ms)\n",
                    result.totalTimeUs / 1000, avgTotalUs / 1000))
                append("Keypoints: ${result.keypointCount}")
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

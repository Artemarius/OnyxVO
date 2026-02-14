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
        binding.debugOverlay.text = "OnyxVO v$version"

        binding.toggleNeonButton.setOnClickListener {
            cameraManager?.let { cam ->
                cam.useNeon = !cam.useNeon
                binding.toggleNeonButton.text = if (cam.useNeon) "NEON" else "Scalar"
            }
        }

        binding.benchmarkButton.setOnClickListener {
            cameraManager?.let { cam ->
                binding.benchmarkResult.text = "Running benchmark..."
                binding.benchmarkResult.visibility = View.VISIBLE
                cam.onBenchmarkResult = { result ->
                    val text = String.format(
                        "Benchmark (100 iters):\nNEON: %.0f us  Scalar: %.0f us\nSpeedup: %.2fx",
                        result[0], result[1], result[2]
                    )
                    Log.i(TAG, text.replace('\n', ' '))
                    runOnUiThread {
                        binding.benchmarkResult.text = text
                    }
                }
                cam.benchmarkRequested = true
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

    private fun startCamera() {
        cameraManager = CameraManager(
            lifecycleOwner = this,
            previewView = binding.cameraPreview,
            nativeBridge = nativeBridge,
            onFrameProcessed = ::onFrameProcessed
        )
        cameraManager?.start()
    }

    private fun onFrameProcessed(result: CameraManager.FrameResult) {
        frameCount++

        // Exponential moving average for smooth display
        avgTotalUs = if (frameCount == 1L) result.totalTimeUs
                     else avgTotalUs * (1 - SMOOTHING) + result.totalTimeUs * SMOOTHING

        val now = System.currentTimeMillis()
        if (now - lastUiUpdate < 200) return // Update UI ~5 times per second
        lastUiUpdate = now

        val formatName = when (result.format) {
            ImageFormat.YUV_420_888 -> "YUV_420_888"
            else -> "fmt=${result.format}"
        }
        val mode = if (result.useNeon) "NEON" else "Scalar"
        val text = "OnyxVO v${nativeBridge.nativeGetVersion()}\n" +
            "${result.width}x${result.height} $formatName -> 640x480\n" +
            "Mode: $mode | Frames: $frameCount\n" +
            String.format("Resize: %.0f us | Norm: %.0f us\n",
                result.resizeTimeUs, result.normalizeTimeUs) +
            String.format("Total: %.0f us (avg %.1f ms)",
                result.totalTimeUs, avgTotalUs / 1000.0)

        runOnUiThread {
            binding.debugOverlay.text = text
        }
    }
}

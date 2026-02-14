package com.onyxvo.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.onyxvo.app.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "OnyxVO.Main"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var nativeBridge: NativeBridge
    private var cameraManager: CameraManager? = null

    private var frameCount = 0L
    private var lastLogTime = 0L

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
            onFrameInfo = ::onFrameInfo
        )
        cameraManager?.start()
    }

    private fun onFrameInfo(width: Int, height: Int, format: Int) {
        frameCount++
        val now = System.currentTimeMillis()

        if (now - lastLogTime >= 1000) {
            val formatName = when (format) {
                ImageFormat.YUV_420_888 -> "YUV_420_888"
                else -> "format=$format"
            }
            Log.i(TAG, "Frame #$frameCount: ${width}x${height} $formatName")

            val text = "OnyxVO v${nativeBridge.nativeGetVersion()}\n" +
                "${width}x${height} $formatName\n" +
                "Frames: $frameCount"

            runOnUiThread {
                binding.debugOverlay.text = text
            }
            lastLogTime = now
        }
    }
}

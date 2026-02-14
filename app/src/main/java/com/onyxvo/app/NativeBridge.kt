package com.onyxvo.app

import java.nio.ByteBuffer

class NativeBridge {

    companion object {
        init {
            System.loadLibrary("onyx_vo")
        }
    }

    external fun nativeInit()
    external fun nativeGetVersion(): String

    // Phase 2: Preprocessing
    // Returns FloatArray [copy_us, resize_us, normalize_us, total_us] or null on error
    external fun nativePreprocessFrame(
        yPlaneBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        useNeon: Boolean
    ): FloatArray?

    // Phase 2: A/B benchmark
    // Returns FloatArray [neon_avg_us, scalar_avg_us, speedup] or null on error
    external fun nativeBenchmarkPreprocessing(
        yPlaneBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        iterations: Int
    ): FloatArray?
}

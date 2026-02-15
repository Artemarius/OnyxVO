package com.onyxvo.app

import android.content.res.AssetManager
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
    // Returns FloatArray [resize_us, normalize_us, total_us] or null on error
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

    // Phase 2: NEON vs scalar validation (no camera needed — uses synthetic data)
    // Returns FloatArray [resize_max_err, normalize_max_err, pipeline_max_err, all_passed]
    external fun nativeValidatePreprocessing(): FloatArray?

    // Phase 3: Initialize XFeat model
    // Returns true on success
    external fun nativeInitModel(assetManager: AssetManager, useInt8: Boolean): Boolean

    // Phase 3+4: Process frame (preprocess + feature extraction + matching)
    // Returns FloatArray [preprocess_us, inference_us, matching_us, kp_count, match_count,
    //                     kp_coords..., match_lines...] or null
    external fun nativeProcessFrame(
        yPlaneBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        useNeon: Boolean
    ): FloatArray?

    // Phase 3: Switch between FP32 and INT8 models
    external fun nativeSwitchModel(assetManager: AssetManager, useInt8: Boolean): Boolean

    // Phase 3: Inference benchmark (FP32 vs INT8)
    // Returns FloatArray [fp32_avg_us, int8_avg_us, speedup, fp32_kp_count, int8_kp_count]
    external fun nativeBenchmarkInference(assetManager: AssetManager, iterations: Int): FloatArray?

    // Phase 3: Inference validation smoke test
    // Returns FloatArray [fp32_kp_count, int8_kp_count, diff_pct, passed]
    external fun nativeValidateInference(assetManager: AssetManager): FloatArray?

    // Phase 5: Initialize visual odometry (pose estimator + trajectory)
    external fun nativeInitVO()

    // Phase 5: Reset trajectory and clear previous frame
    external fun nativeResetTrajectory()

    // Phase 4: Initialize matcher (GPU + CPU fallback)
    // Returns true if GPU matcher is available, false if CPU-only
    external fun nativeInitMatcher(): Boolean

    // Phase 4: Toggle GPU/CPU matching
    external fun nativeSetMatcherUseGpu(useGpu: Boolean)

    // Phase 8: Toggle adaptive frame skipping (disabled during benchmarking)
    external fun nativeSetFrameSkipEnabled(enabled: Boolean)

    // Phase 4: Matching benchmark (GPU vs CPU)
    // Returns FloatArray [gpu_avg_us, cpu_avg_us, speedup, match_count]
    external fun nativeBenchmarkMatching(iterations: Int): FloatArray?

    // Phase 4: Matching validation (GPU vs CPU correctness)
    // Returns FloatArray [gpu_matches, cpu_matches, mismatches, passed]
    external fun nativeValidateMatching(): FloatArray?

    // Phase 6: Release heavyweight compute resources (ORT session, Vulkan) on pause
    external fun nativePause()

    // Phase 6: Destroy pipeline and free all native resources
    external fun nativeDestroy()

    // Phase 6: Check if pipeline is fully initialized and ready
    external fun nativeIsReady(): Boolean
}

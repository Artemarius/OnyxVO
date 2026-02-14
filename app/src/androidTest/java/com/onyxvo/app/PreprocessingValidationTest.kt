package com.onyxvo.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Phase 2 validation: NEON vs scalar preprocessing correctness.
 *
 * Runs on-device (arm64 required for NEON intrinsics). Generates synthetic
 * test images in native code — no camera or permissions needed.
 *
 * Thresholds:
 *   - Resize:    max |NEON - scalar| <= 1  (fixed-point rounding)
 *   - Normalize: max |NEON - scalar| < 1e-5 (float precision)
 *   - Pipeline:  max |NEON - scalar| < 0.005 (resize error propagated through normalize)
 */
@RunWith(AndroidJUnit4::class)
class PreprocessingValidationTest {

    private lateinit var bridge: NativeBridge

    @Before
    fun setUp() {
        bridge = NativeBridge()
        bridge.nativeInit()
    }

    @Test
    fun neonMatchesScalar() {
        val result = bridge.nativeValidatePreprocessing()
        assertNotNull("nativeValidatePreprocessing returned null", result)
        result!!

        val resizeMaxErr    = result[0]
        val normalizeMaxErr = result[1]
        val pipelineMaxErr  = result[2]
        val allPassed       = result[3]

        // Resize: NEON fixed-point vs scalar float — at most ±1 in uint8
        assertTrue(
            "Resize max error $resizeMaxErr exceeds threshold 1",
            resizeMaxErr <= 1.0f
        )

        // Normalize: both use float math, only vectorization differs
        assertTrue(
            "Normalize max error $normalizeMaxErr exceeds threshold 1e-5",
            normalizeMaxErr < 1e-5f
        )

        // Full pipeline: resize rounding propagated through normalize (1/255 ≈ 0.004)
        assertTrue(
            "Pipeline max error $pipelineMaxErr exceeds threshold 0.005",
            pipelineMaxErr < 0.005f
        )

        assertEquals("Not all validation checks passed", 1.0f, allPassed)
    }
}

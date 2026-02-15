package com.onyxvo.app

import android.content.res.AssetManager
import android.os.Build
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Automated benchmark that cycles through all 4 mode combinations
 * (FP32+GPU, FP32+CPU, INT8+GPU, INT8+CPU), collects per-frame stats
 * from live camera frames, and outputs a markdown table.
 */
class BenchmarkRunner(
    private val nativeBridge: NativeBridge,
    private val assetManager: AssetManager,
    private val outputDir: File,
    private val gpuAvailable: Boolean,
    private val onProgress: (String) -> Unit,
    private val onComplete: (String) -> Unit,
    private val onModeSwitch: (useInt8: Boolean, useGpu: Boolean, epType: Int) -> Unit
) {
    companion object {
        private const val TAG = "OnyxVO"
        private const val WARMUP_FRAMES = 30
        private const val MEASURE_FRAMES = 300
    }

    data class ModeConfig(
        val label: String,
        val useInt8: Boolean,
        val useGpu: Boolean,
        val epType: Int = 1  // 0=CPU, 1=XNNPACK, 2=NNAPI
    )

    private enum class Phase {
        IDLE, SWITCHING, WARMUP, MEASURING, DONE
    }

    // Ordered to minimize model reloads: FP32 group (XNNPACK pair, then NNAPI), then INT8 group
    private val modes: List<ModeConfig> = buildList {
        // FP32 + XNNPACK EP
        add(ModeConfig("FP32+XNNPACK+GPU", useInt8 = false, useGpu = true, epType = 1))
        add(ModeConfig("FP32+XNNPACK+CPU", useInt8 = false, useGpu = false, epType = 1))
        // FP32 + NNAPI EP (CPU matcher — NNAPI replaces XNNPACK for inference)
        add(ModeConfig("FP32+NNAPI+CPU", useInt8 = false, useGpu = false, epType = 2))
        // INT8 + default CPU EP
        add(ModeConfig("INT8+CPU+GPU", useInt8 = true, useGpu = true, epType = 0))
        add(ModeConfig("INT8+CPU+CPU", useInt8 = true, useGpu = false, epType = 0))
        // INT8 + NNAPI EP
        add(ModeConfig("INT8+NNAPI+CPU", useInt8 = true, useGpu = false, epType = 2))
    }.filter { !it.useGpu || gpuAvailable }

    private var phase = Phase.IDLE
    private var currentModeIndex = 0
    private var frameCounter = 0
    private var currentModelIsInt8 = false // tracks which model is currently loaded
    private var currentEpType = 1 // tracks which EP is currently active

    // Raw timing samples for current mode (microseconds)
    private val preprocessSamples = mutableListOf<Double>()
    private val inferenceSamples = mutableListOf<Double>()
    private val matchingSamples = mutableListOf<Double>()
    private val poseSamples = mutableListOf<Double>()
    private val totalSamples = mutableListOf<Double>()

    // Feature count accumulators
    private var totalKeypoints = 0L
    private var totalMatches = 0L
    private var totalInliers = 0L

    // Collected results per mode
    private data class ModeResult(
        val label: String,
        val preprocess: Stats,
        val inference: Stats,
        val matching: Stats,
        val pose: Stats,
        val total: Stats,
        val avgKeypoints: Double,
        val avgMatches: Double,
        val avgInliers: Double,
        val frameCount: Int
    )

    private data class Stats(
        val mean: Double,
        val median: Double,
        val p95: Double,
        val min: Double,
        val max: Double
    )

    private val results = mutableListOf<ModeResult>()

    @Volatile
    var running = false
        private set

    // Initial mode to restore after benchmark
    private var originalUseInt8 = false
    private var originalUseGpu = false
    private var originalEpType = 1

    fun start(currentUseInt8: Boolean, currentUseGpu: Boolean, currentEp: Int = 1) {
        if (running) return

        originalUseInt8 = currentUseInt8
        originalUseGpu = currentUseGpu
        originalEpType = currentEp
        currentModelIsInt8 = currentUseInt8
        currentEpType = currentEp

        running = true
        phase = Phase.IDLE
        currentModeIndex = 0
        results.clear()

        // Disable adaptive frame skipping for consistent measurements
        nativeBridge.nativeSetFrameSkipEnabled(false)

        switchToNextMode()
    }

    fun cancel() {
        if (!running) return
        running = false
        phase = Phase.DONE

        // Re-enable adaptive frame skipping
        nativeBridge.nativeSetFrameSkipEnabled(true)

        // Restore original mode
        restoreOriginalMode()

        onProgress("Benchmark cancelled")
    }

    /**
     * Called from onFrameProcessed() for every frame while benchmark is running.
     * Must be called on the UI thread (or at least the thread that calls onFrameProcessed).
     */
    fun onFrame(result: CameraManager.FrameResult) {
        if (!running) return

        when (phase) {
            Phase.WARMUP -> {
                frameCounter++
                val remaining = WARMUP_FRAMES - frameCounter
                if (frameCounter % 10 == 0) {
                    val mode = modes[currentModeIndex]
                    onProgress("${mode.label}: warming up ($remaining frames left)")
                }
                if (frameCounter >= WARMUP_FRAMES) {
                    // Transition to measuring
                    phase = Phase.MEASURING
                    frameCounter = 0
                    clearSamples()
                    val mode = modes[currentModeIndex]
                    onProgress("${mode.label}: measuring (0/${MEASURE_FRAMES})")
                }
            }
            Phase.MEASURING -> {
                // Collect sample
                preprocessSamples.add(result.resizeTimeUs.toDouble())
                inferenceSamples.add(result.inferenceTimeUs.toDouble())
                matchingSamples.add(result.matchingTimeUs.toDouble())
                poseSamples.add(result.poseTimeUs.toDouble())
                totalSamples.add(result.totalTimeUs.toDouble())

                totalKeypoints += result.keypointCount
                totalMatches += result.matchCount
                totalInliers += result.inlierCount

                frameCounter++
                if (frameCounter % 50 == 0) {
                    val mode = modes[currentModeIndex]
                    onProgress("${mode.label}: measuring ($frameCounter/${MEASURE_FRAMES})")
                }

                if (frameCounter >= MEASURE_FRAMES) {
                    // Collect results for this mode
                    finalizeModeResults()
                    currentModeIndex++

                    if (currentModeIndex >= modes.size) {
                        // All modes done
                        finishBenchmark()
                    } else {
                        switchToNextMode()
                    }
                }
            }
            else -> { /* IDLE, SWITCHING, DONE — ignore frames */ }
        }
    }

    private fun switchToNextMode() {
        phase = Phase.SWITCHING
        val mode = modes[currentModeIndex]
        val modeProgress = "${currentModeIndex + 1}/${modes.size}"
        onProgress("[$modeProgress] Switching to ${mode.label}...")

        Thread {
            try {
                // Switch model/EP if needed
                val needSwitch = mode.useInt8 != currentModelIsInt8 || mode.epType != currentEpType
                if (needSwitch) {
                    val success = nativeBridge.nativeSwitchModel(assetManager, mode.useInt8, mode.epType)
                    if (!success) {
                        Log.e(TAG, "BENCH: Model switch to ${mode.label} failed")
                        onProgress("ERROR: Model switch failed for ${mode.label}")
                        // Skip this mode
                        currentModeIndex++
                        if (currentModeIndex >= modes.size) {
                            finishBenchmark()
                        } else {
                            switchToNextMode()
                        }
                        return@Thread
                    }
                    currentModelIsInt8 = mode.useInt8
                    currentEpType = mode.epType
                }

                // Switch matcher
                nativeBridge.nativeSetMatcherUseGpu(mode.useGpu)

                // Reset trajectory for clean state
                nativeBridge.nativeResetTrajectory()
                // Re-disable frame skip (resetTrajectory resets skip state)
                nativeBridge.nativeSetFrameSkipEnabled(false)

                // Begin warmup on UI thread
                phase = Phase.WARMUP
                frameCounter = 0
                clearSamples()

                onModeSwitch(mode.useInt8, mode.useGpu, mode.epType)
                onProgress("${mode.label}: warming up ($WARMUP_FRAMES frames)")

            } catch (e: Exception) {
                Log.e(TAG, "BENCH: Error switching to ${mode.label}", e)
                onProgress("ERROR: ${e.message}")
                cancel()
            }
        }.start()
    }

    private fun clearSamples() {
        preprocessSamples.clear()
        inferenceSamples.clear()
        matchingSamples.clear()
        poseSamples.clear()
        totalSamples.clear()
        totalKeypoints = 0L
        totalMatches = 0L
        totalInliers = 0L
    }

    private fun finalizeModeResults() {
        val mode = modes[currentModeIndex]
        val n = totalSamples.size

        val modeResult = ModeResult(
            label = mode.label,
            preprocess = computeStats(preprocessSamples),
            inference = computeStats(inferenceSamples),
            matching = computeStats(matchingSamples),
            pose = computeStats(poseSamples),
            total = computeStats(totalSamples),
            avgKeypoints = if (n > 0) totalKeypoints.toDouble() / n else 0.0,
            avgMatches = if (n > 0) totalMatches.toDouble() / n else 0.0,
            avgInliers = if (n > 0) totalInliers.toDouble() / n else 0.0,
            frameCount = n
        )
        results.add(modeResult)

        // Emit logcat lines
        emitLogcatResults(mode.label, "preprocess", modeResult.preprocess)
        emitLogcatResults(mode.label, "inference", modeResult.inference)
        emitLogcatResults(mode.label, "matching", modeResult.matching)
        emitLogcatResults(mode.label, "pose", modeResult.pose)
        emitLogcatResults(mode.label, "total", modeResult.total)

        Log.i(TAG, "BENCH_RESULT|${mode.label}|counts|" +
            "keypoints=%.0f|matches=%.0f|inliers=%.0f|frames=%d".format(
                modeResult.avgKeypoints, modeResult.avgMatches,
                modeResult.avgInliers, modeResult.frameCount))
    }

    private fun emitLogcatResults(mode: String, stage: String, stats: Stats) {
        Log.i(TAG, "BENCH_RESULT|$mode|$stage|" +
            "mean=%.0f|median=%.0f|p95=%.0f|min=%.0f|max=%.0f".format(
                stats.mean, stats.median, stats.p95, stats.min, stats.max))
    }

    private fun finishBenchmark() {
        phase = Phase.DONE
        running = false

        // Re-enable adaptive frame skipping
        nativeBridge.nativeSetFrameSkipEnabled(true)

        // Generate markdown
        val markdown = generateMarkdown()

        // Write to file
        try {
            val file = File(outputDir, "onyxvo_benchmark.md")
            file.writeText(markdown)
            Log.i(TAG, "BENCH: Results written to ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "BENCH: Failed to write results file", e)
        }

        // Restore original mode
        restoreOriginalMode()

        onComplete(markdown)
    }

    private fun restoreOriginalMode() {
        Thread {
            try {
                if (currentModelIsInt8 != originalUseInt8 || currentEpType != originalEpType) {
                    nativeBridge.nativeSwitchModel(assetManager, originalUseInt8, originalEpType)
                    currentModelIsInt8 = originalUseInt8
                    currentEpType = originalEpType
                }
                nativeBridge.nativeSetMatcherUseGpu(originalUseGpu)
                onModeSwitch(originalUseInt8, originalUseGpu, originalEpType)
            } catch (e: Exception) {
                Log.e(TAG, "BENCH: Failed to restore original mode", e)
            }
        }.start()
    }

    private fun computeStats(samples: List<Double>): Stats {
        if (samples.isEmpty()) return Stats(0.0, 0.0, 0.0, 0.0, 0.0)

        val sorted = samples.sorted()
        val n = sorted.size
        val mean = sorted.sum() / n
        val median = if (n % 2 == 0) (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 else sorted[n / 2]
        val p95idx = ((n - 1) * 0.95).toInt().coerceIn(0, n - 1)
        val p95 = sorted[p95idx]

        return Stats(
            mean = mean,
            median = median,
            p95 = p95,
            min = sorted.first(),
            max = sorted.last()
        )
    }

    private fun generateMarkdown(): String {
        val deviceModel = Build.MODEL
        val dateStr = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())

        val sb = StringBuilder()
        sb.appendLine("## OnyxVO Benchmark Results")
        sb.appendLine("- **Device:** $deviceModel")
        sb.appendLine("- **Date:** $dateStr")
        sb.appendLine("- **Warmup:** $WARMUP_FRAMES frames, **Measurement:** $MEASURE_FRAMES frames per mode")
        sb.appendLine()

        // Pipeline Total table
        sb.appendLine("### Pipeline Total (ms)")
        sb.appendLine("| Mode | Mean | Median | P95 | Min | Max |")
        sb.appendLine("|------|------|--------|-----|-----|-----|")
        for (r in results) {
            sb.appendLine("| ${r.label} | ${fmtMs(r.total.mean)} | ${fmtMs(r.total.median)} | " +
                "${fmtMs(r.total.p95)} | ${fmtMs(r.total.min)} | ${fmtMs(r.total.max)} |")
        }
        sb.appendLine()

        // Per-Stage Mean table
        sb.appendLine("### Per-Stage Mean (ms)")
        sb.appendLine("| Mode | Preprocess | Inference | Matching | Pose | Total |")
        sb.appendLine("|------|------------|-----------|----------|------|-------|")
        for (r in results) {
            sb.appendLine("| ${r.label} | ${fmtMs(r.preprocess.mean)} | ${fmtMs(r.inference.mean)} | " +
                "${fmtMs(r.matching.mean)} | ${fmtMs(r.pose.mean)} | ${fmtMs(r.total.mean)} |")
        }
        sb.appendLine()

        // Feature Counts table
        sb.appendLine("### Feature Counts (avg)")
        sb.appendLine("| Mode | Keypoints | Matches | Inliers | Frames |")
        sb.appendLine("|------|-----------|---------|---------|--------|")
        for (r in results) {
            sb.appendLine("| ${r.label} | %.0f | %.0f | %.0f | %d |".format(
                r.avgKeypoints, r.avgMatches, r.avgInliers, r.frameCount))
        }
        sb.appendLine()

        return sb.toString()
    }

    /** Format microseconds as milliseconds with 1 decimal. */
    private fun fmtMs(us: Double): String = "%.1f".format(us / 1000.0)
}

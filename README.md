# OnyxVO

Real-time visual odometry on Android using learned features (XFeat via ONNX Runtime), Vulkan compute (Kompute) for descriptor matching, and ARM NEON SIMD for image preprocessing. Thin Kotlin UI layer, heavy C++ native core.

## Development Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project skeleton, CameraX preview, JNI bridge | Done |
| 2 | NEON image preprocessing (resize, normalize) with benchmarks | Done |
| 3 | XFeat ONNX integration (FP32 + INT8) with keypoint overlay | Done |
| 4 | Kompute descriptor matching (Vulkan compute shader) | Done |
| 5 | Pose estimation (RANSAC + essential matrix) with trajectory | Done |
| 6 | Full pipeline integration + performance dashboard | Done |
| 7 | Enhanced visualization (quality-colored keypoints, match lines, trajectory) | Done |
| 8 | Optimization (XNNPACK, NNAPI, shared memory tiling, adaptive frame skip) | Done |
| 9 | Polish, documentation, demo recording | In Progress |

## Architecture

```
Camera Frame (CameraX YUV_420_888)
    |
    v
+---------------------------------+
|  NEON Preprocessing   ~0.3 ms   |  Y-plane extract, bilinear resize, float normalize
+---------------------------------+  (fixed-point NEON intrinsics, 16 px/iter)
    |
    v
+---------------------------------+
|  XFeat (ONNX Runtime)  ~60 ms  |  Backbone + NMS + descriptor interpolation
+---------------------------------+  (FP32/XNNPACK or INT8/CPU, 500 keypoints)
    |
    v
+---------------------------------+
|  Descriptor Matching   ~15 ms  |  Brute-force L2, ratio test
+---------------------------------+  (Vulkan compute or CPU fallback)
    |
    v
+---------------------------------+
|  Pose Estimation       ~5 ms   |  5-point RANSAC + essential matrix (Eigen)
+---------------------------------+  (cheirality check, keyframe management)
    |
    v
+---------------------------------+
|  Trajectory + UI               |  Quality-colored path, heading arrow, dashboard
+---------------------------------+
```

## Benchmarks

Samsung Galaxy S21 (Exynos 2100), 300 live frames per mode, adaptive frame skip disabled:

### Pipeline Total (ms)

| Mode | Mean | Median | P95 | Min | Max |
|------|------|--------|-----|-----|-----|
| FP32 + XNNPACK + GPU | 85.7 | 84.7 | 101.9 | 71.3 | 105.7 |
| FP32 + XNNPACK + CPU | 87.2 | 81.3 | 110.0 | 66.2 | 150.1 |
| INT8 + CPU + GPU | 104.5 | 103.0 | 114.7 | 86.1 | 295.9 |
| INT8 + CPU + CPU | 82.0 | 80.0 | 92.1 | 72.1 | 361.5 |

### Per-Stage Breakdown (mean, ms)

| Mode | Preprocess | Inference | Matching | Pose | Total |
|------|------------|-----------|----------|------|-------|
| FP32 + XNNPACK + GPU | 0.2 | 58.8 | 22.6 | 4.0 | 85.7 |
| FP32 + XNNPACK + CPU | 0.3 | 68.5 | 14.0 | 4.4 | 87.2 |
| INT8 + CPU + GPU | 0.3 | 60.0 | 38.5 | 5.7 | 104.5 |
| INT8 + CPU + CPU | 0.4 | 59.4 | 16.9 | 5.3 | 82.0 |

### Feature Counts (average)

| Mode | Keypoints | Matches | Inliers |
|------|-----------|---------|---------|
| FP32 + GPU | 500 | 309 | 171 |
| FP32 + CPU | 500 | 317 | 174 |
| INT8 + GPU | 500 | 300 | 155 |
| INT8 + CPU | 500 | 299 | 151 |

### Key Findings

- **Inference dominates** (~70% of frame time). The XFeat backbone is the bottleneck, not matching or pose estimation.
- **INT8 provides no speedup** over FP32 with XNNPACK on Exynos 2100. The XNNPACK EP does not accelerate quantized ops on this SoC.
- **GPU matching is slower than CPU** at 500 descriptors. Vulkan dispatch overhead exceeds the compute savings at this scale. GPU matching would win at higher descriptor counts (~2000+).
- **NNAPI (Exynos NPU)** was tested but showed no improvement; the NPU rejected most XFeat ops, falling back to CPU.
- **Best mode:** INT8 + CPU matching (80 ms median, ~12.5 FPS).

## Building

### Prerequisites

- Android Studio (Hedgehog 2023.1.1+)
- Android NDK 25.1.8937393 (required for Kompute Vulkan wrapper compatibility)
- CMake 3.22+ (install via SDK Manager)
- Physical `arm64-v8a` Android device (emulator lacks Vulkan)

### Build

```bash
git clone --recurse-submodules https://github.com/artem-shamsuarov/onyx-vo.git
cd onyx-vo
./gradlew assembleDebug
```

Or open in Android Studio and run on a connected device.

The first build fetches Eigen 3.4 via CMake `FetchContent` and compiles the GLSL compute shader to SPIR-V using the NDK's `glslc`.

### Model Optimization (optional)

Pre-bake ONNX Runtime graph optimizations to reduce on-device session creation time:

```bash
pip install onnxruntime numpy
python scripts/optimize_onnx.py --verify
```

This saves `xfeat_fp32_opt.onnx` / `xfeat_int8_opt.onnx` in the assets directory. The native code automatically prefers optimized models when present.

## Project Structure

```
OnyxVO/
+-- app/src/main/
|   +-- java/com/onyxvo/app/
|   |   +-- MainActivity.kt              # Lifecycle, permissions, UI setup
|   |   +-- CameraManager.kt             # CameraX frame delivery + JNI pipeline calls
|   |   +-- NativeBridge.kt              # JNI declarations
|   |   +-- KeypointOverlayView.kt       # Quality-colored keypoints + match lines
|   |   +-- TrajectoryView.kt            # Heading arrow, quality-colored path, grid
|   |   +-- PerformanceDashboardView.kt  # FPS, per-stage timing bars, mode indicators
|   |   +-- BenchmarkRunner.kt           # Automated 6-mode benchmark orchestration
|   |
|   +-- cpp/
|   |   +-- jni_bridge.cpp               # Single JNI entry point
|   |   +-- pipeline.h / pipeline.cpp    # Frame pipeline orchestrator
|   |   +-- preprocessing/
|   |   |   +-- neon_ops.h / .cpp        # ARM NEON: Y-plane resize, normalize
|   |   +-- feature/
|   |   |   +-- xfeat_extractor.h / .cpp # ONNX Runtime session, keypoint extraction
|   |   +-- matching/
|   |   |   +-- cpu_matcher.h / .cpp     # CPU brute-force L2 + ratio test
|   |   |   +-- gpu_matcher.h / .cpp     # Kompute Vulkan compute dispatch
|   |   |   +-- shaders/
|   |   |       +-- match_descriptors.comp  # GLSL compute kernel (shared mem tiling)
|   |   +-- vo/
|   |   |   +-- pose_estimator.h / .cpp  # 5-point RANSAC + essential matrix
|   |   |   +-- trajectory.h / .cpp      # Cumulative pose tracking
|   |   +-- utils/
|   |   |   +-- android_log.h            # __android_log_print wrapper
|   |   |   +-- timer.h                  # ScopedTimer (RAII microsecond timing)
|   |   |   +-- trace.h                  # ScopedTrace (ATrace/Perfetto markers)
|   |   +-- cmake/
|   |   |   +-- spv_to_header.cmake      # SPIR-V -> C++ uint32_t[] header
|   |   +-- third_party/
|   |       +-- kompute/                 # Vulkan compute framework (git submodule)
|   |       +-- onnxruntime/include/     # ORT C++ headers (vendored)
|   |
|   +-- assets/
|       +-- xfeat_fp32.onnx             # XFeat FP32 model
|       +-- xfeat_int8.onnx             # XFeat INT8 quantized model
|
+-- scripts/
|   +-- export_xfeat_onnx.py            # PyTorch -> ONNX export
|   +-- quantize_xfeat.py               # FP32 -> INT8 static quantization
|   +-- optimize_onnx.py                # ORT graph optimization (fusions, folding)
|
+-- onyxvo_benchmark_baseline.md        # Benchmark results (S21 / Exynos 2100)
```

## Technical Highlights

### ARM NEON SIMD Preprocessing
Hand-written NEON intrinsics for the image preprocessing hot path. 14-bit fixed-point bilinear resize handles non-integer scale factors with stride-aware Y-plane access. Float normalization processes 16 pixels per iteration via `float32x4` operations. No OpenCV dependency.

### ONNX Runtime Inference
XFeat backbone runs via ONNX Runtime with configurable execution providers: XNNPACK (FP32), CPU (INT8), or NNAPI (experimental NPU offload). Session options use `ORT_ENABLE_ALL` graph optimization with single-threaded execution. Backbone-only model output is decoded with SuperPoint-style softmax, full-resolution heatmap, 5x5 NMS, and bilinear descriptor interpolation.

### Vulkan Compute Matching
Brute-force L2 descriptor matching via a GLSL compute shader dispatched through Kompute. Shared memory tiling (TILE_SIZE=32, 8 KB) cooperatively loads desc2 tiles for coalesced global memory access, reducing bandwidth by up to 256x. Workgroup size is configurable (64/128/256) via Vulkan specialization constants without shader recompilation. Ratio test applied CPU-side after GPU readback.

### Pose Estimation
Custom 5-point algorithm (Nister) with RANSAC using Eigen for all linear algebra. Essential matrix decomposed via `JacobiSVD`, 4-solution cheirality check selects the geometrically valid pose. Keyframe management triggers on low inlier ratio or high median feature displacement.

### Adaptive Frame Skipping
EMA-tracked processing time with hysteresis-based skip interval adjustment (1-3 frames). Skipped frames return cached results. Prevents UI stutter when pipeline exceeds the frame budget.

## Profiling with Perfetto

The pipeline has ATrace markers on all stages. To capture a trace:

```bash
# Record a 10-second trace (on device or via adb)
adb shell perfetto -o /data/misc/perfetto-traces/onyxvo.perfetto-trace -t 10s \
  -c - <<EOF
buffers: { size_kb: 32768 }
data_sources: {
  config {
    name: "linux.ftrace"
    ftrace_config {
      atrace_categories: "view"
      atrace_apps: "com.onyxvo.app"
    }
  }
}
EOF

# Pull and open in Perfetto UI
adb pull /data/misc/perfetto-traces/onyxvo.perfetto-trace .
# Open https://ui.perfetto.dev and load the trace file
```

Look for `OnyxVO::processFrame`, `OnyxVO::preprocess`, `OnyxVO::extract`, `OnyxVO::ort_inference`, `OnyxVO::match`, `OnyxVO::gpu_dispatch`, and `OnyxVO::pose` sections.

## Dependencies

| Library | Version | Role |
|---------|---------|------|
| [CameraX](https://developer.android.com/media/camera/camerax) | 1.3.1 | Camera frame capture |
| [ONNX Runtime](https://onnxruntime.ai/) | 1.20.0 | Neural network inference |
| [Kompute](https://github.com/KomputeProject/kompute) | 0.8.1 | Vulkan compute abstraction |
| [Eigen](https://eigen.tuxfamily.org/) | 3.4.0 | Linear algebra (header-only) |
| [XFeat](https://github.com/verlab/accelerated_features) | CVPR 2024 | Learned feature extraction model |
| Android NDK | 25.1 | ARM64 native compilation, NEON, Vulkan |

## Performance Analysis

The pipeline runs at ~12.5 FPS (80 ms/frame) against an original target of 25+ FPS. Here's where the time goes and what it would take to close the gap:

```
Frame budget breakdown (80 ms total):

  Preprocessing   ███                                              0.3 ms  ( 0.4%)
  Inference       ████████████████████████████████████████████████  59  ms  (74  %)
  Matching        ██████████████                                   17  ms  (21  %)
  Pose            ████                                              5  ms  ( 6  %)
                  |---------|---------|---------|---------|---------|
                  0        12        24        36        48       60 ms
```

Inference consumes ~74% of each frame. The non-inference pipeline (preprocessing + matching + pose) totals ~22 ms, which alone would sustain 45 FPS. The bottleneck is entirely in the XFeat backbone forward pass.

### What was tried

| Optimization | Result |
|---|---|
| XNNPACK EP (FP32) | Reduced inference from ~90 ms to ~59 ms — largest single win |
| INT8 quantization | No speedup on Exynos 2100 (XNNPACK doesn't accelerate INT8 on this SoC) |
| NNAPI (Exynos NPU) | NPU rejected most XFeat ops; fell back to CPU |
| ORT graph optimization | `ORT_ENABLE_ALL` fusions applied; marginal improvement |
| Shared memory tiling | Matching improved but matching isn't the bottleneck |
| Adaptive frame skip | Smooths UI cadence but doesn't reduce per-frame cost |

### What would close the gap

Reaching 25+ FPS (40 ms budget) requires cutting inference from 59 ms to ~18 ms — a 3.3x reduction. Realistic paths:

1. **Smaller backbone** — Distill XFeat to a MobileNetV3-Small or EfficientNet-Lite0 backbone. Trades keypoint quality for speed. Estimated inference: 15-25 ms.
2. **ORT mobile format** — Convert to `.ort` flatbuffer format with pre-optimization. Eliminates session startup overhead but minimal per-frame gain (~5-10%).
3. **Qualcomm QNN EP** — On Snapdragon 8 Gen 2+ with Hexagon DSP, QNN can accelerate the full backbone. Requires Snapdragon-specific toolchain and model compilation.
4. **Pipelined inference** — Overlap frame N inference with frame N-1 matching/pose on separate threads. Doubles throughput at the cost of one frame latency.
5. **Reduced resolution** — Drop from 640x480 to 320x240 input. Roughly 4x fewer backbone FLOPs. Significant keypoint quality loss.

None of these were pursued because the current project demonstrates the full technical stack (NEON, ONNX Runtime, Vulkan compute, geometric VO) at a level that meets the portfolio goals. Real-time performance on constrained hardware is an engineering tradeoff, not a missing capability.

## Known Limitations

- **Monocular scale ambiguity:** No absolute scale without IMU or known geometry. Trajectory shows relative motion only.
- **Drift:** No loop closure or bundle adjustment. Trajectory drifts over extended sequences.
- **Lighting sensitivity:** XFeat keypoint quality degrades in low-light or motion-blurred frames.
- **Single device tested:** Benchmarks are from Samsung Galaxy S21 (Exynos 2100) only. Performance varies across SoCs.

## Future Work

- **Smaller backbone:** Distill XFeat to a lighter architecture for sub-30 ms inference
- **Pipelined inference:** Overlap inference with matching/pose on separate threads
- **Loop closure:** Detect revisited places and correct accumulated drift
- **IMU fusion:** Accelerometer/gyroscope pre-integration for absolute scale and inter-frame prediction
- **QNN EP:** Qualcomm QNN execution provider for Snapdragon DSP/NPU acceleration

## References

- Potje et al., "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024
- Nister, "An Efficient Solution to the Five-Point Relative Pose Problem", IEEE TPAMI 2004
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision", Cambridge University Press
- [Kompute Documentation](https://kompute.cc/)
- [ONNX Runtime Mobile Guide](https://onnxruntime.ai/docs/tutorials/mobile/)

## License

MIT

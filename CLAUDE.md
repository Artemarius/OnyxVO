# CLAUDE.md — OnyxVO

## What This Project Is

An Android application implementing real-time visual odometry using XFeat (learned features via ONNX Runtime), Vulkan compute (via Kompute) for descriptor matching, and ARM NEON SIMD for image preprocessing. Thin Kotlin UI layer, heavy C++ native core.

**Read PROJECT.md for full roadmap, phase definitions, and strategic context.**

## Developer Background

Senior C++/CUDA/Computer Vision engineer (15+ years). Expert in 3D reconstruction, SLAM, GPU optimization, real-time image processing. Built production SLAM systems at Artec3D and Samsung. Has Vulkan compute experience. New to Android NDK development (learning), but deeply experienced with cross-platform C++ and mobile GPU optimization (OpenCL/Metal on iOS/Android at Samsung).

## Architecture Rules

### Language Split
- **Kotlin:** UI only — CameraX setup, trajectory rendering canvas, JNI calls, lifecycle management
- **C++ (NDK):** All compute — preprocessing, inference, matching, pose estimation
- **GLSL:** Vulkan compute shaders for descriptor matching (compiled to SPIR-V)

### JNI Bridge
- Single JNI bridge file: `jni_bridge.cpp`
- Minimize JNI crossings per frame — one call in (frame data), one call out (pose + keypoints for visualization)
- Frame data passed as direct ByteBuffer, not copied
- Native side owns all compute state (ORT session, Kompute manager, pose history)

### Module Structure (C++ side)
```
cpp/
├── jni_bridge.cpp              # Single JNI entry point
├── pipeline.h / pipeline.cpp   # Orchestrates full frame pipeline
├── preprocessing/
│   ├── neon_ops.h              # NEON intrinsic declarations
│   └── neon_ops.cpp            # RGB→gray, resize, normalize (ARM NEON)
├── feature/
│   ├── xfeat_extractor.h       # ONNX Runtime session wrapper
│   └── xfeat_extractor.cpp     # Load model, run inference, extract keypoints + descriptors
├── matching/
│   ├── gpu_matcher.h           # Kompute-based matching
│   ├── gpu_matcher.cpp         # Setup Kompute manager, dispatch compute, read results
│   └── shaders/                # Pre-compiled SPIR-V or raw GLSL
├── vo/
│   ├── pose_estimator.h        # RANSAC + essential matrix
│   ├── pose_estimator.cpp      # 5-point algorithm, cheirality check
│   ├── trajectory.h            # Cumulative pose tracking
│   └── trajectory.cpp
└── utils/
    ├── timer.h                 # Scoped timing for each pipeline stage
    └── android_log.h           # __android_log_print wrapper
```

### Memory & Performance Rules
- **Zero-copy camera path:** CameraX YUV buffer → JNI direct ByteBuffer → NEON processing. No Java-side pixel copies
- **Pre-allocate everything:** ONNX session, Kompute tensors, descriptor buffers — all created at init, reused every frame
- **NEON intrinsics only:** No OpenCV for preprocessing. Hand-written NEON for RGB→gray, bilinear resize, float normalize
- **Kompute buffer lifecycle:** Create `kp::Tensor` once at init with max descriptor count, sync only used portion
- **Eigen for geometry:** Essential matrix, SVD decomposition, pose composition — all Eigen. No OpenCV geometry

### ONNX Runtime Configuration
- **Execution provider:** CPU (XNNPACK) as primary. NNAPI as optional bonus (not all ops may be supported)
- **Session options:** `ORT_ENABLE_ALL` graph optimization, single intra-op thread, `ORT_DISABLE_ALL` for inter-op
- **Model format:** Standard ONNX (not ORT mobile format initially — can optimize later)
- **Input:** `[1, 1, H, W]` float32 grayscale, fixed resolution (e.g., 640×480)
- **Outputs:** keypoints `[N, 2]`, descriptors `[N, 64]`, scores `[N]`
- **Quantized model:** INT8 (uint8 weights, uint8 activations) via ONNX Runtime static quantization

### Kompute / Vulkan Compute Configuration
- **Android Vulkan:** Dynamic loading via Kompute's NDK wrapper (`VK_USE_PLATFORM_ANDROID_KHR`)
- **Compute shader:** GLSL → SPIR-V (pre-compiled, stored as raw asset or embedded header)
- **Matching algorithm:** Brute-force L2 with ratio test, workgroup size tuned per device (start with 256)
- **Buffer layout:** Descriptors as `float[N][64]`, output as `int[N]` match indices + `float[N]` distances
- **Fallback:** If Vulkan unavailable, CPU brute-force matching (scalar with optional NEON dot product)

### Pose Estimation
- **Algorithm:** 5-point algorithm (Nister) with RANSAC
- **Eigen usage:** `Eigen::JacobiSVD` for essential matrix decomposition, `Eigen::Matrix3d` / `Eigen::Vector3d` for poses
- **Scale:** Relative scale only (monocular VO). No absolute scale without IMU or known geometry
- **Coordinate frame:** Camera-centric, right-handed (X-right, Y-down, Z-forward, OpenCV convention)
- **Keyframe strategy:** New keyframe when number of inlier matches drops below threshold or median parallax exceeds threshold

## Build System

### CMake (native)
```
# Top-level native CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(onyx_vo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Kompute via add_subdirectory
add_subdirectory(${KOMPUTE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)

# ONNX Runtime: pre-built AAR extracts .so + headers
# Eigen: header-only, FetchContent or vendored

# Target: single shared library
add_library(onyx_vo SHARED
    jni_bridge.cpp
    pipeline.cpp
    preprocessing/neon_ops.cpp
    feature/xfeat_extractor.cpp
    matching/gpu_matcher.cpp
    vo/pose_estimator.cpp
    vo/trajectory.cpp
)

target_link_libraries(onyx_vo
    kompute
    kompute_vk_ndk_wrapper
    onnxruntime
    log
    android
)
```

### Gradle (app)
- `minSdk 28` (Vulkan 1.1 widely available from API 28+)
- `targetSdk 34`
- `externalNativeBuild` with CMake
- ONNX Runtime Android via Maven: `com.microsoft.onnxruntime:onnxruntime-android`
- NDK ABI filter: `arm64-v8a` only (NEON guaranteed, simplifies builds)

## Phase Boundaries

Claude Code should implement ONE phase at a time. Do not pull in components from later phases.

- **Phase 1:** Android project skeleton + CameraX preview + JNI hello-world
- **Phase 2:** NEON preprocessing (RGB→gray, resize, normalize) with benchmarks
- **Phase 3:** XFeat ONNX integration (FP32 first, then INT8) with keypoint overlay
- **Phase 4:** Kompute setup + descriptor matching compute shader
- **Phase 5:** Pose estimation (RANSAC + essential matrix) with trajectory display
- **Phase 6:** Pipeline integration — full frame pipeline, performance dashboard
- **Phase 7:** Optimization — profiling, memory tuning, shader workgroup tuning
- **Phase 8:** Polish — UI, demo recording, README screenshots, benchmarks table

## File Naming Conventions

- C++ headers: `.h` (not `.hpp` — consistency with Android NDK style)
- C++ source: `.cpp`
- GLSL shaders: `.comp` (compute)
- SPIR-V: `.spv`
- Kotlin: standard Android conventions (`PascalCase` classes)
- No underscores in Kotlin filenames, underscores in C++ filenames

## Testing Strategy

- **Unit tests (C++):** Google Test via CMake for pose estimation math, NEON output validation
- **Instrumented tests (Android):** Keypoint count sanity, pipeline latency assertions
- **Visual validation:** Record known trajectory (e.g., phone on desk, rotate 360°), verify closed loop
- **A/B comparison:** FP32 vs INT8 keypoint quality (reprojection error on calibration images)

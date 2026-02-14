# OnyxVO

Real-time visual odometry on Android, powered by learned features (XFeat), Vulkan compute (Kompute), and NEON SIMD — all running on-device.

## Motivation

Most visual odometry implementations target desktop GPUs or ROS-based robot platforms. I wanted to build one that runs entirely on a phone — using Vulkan compute for GPU-accelerated descriptor matching, a quantized neural feature extractor via ONNX Runtime, and ARM NEON intrinsics for the image preprocessing hot path. The goal is a self-contained Android app that tracks camera motion in real-time and visualizes the trajectory on screen.

## Architecture

```
Camera Frame (CameraX)
    │
    ▼
┌─────────────────────────┐
│  NEON Preprocessing     │  ARM SIMD: Y-plane extract, resize, normalize
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  XFeat (ONNX Runtime)   │  INT8 quantized model → keypoints + 64-dim descriptors
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Kompute Matching       │  Vulkan compute shader: brute-force L2 descriptor matching
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Pose Estimation        │  RANSAC + Essential matrix decomposition (Eigen)
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Trajectory Viewer      │  Real-time 2D/3D trajectory overlay on camera feed
└─────────────────────────┘
```

## Development Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project skeleton, CameraX preview, JNI bridge | Done |
| 2 | NEON image preprocessing with benchmarks | Planned |
| 3 | XFeat ONNX integration (FP32 + INT8) | Planned |
| 4 | Kompute descriptor matching (Vulkan compute) | Planned |
| 5 | Pose estimation (RANSAC + essential matrix) | Planned |
| 6 | Full pipeline integration + performance dashboard | Planned |
| 7 | Optimization + multi-device benchmarks | Planned |
| 8 | Polish, demo recording, documentation | Planned |

### What's Working Now

- Android project with Gradle + CMake NDK build pipeline
- CameraX live preview with `ImageAnalysis` frame callback
- JNI bridge: native C++ library loaded, `nativeInit()` / `nativeGetVersion()` round-trip
- Frame metadata (resolution, YUV format) logged from analyzer callback
- Debug overlay displaying native version string and frame info

## Performance Targets

| Stage | Target | Device |
|---|---|---|
| NEON preprocessing (VGA) | < 2 ms | Snapdragon 8 Gen 1 |
| XFeat INT8 inference | < 15 ms | CPU (XNNPACK) |
| Kompute matching (500 kps) | < 5 ms | Adreno 730 |
| Pose estimation | < 3 ms | CPU |
| **Total pipeline** | **< 30 ms (30+ FPS)** | |

## Building

### Prerequisites
- Android Studio Hedgehog+
- Android NDK (installed via SDK Manager)
- CMake 3.22+ (installed via SDK Manager)
- Android device with `arm64-v8a` (Vulkan 1.1+ recommended)

### Build Steps
```bash
git clone https://github.com/artem-shamsuarov/onyx-vo.git
cd onyx-vo
./gradlew assembleDebug
```

Or open in Android Studio and run on a physical device.

## Project Structure

```
OnyxVO/
├── app/
│   ├── build.gradle.kts          # minSdk 28, targetSdk 34, arm64-v8a, CameraX
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/onyxvo/app/
│       │   ├── MainActivity.kt   # Permission handling, debug overlay
│       │   ├── CameraManager.kt  # CameraX Preview + ImageAnalysis
│       │   └── NativeBridge.kt   # JNI wrapper
│       ├── cpp/
│       │   ├── CMakeLists.txt    # C++17 native build
│       │   ├── jni_bridge.cpp    # Native entry point
│       │   └── utils/
│       │       └── android_log.h # Logging macros
│       └── res/
├── build.gradle.kts              # AGP 8.2.2, Kotlin 1.9.22
└── settings.gradle.kts
```

## Dependencies

| Library | Version | Role |
|---|---|---|
| CameraX | 1.3.1 | Android camera API |
| [Kompute](https://github.com/KomputeProject/kompute) | TBD | Vulkan compute abstraction (Phase 4) |
| [ONNX Runtime](https://onnxruntime.ai/) | TBD | Neural network inference (Phase 3) |
| [XFeat](https://github.com/verlab/accelerated_features) | CVPR 2024 | Learned feature extraction (Phase 3) |
| [Eigen](https://eigen.tuxfamily.org/) | 3.4+ | Linear algebra (Phase 5) |

## References

- Potje et al., "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024
- Nister, "An Efficient Solution to the Five-Point Relative Pose Problem", IEEE TPAMI 2004
- [Kompute Documentation](https://kompute.cc/)
- [ONNX Runtime Mobile Guide](https://onnxruntime.ai/docs/tutorials/mobile/)

## License

MIT

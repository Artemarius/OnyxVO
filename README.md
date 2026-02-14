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
│  NEON Preprocessing     │  ARM SIMD: RGB→Gray, resize, normalize
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

## What's Implemented

**Feature Extraction**
- XFeat (CVPR 2024) exported to ONNX, quantized to INT8 for mobile deployment
- FP32 and INT8 models included with runtime comparison benchmarks
- ONNX Runtime C++ API with CPU execution provider (NNAPI optional)

**GPU-Accelerated Matching**
- Kompute (Vulkan compute framework) for descriptor matching on mobile GPU
- Custom GLSL compute shaders for brute-force L2 nearest neighbor search
- Lowe's ratio test filtering implemented in shader

**Image Preprocessing (NEON)**
- ARM NEON SIMD intrinsics for RGB-to-grayscale conversion
- Vectorized image resize and float normalization
- Benchmarked against scalar baseline

**Visual Odometry Pipeline**
- 5-point algorithm with RANSAC for essential matrix estimation
- Cheirality check for correct pose recovery
- Cumulative pose tracking with scale estimation
- Real-time trajectory visualization overlaid on camera preview

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
- Android NDK r26+
- CMake 3.22+
- Android device with Vulkan 1.1+ support

### Build Steps
```bash
# Clone with submodules
git clone --recursive https://github.com/artem-shamsuarov/onyx-vo.git

# Open in Android Studio → Build → Run
# Or from command line:
cd onyx-vo
./gradlew assembleDebug
```

### Model Preparation (one-time)
```bash
# Export and quantize XFeat model
cd scripts/
python export_xfeat_onnx.py --output ../app/src/main/assets/xfeat_fp32.onnx
python quantize_xfeat.py --input ../app/src/main/assets/xfeat_fp32.onnx \
                         --output ../app/src/main/assets/xfeat_int8.onnx
```

## Project Structure

```
onyx-vo/
├── app/
│   ├── src/main/
│   │   ├── java/com/onyxvo/         # Kotlin UI layer
│   │   │   ├── MainActivity.kt
│   │   │   ├── CameraManager.kt
│   │   │   └── TrajectoryView.kt
│   │   ├── cpp/                      # Native C++ core
│   │   │   ├── CMakeLists.txt
│   │   │   ├── jni_bridge.cpp
│   │   │   ├── preprocessing/        # NEON image ops
│   │   │   ├── feature/              # ONNX Runtime XFeat wrapper
│   │   │   ├── matching/             # Kompute GPU matching
│   │   │   ├── vo/                   # Pose estimation & tracking
│   │   │   └── utils/                # Timing, logging
│   │   ├── assets/                   # ONNX models
│   │   └── res/
│   └── build.gradle.kts
├── shaders/                          # GLSL compute shaders
│   └── match_descriptors.comp
├── scripts/                          # Model export & quantization
├── third_party/                      # Kompute, Eigen
└── build.gradle.kts
```

## Dependencies

| Library | Version | Role |
|---|---|---|
| [Kompute](https://github.com/KomputeProject/kompute) | latest | Vulkan compute abstraction |
| [ONNX Runtime](https://onnxruntime.ai/) | 1.17+ | Neural network inference |
| [XFeat](https://github.com/verlab/accelerated_features) | CVPR 2024 | Learned feature extraction |
| [Eigen](https://eigen.tuxfamily.org/) | 3.4+ | Linear algebra (pose estimation) |
| CameraX | latest | Android camera API |

## References

- Potje et al., "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024
- Nister, "An Efficient Solution to the Five-Point Relative Pose Problem", IEEE TPAMI 2004
- [Kompute Documentation](https://kompute.cc/)
- [ONNX Runtime Mobile Guide](https://onnxruntime.ai/docs/tutorials/mobile/)

## License

MIT

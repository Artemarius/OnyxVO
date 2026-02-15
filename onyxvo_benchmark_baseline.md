## OnyxVO Benchmark Results
- **Device:** SM-G991B
- **Date:** 2026-02-15 16:10
- **Warmup:** 30 frames, **Measurement:** 300 frames per mode

### Pipeline Total (ms)
| Mode | Mean | Median | P95 | Min | Max |
|------|------|--------|-----|-----|-----|
| FP32+GPU | 85.7 | 84.7 | 101.9 | 71.3 | 105.7 |
| FP32+CPU | 87.2 | 81.3 | 110.0 | 66.2 | 150.1 |
| INT8+GPU | 104.5 | 103.0 | 114.7 | 86.1 | 295.9 |
| INT8+CPU | 82.0 | 80.0 | 92.1 | 72.1 | 361.5 |

### Per-Stage Mean (ms)
| Mode | Preprocess | Inference | Matching | Pose | Total |
|------|------------|-----------|----------|------|-------|
| FP32+GPU | 0.2 | 58.8 | 22.6 | 4.0 | 85.7 |
| FP32+CPU | 0.3 | 68.5 | 14.0 | 4.4 | 87.2 |
| INT8+GPU | 0.3 | 60.0 | 38.5 | 5.7 | 104.5 |
| INT8+CPU | 0.4 | 59.4 | 16.9 | 5.3 | 82.0 |

### Feature Counts (avg)
| Mode | Keypoints | Matches | Inliers | Frames |
|------|-----------|---------|---------|--------|
| FP32+GPU | 500 | 309 | 171 | 300 |
| FP32+CPU | 500 | 317 | 174 | 300 |
| INT8+GPU | 500 | 300 | 155 | 300 |
| INT8+CPU | 500 | 299 | 151 | 300 |


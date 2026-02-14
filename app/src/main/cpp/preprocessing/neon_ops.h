#ifndef ONYX_VO_NEON_OPS_H
#define ONYX_VO_NEON_OPS_H

#include <cstdint>

namespace onyx {
namespace preprocessing {

// Fixed output resolution for XFeat model input
static constexpr int kTargetWidth = 640;
static constexpr int kTargetHeight = 480;

// --- Bilinear resize (uint8 -> uint8) ---
// src_stride: row stride in bytes (may differ from src_w if padding exists).

void neon_resize_bilinear(const uint8_t* src, int src_stride,
                          uint8_t* dst,
                          int src_w, int src_h, int dst_w, int dst_h);

void scalar_resize_bilinear(const uint8_t* src, int src_stride,
                            uint8_t* dst,
                            int src_w, int src_h, int dst_w, int dst_h);

// --- Normalize (uint8 -> float32) ---
// out[i] = (src[i] - mean) / std_dev

void neon_normalize(const uint8_t* src, float* dst, int count,
                    float mean, float std_dev);

void scalar_normalize(const uint8_t* src, float* dst, int count,
                      float mean, float std_dev);

// --- Per-stage timing ---
struct PreprocessTiming {
    double resize_us;
    double normalize_us;
    double total_us;
};

// Full preprocessing pipeline:
//   Y-plane (with stride) -> resize to dst_w x dst_h -> normalize to float32
//
// Caller owns:
//   resized: dst_w * dst_h  uint8
//   output:  dst_w * dst_h  float
PreprocessTiming preprocess_frame(
    const uint8_t* y_plane, int src_w, int src_h, int row_stride,
    uint8_t* resized, float* output,
    int dst_w, int dst_h,
    float mean, float std_dev,
    bool use_neon);

} // namespace preprocessing
} // namespace onyx

#endif // ONYX_VO_NEON_OPS_H

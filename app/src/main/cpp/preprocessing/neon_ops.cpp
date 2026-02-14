#include "neon_ops.h"
#include "../utils/timer.h"
#include "../utils/android_log.h"

#include <algorithm>
#include <cstring>
#include <cmath>

#if defined(__aarch64__)
#include <arm_neon.h>
#else
#error "neon_ops.cpp requires ARM64 (aarch64) target — build with arm64-v8a ABI only"
#endif

namespace onyx {
namespace preprocessing {

// ============================================================================
// Scalar bilinear resize — reference implementation
//
// Reads directly from source with stride, no intermediate copy needed.
// ============================================================================

void scalar_resize_bilinear(const uint8_t* src, int src_stride,
                            uint8_t* dst,
                            int src_w, int src_h, int dst_w, int dst_h) {
    if (src_w == dst_w && src_h == dst_h && src_stride == src_w) {
        std::memcpy(dst, src, static_cast<size_t>(dst_w) * dst_h);
        return;
    }

    const float x_scale = (dst_w > 1) ? static_cast<float>(src_w - 1) / (dst_w - 1) : 0.0f;
    const float y_scale = (dst_h > 1) ? static_cast<float>(src_h - 1) / (dst_h - 1) : 0.0f;

    for (int oy = 0; oy < dst_h; ++oy) {
        const float fy = oy * y_scale;
        const int y0 = static_cast<int>(fy);
        const int y1 = std::min(y0 + 1, src_h - 1);
        const float wy = fy - y0;
        const float inv_wy = 1.0f - wy;

        const uint8_t* row0 = src + y0 * src_stride;
        const uint8_t* row1 = src + y1 * src_stride;
        uint8_t* out_row = dst + oy * dst_w;

        for (int ox = 0; ox < dst_w; ++ox) {
            const float fx = ox * x_scale;
            const int x0 = static_cast<int>(fx);
            const int x1 = std::min(x0 + 1, src_w - 1);
            const float wx = fx - x0;

            const float top = row0[x0] + (row0[x1] - row0[x0]) * wx;
            const float bot = row1[x0] + (row1[x1] - row1[x0]) * wx;
            out_row[ox] = static_cast<uint8_t>(top + (bot - top) * wy + 0.5f);
        }
    }
}

// ============================================================================
// NEON bilinear resize — direct integer fixed-point
//
// Same direct bilinear algorithm as scalar but using 14-bit integer
// fixed-point arithmetic. Avoids float-to-int conversions per pixel.
// Source rows are accessed with stride — no intermediate Y-plane copy.
//
// Fixed-point scheme:
//   Horizontal weights: 7-bit [0..128], products max 255*128 = 32640 (int16)
//   Vertical: (top*inv_wy + bot*wy) max 32640*128 = 4,177,920 (int32)
//   Final: >> 14 recovers [0..255]
// ============================================================================

void neon_resize_bilinear(const uint8_t* src, int src_stride,
                          uint8_t* dst,
                          int src_w, int src_h, int dst_w, int dst_h) {
    if (src_w == dst_w && src_h == dst_h && src_stride == src_w) {
        std::memcpy(dst, src, static_cast<size_t>(dst_w) * dst_h);
        return;
    }

    const float x_scale = (dst_w > 1) ? static_cast<float>(src_w - 1) / (dst_w - 1) : 0.0f;
    const float y_scale = (dst_h > 1) ? static_cast<float>(src_h - 1) / (dst_h - 1) : 0.0f;

    for (int oy = 0; oy < dst_h; ++oy) {
        const float fy = oy * y_scale;
        const int y0 = static_cast<int>(fy);
        const int y1 = std::min(y0 + 1, src_h - 1);
        const int wy = static_cast<int>((fy - y0) * 128.0f);
        const int inv_wy = 128 - wy;

        const uint8_t* row0 = src + y0 * src_stride;
        const uint8_t* row1 = src + y1 * src_stride;
        uint8_t* out_row = dst + oy * dst_w;

        for (int ox = 0; ox < dst_w; ++ox) {
            const float fx = ox * x_scale;
            const int x0 = static_cast<int>(fx);
            const int x1 = std::min(x0 + 1, src_w - 1);
            const int wx = static_cast<int>((fx - x0) * 128.0f);
            const int inv_wx = 128 - wx;

            const int top = row0[x0] * inv_wx + row0[x1] * wx;
            const int bot = row1[x0] * inv_wx + row1[x1] * wx;
            out_row[ox] = static_cast<uint8_t>((top * inv_wy + bot * wy + 8192) >> 14);
        }
    }
}

// ============================================================================
// Scalar normalize — reference implementation
// ============================================================================

void scalar_normalize(const uint8_t* src, float* dst, int count,
                      float mean, float std_dev) {
    const float inv_std = 1.0f / std_dev;
    for (int i = 0; i < count; ++i) {
        dst[i] = (static_cast<float>(src[i]) - mean) * inv_std;
    }
}

// ============================================================================
// NEON normalize — 16 uint8 pixels per iteration
// ============================================================================

void neon_normalize(const uint8_t* src, float* dst, int count,
                    float mean, float std_dev) {
    const float inv_std = 1.0f / std_dev;
    const float32x4_t mean_v    = vdupq_n_f32(mean);
    const float32x4_t inv_std_v = vdupq_n_f32(inv_std);

    int i = 0;
    for (; i + 15 < count; i += 16) {
        uint8x16_t pixels = vld1q_u8(src + i);

        uint16x8_t lo16 = vmovl_u8(vget_low_u8(pixels));
        uint16x8_t hi16 = vmovl_u8(vget_high_u8(pixels));

        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
        float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
        float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

        f0 = vmulq_f32(vsubq_f32(f0, mean_v), inv_std_v);
        f1 = vmulq_f32(vsubq_f32(f1, mean_v), inv_std_v);
        f2 = vmulq_f32(vsubq_f32(f2, mean_v), inv_std_v);
        f3 = vmulq_f32(vsubq_f32(f3, mean_v), inv_std_v);

        vst1q_f32(dst + i,      f0);
        vst1q_f32(dst + i + 4,  f1);
        vst1q_f32(dst + i + 8,  f2);
        vst1q_f32(dst + i + 12, f3);
    }

    for (; i < count; ++i) {
        dst[i] = (static_cast<float>(src[i]) - mean) * inv_std;
    }
}

// ============================================================================
// Full preprocessing pipeline
// ============================================================================

PreprocessTiming preprocess_frame(
    const uint8_t* y_plane, int src_w, int src_h, int row_stride,
    uint8_t* resized, float* output,
    int dst_w, int dst_h,
    float mean, float std_dev,
    bool use_neon) {

    PreprocessTiming timing = {};

    {
        ScopedTimer timer_total(timing.total_us);

        // Step 1: Bilinear resize (reads Y-plane directly with stride)
        {
            ScopedTimer timer_resize(timing.resize_us);
            if (use_neon) {
                neon_resize_bilinear(y_plane, row_stride, resized,
                                     src_w, src_h, dst_w, dst_h);
            } else {
                scalar_resize_bilinear(y_plane, row_stride, resized,
                                       src_w, src_h, dst_w, dst_h);
            }
        }

        // Step 2: Normalize to float32
        {
            ScopedTimer timer_norm(timing.normalize_us);
            const int pixel_count = dst_w * dst_h;
            if (use_neon) {
                neon_normalize(resized, output, pixel_count, mean, std_dev);
            } else {
                scalar_normalize(resized, output, pixel_count, mean, std_dev);
            }
        }
    }

    return timing;
}

} // namespace preprocessing
} // namespace onyx

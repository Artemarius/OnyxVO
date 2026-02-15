#include "neon_ops.h"
#include "../utils/timer.h"
#include "../utils/android_log.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <memory>

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
// NEON bilinear resize — two-pass with vectorized vertical interpolation
//
// Two-pass approach exploiting the structure of bilinear interpolation:
//   Pass 1 (scalar): Horizontal interpolation with pre-computed x-mapping
//     tables. Non-uniform source spacing (downscale) makes NEON gather
//     impractical, but table lookups avoid redundant float->int per row.
//   Pass 2 (NEON): Vertical interpolation, 8 pixels at a time. Uniform
//     weights across the row make this a clean vectorization target.
//
// Fixed-point scheme (14-bit):
//   Horizontal weights: 7-bit [0..128], products max 255*128 = 32640 (int16)
//   Vertical: (top*inv_wy + bot*wy) max 32640*128 = 4,177,920 (int32)
//   Final: (result + 8192) >> 14 recovers [0..255]
//
// NEON intrinsics used: vmull_s16, vmlal_s16 (widening multiply-accumulate),
//   vshrq_n_s32 (shift), vmovn_s32 + vqmovun_s16 (saturating narrow),
//   vld1q_s16 / vst1_u8 (contiguous load/store).
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

    // Pre-compute per-output-column x-mapping tables (constant across all rows).
    // Max dst_w is 640 for our pipeline; use fixed-size arrays to avoid allocation.
    // For safety, support up to 2048 output width with static buffers.
    static constexpr int kMaxDstW = 2048;
    // If dst_w exceeds static buffer, fall back to dynamic allocation
    int16_t x0_static[kMaxDstW];
    int16_t x1_static[kMaxDstW];
    int16_t wx_static[kMaxDstW];
    int16_t inv_wx_static[kMaxDstW];

    int16_t* x0_tab = x0_static;
    int16_t* x1_tab = x1_static;
    int16_t* wx_tab = wx_static;
    int16_t* inv_wx_tab = inv_wx_static;

    // Dynamic fallback for unusually large output widths
    std::unique_ptr<int16_t[]> x0_dyn, x1_dyn, wx_dyn, inv_wx_dyn;
    if (dst_w > kMaxDstW) {
        x0_dyn.reset(new int16_t[dst_w]);
        x1_dyn.reset(new int16_t[dst_w]);
        wx_dyn.reset(new int16_t[dst_w]);
        inv_wx_dyn.reset(new int16_t[dst_w]);
        x0_tab = x0_dyn.get();
        x1_tab = x1_dyn.get();
        wx_tab = wx_dyn.get();
        inv_wx_tab = inv_wx_dyn.get();
    }

    for (int ox = 0; ox < dst_w; ++ox) {
        const float fx = ox * x_scale;
        const int ix = static_cast<int>(fx);
        x0_tab[ox] = static_cast<int16_t>(ix);
        x1_tab[ox] = static_cast<int16_t>(std::min(ix + 1, src_w - 1));
        wx_tab[ox] = static_cast<int16_t>((fx - ix) * 128.0f);
        inv_wx_tab[ox] = static_cast<int16_t>(128 - wx_tab[ox]);
    }

    // Temporary buffers for horizontal interpolation results (int16 range: 0..32640).
    // These hold one row's worth of horizontally-interpolated values for top and bottom.
    int16_t top_static[kMaxDstW];
    int16_t bot_static[kMaxDstW];
    int16_t* top_buf = top_static;
    int16_t* bot_buf = bot_static;
    std::unique_ptr<int16_t[]> top_dyn, bot_dyn;
    if (dst_w > kMaxDstW) {
        top_dyn.reset(new int16_t[dst_w]);
        bot_dyn.reset(new int16_t[dst_w]);
        top_buf = top_dyn.get();
        bot_buf = bot_dyn.get();
    }

    for (int oy = 0; oy < dst_h; ++oy) {
        const float fy = oy * y_scale;
        const int y0 = static_cast<int>(fy);
        const int y1 = std::min(y0 + 1, src_h - 1);
        const int16_t wy = static_cast<int16_t>((fy - y0) * 128.0f);
        const int16_t inv_wy = static_cast<int16_t>(128 - wy);

        const uint8_t* row0 = src + y0 * src_stride;
        const uint8_t* row1 = src + y1 * src_stride;
        uint8_t* out_row = dst + oy * dst_w;

        // Pass 1: Scalar horizontal interpolation (non-uniform x spacing,
        // no efficient NEON gather on ARM). Store int16 results in temp buffers.
        // Max value: 255 * 128 = 32640, fits in int16_t (max 32767).
        for (int ox = 0; ox < dst_w; ++ox) {
            const int sx0 = x0_tab[ox];
            const int sx1 = x1_tab[ox];
            const int iwx = inv_wx_tab[ox];
            const int wxv = wx_tab[ox];
            top_buf[ox] = static_cast<int16_t>(row0[sx0] * iwx + row0[sx1] * wxv);
            bot_buf[ox] = static_cast<int16_t>(row1[sx0] * iwx + row1[sx1] * wxv);
        }

        // Pass 2: NEON vertical interpolation, 8 output pixels at a time.
        // Computes: result = (top * inv_wy + bot * wy + 8192) >> 14
        // Product range: 32640 * 128 = 4,177,920, fits in int32.
        const int16x8_t wy_v = vdupq_n_s16(wy);
        const int16x8_t inv_wy_v = vdupq_n_s16(inv_wy);
        const int32x4_t round_v = vdupq_n_s32(8192);

        int ox = 0;
        for (; ox + 7 < dst_w; ox += 8) {
            // Load 8 horizontally-interpolated int16 values for top and bottom rows
            const int16x8_t top_v = vld1q_s16(top_buf + ox);
            const int16x8_t bot_v = vld1q_s16(bot_buf + ox);

            // Vertical blend: top * inv_wy + bot * wy (widened to int32)
            // Low 4 elements
            int32x4_t res_lo = vmull_s16(vget_low_s16(top_v), vget_low_s16(inv_wy_v));
            res_lo = vmlal_s16(res_lo, vget_low_s16(bot_v), vget_low_s16(wy_v));
            // High 4 elements
            int32x4_t res_hi = vmull_s16(vget_high_s16(top_v), vget_high_s16(inv_wy_v));
            res_hi = vmlal_s16(res_hi, vget_high_s16(bot_v), vget_high_s16(wy_v));

            // Add rounding bias and shift right by 14
            res_lo = vshrq_n_s32(vaddq_s32(res_lo, round_v), 14);
            res_hi = vshrq_n_s32(vaddq_s32(res_hi, round_v), 14);

            // Narrow int32 -> int16 -> uint8 (saturating for safety)
            const int16x4_t narrow_lo = vmovn_s32(res_lo);
            const int16x4_t narrow_hi = vmovn_s32(res_hi);
            const int16x8_t narrow = vcombine_s16(narrow_lo, narrow_hi);
            const uint8x8_t result = vqmovun_s16(narrow);

            vst1_u8(out_row + ox, result);
        }

        // Scalar tail for remaining pixels (when dst_w is not a multiple of 8)
        for (; ox < dst_w; ++ox) {
            out_row[ox] = static_cast<uint8_t>(
                (top_buf[ox] * inv_wy + bot_buf[ox] * wy + 8192) >> 14);
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

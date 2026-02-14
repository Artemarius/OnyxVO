#include <jni.h>
#include <string>
#include <memory>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "utils/android_log.h"
#include "preprocessing/neon_ops.h"

#define ONYX_VO_VERSION "0.2.0-phase2"

namespace {

// Pre-allocated buffers for frame preprocessing — created once, reused every frame.
struct PreprocessState {
    std::unique_ptr<uint8_t[]> resized;  // dst_w * dst_h
    std::unique_ptr<float[]>   output;   // dst_w * dst_h
    int dst_w = 0;
    int dst_h = 0;
    bool initialized = false;

    void ensure_init(int dw, int dh) {
        if (initialized && dw == dst_w && dh == dst_h) return;
        dst_w = dw;
        dst_h = dh;
        resized.reset(new uint8_t[dw * dh]);
        output.reset(new float[dw * dh]);
        initialized = true;
        LOGI("PreprocessState init: output %dx%d", dw, dh);
    }
};

PreprocessState g_preprocess;

// Default normalization: maps [0, 255] -> [0.0, 1.0]
constexpr float kDefaultMean = 0.0f;
constexpr float kDefaultStd  = 255.0f;

} // anonymous namespace

extern "C" {

JNIEXPORT void JNICALL
Java_com_onyxvo_app_NativeBridge_nativeInit(JNIEnv* /* env */, jobject /* thiz */) {
    LOGI("OnyxVO native init — version %s", ONYX_VO_VERSION);
}

JNIEXPORT jstring JNICALL
Java_com_onyxvo_app_NativeBridge_nativeGetVersion(JNIEnv* env, jobject /* thiz */) {
    return env->NewStringUTF(ONYX_VO_VERSION);
}

// ---------------------------------------------------------------------------
// Phase 2: Frame preprocessing
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativePreprocessFrame(
    JNIEnv* env, jobject /* thiz */,
    jobject y_plane_buffer,
    jint width, jint height, jint row_stride,
    jboolean use_neon) {

    using namespace onyx::preprocessing;

    auto* y_plane = static_cast<const uint8_t*>(
        env->GetDirectBufferAddress(y_plane_buffer));
    if (!y_plane) {
        LOGE("nativePreprocessFrame: GetDirectBufferAddress returned null");
        return nullptr;
    }

    g_preprocess.ensure_init(kTargetWidth, kTargetHeight);

    auto timing = preprocess_frame(
        y_plane, width, height, row_stride,
        g_preprocess.resized.get(),
        g_preprocess.output.get(),
        kTargetWidth, kTargetHeight,
        kDefaultMean, kDefaultStd,
        use_neon);

    // Return [resize_us, normalize_us, total_us]
    jfloatArray result = env->NewFloatArray(3);
    if (result) {
        float data[3] = {
            static_cast<float>(timing.resize_us),
            static_cast<float>(timing.normalize_us),
            static_cast<float>(timing.total_us)
        };
        env->SetFloatArrayRegion(result, 0, 3, data);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Phase 2: A/B benchmark (NEON vs scalar)
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeBenchmarkPreprocessing(
    JNIEnv* env, jobject /* thiz */,
    jobject y_plane_buffer,
    jint width, jint height, jint row_stride,
    jint iterations) {

    using namespace onyx::preprocessing;

    auto* y_plane = static_cast<const uint8_t*>(
        env->GetDirectBufferAddress(y_plane_buffer));
    if (!y_plane) {
        LOGE("nativeBenchmarkPreprocessing: GetDirectBufferAddress returned null");
        return nullptr;
    }

    g_preprocess.ensure_init(kTargetWidth, kTargetHeight);

    double neon_total = 0.0, scalar_total = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto t = preprocess_frame(
            y_plane, width, height, row_stride,
            g_preprocess.resized.get(), g_preprocess.output.get(),
            kTargetWidth, kTargetHeight, kDefaultMean, kDefaultStd, true);
        neon_total += t.total_us;
    }

    for (int i = 0; i < iterations; ++i) {
        auto t = preprocess_frame(
            y_plane, width, height, row_stride,
            g_preprocess.resized.get(), g_preprocess.output.get(),
            kTargetWidth, kTargetHeight, kDefaultMean, kDefaultStd, false);
        scalar_total += t.total_us;
    }

    float neon_avg   = static_cast<float>(neon_total / iterations);
    float scalar_avg = static_cast<float>(scalar_total / iterations);
    float speedup    = (neon_avg > 0.0f) ? scalar_avg / neon_avg : 0.0f;

    LOGI("Benchmark (%d iters): NEON=%.0f us, Scalar=%.0f us, Speedup=%.2fx",
         iterations, neon_avg, scalar_avg, speedup);

    jfloatArray result = env->NewFloatArray(3);
    if (result) {
        float data[3] = { neon_avg, scalar_avg, speedup };
        env->SetFloatArrayRegion(result, 0, 3, data);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Phase 2: NEON vs scalar validation
//
// Generates synthetic test images (no camera needed), runs both paths,
// compares outputs. Returns float[4]:
//   [0] resize_max_err    — max |neon - scalar| over uint8 resize outputs
//   [1] normalize_max_err — max |neon - scalar| over float normalize outputs
//                           (same uint8 input to isolate normalize precision)
//   [2] pipeline_max_err  — max |neon - scalar| over full pipeline float outputs
//   [3] all_passed        — 1.0 if all thresholds met, 0.0 otherwise
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeValidatePreprocessing(
    JNIEnv* env, jobject /* thiz */) {

    using namespace onyx::preprocessing;

    // Test configurations: {src_w, src_h, label}
    struct TestCase { int w; int h; const char* label; };
    const TestCase cases[] = {
        {1920, 1080, "1080p"},
        {1280,  720, "720p"},
        { 640,  480, "identity"},
        { 800,  600, "800x600"},
    };

    float worst_resize_err    = 0.0f;
    float worst_normalize_err = 0.0f;
    float worst_pipeline_err  = 0.0f;

    for (const auto& tc : cases) {
        const int src_size = tc.w * tc.h;
        const int dst_size = kTargetWidth * kTargetHeight;

        // Allocate buffers
        auto src = std::make_unique<uint8_t[]>(src_size);
        auto resize_neon   = std::make_unique<uint8_t[]>(dst_size);
        auto resize_scalar = std::make_unique<uint8_t[]>(dst_size);
        auto norm_neon     = std::make_unique<float[]>(dst_size);
        auto norm_scalar   = std::make_unique<float[]>(dst_size);
        auto pipe_neon     = std::make_unique<float[]>(dst_size);
        auto pipe_scalar   = std::make_unique<float[]>(dst_size);
        auto resize_shared = std::make_unique<uint8_t[]>(dst_size);

        // --- Pattern 1: Horizontal gradient ---
        for (int y = 0; y < tc.h; ++y)
            for (int x = 0; x < tc.w; ++x)
                src[y * tc.w + x] = static_cast<uint8_t>((x * 255) / std::max(tc.w - 1, 1));

        // Test resize
        neon_resize_bilinear(src.get(), tc.w, resize_neon.get(),
                             tc.w, tc.h, kTargetWidth, kTargetHeight);
        scalar_resize_bilinear(src.get(), tc.w, resize_scalar.get(),
                               tc.w, tc.h, kTargetWidth, kTargetHeight);

        int resize_err = 0;
        for (int i = 0; i < dst_size; ++i)
            resize_err = std::max(resize_err,
                std::abs(static_cast<int>(resize_neon[i]) - static_cast<int>(resize_scalar[i])));
        worst_resize_err = std::max(worst_resize_err, static_cast<float>(resize_err));

        // Test normalize (same input to isolate float precision)
        scalar_resize_bilinear(src.get(), tc.w, resize_shared.get(),
                               tc.w, tc.h, kTargetWidth, kTargetHeight);
        neon_normalize(resize_shared.get(), norm_neon.get(), dst_size,
                       kDefaultMean, kDefaultStd);
        scalar_normalize(resize_shared.get(), norm_scalar.get(), dst_size,
                         kDefaultMean, kDefaultStd);

        float norm_err = 0.0f;
        for (int i = 0; i < dst_size; ++i)
            norm_err = std::max(norm_err, std::abs(norm_neon[i] - norm_scalar[i]));
        worst_normalize_err = std::max(worst_normalize_err, norm_err);

        // Test full pipeline
        preprocess_frame(src.get(), tc.w, tc.h, tc.w,
                         resize_neon.get(), pipe_neon.get(),
                         kTargetWidth, kTargetHeight, kDefaultMean, kDefaultStd, true);
        preprocess_frame(src.get(), tc.w, tc.h, tc.w,
                         resize_scalar.get(), pipe_scalar.get(),
                         kTargetWidth, kTargetHeight, kDefaultMean, kDefaultStd, false);

        float pipe_err = 0.0f;
        for (int i = 0; i < dst_size; ++i)
            pipe_err = std::max(pipe_err, std::abs(pipe_neon[i] - pipe_scalar[i]));
        worst_pipeline_err = std::max(worst_pipeline_err, pipe_err);

        LOGI("Validate [%s gradient]: resize_err=%d norm_err=%.2e pipe_err=%.6f",
             tc.label, resize_err, norm_err, pipe_err);

        // --- Pattern 2: Uniform (min/max) ---
        for (uint8_t val : {uint8_t(0), uint8_t(128), uint8_t(255)}) {
            std::memset(src.get(), val, src_size);

            neon_resize_bilinear(src.get(), tc.w, resize_neon.get(),
                                 tc.w, tc.h, kTargetWidth, kTargetHeight);
            scalar_resize_bilinear(src.get(), tc.w, resize_scalar.get(),
                                   tc.w, tc.h, kTargetWidth, kTargetHeight);

            int err = 0;
            for (int i = 0; i < dst_size; ++i)
                err = std::max(err,
                    std::abs(static_cast<int>(resize_neon[i]) - static_cast<int>(resize_scalar[i])));
            worst_resize_err = std::max(worst_resize_err, static_cast<float>(err));

            LOGI("Validate [%s uniform=%d]: resize_err=%d", tc.label, val, err);
        }

        // --- Pattern 3: Checkerboard ---
        for (int y = 0; y < tc.h; ++y)
            for (int x = 0; x < tc.w; ++x)
                src[y * tc.w + x] = ((x + y) & 1) ? 255 : 0;

        neon_resize_bilinear(src.get(), tc.w, resize_neon.get(),
                             tc.w, tc.h, kTargetWidth, kTargetHeight);
        scalar_resize_bilinear(src.get(), tc.w, resize_scalar.get(),
                               tc.w, tc.h, kTargetWidth, kTargetHeight);

        resize_err = 0;
        for (int i = 0; i < dst_size; ++i)
            resize_err = std::max(resize_err,
                std::abs(static_cast<int>(resize_neon[i]) - static_cast<int>(resize_scalar[i])));
        worst_resize_err = std::max(worst_resize_err, static_cast<float>(resize_err));

        LOGI("Validate [%s checker]: resize_err=%d", tc.label, resize_err);
    }

    // --- Normalize tail test: count not divisible by 16 ---
    for (int count : {1, 7, 15, 16, 17, 31, 33, 100}) {
        auto tail_src = std::make_unique<uint8_t[]>(count);
        auto tail_neon   = std::make_unique<float[]>(count);
        auto tail_scalar = std::make_unique<float[]>(count);

        for (int i = 0; i < count; ++i)
            tail_src[i] = static_cast<uint8_t>(i % 256);

        neon_normalize(tail_src.get(), tail_neon.get(), count, kDefaultMean, kDefaultStd);
        scalar_normalize(tail_src.get(), tail_scalar.get(), count, kDefaultMean, kDefaultStd);

        float err = 0.0f;
        for (int i = 0; i < count; ++i)
            err = std::max(err, std::abs(tail_neon[i] - tail_scalar[i]));
        worst_normalize_err = std::max(worst_normalize_err, err);

        LOGI("Validate [normalize tail=%d]: err=%.2e", count, err);
    }

    // Thresholds
    const bool resize_ok    = worst_resize_err <= 1.0f;    // ±1 uint8 from fixed-point rounding
    const bool normalize_ok = worst_normalize_err < 1e-5f; // float precision only
    const bool pipeline_ok  = worst_pipeline_err < 0.005f; // resize rounding propagated: ≤ 1/255 ≈ 0.004
    const bool all_passed   = resize_ok && normalize_ok && pipeline_ok;

    LOGI("Validation result: resize=%.0f (max 1) %s, normalize=%.2e (max 1e-5) %s, "
         "pipeline=%.6f (max 0.005) %s → %s",
         worst_resize_err, resize_ok ? "PASS" : "FAIL",
         worst_normalize_err, normalize_ok ? "PASS" : "FAIL",
         worst_pipeline_err, pipeline_ok ? "PASS" : "FAIL",
         all_passed ? "ALL PASS" : "FAIL");

    jfloatArray result = env->NewFloatArray(4);
    if (result) {
        float data[4] = {
            worst_resize_err,
            worst_normalize_err,
            worst_pipeline_err,
            all_passed ? 1.0f : 0.0f
        };
        env->SetFloatArrayRegion(result, 0, 4, data);
    }
    return result;
}

} // extern "C"

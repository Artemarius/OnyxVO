#include <jni.h>
#include <string>
#include <memory>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <android/asset_manager_jni.h>
#include "utils/android_log.h"
#include "preprocessing/neon_ops.h"
#include "feature/xfeat_extractor.h"

#define ONYX_VO_VERSION "0.3.0-phase3"

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

// Phase 3: XFeat extractor
std::unique_ptr<onyx::feature::XFeatExtractor> g_extractor;

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
// Phase 2: Frame preprocessing (kept for backward compatibility)
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
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeValidatePreprocessing(
    JNIEnv* env, jobject /* thiz */) {

    using namespace onyx::preprocessing;

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

        neon_resize_bilinear(src.get(), tc.w, resize_neon.get(),
                             tc.w, tc.h, kTargetWidth, kTargetHeight);
        scalar_resize_bilinear(src.get(), tc.w, resize_scalar.get(),
                               tc.w, tc.h, kTargetWidth, kTargetHeight);

        int resize_err = 0;
        for (int i = 0; i < dst_size; ++i)
            resize_err = std::max(resize_err,
                std::abs(static_cast<int>(resize_neon[i]) - static_cast<int>(resize_scalar[i])));
        worst_resize_err = std::max(worst_resize_err, static_cast<float>(resize_err));

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

        // --- Pattern 2: Uniform ---
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

    // --- Normalize tail test ---
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

    const bool resize_ok    = worst_resize_err <= 1.0f;
    const bool normalize_ok = worst_normalize_err < 1e-5f;
    const bool pipeline_ok  = worst_pipeline_err < 0.005f;
    const bool all_passed   = resize_ok && normalize_ok && pipeline_ok;

    LOGI("Validation result: resize=%.0f (max 1) %s, normalize=%.2e (max 1e-5) %s, "
         "pipeline=%.6f (max 0.005) %s -> %s",
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

// ---------------------------------------------------------------------------
// Phase 3: Model initialization
// ---------------------------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_onyxvo_app_NativeBridge_nativeInitModel(
    JNIEnv* env, jobject /* thiz */,
    jobject asset_manager, jboolean use_int8) {

    auto* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        LOGE("nativeInitModel: AAssetManager_fromJava returned null");
        return JNI_FALSE;
    }

    auto model_type = use_int8
        ? onyx::feature::XFeatExtractor::ModelType::INT8
        : onyx::feature::XFeatExtractor::ModelType::FP32;

    try {
        g_extractor = std::make_unique<onyx::feature::XFeatExtractor>(mgr, model_type);
        LOGI("Model initialized: %s", g_extractor->modelName());
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("nativeInitModel failed: %s", e.what());
        g_extractor.reset();
        return JNI_FALSE;
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Process frame (preprocess + extract features)
//
// Returns FloatArray:
//   [0] preprocess_us
//   [1] inference_us
//   [2] keypoint_count
//   [3..3+2*N-1] keypoint coordinates (x0, y0, x1, y1, ...)
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeProcessFrame(
    JNIEnv* env, jobject /* thiz */,
    jobject y_plane_buffer,
    jint width, jint height, jint row_stride,
    jboolean use_neon) {

    using namespace onyx::preprocessing;

    auto* y_plane = static_cast<const uint8_t*>(
        env->GetDirectBufferAddress(y_plane_buffer));
    if (!y_plane) {
        LOGE("nativeProcessFrame: GetDirectBufferAddress returned null");
        return nullptr;
    }
    if (!g_extractor) {
        LOGE("nativeProcessFrame: model not initialized");
        return nullptr;
    }

    g_preprocess.ensure_init(kTargetWidth, kTargetHeight);

    // Step 1: Preprocess (resize + normalize)
    auto timing = preprocess_frame(
        y_plane, width, height, row_stride,
        g_preprocess.resized.get(),
        g_preprocess.output.get(),
        kTargetWidth, kTargetHeight,
        kDefaultMean, kDefaultStd,
        use_neon);

    // Step 2: Feature extraction
    auto features = g_extractor->extract(
        g_preprocess.output.get(), kTargetWidth, kTargetHeight);

    // Pack result: [preprocess_us, inference_us, kp_count, x0, y0, x1, y1, ...]
    int n = features.count;
    int result_size = 3 + n * 2;
    jfloatArray result = env->NewFloatArray(result_size);
    if (!result) return nullptr;

    auto data = std::make_unique<float[]>(result_size);
    data[0] = static_cast<float>(timing.total_us);
    data[1] = static_cast<float>(features.inference_us);
    data[2] = static_cast<float>(n);

    for (int i = 0; i < n; ++i) {
        data[3 + i * 2]     = features.keypoints[i].x();
        data[3 + i * 2 + 1] = features.keypoints[i].y();
    }

    env->SetFloatArrayRegion(result, 0, result_size, data.get());
    return result;
}

// ---------------------------------------------------------------------------
// Phase 3: Switch model (FP32 / INT8)
// ---------------------------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_onyxvo_app_NativeBridge_nativeSwitchModel(
    JNIEnv* env, jobject /* thiz */,
    jobject asset_manager, jboolean use_int8) {

    if (!g_extractor) {
        LOGE("nativeSwitchModel: model not initialized");
        return JNI_FALSE;
    }

    auto* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        LOGE("nativeSwitchModel: AAssetManager_fromJava returned null");
        return JNI_FALSE;
    }

    auto model_type = use_int8
        ? onyx::feature::XFeatExtractor::ModelType::INT8
        : onyx::feature::XFeatExtractor::ModelType::FP32;

    try {
        g_extractor->switchModel(mgr, model_type);
        LOGI("Switched to model: %s", g_extractor->modelName());
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("nativeSwitchModel failed: %s", e.what());
        return JNI_FALSE;
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Inference benchmark (FP32 vs INT8)
//
// Returns FloatArray:
//   [0] fp32_avg_us  — average FP32 inference time
//   [1] int8_avg_us  — average INT8 inference time
//   [2] speedup      — fp32/int8 ratio
//   [3] fp32_kp_count — keypoints from FP32
//   [4] int8_kp_count — keypoints from INT8
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeBenchmarkInference(
    JNIEnv* env, jobject /* thiz */,
    jobject asset_manager, jint iterations) {

    auto* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        LOGE("nativeBenchmarkInference: AAssetManager_fromJava returned null");
        return nullptr;
    }

    using namespace onyx::preprocessing;
    using namespace onyx::feature;

    // Generate a synthetic test image (gradient)
    const int w = kTargetWidth;
    const int h = kTargetHeight;
    const int pixels = w * h;
    auto test_image = std::make_unique<float[]>(pixels);
    for (int i = 0; i < pixels; ++i) {
        test_image[i] = static_cast<float>(i % w) / static_cast<float>(w);
    }

    // Benchmark FP32
    double fp32_total = 0.0;
    int fp32_kp = 0;
    {
        XFeatExtractor extractor(mgr, XFeatExtractor::ModelType::FP32);
        // Warmup
        extractor.extract(test_image.get(), w, h);

        for (int i = 0; i < iterations; ++i) {
            auto r = extractor.extract(test_image.get(), w, h);
            fp32_total += r.inference_us;
            if (i == 0) fp32_kp = r.count;
        }
    }

    // Benchmark INT8
    double int8_total = 0.0;
    int int8_kp = 0;
    {
        XFeatExtractor extractor(mgr, XFeatExtractor::ModelType::INT8);
        // Warmup
        extractor.extract(test_image.get(), w, h);

        for (int i = 0; i < iterations; ++i) {
            auto r = extractor.extract(test_image.get(), w, h);
            int8_total += r.inference_us;
            if (i == 0) int8_kp = r.count;
        }
    }

    float fp32_avg = static_cast<float>(fp32_total / iterations);
    float int8_avg = static_cast<float>(int8_total / iterations);
    float speedup = (int8_avg > 0.0f) ? fp32_avg / int8_avg : 0.0f;

    LOGI("Inference benchmark (%d iters): FP32=%.0f us (%d kp), INT8=%.0f us (%d kp), Speedup=%.2fx",
         iterations, fp32_avg, fp32_kp, int8_avg, int8_kp, speedup);

    jfloatArray result = env->NewFloatArray(5);
    if (result) {
        float data[5] = { fp32_avg, int8_avg, speedup,
                          static_cast<float>(fp32_kp), static_cast<float>(int8_kp) };
        env->SetFloatArrayRegion(result, 0, 5, data);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Phase 3: Inference validation (smoke test)
//
// Returns FloatArray:
//   [0] fp32_kp_count
//   [1] int8_kp_count
//   [2] kp_count_diff_pct — percentage difference
//   [3] passed — 1.0 if both produce keypoints and diff < 20%
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeValidateInference(
    JNIEnv* env, jobject /* thiz */,
    jobject asset_manager) {

    auto* mgr = AAssetManager_fromJava(env, asset_manager);
    if (!mgr) {
        LOGE("nativeValidateInference: AAssetManager_fromJava returned null");
        return nullptr;
    }

    using namespace onyx::preprocessing;
    using namespace onyx::feature;

    const int w = kTargetWidth;
    const int h = kTargetHeight;
    const int pixels = w * h;

    // Synthetic gradient image (should produce some keypoints)
    auto test_image = std::make_unique<float[]>(pixels);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Diagonal gradient with some texture
            float gx = static_cast<float>(x) / w;
            float gy = static_cast<float>(y) / h;
            test_image[y * w + x] = (gx + gy) * 0.5f;
        }
    }

    int fp32_kp = 0, int8_kp = 0;

    // FP32
    {
        XFeatExtractor extractor(mgr, XFeatExtractor::ModelType::FP32);
        auto r = extractor.extract(test_image.get(), w, h);
        fp32_kp = r.count;
        LOGI("Validate FP32: %d keypoints, inference=%.0f us", r.count, r.inference_us);
    }

    // INT8
    {
        XFeatExtractor extractor(mgr, XFeatExtractor::ModelType::INT8);
        auto r = extractor.extract(test_image.get(), w, h);
        int8_kp = r.count;
        LOGI("Validate INT8: %d keypoints, inference=%.0f us", r.count, r.inference_us);
    }

    float diff_pct = (fp32_kp > 0)
        ? std::abs(fp32_kp - int8_kp) / static_cast<float>(fp32_kp) * 100.0f
        : 100.0f;

    bool passed = (fp32_kp > 0) && (int8_kp > 0) && (diff_pct < 20.0f);

    LOGI("Validate inference: FP32=%d kp, INT8=%d kp, diff=%.1f%% -> %s",
         fp32_kp, int8_kp, diff_pct, passed ? "PASS" : "FAIL");

    jfloatArray result = env->NewFloatArray(4);
    if (result) {
        float data[4] = {
            static_cast<float>(fp32_kp),
            static_cast<float>(int8_kp),
            diff_pct,
            passed ? 1.0f : 0.0f
        };
        env->SetFloatArrayRegion(result, 0, 4, data);
    }
    return result;
}

} // extern "C"

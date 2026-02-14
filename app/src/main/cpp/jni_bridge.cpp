#include <jni.h>
#include <string>
#include <memory>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <set>
#include <android/asset_manager_jni.h>
#include "utils/android_log.h"
#include "preprocessing/neon_ops.h"
#include "feature/xfeat_extractor.h"
#include "matching/gpu_matcher.h"
#include "matching/cpu_matcher.h"

#define ONYX_VO_VERSION "0.4.0-phase4"

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

// Phase 4: Matching state
struct MatchState {
    std::unique_ptr<onyx::matching::GpuMatcher> gpu_matcher;
    onyx::matching::CpuMatcher cpu_matcher;
    bool use_gpu = true;
    bool initialized = false;

    // Previous frame features for frame-to-frame matching
    onyx::feature::FeatureResult prev_features;
    bool has_prev_frame = false;
};

MatchState g_match;

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
// Phase 4: Matcher initialization
// ---------------------------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_onyxvo_app_NativeBridge_nativeInitMatcher(
    JNIEnv* /* env */, jobject /* thiz */) {

    try {
        g_match.gpu_matcher = std::make_unique<onyx::matching::GpuMatcher>(600);
        g_match.use_gpu = g_match.gpu_matcher->isAvailable();
        g_match.initialized = true;
        g_match.has_prev_frame = false;

        LOGI("Matcher init: GPU=%s", g_match.use_gpu ? "yes" : "no (CPU fallback)");
        return g_match.use_gpu ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGE("nativeInitMatcher failed: %s", e.what());
        g_match.use_gpu = false;
        g_match.initialized = true;  // CPU matcher is always available
        g_match.has_prev_frame = false;
        return JNI_FALSE;
    }
}

// ---------------------------------------------------------------------------
// Phase 3+4: Process frame (preprocess + extract + match)
//
// Returns FloatArray:
//   [0] preprocess_us
//   [1] inference_us
//   [2] matching_us
//   [3] keypoint_count (N)
//   [4] match_count (M)
//   [5..5+2*N-1]         keypoint coordinates (x0, y0, x1, y1, ...)
//   [5+2*N..5+2*N+4*M-1] match lines (prev_x, prev_y, curr_x, curr_y per match)
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

    // Step 3: Matching (if we have a previous frame and matcher is initialized)
    double matching_us = 0.0;
    std::vector<onyx::matching::Match> matches;

    if (g_match.initialized && g_match.has_prev_frame &&
        features.count > 0 && g_match.prev_features.count > 0) {

        if (g_match.use_gpu && g_match.gpu_matcher && g_match.gpu_matcher->isAvailable()) {
            matches = g_match.gpu_matcher->match(
                g_match.prev_features.descriptors, g_match.prev_features.count,
                features.descriptors, features.count,
                0.8f, &matching_us);
        } else {
            matches = g_match.cpu_matcher.match(
                g_match.prev_features.descriptors, g_match.prev_features.count,
                features.descriptors, features.count,
                0.8f, &matching_us);
        }
    }

    int n = features.count;
    int m = static_cast<int>(matches.size());

    // Pack result: [preprocess_us, inference_us, matching_us, kp_count, match_count,
    //               kp_coords..., match_lines...]
    int result_size = 5 + n * 2 + m * 4;
    jfloatArray result = env->NewFloatArray(result_size);
    if (!result) return nullptr;

    auto data = std::make_unique<float[]>(result_size);
    data[0] = static_cast<float>(timing.total_us);
    data[1] = static_cast<float>(features.inference_us);
    data[2] = static_cast<float>(matching_us);
    data[3] = static_cast<float>(n);
    data[4] = static_cast<float>(m);

    // Keypoint coordinates
    for (int i = 0; i < n; ++i) {
        data[5 + i * 2]     = features.keypoints[i].x();
        data[5 + i * 2 + 1] = features.keypoints[i].y();
    }

    // Match lines: (prev_x, prev_y, curr_x, curr_y) per match
    int match_offset = 5 + n * 2;
    for (int i = 0; i < m; ++i) {
        const auto& match = matches[i];
        data[match_offset + i * 4]     = g_match.prev_features.keypoints[match.idx1].x();
        data[match_offset + i * 4 + 1] = g_match.prev_features.keypoints[match.idx1].y();
        data[match_offset + i * 4 + 2] = features.keypoints[match.idx2].x();
        data[match_offset + i * 4 + 3] = features.keypoints[match.idx2].y();
    }

    env->SetFloatArrayRegion(result, 0, result_size, data.get());

    // Cache current features for next frame matching
    g_match.prev_features = std::move(features);
    g_match.has_prev_frame = true;

    return result;
}

// ---------------------------------------------------------------------------
// Phase 4: Toggle GPU/CPU matching
// ---------------------------------------------------------------------------

JNIEXPORT void JNICALL
Java_com_onyxvo_app_NativeBridge_nativeSetMatcherUseGpu(
    JNIEnv* /* env */, jobject /* thiz */, jboolean use_gpu) {

    if (!g_match.initialized) {
        LOGW("nativeSetMatcherUseGpu: matcher not initialized");
        return;
    }

    bool can_use_gpu = g_match.gpu_matcher && g_match.gpu_matcher->isAvailable();
    g_match.use_gpu = use_gpu && can_use_gpu;
    LOGI("Matcher mode: %s", g_match.use_gpu ? "GPU" : "CPU");
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
        // Invalidate previous frame since model changed
        g_match.has_prev_frame = false;
        LOGI("Switched to model: %s", g_extractor->modelName());
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("nativeSwitchModel failed: %s", e.what());
        return JNI_FALSE;
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Inference benchmark (FP32 vs INT8)
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

// ---------------------------------------------------------------------------
// Phase 4: Matching benchmark (GPU vs CPU)
//
// Returns FloatArray:
//   [0] gpu_avg_us
//   [1] cpu_avg_us
//   [2] speedup (cpu/gpu)
//   [3] match_count
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeBenchmarkMatching(
    JNIEnv* env, jobject /* thiz */,
    jint iterations) {

    using namespace onyx::matching;

    const int N = 500;
    const int D = 64;

    // Generate synthetic L2-normalized descriptors
    Eigen::MatrixXf desc1 = Eigen::MatrixXf::Random(N, D);
    Eigen::MatrixXf desc2 = Eigen::MatrixXf::Random(N, D);
    // Normalize rows
    for (int i = 0; i < N; ++i) {
        desc1.row(i).normalize();
        desc2.row(i).normalize();
    }
    // Make some descriptors similar to ensure matches exist
    for (int i = 0; i < N / 5; ++i) {
        desc2.row(i) = desc1.row(i) + Eigen::RowVectorXf::Random(D) * 0.05f;
        desc2.row(i).normalize();
    }

    float gpu_avg = 0.0f;
    float cpu_avg = 0.0f;
    int match_count = 0;

    // Benchmark GPU
    if (g_match.gpu_matcher && g_match.gpu_matcher->isAvailable()) {
        double total = 0.0;
        for (int i = 0; i < iterations; ++i) {
            double t = 0.0;
            auto m = g_match.gpu_matcher->match(desc1, N, desc2, N, 0.8f, &t);
            total += t;
            if (i == 0) match_count = static_cast<int>(m.size());
        }
        gpu_avg = static_cast<float>(total / iterations);
    }

    // Benchmark CPU
    {
        CpuMatcher cpu;
        double total = 0.0;
        for (int i = 0; i < iterations; ++i) {
            double t = 0.0;
            auto m = cpu.match(desc1, N, desc2, N, 0.8f, &t);
            total += t;
            if (match_count == 0 && i == 0) match_count = static_cast<int>(m.size());
        }
        cpu_avg = static_cast<float>(total / iterations);
    }

    float speedup = (gpu_avg > 0.0f) ? cpu_avg / gpu_avg : 0.0f;

    LOGI("Matching benchmark (%d iters, %d descs): GPU=%.0f us, CPU=%.0f us, "
         "Speedup=%.2fx, matches=%d",
         iterations, N, gpu_avg, cpu_avg, speedup, match_count);

    jfloatArray result = env->NewFloatArray(4);
    if (result) {
        float data[4] = { gpu_avg, cpu_avg, speedup, static_cast<float>(match_count) };
        env->SetFloatArrayRegion(result, 0, 4, data);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Phase 4: Matching validation (GPU vs CPU correctness)
//
// Returns FloatArray:
//   [0] gpu_matches
//   [1] cpu_matches
//   [2] mismatches (GPU matches not in CPU matches)
//   [3] passed (1.0 if mismatches == 0)
// ---------------------------------------------------------------------------

JNIEXPORT jfloatArray JNICALL
Java_com_onyxvo_app_NativeBridge_nativeValidateMatching(
    JNIEnv* env, jobject /* thiz */) {

    using namespace onyx::matching;

    const int N = 200;
    const int D = 64;

    // Generate synthetic descriptors with known matches
    Eigen::MatrixXf desc1 = Eigen::MatrixXf::Random(N, D);
    Eigen::MatrixXf desc2 = Eigen::MatrixXf::Random(N, D);
    for (int i = 0; i < N; ++i) {
        desc1.row(i).normalize();
        desc2.row(i).normalize();
    }
    // Plant known matches: first 50 descriptors in desc2 are close copies of desc1
    for (int i = 0; i < 50; ++i) {
        desc2.row(i) = desc1.row(i) + Eigen::RowVectorXf::Random(D) * 0.02f;
        desc2.row(i).normalize();
    }

    // Run CPU matcher
    CpuMatcher cpu;
    auto cpu_matches = cpu.match(desc1, N, desc2, N, 0.8f);

    // Run GPU matcher (if available)
    std::vector<Match> gpu_matches;
    bool gpu_available = g_match.gpu_matcher && g_match.gpu_matcher->isAvailable();
    if (gpu_available) {
        gpu_matches = g_match.gpu_matcher->match(desc1, N, desc2, N, 0.8f);
    }

    // Compare: count mismatches
    int mismatches = 0;
    if (gpu_available) {
        // Build set of CPU match pairs
        std::set<std::pair<int, int>> cpu_set;
        for (const auto& m : cpu_matches) {
            cpu_set.insert({m.idx1, m.idx2});
        }
        for (const auto& m : gpu_matches) {
            if (cpu_set.find({m.idx1, m.idx2}) == cpu_set.end()) {
                mismatches++;
            }
        }
    }

    bool passed = gpu_available
        ? (mismatches == 0 && gpu_matches.size() == cpu_matches.size())
        : (cpu_matches.size() > 0);  // CPU-only: just verify it produces matches

    LOGI("Validate matching: GPU=%zu matches, CPU=%zu matches, mismatches=%d -> %s",
         gpu_matches.size(), cpu_matches.size(), mismatches,
         passed ? "PASS" : "FAIL");

    jfloatArray result = env->NewFloatArray(4);
    if (result) {
        float data[4] = {
            static_cast<float>(gpu_matches.size()),
            static_cast<float>(cpu_matches.size()),
            static_cast<float>(mismatches),
            passed ? 1.0f : 0.0f
        };
        env->SetFloatArrayRegion(result, 0, 4, data);
    }
    return result;
}

} // extern "C"

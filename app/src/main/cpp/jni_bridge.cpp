#include <jni.h>
#include <string>
#include <memory>
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

} // extern "C"

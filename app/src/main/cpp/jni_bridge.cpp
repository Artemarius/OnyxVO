#include <jni.h>
#include <string>
#include "utils/android_log.h"

#define ONYX_VO_VERSION "0.1.0-phase1"

extern "C" {

JNIEXPORT void JNICALL
Java_com_onyxvo_app_NativeBridge_nativeInit(JNIEnv* /* env */, jobject /* thiz */) {
    LOGI("OnyxVO native init — version %s", ONYX_VO_VERSION);
}

JNIEXPORT jstring JNICALL
Java_com_onyxvo_app_NativeBridge_nativeGetVersion(JNIEnv* env, jobject /* thiz */) {
    return env->NewStringUTF(ONYX_VO_VERSION);
}

} // extern "C"

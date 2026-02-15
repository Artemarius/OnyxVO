import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Resolve ONNX Runtime native library from the AAR for CMake linking.
// The AAR bundles jni/<abi>/libonnxruntime.so. We extract it to a stable
// directory outside build/ so it survives `clean`.
val ortNativeDir: String by lazy {
    val ortConfig = configurations.detachedConfiguration(
        dependencies.create("com.microsoft.onnxruntime:onnxruntime-android:1.20.0@aar")
    )
    val aarFile = ortConfig.resolve().first()
    val extractDir = file("${projectDir}/.ort-native")
    val targetSo = File(extractDir, "jni/arm64-v8a/libonnxruntime.so")
    if (!targetSo.exists()) {
        copy {
            from(zipTree(aarFile))
            into(extractDir)
            include("jni/arm64-v8a/**")
        }
    }
    "${extractDir.absolutePath}/jni/arm64-v8a".replace("\\", "/")
}

val keystorePropertiesFile = rootProject.file("app/keystore.properties")
val keystoreProperties = Properties().apply {
    if (keystorePropertiesFile.exists()) {
        keystorePropertiesFile.inputStream().use { load(it) }
    }
}

android {
    namespace = "com.onyxvo.app"
    compileSdk = 34

    signingConfigs {
        create("release") {
            if (keystorePropertiesFile.exists()) {
                storeFile = file(keystoreProperties["storeFile"] as String)
                storePassword = keystoreProperties["storePassword"] as String
                keyAlias = keystoreProperties["keyAlias"] as String
                keyPassword = keystoreProperties["keyPassword"] as String
            }
        }
    }

    defaultConfig {
        applicationId = "com.onyxvo.app"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += "-DANDROID_STL=c++_shared"
                arguments += "-DORT_LIB_DIR=$ortNativeDir"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs {
            // Kompute's Vulkan NDK wrapper resolves function pointers via dlopen/dlsym.
            // Without extraction, the .so pages inside the compressed APK lack execute
            // permission, causing SIGSEGV in kp::Manager::createInstance().
            useLegacyPackaging = true
        }
    }

    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // ONNX Runtime for on-device ML inference (XFeat feature extraction)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
}

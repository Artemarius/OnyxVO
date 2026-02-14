cmake_minimum_required(VERSION 3.22)
project(onyx_vo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Eigen 3.4 (header-only)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
# Eigen has a heavy build system; we only need headers.
# Disable everything except the header-only target.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
FetchContent_Populate(eigen)
# Eigen 3.4 provides the Eigen3::Eigen target via add_subdirectory,
# but that pulls in too much. Just create a minimal interface target.
add_library(Eigen3_Eigen INTERFACE)
target_include_directories(Eigen3_Eigen INTERFACE ${eigen_SOURCE_DIR})
add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)

# ---------------------------------------------------------------------------
# ONNX Runtime (headers vendored, .so from AAR via Gradle)
# ---------------------------------------------------------------------------
# ORT_LIB_DIR is passed from build.gradle.kts via cmake arguments,
# pointing to the extracted AAR's jni/arm64-v8a/ directory.
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB_DIR}/libonnxruntime.so"
)
target_include_directories(onnxruntime INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
)

# ---------------------------------------------------------------------------
# Main library
# ---------------------------------------------------------------------------
add_library(onyx_vo SHARED
    jni_bridge.cpp
    preprocessing/neon_ops.cpp
    feature/xfeat_extractor.cpp
)

target_include_directories(onyx_vo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optimize native code even in debug builds — NEON intrinsics need
# proper instruction scheduling to outperform scalar.
target_compile_options(onyx_vo PRIVATE -O2)

target_link_libraries(onyx_vo
    Eigen3::Eigen
    onnxruntime
    log
    android
)

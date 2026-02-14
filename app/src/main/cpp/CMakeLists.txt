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
# Kompute v0.8.1 (Vulkan compute via submodule)
# ---------------------------------------------------------------------------
# Kompute v0.8.1's NDK wrapper is incompatible with NDK 25+ Vulkan headers
# (removed NVX extensions, vulkan.hpp DispatchLoaderStatic references 1.2+
# functions). We pin Vulkan-Headers to v1.1.126 and patch the wrapper to add
# 3 missing VK_KHR_timeline_semaphore function pointers.
# The patch is applied automatically on first configure after submodule init.
set(KOMPUTE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kompute")
set(KOMPUTE_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/patches/kompute-ndk25-compat.patch")

# Check if patch has already been applied (marker: timeline semaphore in wrapper)
file(STRINGS "${KOMPUTE_DIR}/vk_ndk_wrapper_include/kompute_vk_ndk_wrapper_patch.hpp"
     _PATCH_CHECK REGEX "vkGetSemaphoreCounterValueKHR" LIMIT_COUNT 1)
if(NOT _PATCH_CHECK)
    message(STATUS "Applying Kompute NDK 25+ compatibility patch...")
    execute_process(
        COMMAND git apply "${KOMPUTE_PATCH}"
        WORKING_DIRECTORY "${KOMPUTE_DIR}"
        RESULT_VARIABLE _PATCH_RESULT
    )
    if(NOT _PATCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to apply Kompute patch. "
                            "Try: cd ${KOMPUTE_DIR} && git checkout . && reconfigure.")
    endif()
endif()

# Pin Vulkan-Headers to v1.1.126 (pre-Vulkan-1.2, compatible with NDK wrapper)
set(VK_HEADERS_DIR "${KOMPUTE_DIR}/external/Vulkan-Headers")
if(EXISTS "${VK_HEADERS_DIR}/.git")
    execute_process(
        COMMAND git describe --tags --exact-match HEAD
        WORKING_DIRECTORY "${VK_HEADERS_DIR}"
        OUTPUT_VARIABLE _VK_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _VK_TAG_RESULT
    )
    if(NOT _VK_TAG STREQUAL "v1.1.126")
        message(STATUS "Checking out Vulkan-Headers v1.1.126...")
        execute_process(
            COMMAND git checkout v1.1.126
            WORKING_DIRECTORY "${VK_HEADERS_DIR}"
            RESULT_VARIABLE _VK_CO_RESULT
        )
        if(NOT _VK_CO_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to checkout Vulkan-Headers v1.1.126")
        endif()
    endif()
endif()

set(KOMPUTE_OPT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_BUILD_SHADERS OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_BUILD_SINGLE_HEADER OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_INSTALL OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_ANDROID_BUILD ON CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS ON CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_ENABLE_SPDLOG OFF CACHE BOOL "" FORCE)
set(KOMPUTE_OPT_REPO_SUBMODULE_BUILD ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/kompute ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)

# ---------------------------------------------------------------------------
# SPIR-V shader compilation (GLSL -> .spv -> C++ header)
# ---------------------------------------------------------------------------
# Find glslc from Android NDK shader-tools
find_program(GLSLC glslc
    HINTS
        "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}"
        "${ANDROID_NDK}/shader-tools/windows-x86_64"
        "${ANDROID_NDK}/shader-tools/linux-x86_64"
        "${ANDROID_NDK}/shader-tools/darwin-x86_64"
)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found — ensure NDK is installed with shader-tools. "
                        "Looked in: ${ANDROID_NDK}/shader-tools/")
endif()
message(STATUS "Found glslc: ${GLSLC}")

set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/matching/shaders/match_descriptors.comp")
set(SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/match_descriptors.spv")
set(SHADER_HDR "${CMAKE_CURRENT_BINARY_DIR}/match_descriptors_comp_spv.h")

# .comp -> .spv
add_custom_command(
    OUTPUT "${SHADER_SPV}"
    COMMAND "${GLSLC}" --target-env=vulkan1.0 -o "${SHADER_SPV}" "${SHADER_SRC}"
    DEPENDS "${SHADER_SRC}"
    COMMENT "Compiling GLSL -> SPIR-V: match_descriptors.comp"
    VERBATIM
)

# .spv -> .h (C++ header with uint32_t array)
add_custom_command(
    OUTPUT "${SHADER_HDR}"
    COMMAND ${CMAKE_COMMAND}
        -DSPV_FILE=${SHADER_SPV}
        -DHEADER_FILE=${SHADER_HDR}
        -DVAR_NAME=match_descriptors_comp_spv
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/spv_to_header.cmake
    DEPENDS "${SHADER_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/spv_to_header.cmake"
    COMMENT "Generating C++ header from SPIR-V: match_descriptors_comp_spv.h"
)

add_custom_target(compile_shaders DEPENDS "${SHADER_HDR}")

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
    matching/cpu_matcher.cpp
    matching/gpu_matcher.cpp
    vo/pose_estimator.cpp
    vo/trajectory.cpp
)

add_dependencies(onyx_vo compile_shaders)

target_include_directories(onyx_vo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}   # For generated SPIR-V header
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kompute/vk_ndk_wrapper_include  # NDK Vulkan wrapper
)

target_compile_definitions(onyx_vo PRIVATE
    VK_USE_PLATFORM_ANDROID_KHR=1
    KOMPUTE_DISABLE_VK_DEBUG_LAYERS=1
)

# Optimize native code even in debug builds — NEON intrinsics need
# proper instruction scheduling to outperform scalar.
target_compile_options(onyx_vo PRIVATE -O2)

target_link_libraries(onyx_vo
    Eigen3::Eigen
    kompute
    kompute_vk_ndk_wrapper
    onnxruntime
    log
    android
)

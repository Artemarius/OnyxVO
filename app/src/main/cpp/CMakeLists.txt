cmake_minimum_required(VERSION 3.22)
project(onyx_vo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(onyx_vo SHARED
    jni_bridge.cpp
    preprocessing/neon_ops.cpp
)

target_include_directories(onyx_vo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optimize native code even in debug builds — NEON intrinsics need
# proper instruction scheduling to outperform scalar.
target_compile_options(onyx_vo PRIVATE -O2)

target_link_libraries(onyx_vo
    log
    android
)

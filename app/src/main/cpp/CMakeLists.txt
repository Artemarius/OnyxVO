cmake_minimum_required(VERSION 3.22)
project(onyx_vo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phase 1: JNI bridge only
add_library(onyx_vo SHARED
    jni_bridge.cpp
)

target_include_directories(onyx_vo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(onyx_vo
    log
    android
)

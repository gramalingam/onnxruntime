# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Builds the toy FeedForward "C++ model" driver. It links against a prebuilt
# ONNX Runtime package (the same way samples/cxx does) so the hand-written model
# can drive ORT's op kernels.
#
#   cmake -S . -B build \
#       -DORT_HEADER_DIR:PATH=/path/to/onnxruntime/include \
#       -DORT_LIBRARY_DIR:PATH=/path/to/onnxruntime/lib
#   cmake --build build --config Release
#   (cd build && ./feed_forward_model)   # model file is copied next to the exe

cmake_minimum_required(VERSION 3.28)

project(onnxruntime_feed_forward_model CXX)

set(CMAKE_CXX_STANDARD 20)

foreach(VAR IN ITEMS ORT_LIBRARY_DIR ORT_HEADER_DIR)
    if (NOT DEFINED ${VAR})
        message(FATAL_ERROR "Required variable ${VAR} is not set. "
            "Set ORT_LIBRARY_DIR to the ONNX Runtime lib directory and "
            "ORT_HEADER_DIR to the ONNX Runtime include directory.")
    endif()
endforeach()

get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY_DIR}" ABSOLUTE)
get_filename_component(ORT_HEADER_DIR "${ORT_HEADER_DIR}" ABSOLUTE)

block()
add_executable(feed_forward_model)

target_sources(feed_forward_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/feed_forward_model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/run_feed_forward.cc)

target_include_directories(feed_forward_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ORT_HEADER_DIR})

target_link_directories(feed_forward_model PRIVATE ${ORT_LIBRARY_DIR})
target_link_libraries(feed_forward_model PRIVATE onnxruntime)

# Copy ONNX Runtime shared libraries next to the executable.
if (WIN32)
    file(GLOB ORT_SHARED_LIBS "${ORT_LIBRARY_DIR}/*.dll")
elseif (APPLE)
    file(GLOB ORT_SHARED_LIBS "${ORT_LIBRARY_DIR}/*.dylib")
else()
    file(GLOB ORT_SHARED_LIBS "${ORT_LIBRARY_DIR}/*.so" "${ORT_LIBRARY_DIR}/*.so.*")
endif()

add_custom_command(TARGET feed_forward_model POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_SHARED_LIBS}
        $<TARGET_FILE_DIR:feed_forward_model>)

# Copy the model file next to the executable so it can be found at run time.
add_custom_command(TARGET feed_forward_model POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/feed_forward.onnx
        $<TARGET_FILE_DIR:feed_forward_model>)
endblock()

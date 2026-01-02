cmake_minimum_required(VERSION 3.24)
project(infercustomparser LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA not found. Please install CUDA.")
endif()
message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
enable_language(CUDA)

add_library(
    nvds_infercustomparser SHARED
    src/postprocess.cu
)

target_link_libraries(
    nvds_infercustomparser
    PRIVATE
    nvinfer
)

include_directories(
    /opt/nvidia/deepstream/deepstream/sources/includes
)

set_target_properties(
    nvds_infercustomparser PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

cmake_minimum_required(VERSION 3.5)
project(libtorch-example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Optional
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Optional

# CMAKE MODULES
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../cmake/")
include(CPM)

# EXTERNAL LIBRARIES
cpmaddpackage("gh:gabime/spdlog#v1.15.2")
cpmaddpackage("gh:jarro2783/cxxopts#v3.2.0")

find_package(OpenCV REQUIRED) # Need `CMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR`

# Find Torch by using pip install torch
execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE Torch_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Torch_DIR: ${Torch_DIR}")
find_package(Torch REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch ${Torch_DIR})
message(STATUS "Torch_LIBRARIES: ${TORCH_LIBRARIES}")

add_executable(main cpp/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp/base.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp)
target_link_libraries(main PUBLIC ${OpenCV_LIBS} spdlog::spdlog cxxopts::cxxopts ${TORCH_LIBRARIES})

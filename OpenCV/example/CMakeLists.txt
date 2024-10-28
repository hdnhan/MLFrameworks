cmake_minimum_required(VERSION 3.5)
project(opencv-example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Optional
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Optional

find_package(OpenCV REQUIRED) # Need `CMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR`

add_executable(main cpp/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp/base.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp)
target_link_libraries(main ${OpenCV_LIBS})

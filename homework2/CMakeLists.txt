cmake_minimum_required(VERSION 3.1)

# require C++17
set(CMAKE_CXX_STANDARD 17)

project(TDCV_HW02)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES hog_visualization.cpp task2.cpp hog_descriptors.cpp)
add_executable(TDCV_HW02 main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(TDCV_HW02 ${OpenCV_LIBS})

# this adds many warnings to your build. They usually help you find some bugs
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options( TDCV_HW02 PUBLIC -Wall -Wextra -Wconversion -pedantic -Wfatal-errors)
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options( TDCV_HW02 PUBLIC /W3 /WX )
endif()

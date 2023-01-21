cmake_minimum_required(VERSION 2.6)

project(qrcode LANGUAGES C CXX CUDA)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# Set output dir
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Set include dir
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)


# Set 3rdParty packages
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib/arrch64-linux-gnu)

add_subdirectory(cuda)
set (EXTRA_LIBS ${EXTRA_LIBS} gpu)


add_executable(demo demo.cpp source/cujpg.cpp)
target_link_libraries(demo ${EXTRA_LIBS})
target_link_libraries(demo ${OpenCV_LIBS})
target_link_libraries(demo nppc nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu nppitc npps)

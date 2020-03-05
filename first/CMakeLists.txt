cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(OpenCV_STATIC ON)
set(OpenCV_DIR E:/thirdparty/build/install/win64)


project(firstcode VERSION 2.7.0)
find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

FILE(GLOB_RECURSE LIB_SRC "src/*.cu")
FILE(GLOB_RECURSE LIB_inc "include/*.h")

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(tt main.cu ${LIB_SRC} ${LIB_inc} )

target_link_libraries(tt ${CUDA_LIBRARIES} ${OpenCV_LIBS})




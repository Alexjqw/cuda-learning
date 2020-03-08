cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(OpenCV_STATIC ON)
set(OpenCV_DIR E:/thirdparty/build/install/win64)

set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")


if(DEFINED AUTO_DETECT_ARCH AND "${AUTO_DETECT_ARCH}")
	CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
	set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
endif()
message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")


project(shared_memory VERSION 2.7.0)
find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

FILE(GLOB_RECURSE LIB_SRC "src/*.cu")
FILE(GLOB_RECURSE LIB_inc "include/*.h")

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(shared_memory main.cu ${LIB_SRC} ${LIB_inc} )

target_link_libraries(shared_memory ${CUDA_LIBRARIES} ${OpenCV_LIBS})




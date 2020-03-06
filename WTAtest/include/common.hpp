/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __COMMON_H__
#define __COMMON_H__



#include <stdio.h>
#include <random>
#include <limits>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "utility.hpp"
#define random(a,b) (rand()%(b-a)+a)



namespace srs {
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)




    extern std::default_random_engine g_random_engine;

    template <typename T>
    inline thrust::host_vector<T> generate_random_sequence(size_t n) {
        std::uniform_int_distribution<T> distribution(
            std::numeric_limits<T>::min(),
            std::numeric_limits<T>::max());
        thrust::host_vector<T> seq(n);
        for (size_t i = 0; i < n; ++i) {
            seq[i] = distribution(g_random_engine);
        }
        return seq;
    }

    template <>
    inline thrust::host_vector<uint8_t> generate_random_sequence<uint8_t>(size_t n) {
        std::uniform_int_distribution<unsigned int> distribution(
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max());
        thrust::host_vector<uint8_t> seq(n);
        for (size_t i = 0; i < n; ++i) {
            seq[i] = static_cast<uint8_t>(distribution(g_random_engine));
        }
        return seq;
    }


    template <typename T>
    inline thrust::host_vector<T> generate_random_data(size_t n) {
        srand((int)time(0));
        T maxV = std::numeric_limits<T>::max();
        T minV = std::numeric_limits<T>::min();
        thrust::host_vector<T> seq(n);
        for (size_t i = 0; i < n; ++i) {
            seq[i] = random(minV, maxV);
        }
        return seq;
    }



    template <typename T>
    inline thrust::host_vector<T> to_host_vector(const thrust::device_vector<T>& src)
    {
        thrust::host_vector<T> dest(src.size());
        cudaMemcpy(
            dest.data(),
            src.data().get(),
            sizeof(T) * src.size(),
            cudaMemcpyDeviceToHost);
        return dest;
    }

    template <typename T>
    inline thrust::device_vector<T> to_device_vector(const thrust::host_vector<T>& src)
    {
        thrust::device_vector<T> dest(src.size());
        cudaMemcpy(
            dest.data().get(),
            src.data(),
            sizeof(T) * src.size(),
            cudaMemcpyHostToDevice);
        return dest;
    }


inline void cuda_safe_call(cudaError error, const char *file, const int line)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "cuda error %s : %d %s\n", file, line, cudaGetErrorString(error));
        exit(-1);
    }
}

#define CudaSafeCall(error) cuda_safe_call(error, __FILE__, __LINE__)

#define CudaKernelCheck() CudaSafeCall(cudaGetLastError())

struct device_buffer
{
    device_buffer() : data(nullptr) {}
    device_buffer(size_t count) { allocate(count); }
    void allocate(size_t count) { cudaMalloc(&data, count); }
    ~device_buffer() { cudaFree(data); }
    void* data;
};

}

#endif

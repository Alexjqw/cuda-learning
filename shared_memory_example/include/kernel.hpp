
#ifndef __KENER_H__
#define __KENER_H__

#include "common.hpp"


#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

const int NUM_REPEATS = 10;
const int N = 100000000;
const int M = sizeof(real) * N;
const int BLOCK_SIZE = 128;

void __global__ reduce_global(real* d_x, real* d_y);
void __global__ reduce_dynamic(real* d_x, real* d_y);
real reduce(real* d_x, const int method);
void __global__ reduce_shared(real* d_x, real* d_y);
void timing(real* h_x, real* d_x, const int method);
#endif

#ifndef __KENER_H__
#define __KENER_H__

#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

void timing(const real* d_A, real* d_B, const int N, const int task);
__global__ void copy(const real* A, real* B, const int N);
__global__ void transpose1(const real* A, real* B, const int N);
__global__ void transpose2(const real* A, real* B, const int N);
__global__ void transpose3(const real* A, real* B, const int N);
void print_matrix(const int N, const real* A);


#endif
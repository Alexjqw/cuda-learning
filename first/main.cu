#include"include/kener.hpp"
#include "include/book.hpp"


int main(void) {
    int a[N], b[N], c[N];
    int* dev_a, * dev_b, * dev_c;

    // allocate the memory on the GPU
    CHECK(cudaMalloc((void**)&dev_a, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&dev_b, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&dev_c, N * sizeof(int)));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    // copy the arrays 'a' and 'b' to the GPU
    CHECK(cudaMemcpy(dev_a, a, N * sizeof(int),
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, N * sizeof(int),
        cudaMemcpyHostToDevice));

    add << <N, 1 >> > (dev_a, dev_b, dev_c);

    // copy the array 'c' back from the GPU to the CPU
    CHECK(cudaMemcpy(c, dev_c, N * sizeof(int),
        cudaMemcpyDeviceToHost));

    // display the results
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // free the memory allocated on the GPU
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_c));

    return 0;
}


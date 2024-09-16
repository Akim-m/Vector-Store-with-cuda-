#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

extern "C" void calculate_dot_product(float* a, float* b, float* result, int n) {
    float *d_a, *d_b;
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_b, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    cublasSdot(handle, n, d_a, 1, d_b, 1, result);

    cudaFree(d_a);
    cudaFree(d_b);

    cublasDestroy(handle);
}

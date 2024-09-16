#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" {

    float cuda_norm_cublas(float* h_vector, int n) {
        float norm_result = 0.0f;

        cublasHandle_t handle;
        cublasCreate(&handle);

        float* d_vector;
        cudaMalloc(&d_vector, n * sizeof(float));

        cublasSetVector(n, sizeof(float), h_vector, 1, d_vector, 1);

        cublasSnrm2(handle, n, d_vector, 1, &norm_result);

        cudaFree(d_vector);

        cublasDestroy(handle);

        return norm_result;
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    cublasHandle_t handle;
    int m, n, k, i, j;
    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;
    double alpha = 1.0, beta = 0.0;

    // Specify the dimensions of the matrices
    m = 64;  // Number of rows of A and C
    n = 64;  // Number of columns of B and C
    k = 64;  // Number of columns of A and rows of B

    // Allocate host memory
    h_A = (double *)malloc(m * k * sizeof(double));
    h_B = (double *)malloc(k * n * sizeof(double));
    h_C = (double *)malloc(m * n * sizeof(double));

    // Initialize host matrices with sample values
    // For simplicity, matrices are filled with sequential numbers
    // A better approach is to fill them with random numbers or actual data
    for (i = 0; i < m; ++i) {
        for (j = 0; j < k; ++j) {
            h_A[i * k + j] = 1;//i * k + j + 1;
        }
    }

    for (i = 0; i < k; ++i) {
        for (j = 0; j < n; ++j) {
            h_B[i * n + j] = 1;//i * n + j + 1;
        }
    }

    // Initialize cuBLAS
    cublasCreate(&handle);

    // Allocate device memory
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    // Copy matrices from host to device
    cublasSetMatrix(m, k, sizeof(double), h_A, m, d_A, m);
    cublasSetMatrix(k, n, sizeof(double), h_B, k, d_B, k);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    // Copy the result back to host
    cublasGetMatrix(m, n, sizeof(double), d_C, m, h_C, m);

    // Print the result
    // printf("Result matrix C:\n");
    // for (i = 0; i < m; ++i) {
    //     for (j = 0; j < n; ++j) {
    //         printf("%f ", h_C[i * n + j]);
    //     }
    //     printf("\n");
    // }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);

    return 0;
}

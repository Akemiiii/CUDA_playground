#include<iostream>
#include<cuda.h>
#include<mma.h>
#include<assert.h>
#include"Common/helper_cuda.h"

#define M 8
#define N 8
#define K 4

int main() {
    double *A_h, *B_h, *C_h;
    checkCudaErrors(cudaMallocHost((void**)&A_h, M*K*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void**)&B_h, K*N*sizeof(double)));
    checkCudaErrors(cudaMallocHost((void**)&C_h, M*N*sizeof(double)));

    for(int i=0; i<M*K; i++) {
        A_h[i] = i;
    }
    for(int i=0; i<K*N; i++) {
        B_h[i] = i;
    }
    for(int i=0; i<M*N; i++) {
        C_h[i] = 0;
    }

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *D = NULL;
    checkCudaErrors(cudaMalloc((void**)&A, M*K*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&B, K*N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&C, M*N*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&D, M*N*sizeof(double)));
    assert(((unsigned long long)A)%128 == 0);
    assert(((unsigned long long)B)%128 == 0);
    assert(((unsigned long long)C)%128 == 0);
    assert(((unsigned long long)D)%128 == 0);

    checkCudaErrors(cudaMemcpy(A, A_h, M*K*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, K*N*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, M*N*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, M*N*sizeof(double)));

    


}
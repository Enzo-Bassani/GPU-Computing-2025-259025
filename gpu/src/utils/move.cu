#include "move.h"
#include "read.h"
#include <cstdio>

void moveCOOToDevice(COOMatrix &h, COOMatrix &d) {
    size_t rowsSize = (d.nnz * sizeof(int));
    size_t colsSize = (d.nnz * sizeof(int));
    size_t valuesSize = (d.nnz * sizeof(float));

    checkCudaError(cudaMalloc(&d.rows, rowsSize));
    checkCudaError(cudaMalloc(&d.cols, colsSize));
    checkCudaError(cudaMalloc(&d.values, valuesSize));

    checkCudaError(cudaMemcpy(d.rows, h.rows, rowsSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d.cols, h.cols, colsSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d.values, h.values, valuesSize, cudaMemcpyHostToDevice));
}

void moveCSRToDevice(CSRMatrix &h, CSRMatrix &d) {
    size_t rowPtrsSize = ((d.numRows + 1) * sizeof(int));
    size_t colsSize = (d.nnz * sizeof(int));
    size_t valuesSize = (d.nnz * sizeof(float));

    checkCudaError(cudaMalloc(&d.rowPtrs, rowPtrsSize));
    checkCudaError(cudaMalloc(&d.cols, colsSize));
    checkCudaError(cudaMalloc(&d.values, valuesSize));

    checkCudaError(cudaMemcpy(d.rowPtrs, h.rowPtrs, rowPtrsSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d.cols, h.cols, colsSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d.values, h.values, valuesSize, cudaMemcpyHostToDevice));
}

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

float *moveArrayToDevice(float *h_array, int length) {
    float *d_array;
    size_t size = sizeof(float) * length;
    checkCudaError(cudaMalloc(&d_array, size));
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    return d_array;
}

#include "move.h"
#include "read.h"
#include "readcu.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

// Constant memory for the vector (up to max size)
// Maximum size is set to accommodate large vectors, but we need to be careful not to exceed constant memory limits
#define MAX_VECTOR_SIZE 0x10000/4
__constant__ float d_const_vector[MAX_VECTOR_SIZE];

// Function to multiply CSR matrix by a vector stored in constant memory
__global__ void multiplyMatrixVector(CSRMatrix matrix, float *result) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= matrix.numRows)
        return;

    float sum = 0.0;

    int start = matrix.rowPtrs[i];
    int end = matrix.rowPtrs[i + 1];

    for (int j = start; j < end; j++) {
        int col = matrix.cols[j];
        float val = matrix.values[j];

        sum += val * d_const_vector[col];
    }

    result[i] = sum;
}

int main() {
    printf("Running!\n");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/36057_onetone1_sorted.mtx";
    // char filename[] = "../matrices/929901_Hardesty2_sorted.mtx";
    // char filename[] = "../matrices/923136_Emilia_923_sorted.mtx";
    // Read the matrix from file
    CSRMatrix h_matrix = readMTXFileCSR(filename);
    CSRMatrix d_matrix = h_matrix;
    moveCSRToDevice(h_matrix, d_matrix);
    // printCSRMatrixHead(h_matrix);

    int cols = h_matrix.numCols, rows = h_matrix.numRows;

    // Ensure our vector fits within the constant memory size limit
    if (cols > MAX_VECTOR_SIZE) {
        fprintf(stderr, "Vector size (%d) exceeds maximum constant memory vector size (%d)\n",
                cols, MAX_VECTOR_SIZE);
        freeCSRMatrixCuda(h_matrix);
        exit(1);
    }

    // Create vector of ones
    float *h_vector = (float *)std::malloc(cols * sizeof(float));
    if (h_vector == NULL) {
        fprintf(stderr, "Memory allocation failed for vector\n");
        freeCSRMatrixCuda(h_matrix);
        exit(1);
    }

    for (int i = 0; i < cols; i++) {
        h_vector[i] = 1.0;
    }

    // Copy vector to constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_const_vector, h_vector, cols * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector to constant memory: %s\n", cudaGetErrorString(err));
        freeCSRMatrixCuda(h_matrix);
        std::free(h_vector);
        exit(1);
    }

    // Multiply matrix by vector
    size_t results_size = rows * sizeof(float);
    float *h_results = (float *)std::malloc(results_size);
    float *d_results;
    cudaMalloc(&d_results, results_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads_per_block = 256;
    int num_blocks = (rows + threads_per_block - 1) / threads_per_block;
    printf("Launching %d blocks of %d threads\n", num_blocks, threads_per_block);

    cudaEventRecord(start);
    multiplyMatrixVector<<<num_blocks, threads_per_block>>>(d_matrix, d_results);
    cudaEventRecord(stop);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Save results
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    char result_path[256];
    sprintf(result_path, "../results/%s_result.txt", strrchr(filename, '/') ? strrchr(filename, '/') + 1 : filename);
    writeVectorToFile(result_path, h_results, rows);

    printf("Kernel Time: %f ms\n", milliseconds);

    // Free allocated memory
    cudaFree(d_results);
    freeCSRMatrixCuda(d_matrix);
    std::free(h_vector);
    std::free(h_results);
    freeCSRMatrix(h_matrix);

    return 0;
}

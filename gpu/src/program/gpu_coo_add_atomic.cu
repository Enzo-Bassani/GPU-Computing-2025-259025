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

// Function to multiply CSR matrix by a vector
__global__ void multiplyMatrixVector(COOMatrix matrix, float *vector, float *result) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= matrix.nnz)
        return;

    int row = matrix.rows[i];
    int col = matrix.cols[i];
    float val = matrix.values[i];

    atomicAdd(&result[row], val * vector[col]);
}

int main() {
    printf("Running!\n");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    // char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/36057_onetone1_sorted.mtx";
    // char filename[] = "../matrices/929901_Hardesty2_sorted.mtx";
    char filename[] = "../matrices/923136_Emilia_923_sorted.mtx";
    // Read the matrix from file
    COOMatrix h_matrix = readMTXFileCOO(filename);
    COOMatrix d_matrix = h_matrix;
    moveCOOToDevice(h_matrix, d_matrix);
    // printCSRMatrixHead(h_matrix);

    int cols = h_matrix.numCols, rows = h_matrix.numRows;

    // Create vector of ones
    float *h_vector = (float *)std::malloc(cols * sizeof(float));
    if (h_vector == NULL) {
        fprintf(stderr, "Memory allocation failed for vector\n");
        freeCOOMatrixCuda(h_matrix);
        exit(1);
    }

    for (int i = 0; i < cols; i++) {
        h_vector[i] = 1.0;
    }

    float *d_vector = moveArrayToDevice(h_vector, cols);

    // Multiply matrix by vector
    size_t results_size = rows * sizeof(float);
    float *h_results = (float *)std::malloc(results_size);
    float *d_results;
    cudaMalloc(&d_results, results_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads_per_block = 256;
    int num_blocks = (d_matrix.nnz + threads_per_block - 1) / threads_per_block;
    printf("Launching %d blocks of %d threads\n", num_blocks, threads_per_block);
    cudaEventRecord(start);
    multiplyMatrixVector<<<num_blocks, 256>>>(d_matrix, d_vector, d_results);
    cudaEventRecord(stop);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    cudaEventSynchronize(stop);
    cudaError_t err = cudaGetLastError();
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
    cudaFree(d_vector);
    cudaFree(d_results);
    freeCOOMatrixCuda(d_matrix);
    std::free(h_vector);
    std::free(h_results);
    freeCOOMatrix(h_matrix);

    return 0;
}

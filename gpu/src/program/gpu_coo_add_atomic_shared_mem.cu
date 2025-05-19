#include "move.h"
#include "timers.h"
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

#define WARMUP 3
#define NITER 10
// Function to multiply CSR matrix by a vector
__global__ void multiplyMatrixVector(COOMatrix matrix, float *vector, float *result) {
    __shared__ float sdata[256];
    __shared__ int firstRowInBlock;

    // Initialize shared memory
    sdata[threadIdx.x] = 0.0f;

    // Get global thread index
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Early return for threads outside matrix range
    if (i >= matrix.nnz)
        return;

    // Get row, column and value for this thread
    int row = matrix.rows[i];
    int col = matrix.cols[i];
    float val = matrix.values[i];

    // Record the first row in the block
    if (threadIdx.x == 0) {
        firstRowInBlock = row;
    }
    __syncthreads();

    // Calculate the shared memory index
    int sdataIndex = row - firstRowInBlock;

    // If the index is within our shared memory range, use shared memory
    // Otherwise, directly update the result
    // if (sdataIndex >= 0 && sdataIndex < 256) {
    if (sdataIndex < 256) {
        atomicAdd(&sdata[sdataIndex], val * vector[col]);
    } else {
        atomicAdd(&result[row], val * vector[col]);
    }

    __syncthreads();

    atomicAdd(&result[row], sdata[threadIdx.x]);

    __syncthreads();
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
    if (h_results == NULL) {
        fprintf(stderr, "Memory allocation failed for results\n");
        cudaFree(d_vector);
        freeCOOMatrixCuda(d_matrix);
        freeCOOMatrix(h_matrix);
        std::free(h_vector);
        exit(1);
    }

    // Initialize h_results to zero
    memset(h_results, 0, results_size);

    float *d_results;
    cudaError_t malloc_error = cudaMalloc(&d_results, results_size);
    if (malloc_error != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(malloc_error));
        std::free(h_results);
        cudaFree(d_vector);
        freeCOOMatrixCuda(d_matrix);
        freeCOOMatrix(h_matrix);
        std::free(h_vector);
        exit(1);
    }

    // Initialize d_results to zero
    cudaMemset(d_results, 0, results_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads_per_block = 256;
    int num_blocks = (d_matrix.nnz + threads_per_block - 1) / threads_per_block;
    printf("Launching %d blocks of %d threads\n", num_blocks, threads_per_block);
    float timers[NITER];
    float iter_time = 0;
    for (int i = -WARMUP; i < NITER; i++) {
        cudaMemset(d_results, 0, results_size);
        cudaEventRecord(start);
        multiplyMatrixVector<<<num_blocks, threads_per_block>>>(d_matrix, d_vector, d_results);
        cudaEventRecord(stop);

        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&iter_time, start, stop);
        if (i >= 0) {
            timers[i] = iter_time;
            printf("%d iter_time %f\n", i, iter_time);
        }
    }

    // Save results
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    char result_path[256];
    sprintf(result_path, "../results/%s_result.txt", strrchr(filename, '/') ? strrchr(filename, '/') + 1 : filename);
    writeVectorToFile(result_path, h_results, rows);

    float geo_avg = geometric_mean(timers, NITER);
    int num_FLOPs = 0;
    int num_bytes_accessed = 0;
    printStats(geo_avg, num_FLOPs, num_bytes_accessed);

    // Free allocated memory
    cudaFree(d_vector);
    cudaFree(d_results);
    freeCOOMatrixCuda(d_matrix);
    std::free(h_vector);
    std::free(h_results);
    freeCOOMatrix(h_matrix);

    return 0;
}

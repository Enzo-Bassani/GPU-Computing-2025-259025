#include "move.h"
#include "read.h"
#include "readcu.h"
#include "timers.h"
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
__global__ void multiplyMatrixVector(CSRMatrix matrix, float *vector, float *result) {
    int firstRow = blockDim.x * blockIdx.x;
    __shared__ float smem[256];
    smem[threadIdx.x] = 0;
    __syncthreads();
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i >= matrix.numRows)
    //     return;

    for (int i = 0; i < 256; i++) {
        int current_row = firstRow + i;
        int start = matrix.rowPtrs[current_row];
        int end = matrix.rowPtrs[current_row + 1];

        if ((int)blockIdx.x < end) {
            int index = start + blockIdx.x;
            int col = matrix.cols[index];
            float val = matrix.values[index];

            float sum = val * vector[col];
            atomicAdd(&smem[i], sum);
        }
    }

    __syncthreads();

    result[firstRow + threadIdx.x] = smem[threadIdx.x];
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
    CSRMatrix h_matrix = readMTXFileCSR(filename);
    CSRMatrix d_matrix = h_matrix;
    moveCSRToDevice(h_matrix, d_matrix);
    // printCSRMatrixHead(h_matrix);

    int cols = h_matrix.numCols, rows = h_matrix.numRows;

    // Create vector of ones
    float *h_vector = (float *)std::malloc(cols * sizeof(float));
    if (h_vector == NULL) {
        fprintf(stderr, "Memory allocation failed for vector\n");
        freeCSRMatrixCuda(h_matrix);
        exit(1);
    }

    // Seed the random number generator
    srand(42);
    for (int i = 0; i < cols; i++) {
        h_vector[i] = (float)rand();
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
    int num_blocks = (rows + threads_per_block - 1) / threads_per_block;
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

    printCSRMatrixHead(h_matrix);

    float geo_avg = geometric_mean(timers, NITER);
    int num_FLOPs = 2 * h_matrix.nnz;
    int num_bytes_accessed = (3 * rows + 3 * h_matrix.nnz) * 4;
    printStats(geo_avg, num_FLOPs, num_bytes_accessed);
    saveStatsToJson(geo_avg, num_FLOPs, num_bytes_accessed, filename, "gpu_csr_coalesced");

    // Free allocated memory
    cudaFree(d_vector);
    cudaFree(d_results);
    freeCSRMatrixCuda(d_matrix);
    std::free(h_vector);
    std::free(h_results);
    freeCSRMatrix(h_matrix);

    return 0;
}

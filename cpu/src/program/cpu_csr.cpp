#include "read.h"
#include "timers.h"
#include "utils.h"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WARMUP 3
#define NITER 10

// Function to multiply CSR matrix by a vector
void multiplyMatrixVector(CSRMatrix matrix, float *vector, float *result) {

    // Iterate over each row in the matrix
    for (int i = 0; i < matrix.numRows; i++) {
        float sum = 0.0;

        // Start index of the current row
        int start = matrix.rowPtrs[i];
        // End index of the current row
        int end = matrix.rowPtrs[i + 1];

        // Perform the sum for the current row
        for (int j = start; j < end; j++) {
            int col = matrix.cols[j];
            float val = matrix.values[j];

            // Multiply the non-zero element by the corresponding element in the
            // vector
            sum += val * vector[col];
        }

        // Store the result in the output vector y
        result[i] = sum;
    }
}

int main() {
    printf("Running!\n");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    // char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/36057_onetone1_sorted.mtx";
    // char filename[] = "../matrices/929901_Hardesty2_sorted.mtx";
    // char filename[] = "../matrices/923136_Emilia_923_sorted.mtx";
    // char filename[] = "../matrices/12057441_hugetrace_00010_sorted.mtx";
    // Read the matrix from file
    auto filenames = getFilenames();
    for (std::string filename : filenames) {

        CSRMatrix matrix = readMTXFileCSR(filename.c_str());
        int cols = matrix.numCols, rows = matrix.numRows;

        // Create vector of ones
        float *vector = (float *)malloc(cols * sizeof(float));
        if (vector == NULL) {
            fprintf(stderr, "Memory allocation failed for vector\n");
            freeCSRMatrix(matrix);
            exit(1);
        }

        // Seed the random number generator
        srand(42);
        for (int i = 0; i < cols; i++) {
            vector[i] = (float)rand();
        }

        TIMER_DEF;
        float timers[NITER];
        float *result = (float *)calloc(matrix.numRows, sizeof(float));
        for (int i = -WARMUP; i < NITER; i++) {
            // Multiply matrix by vector
            TIMER_START;
            multiplyMatrixVector(matrix, vector, result);
            TIMER_STOP;

            float iter_time = TIMER_ELAPSED;

            if (i >= 0) {
                timers[i] = iter_time;
                printf("%d iter_time %f\n", i, iter_time);
            }

            if (i != NITER - 1) {
                memset(result, 0, rows * sizeof(float));
            }
        }

        // Print the result
        // printf("Result vector after multiplication:\n");
        // for (int i = 0; i < rows; i++) {
        //     printf("%f\n", result[i]);
        // }

        // Create path for results file
        char result_path[256];
        sprintf(result_path, "../results/%s_result.txt",
                strrchr(filename.c_str(), '/') ? strrchr(filename.c_str(), '/') + 1 : filename.c_str());
        // printf("%s", result_path);
        // Write results to file
        writeVectorToFile(result_path, result, rows);

        float geo_avg = geometric_mean(timers, NITER);
        int num_FLOPs = 2 * matrix.nnz;
        int num_bytes_accessed = (3 * rows + 3 * matrix.nnz) * 4;
        printStats(geo_avg, num_FLOPs, num_bytes_accessed);
        saveStatsToJson(geo_avg, num_FLOPs, num_bytes_accessed, filename, "cpu_csr");

        // Free allocated memory
        free(vector);
        free(result);
        freeCSRMatrix(matrix);
    }
    return 0;
}

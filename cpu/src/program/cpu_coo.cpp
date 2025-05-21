#include "read.h"
#include "timers.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WARMUP 3
#define NITER 10

// Function to multiply matrix by vector
void multiplyMatrixVector(COOMatrix matrix, float *vector, float *result) {
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed for result vector\n");
        exit(1);
    }

    // Perform matrix-vector multiplication using COO format
    for (int i = 0; i < matrix.nnz; i++) {
        int row = matrix.rows[i];
        int col = matrix.cols[i];
        float val = matrix.values[i];

        result[row] += val * vector[col];
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
    auto filenames = getFilenames();
    for (std::string filename : filenames) { // Read the matrix from file
        COOMatrix matrix = readMTXFileCOO(filename.c_str());
        int cols = matrix.numCols, rows = matrix.numRows;

        // Create vector of ones
        float *vector = (float *)malloc(cols * sizeof(float));
        if (vector == NULL) {
            fprintf(stderr, "Memory allocation failed for vector\n");
            freeCOOMatrix(matrix);
            exit(1);
        }

        // Seed the random number generator
        srand(42);
        for (int i = 0; i < cols; i++) {
            vector[i] = (float)rand();
        }

        TIMER_DEF;
        float timers[NITER];
        // Allocate memory for result vector
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

        // Create path for results file
        char result_path[256];
        sprintf(result_path, "../results/%s_result.txt",
                strrchr(filename.c_str(), '/') ? strrchr(filename.c_str(), '/') + 1 : filename.c_str());
        // printf("%s", result_path);
        // Write results to file
        writeVectorToFile(result_path, result, rows);

        // Print the result
        // printf("Result vector after multiplication:\n");
        // for (int i = 0; i < rows; i++) {
        //     printf("%f\n", result[i]);
        // }

        float geo_avg = geometric_mean(timers, NITER);
        int num_FLOPs = 2 * matrix.nnz;
        int num_bytes_accessed = 5 * matrix.nnz * 4;
        printStats(geo_avg, num_FLOPs, num_bytes_accessed);
        saveStatsToJson(geo_avg, num_FLOPs, num_bytes_accessed, filename, "cpu_coo");

        // Free allocated memory
        free(vector);
        free(result);
        freeCOOMatrix(matrix);
    }
    return 0;
}

#include "read.h"
#include "timers.h"
#include "utils.h"
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WARMUP 3
#define NITER 10
// Function to multiply matrix by vector
void multiplyMatrixVector(float *matrix, float *vector, int rows, int cols, float *result) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

int main() {

    printf("Running!");
    // Read the matrix from file
    auto filenames = getFilenames();
    for (std::string filename : filenames) {
        int rows, cols, nnz;
        float *matrix = readMTXFile(filename.c_str(), &rows, &cols, &nnz);
        if (matrix == NULL) {
            continue;
        }
        // Create vector of ones
        float *vector = (float *)malloc(cols * sizeof(float));
        if (vector == NULL) {
            fprintf(stderr, "Memory allocation failed for vector\n");
            free(matrix);
            exit(1);
        }

        for (int i = 0; i < cols; i++) {
            vector[i] = 1.0;
        }

        TIMER_DEF;
        float timers[NITER];
        float *result = (float *)malloc(rows * sizeof(float));
        if (result == NULL) {
            fprintf(stderr, "Memory allocation failed for result vector\n");
            exit(1);
        }

        for (int i = -WARMUP; i < NITER; i++) {
            // Multiply matrix by vector
            TIMER_START;
            multiplyMatrixVector(matrix, vector, rows, cols, result);
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

        float geo_avg = geometric_mean(timers, NITER);
        int num_FLOPs = 2 * rows * cols;
        int num_bytes_accessed = (rows * cols * 2 + rows) * 4;
        printStats(geo_avg, num_FLOPs, num_bytes_accessed);
        saveStatsToJson(geo_avg, num_FLOPs, num_bytes_accessed, filename, "cpu_naive");

        // Free allocated memory
        free(matrix);
        free(vector);
        free(result);
    }
    return 0;
}

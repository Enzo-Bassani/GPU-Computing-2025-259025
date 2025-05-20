#include "read.h"
#include "timers.h"
#include "utils.h"
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WARMUP 3
#define NITER 10

// Define a properly packed struct to avoid padding
#pragma pack(push, 1) // Set 1-byte alignment
typedef struct {
    int row;
    int col;
    float val;
} COOUnit;
#pragma pack(pop) // Restore original alignment

// Function to multiply matrix by vector
float *multiplyMatrixVector(COOUnit *matrix, int nnz, float *vector) {
    // Allocate memory for result vector
    float *result = (float *)calloc(nnz, sizeof(float));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed for result vector\n");
        exit(1);
    }

    // Perform matrix-vector multiplication using COO format
    for (int i = 0; i < nnz; i++) {
        result[matrix[i].row] += matrix[i].val * vector[matrix[i].col];
    }

    return result;
}

int main() {
    printf("Running!\n");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    // char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/36057_onetone1_sorted.mtx";
    // char filename[] = "../matrices/929901_Hardesty2_sorted.mtx";
    // char filename[] = "../matrices/923136_Emilia_923_sorted.mtx";
    // Read the matrix from file

    auto filenames = getFilenames();
    for (std::string filename : filenames) { // Read the matrix from file
        COOMatrix matrix = readMTXFileCOO(filename.c_str());
        int cols = matrix.numCols, rows = matrix.numRows;
        // Transform into vector of COOUnits
        // Transform into vector of COOUnits
        COOUnit *matrix_struct = (COOUnit *)malloc(matrix.nnz * sizeof(COOUnit));
        if (matrix_struct == NULL) {
            fprintf(stderr, "Memory allocation failed for matrix data\n");
            exit(1);
        }

        printf("Size of packed COOUnit struct: %lu bytes\n", sizeof(COOUnit));

        // Copy data into packed structs
        for (int i = 0; i < matrix.nnz; i++) {
            matrix_struct[i].row = matrix.rows[i];
            matrix_struct[i].col = matrix.cols[i];
            matrix_struct[i].val = matrix.values[i];
        }
        freeCOOMatrix(matrix);

        // Create vector of ones
        float *vector = (float *)malloc(cols * sizeof(float));
        if (vector == NULL) {
            fprintf(stderr, "Memory allocation failed for vector\n");
            free(matrix_struct);
            exit(1);
        }

        for (int i = 0; i < cols; i++) {
            vector[i] = 1.0;
        }

        TIMER_DEF;
        float timers[NITER];
        float *result;
        for (int i = -WARMUP; i < NITER; i++) {
            // Multiply matrix by vector
            TIMER_START;
            result = multiplyMatrixVector(matrix_struct, matrix.nnz, vector);
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
        saveStatsToJson(geo_avg, num_FLOPs, num_bytes_accessed, filename, "cpu_coo_struct");

        // Free allocated memory
        free(vector);
        free(result);
        free(matrix_struct);
    }
    return 0;
}

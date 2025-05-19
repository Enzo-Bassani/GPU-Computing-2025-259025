#include "read.h"
#include "timers.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to multiply matrix by vector
float *multiplyMatrixVector(COOMatrix matrix, float *vector) {
    // Allocate memory for result vector
    float *result = (float *)calloc(matrix.numRows, sizeof(float));
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

    return result;
}

int main() {
    printf("Running!\n");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    // char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/36057_onetone1_sorted.mtx";
    char filename[] = "../matrices/929901_Hardesty2_sorted.mtx";
    // Read the matrix from file
    COOMatrix matrix = readMTXFileCOO(filename);
    int cols = matrix.numCols, rows = matrix.numRows;

    // Create vector of ones
    float *vector = (float *)malloc(cols * sizeof(float));
    if (vector == NULL) {
        fprintf(stderr, "Memory allocation failed for vector\n");
        freeCOOMatrix(matrix);
        exit(1);
    }

    for (int i = 0; i < cols; i++) {
        vector[i] = 1.0;
    }

    // Multiply matrix by vector
    TIMER_DEF;
    TIMER_START;
    float *result = multiplyMatrixVector(matrix, vector);
    TIMER_STOP;

    float elapsed_time = TIMER_ELAPSED;

    // Create path for results file
    char result_path[256];
    sprintf(result_path, "../results/%s_result.txt", strrchr(filename, '/') ? strrchr(filename, '/') + 1 : filename);
    // printf("%s", result_path);
    // Write results to file
    writeVectorToFile(result_path, result, rows);

    // Print the result
    // printf("Result vector after multiplication:\n");
    // for (int i = 0; i < rows; i++) {
    //     printf("%f\n", result[i]);
    // }

    printf("Elapsed time: %.6fms\n", elapsed_time);

    // Free allocated memory
    free(vector);
    free(result);
    freeCOOMatrix(matrix);
    return 0;
}

#include "read.h"
#include "timers.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to multiply CSR matrix by a vector
float *multiplyMatrixVector(CSRMatrix matrix, float *vector) {
    // Allocate memory for result vector
    float *result = (float *)calloc(matrix.numRows, sizeof(float));
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

    return result;
}

int main() {
    printf("Running!\n");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/36057_onetone1_sorted.mtx";
    // char filename[] = "../matrices/929901_Hardesty2_sorted.mtx";
    // char filename[] = "../matrices/923136_Emilia_923_sorted.mtx";
    // char filename[] = "../matrices/12057441_hugetrace_00010_sorted.mtx";
    // Read the matrix from file
    CSRMatrix matrix = readMTXFileCSR(filename);
    int cols = matrix.numCols, rows = matrix.numRows;

    // Create vector of ones
    float *vector = (float *)malloc(cols * sizeof(float));
    if (vector == NULL) {
        fprintf(stderr, "Memory allocation failed for vector\n");
        freeCSRMatrix(matrix);
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

    // Print the result
    // printf("Result vector after multiplication:\n");
    // for (int i = 0; i < rows; i++) {
    //     printf("%f\n", result[i]);
    // }

    // Create path for results file
    char result_path[256];
    sprintf(result_path, "../results/%s_result.txt", strrchr(filename, '/') ? strrchr(filename, '/') + 1 : filename);
    // printf("%s", result_path);
    // Write results to file
    writeVectorToFile(result_path, result, rows);
    printf("Elapsed time: %.6fms\n", elapsed_time);

    // Free allocated memory
    free(vector);
    free(result);
    freeCSRMatrix(matrix);
    return 0;
}

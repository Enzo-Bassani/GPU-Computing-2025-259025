#include <stdio.h>
#include <stdlib.h>
#include "read.h"
#include "timers.h"

// Function to multiply matrix by vector
double* multiplyMatrixVector(COOMatrix matrix, double* vector) {
    // Allocate memory for result vector
    double* result = (double*)calloc(matrix.numRows, sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed for result vector\n");
        exit(1);
    }

    // Perform matrix-vector multiplication using COO format
    for (int i = 0; i < matrix.nnz; i++) {
        int row = matrix.rows[i];
        int col = matrix.cols[i];
        double val = matrix.values[i];

        result[row] += val * vector[col];
    }

    return result;
}

int main() {
    printf("Running!\n");


    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    // char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    char filename[] = "../matrices/35588_bcsstk31_sorted.mtx";
    // char filename[] = "../matrices/12057441_hugetrace_00010_sorted.mtx";
    // Read the matrix from file
    COOMatrix matrix = readMTXFileCOO(filename);
    int cols = matrix.numCols, rows = matrix.numRows;

    // Create vector of ones
    double* vector = (double*)malloc(cols * sizeof(double));
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
    double* result = multiplyMatrixVector(matrix, vector);
    TIMER_STOP;

    double elapsed_time = TIMER_ELAPSED;

    // Print the result
    printf("Result vector after multiplication:\n");
    for (int i = 0; i < rows; i++) {
        printf("%f\n", result[i]);
    }

    printf("Elapsed time: %.5f\n", elapsed_time);

    // Free allocated memory
    free(vector);
    free(result);
    freeCOOMatrix(matrix);
    return 0;
}

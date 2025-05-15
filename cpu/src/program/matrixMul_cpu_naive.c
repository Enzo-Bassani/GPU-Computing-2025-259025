#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read.h"

// Function to multiply matrix by vector
double* multiplyMatrixVector(double* matrix, double* vector, int rows, int cols) {
    double* result = (double*)malloc(rows * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed for result vector\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }

    return result;
}

int main() {
    int rows, cols, nnz;

    printf("Running!");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/35588_bcsstk31_sorted.mtx";
    // char filename[] = "../matrices/12057441_hugetrace_00010_sorted.mtx";
    // Read the matrix from file
    double* matrix = readMTXFile(filename, &rows, &cols, &nnz);

    // Create vector of ones
    double* vector = (double*)malloc(cols * sizeof(double));
    if (vector == NULL) {
        fprintf(stderr, "Memory allocation failed for vector\n");
        free(matrix);
        exit(1);
    }

    for (int i = 0; i < cols; i++) {
        vector[i] = 1.0;
    }

    // Multiply matrix by vector
    double* result = multiplyMatrixVector(matrix, vector, rows, cols);

    // Print the result
    printf("Result vector after multiplication:\n");
    for (int i = 0; i < rows; i++) {
        printf("%f\n", result[i]);
    }

    // Free allocated memory
    free(matrix);
    free(vector);
    free(result);

    return 0;
}

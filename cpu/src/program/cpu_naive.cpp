#include "read.h"
#include "timers.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to multiply matrix by vector
float *multiplyMatrixVector(float *matrix, float *vector, int rows, int cols) {
    float *result = (float *)malloc(rows * sizeof(float));
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

    printf("Running!");

    // char filename[] = "../matrices/1138_bus_sorted.mtx";
    // char filename[] = "../matrices/4884_bcsstk16_sorted.mtx";
    char filename[] = "../matrices/10974_bcsstk17_sorted.mtx";
    // char filename[] = "../matrices/35588_bcsstk31_sorted.mtx";
    // char filename[] = "../matrices/12057441_hugetrace_00010_sorted.mtx";
    // Read the matrix from file
    int rows, cols, nnz;
    float *matrix = readMTXFile(filename, &rows, &cols, &nnz);

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

    // Multiply matrix by vector

    // Multiply matrix by vector
    TIMER_DEF;
    TIMER_START;
    float *result = multiplyMatrixVector(matrix, vector, rows, cols);
    TIMER_STOP;

    double elapsed_time = TIMER_ELAPSED;



    // Create path for results file
    char result_path[256];
    sprintf(result_path, "../results/%s_result.txt", strrchr(filename, '/') ? strrchr(filename, '/') + 1 : filename);
    // printf("%s", result_path);
    // Write results to file
    writeVectorToFile(result_path, result, rows);

    printf("Elapsed time: %.6fms\n", elapsed_time);
    // Print the result
    // printf("Result vector after multiplication:\n");
    // for (int i = 0; i < rows; i++) {
    //     printf("%f\n", result[i]);
    // }

    // Free allocated memory
    free(matrix);
    free(vector);
    free(result);

    return 0;
}

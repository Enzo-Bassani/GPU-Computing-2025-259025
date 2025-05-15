#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read.h"

// Function to read the matrix from the MTX file
double *readMTXFile(const char *filename, int *rows, int *cols, int *nnz) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    // Skip comment lines
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] != '%') {
            break;
        }
    }

    // Read matrix dimensions and number of non-zeros
    sscanf(line, "%d %d %d", rows, cols, nnz);

    // Allocate memory for the matrix (full size, initialized to zero)
    double *matrix = (double *)calloc((*rows) * (*cols), sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix\n");
        fclose(file);
        exit(1);
    }

    printf("Successfully allocated matrix");

    // Read non-zero elements
    for (int i = 0; i < *nnz; i++) {
        int row, col;
        double value;
        fscanf(file, "%d %d %lf", &row, &col, &value);

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        // Store value in the matrix
        matrix[row * (*cols) + col] = value;
    }

    fclose(file);
    return matrix;
}

// Function to read the matrix from the MTX file in COO format
COOMatrix readMTXFileCOO(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    // Skip comment lines
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] != '%') {
            break;
        }
    }

    // Read matrix dimensions and number of non-zeros
    int rows, cols, nnz;
    sscanf(line, "%d %d %d", &rows, &cols, &nnz);

    // Allocate memory for the COO matrix structure
    COOMatrix matrix;

    // Allocate memory for the arrays
    matrix.rows = (int *)malloc(nnz * sizeof(int));
    matrix.cols = (int *)malloc(nnz * sizeof(int));
    matrix.values = (double *)malloc(nnz * sizeof(double));

    if (matrix.rows == NULL || matrix.cols == NULL || matrix.values == NULL) {
        fprintf(stderr, "Memory allocation failed for COO matrix arrays\n");
        free(matrix.rows);
        free(matrix.cols);
        free(matrix.values);
        fclose(file);
        exit(1);
    }

    // Set matrix properties
    matrix.nnz = nnz;
    matrix.numRows = rows;
    matrix.numCols = cols;

    printf("Successfully allocated COO matrix structure\n");

    // Read non-zero elements
    for (int i = 0; i < nnz; i++) {
        int row, col;
        double value;
        fscanf(file, "%d %d %lf", &row, &col, &value);

        // Adjust for 1-based indexing in MTX format
        row--;
        col--;

        // Store values in the COO format
        matrix.rows[i] = row;
        matrix.cols[i] = col;
        matrix.values[i] = value;
    }

    fclose(file);
    return matrix;
}

// Function to free COO matrix memory
void freeCOOMatrix(COOMatrix matrix) {
    free(matrix.rows);
    free(matrix.cols);
    free(matrix.values);
}

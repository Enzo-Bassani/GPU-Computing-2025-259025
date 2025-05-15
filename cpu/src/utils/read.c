#include "read.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Function to read the matrix from the MTX file into CSR format
CSRMatrix readMTXFileCSR(const char *filename) {
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

    // Allocate memory for the CSR matrix structure
    CSRMatrix matrix;

    // Allocate memory for the arrays
    matrix.rowPtrs = (int *)malloc(
        (rows + 1) * sizeof(int)); // One extra for the last row pointer
    matrix.cols = (int *)malloc(nnz * sizeof(int));         // Column indices
    matrix.values = (double *)malloc(nnz * sizeof(double)); // Non-zero values

    if (matrix.rowPtrs == NULL || matrix.cols == NULL ||
        matrix.values == NULL) {
        fprintf(stderr, "Memory allocation failed for CSR matrix arrays\n");
        free(matrix.rowPtrs);
        free(matrix.cols);
        free(matrix.values);
        fclose(file);
        exit(1);
    }

    // Set matrix properties
    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    // Initialize row pointers to zero
    memset(matrix.rowPtrs, 0, (rows + 1) * sizeof(int));

    // Read non-zero elements and build the CSR format
    int elementIndex = 0;
    int row, col = 0;
    double value = 0;
    while (fscanf(file, "%d %d %lf", &row, &col, &value) != EOF) {
        row--; // Adjust for 1-based indexing
        col--; // Adjust for 1-based indexing

        // Increment the row pointer for the current row
        matrix.rowPtrs[row + 1]++;

        // Store the column index and the value
        matrix.cols[elementIndex] = col;
        matrix.values[elementIndex] = value;
        elementIndex++;
    }

    // Convert rowPtrs to contain actual starting indices of each row
    for (int i = 1; i <= rows; i++) {
        matrix.rowPtrs[i] += matrix.rowPtrs[i - 1];
    }

    fclose(file);
    return matrix;
}

// Function to free CSR matrix memory
void freeCSRMatrix(CSRMatrix matrix) {
    free(matrix.rowPtrs);  // Free the row pointers array
    free(matrix.cols);     // Free the column indices array
    free(matrix.values);   // Free the values array
}

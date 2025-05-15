#pragma once

double* readMTXFile(const char* filename, int* rows, int* cols, int* nnz);

// COO format structure
typedef struct {
    int *rows;      // Row indices
    int *cols;      // Column indices
    double *values; // Non-zero values
    int nnz;        // Number of non-zero elements
    int numRows;    // Total number of rows
    int numCols;    // Total number of columns
} COOMatrix;

COOMatrix readMTXFileCOO(const char *filename);

void freeCOOMatrix(COOMatrix matrix);

#pragma once

float* readMTXFile(const char* filename, int* rows, int* cols, int* nnz);

// COO format structure
typedef struct {
    int *rows;      // Row indices
    int *cols;      // Column indices
    float *values; // Non-zero values
    int nnz;        // Number of non-zero elements
    int numRows;    // Total number of rows
    int numCols;    // Total number of columns
} COOMatrix;

COOMatrix readMTXFileCOO(const char *filename);
void freeCOOMatrix(COOMatrix matrix);

// Structure to store the matrix in CSR format
typedef struct {
    int *rowPtrs;    // Array to store the row pointers
    int *cols;       // Array to store the column indices
    float *values;  // Array to store the non-zero values
    int numRows;     // Number of rows in the matrix
    int numCols;     // Number of columns in the matrix
    int nnz;         // Number of non-zero elements in the matrix
} CSRMatrix;

CSRMatrix readMTXFileCSR(const char *filename);
void freeCSRMatrix(CSRMatrix matrix);

void printCSRMatrixHead(CSRMatrix matrix);

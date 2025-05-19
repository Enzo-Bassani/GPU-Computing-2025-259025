#include "read.h"
#include "readcu.h"
#include <stdlib.h>

// Function to free COO matrix memory
void freeCOOMatrixCuda(COOMatrix matrix) {
    cudaFree(matrix.rows);
    cudaFree(matrix.cols);
    cudaFree(matrix.values);
}

// Function to free CSR matrix memory
void freeCSRMatrixCuda(CSRMatrix matrix) {
    cudaFree(matrix.rowPtrs); // Free the row pointers array
    cudaFree(matrix.cols);    // Free the column indices array
    cudaFree(matrix.values);  // Free the values array
}

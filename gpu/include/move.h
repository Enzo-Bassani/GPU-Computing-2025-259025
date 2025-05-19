#pragma once
#include "read.h"

void moveCSRToDevice(CSRMatrix &h, CSRMatrix &d);
float *moveArrayToDevice(float *h_array, int length);
void checkCudaError(cudaError_t err);
void moveCOOToDevice(COOMatrix &h, COOMatrix &d);

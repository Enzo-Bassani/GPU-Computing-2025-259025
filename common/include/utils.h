#pragma once

void writeVectorToFile(const char *filename, float *vector, int length);
void printStats(double milliseconds, int num_FLOPs, int num_bytes_acessed);

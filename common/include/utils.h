#pragma once
#include <string>
#include <vector>

void writeVectorToFile(const char *filename, float *vector, int length);
void printStats(double milliseconds, int num_FLOPs, int num_bytes_acessed);
std::vector<std::string> getFilenames();
void saveStatsToJson(double milliseconds, int num_FLOPs, int num_bytes_accessed, std::string matrix_file,
                     const std::string &algorithm_name);

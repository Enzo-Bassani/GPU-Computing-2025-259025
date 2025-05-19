#include <cstdio>
#include <stdio.h>
#include <stdlib.h>

// Function to write a vector to a file
void writeVectorToFile(const char *filename, float *vector, int length) {
    printf("Writing results to %s\n", filename);

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening output file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < length; i++) {
        fprintf(file, "%f\n", vector[i]);
    }

    fclose(file);
}

void printStats(double milliseconds, int num_FLOPs, int num_bytes_accessed) {
    double FLOP_per_sec = num_FLOPs / (milliseconds / 1000) / 1000000000;
    double bandwidth_gbps = (num_bytes_accessed / (milliseconds / 1000)) / 1000000000;

    printf("Stats ---\n");
    printf("Time(geo_avg)(ms): %f\n", milliseconds);
    printf("FLOP/s(bytes): %f\n", FLOP_per_sec);
    printf("Bandwidth(Gbps): %f\n", bandwidth_gbps);
    printf("Number of FLOPs: %d\n", num_FLOPs);
    printf("Number of bytes accessed: %d\n", num_bytes_accessed);
}

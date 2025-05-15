#include <stdio.h>
#include <stdlib.h>

// Function to write a vector to a file
void writeVectorToFile(const char *filename, double *vector, int length) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening output file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < length; i++) {
        fprintf(file, "%lf\n", vector[i]);
    }

    fclose(file);
}

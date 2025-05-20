#include <cstdio>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

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

std::string extractFilenameNoExtension(std::string filepath) {
    // Remove path if present
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        filepath = filepath.substr(lastSlash + 1);
    }

    // Remove extension if present
    size_t lastDot = filepath.find_last_of(".");
    if (lastDot != std::string::npos) {
        filepath = filepath.substr(0, lastDot);
    }

    return filepath;
}

void saveStatsToJson(double milliseconds, int num_FLOPs, int num_bytes_accessed, std::string matrix_file,
                     const std::string &algorithm_name) {
    double FLOP_per_sec = num_FLOPs / (milliseconds / 1000) / 1000000000;
    double bandwidth_gbps = (num_bytes_accessed / (milliseconds / 1000)) / 1000000000;

    // Create filename for the JSON output
    std::string matrix_name = extractFilenameNoExtension(matrix_file);
    std::string filename = "../stats/" + algorithm_name + matrix_name + "_stats.json";

    FILE *file = fopen(filename.c_str(), "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening output file %s\n", filename.c_str());
        exit(1);
    }

    // Manually construct JSON string
    fprintf(file, "{\n");
    fprintf(file, "  \"matrix\": \"%s\",\n", matrix_name.c_str());
    fprintf(file, "  \"algorithm\": \"%s\",\n", algorithm_name.c_str());
    fprintf(file, "  \"time_ms\": %f,\n", milliseconds);
    fprintf(file, "  \"gflops\": %f,\n", FLOP_per_sec);
    fprintf(file, "  \"bandwidth_gbps\": %f,\n", bandwidth_gbps);
    fprintf(file, "  \"total_flops\": %d,\n", num_FLOPs);
    fprintf(file, "  \"total_bytes_accessed\": %d\n", num_bytes_accessed);
    fprintf(file, "}\n");

    fclose(file);

    printf("Statistics for %s on matrix %s saved to %s\n", algorithm_name.c_str(), matrix_name.c_str(),
           filename.c_str());
}

std::vector<std::string> getFilenames() {
    std::vector<std::string> filenames;

    // Unix/Linux/Mac implementation
    DIR *dir;
    struct dirent *entry;

    if ((dir = opendir("../matrices/")) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            // Skip directories (including "." and "..")
            if (entry->d_type != DT_DIR) {
                std::string path = "../matrices/";
                path += entry->d_name;
                filenames.push_back(path);
            }
        }
        closedir(dir);
    }

    return filenames;
}

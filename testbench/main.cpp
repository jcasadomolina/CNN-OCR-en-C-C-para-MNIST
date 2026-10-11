#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <dirent.h>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include "fc_weights.h"
#include "conv_weights.h"
#include "params.h"
#include "top_cnn.h"

// Maximum number of test files to handle
#define MAX_TEST_FILES 1000

// Structure to hold file information
typedef struct {
    char path[256];
    int label;
} TestFile;

// Collect files from subdirectories 0/..9/
int collect_files(const char* base_dir, TestFile* files) {
    int count = 0;
    char path[256];
    
    for (int label = 0; label <= 9 && count < MAX_TEST_FILES; ++label) {
        snprintf(path, sizeof(path), "%s/%d", base_dir, label);
        DIR* dir = opendir(path);
        if (!dir) continue;
        
        dirent* entry;
        while ((entry = readdir(dir)) != nullptr && count < MAX_TEST_FILES) {
            if (strstr(entry->d_name, ".txt")) {
                int written = snprintf(files[count].path, sizeof(files[count].path), "%s/%s", path, entry->d_name);
                if (written < 0 || written >= sizeof(files[count].path)) {
                    std::cerr << "Path truncated for file: " << entry->d_name << std::endl;
                    continue;
                }
                files[count].label = label;
                count++;
            }
        }
        closedir(dir);
    }
    return count;
}

int main(int argc, char **argv) {
    const char* dataset_dir = argc > 1 ? argv[1] : "mnist_txt_1000";
    TestFile files[MAX_TEST_FILES];
    int file_count = collect_files(dataset_dir, files);
    if (file_count == 0) {
        std::cerr << "No test images found in: " << dataset_dir << std::endl;
        return 1;
    }
    int total_tests = 0;
    int correct_predictions = 0;

    auto global_start = std::chrono::steady_clock::now();

    for (int i = 0; i < file_count; i++) {
        FILE* infile = fopen(files[i].path, "r");
        if (!infile) {
            std::cerr << "Couldn't open: " << files[i].path << std::endl;
            continue;
        }

        float image[IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS] = {0};
        for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS; j++) {
            if (fscanf(infile, "%f,", &image[j]) != 1) {
                std::cerr << "Error reading image data from file" << std::endl;
                fclose(infile);
                return 1;
            }
        }
        fclose(infile);

        float fc_out[FC_ACT_SIZE];
        top_cnn(image, fc_out, conv_weights, conv_bias, fc_weights, fc_bias);

        int prediction = 0;
        for (int j = 1; j < FC_ACT_SIZE; j++) {
            if (fc_out[j] > fc_out[prediction]) {
                prediction = j;
            }
        }

        if (prediction == files[i].label) {
            correct_predictions++;
        } else {
            std::cout << "Misclassified: " << files[i].path << " -> predicted " << prediction
                      << ", true " << files[i].label << std::endl;
        }

        total_tests++;
    }

    auto global_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(global_end - global_start).count();

    std::cout << "###################### SUMMARY ######################" << std::endl;
    std::cout << "Total images tested: " << total_tests << std::endl;
    std::cout << "Correct predictions: " << correct_predictions << std::endl;
    std::cout << "Accuracy: " << 100.0 * correct_predictions / total_tests << "%" << std::endl;
    std::cout << "Total execution time: " << duration << " ms" << std::endl;

    return 0;
}
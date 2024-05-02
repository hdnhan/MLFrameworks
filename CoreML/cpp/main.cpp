#include <stdlib.h>

#include <fstream>
#include <iostream>

#include "bindings.h"
using namespace std;

int main() {
    const void *model = load("/Users/nhan/dev/CVExamples/CoreML/src/yolov8n.mlmodelc");
    if (model == NULL) {
        cout << "Failed to load model" << endl;
        return 1;
    }
    float *input = (float *)malloc(sizeof(float) * 1 * 3 * 640 * 640);
    // random input from 0 -> 255 and write to file txt
    ofstream file;
    file.open("/Users/nhan/dev/CVExamples/CoreML/src/input.txt");
    for (int i = 0; i < 1 * 3 * 640 * 640; ++i) {
        input[i] = (float)(rand() % 256);
        file << input[i] << " ";
    }
    file.close();

    float *output = (float *)malloc(sizeof(float) * 1 * 84 * 8400);
    predict(model, input, output);

    // write output to file txt
    ofstream wfile;
    wfile.open("/Users/nhan/dev/CVExamples/CoreML/src/output.txt");
    for (int i = 0; i < 1 * 84 * 8400; ++i) {
        wfile << output[i] << " ";
    }
    wfile.close();

    free(input);
    free(output);
    return 0;
}
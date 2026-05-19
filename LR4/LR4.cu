#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <vector>

unsigned char* loadBMP(const char* path, int* w, int* h) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

    *w = img.cols;
    *h = img.rows;

    unsigned char* data = new unsigned char[(*w) * (*h)];
    std::memcpy(data, img.data, (*w) * (*h));
    return data;
}

bool saveBMP(const char* path, const unsigned char* data, int w, int h) {
    cv::Mat img(h, w, CV_8UC1, const_cast<unsigned char*>(data));
    return cv::imwrite(path, img);
}


int main(int argc, char** argv) {
    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];
    float threshold = std::atof(argv[3]);

    int width=0, height=0;
    unsigned char* input = loadBMP(inputPath, &width, &height);

    saveBMP(outputPath, &width, &height);

    delete[] input;
    return 0;
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>

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
    const char* inputPath  = "/content/noisy.bmp";
    const char* outputPath = "/content/result.bmp";

    int width = 0, height = 0;
    unsigned char* input = loadBMP(inputPath, &width, &height);
    unsigned char* output = new unsigned char[width * height];

    saveBMP(outputPath, output, width, height);

    delete[] input;
    delete[] output;
    return 0;
}
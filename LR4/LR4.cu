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

void saveImageWithCross(const char* path, const unsigned char* gray, const std::vector<cv::Point>& corners, int w, int h) {
    cv::Mat img(h, w, CV_8UC1, const_cast<unsigned char*>(gray));
    cv::Mat color;

    cv::cvtColor(img, color, cv::COLOR_GRAY2BGR);

    for (const auto& pt : corners){
        cv::drawMarker(color, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 15, 2);
    }

    cv::imwrite(path, color);
}


float px(const unsigned char* input, int x, int y, int width, int height) {
    x = std::max(0, std::min(x, width  - 1));
    y = std::max(0, std::min(y, height - 1));
    return (float)input[y * width + x];
}

double harrisCPU(const unsigned char* input, std::vector<cv::Point>& corners, int width, int height, float threshold, float alpha)
{
    clock_t t0 = clock();

    std::vector<float> response(width * height);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float Sxx = 0, Sxy = 0, Syy = 0;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int x = col + dx, y = row + dy;

                    float Ix = (px(input, x+1,y-1, width,height) + 2*px(input, x+1,y, width,height) + px(input, x+1,y+1, width,height)
                                - px(input, x-1,y-1, width,height) - 2*px(input, x-1,y, width,height) - px(input, x-1,y+1, width,height)) / 8;
                    float Iy = (px(input, x-1,y+1, width,height) + 2*px(input, x,y+1, width,height) + px(input, x+1,y+1, width,height)
                                - px(input, x-1,y-1, width,height) - 2*px(input, x,y-1, width,height) - px(input, x+1,y-1, width,height)) / 8;

                    Sxx += Ix * Ix;
                    Sxy += Ix * Iy;
                    Syy += Iy * Iy;
                }
            }

            float det   = Sxx * Syy - Sxy * Sxy;
            float trace = Sxx + Syy;
            response[row * width + col] = det - alpha * trace * trace;
        }
    }

    corners.clear();

    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            float r = response[row * width + col];
            if (r <= threshold) continue;

            bool isMax = true;
            for (int dy = -1; dy <= 1 && isMax; dy++){
                for (int dx = -1; dx <= 1 && isMax; dx++){
                    if (!(dx == 0 && dy == 0)){
                        isMax = r > response[(row+dy) * width + (col+dx)];
                    }
                }
            }
            if (isMax) corners.push_back({col, row});
        }
    }

    return (double)(clock() - t0) / CLOCKS_PER_SEC * 1000;
}


int main(int argc, char** argv) {
    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];
    float threshold = std::atof(argv[3]);
    float alpha = 0.05;

    int width=0, height=0;
    unsigned char* input = loadBMP(inputPath, &width, &height);

    std::vector<cv::Point> cpuCorners;

    double cpuTime = harrisCPU(input, cpuCorners, width, height, threshold, alpha);
    std::cout << "CPU time: " << cpuTime << " ms" << std::endl;

    saveImageWithCross(outputPath, input, cpuCorners, width, height);

    delete[] input;
    return 0;
}
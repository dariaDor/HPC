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


__device__ float px_gpu(cudaTextureObject_t texObj, int x, int y) {
    return tex2D<unsigned char>(texObj, x, y);
}

__global__ void harrisKernel(cudaTextureObject_t texObj, float* response, int width, int height, float alpha){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;



    float Sxx = 0, Sxy = 0, Syy = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int x = col + dx, y = row + dy;

            float Ix = (-px_gpu(texObj,x-1,y-1) - 2*px_gpu(texObj,x-1,y) - px_gpu(texObj,x-1,y+1)
                        +px_gpu(texObj,x+1,y-1) + 2*px_gpu(texObj,x+1,y) + px_gpu(texObj,x+1,y+1)) / 8;

            float Iy = (-px_gpu(texObj,x-1,y-1) - 2*px_gpu(texObj,x,y-1) - px_gpu(texObj,x+1,y-1)
                        +px_gpu(texObj,x-1,y+1) + 2*px_gpu(texObj,x,y+1) + px_gpu(texObj,x+1,y+1)) / 8;

            Sxx += Ix * Ix;
            Sxy += Ix * Iy;
            Syy += Iy * Iy;
        }
    }

    float det   = Sxx * Syy - Sxy * Sxy;
    float trace = Sxx + Syy;
    response[row * width + col] = det - alpha * trace * trace;
}


__global__ void harrisNMSKernel(const float* response, int* cornerX, int* cornerY, int* cornerCount, int maxCorners, int width, int height, float threshold){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    float r = response[row * width + col];
    if (r <= threshold) return;

    for (int dy = -1; dy <= 1; dy++){
        for (int dx = -1; dx <= 1; dx++) {

            if (dx == 0 && dy == 0) continue;

            int nx = col + dx, ny = row + dy;

            if ((nx >= 0 && nx < width && ny >= 0 && ny < height) && response[ny * width + nx] >= r) return;
        }
    }

    int idx = atomicAdd(cornerCount, 1);
    if (idx < maxCorners) {
        cornerX[idx] = col;
        cornerY[idx] = row;
    }
}

float harrisGPU(const unsigned char* h_input, std::vector<cv::Point>& corners, int width, int height, float threshold, float alpha) {
    const int MAX_CORNERS = 100000;
    size_t bytes = (size_t)width * height;

    unsigned char* d_input = nullptr;
    float* d_response = nullptr;
    int *d_cornerX=nullptr, *d_cornerY=nullptr, *d_count=nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_response, bytes * sizeof(float));
    cudaMalloc(&d_cornerX, MAX_CORNERS * sizeof(int));
    cudaMalloc(&d_cornerY, MAX_CORNERS * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaArray_t cuArray;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(8,0,0,0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&cuArray, &desc, width, height);
    cudaMemcpy2DToArray(cuArray,0,0, d_input, width, width, height, cudaMemcpyDeviceToDevice);

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    dim3 block(16, 16);
    dim3 grid((width+15)/16, (height+15)/16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    harrisKernel<<<grid,block>>>(texObj, d_response, width, height, alpha);
    cudaDeviceSynchronize();

    harrisNMSKernel<<<grid,block>>>(d_response, d_cornerX, d_cornerY, d_count, MAX_CORNERS, width, height, threshold);
    cudaDeviceSynchronize();

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms=0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    h_count = std::min(h_count, MAX_CORNERS);

    std::vector<int> hX(h_count), hY(h_count);
    cudaMemcpy(hX.data(), d_cornerX, h_count*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hY.data(), d_cornerY, h_count*sizeof(int), cudaMemcpyDeviceToHost);

    corners.clear();
    for (int i=0; i<h_count; i++){
        corners.push_back({hX[i], hY[i]});
    }

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_input); cudaFree(d_response);
    cudaFree(d_cornerX); cudaFree(d_cornerY); cudaFree(d_count);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    return ms;
}

bool isResultsMatch(const std::vector<cv::Point>& gpuCorners, const std::vector<cv::Point>& cpuCorners, int width, int height){
    std::vector<bool> gpuMask(width * height, false);
    std::vector<bool> cpuMask(width * height, false);

    for(const cv::Point& p : gpuCorners){
      gpuMask[p.y * width + p.x] = true;
    }

    for(const cv::Point& p : cpuCorners){
      cpuMask[p.y * width + p.x] = true;
    }

    int diffs = 0;
    for (int i = 0; i < width * height; i++){
      if (gpuMask[i] != cpuMask[i]) diffs++;
    }

    return diffs == 0;
}


int main(int argc, char** argv) {
    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];
    float threshold = std::atof(argv[3]);
    float alpha = 0.05;

    int width=0, height=0;
    unsigned char* input = loadBMP(inputPath, &width, &height);

    std::vector<cv::Point> cpuCorners, gpuCorners;

    float gpuTime = harrisGPU(input, gpuCorners, width, height, threshold, alpha);
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;
   
    double cpuTime = harrisCPU(input, cpuCorners, width, height, threshold, alpha);
    std::cout << "CPU time: " << cpuTime << " ms" << std::endl;

    std::cout << "CPU == GPU: " << (isResultsMatch(gpuCorners, cpuCorners, width, height) ? "true" : "false") << std::endl;

    saveImageWithCross(outputPath, input, cpuCorners, width, height);

    delete[] input;
    return 0;
}
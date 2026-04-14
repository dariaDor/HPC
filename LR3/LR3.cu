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

__global__ void medianFilterKernel(cudaTextureObject_t texObj, unsigned char* __restrict__ output, int width, int height){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    unsigned char window[25];
    int k = 0;

    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
            window[k++] = tex2D<unsigned char>(texObj, col + dx, row + dy);

    for (int i = 0; i < 25; ++i)
        for (int j = i + 1; j < 25; ++j)
            if (window[i] > window[j]) {
                unsigned char t = window[i];
                window[i] = window[j];
                window[j] = t;
            }

    output[row * width + col] = window[12];
}

float runMedianFilterGPU(const unsigned char* h_input, unsigned char* h_output, int width, int height){
  float runMedianFilterGPU(const unsigned char* h_input, unsigned char* h_output, int width, int height, int passes){
    size_t bytes = (size_t)width * height;

    unsigned char *d_buf1 = nullptr, *d_buf2 = nullptr;
    cudaMalloc(&d_buf1, bytes);
    cudaMalloc(&d_buf2, bytes);
    cudaMemcpy(d_buf1, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
        
    cudaArray_t cuArray;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

    cudaMallocArray(&cuArray, &desc, width, height);
    cudaMemcpy2DToArray(cuArray, 0, 0, d_buf1, width,width, height, cudaMemcpyDeviceToDevice);

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

    medianFilterKernel<<<grid, block>>>(texObj, d_buf2, width, height);
    cudaDeviceSynchronize();

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);

    unsigned char* tmp = d_buf1;
    d_buf1 = d_buf2;
    d_buf2 = tmp;

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(h_output, d_buf1, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return ms;
}

int main(int argc, char** argv) {
    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];

    int width = 0, height = 0;
    unsigned char* input = loadBMP(inputPath, &width, &height);

    unsigned char* output = new unsigned char[width * height];

    float gpuTime = runMedianFilterGPU(input, output, width, height);
    std::cout << "GPU time: " << gpuTime << " ms" << std::endl;

    saveBMP(outputPath, output, width, height);

    delete[] input;
    delete[] output;
    return 0;
}
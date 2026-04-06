#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void CPU_vsum(const int* A, int* result, int N) {
    int sum = 0;

    for (int i = 0; i < N; ++i) {
        sum += A[i];
    }

    *result = sum;
}


__global__ void GPU_vsum_kernel(const int* A, int* partial, int N) {

    extern __shared__ int shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (i < N) ? A[i] : 0;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

void GPU_vsum(const int* d_A, int* d_result, int N) {

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    int* d_partial;

    cudaMalloc(&d_partial, gridSize * sizeof(int));

    GPU_vsum_kernel<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_A, d_partial, N);

    int* h_partial = (int*)malloc(gridSize * sizeof(int));

    cudaMemcpy(h_partial, d_partial, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;

    for (int i = 0; i < gridSize; ++i) {
        sum += h_partial[i];
    }

    cudaMemcpy(d_result, &sum, sizeof(int), cudaMemcpyHostToDevice);

    cudaFree(d_partial);
    free(h_partial);
}


void fillRandVector(int* V, int N, int maxVal = 10) {
    for (int i = 0; i < N; ++i) {
        V[i] = rand() % maxVal;
    }
}

double recordEvent(clock_t& t0) {
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
    t0 = clock();
    return elapsed;
}


int main() {

    srand(time(NULL));

    const int vectorSizes[] = {
        1,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
    }; 

    for (int s = 0; s < size(vectorSizes); ++s) {

        int N = vectorSizes[s];
 
        cout << "Vector size: " << N << endl;

        size_t memSize = sizeof(int) * N;

        int* A = (int*)malloc(memSize);

        int cpu_result = 0;
        int gpu_result = 0;

        fillRandVector(A, N);
        clock_t t0 = clock();
        CPU_vsum(A, &cpu_result, N);
        double cpuTime = recordEvent(t0);

        int *d_A, *d_result;

        cudaMalloc(&d_A, memSize);
        cudaMalloc(&d_result, sizeof(int));

        cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);

        t0 = clock();
        GPU_vsum(d_A, d_result, N);
        double gpuTime = recordEvent(t0);

        cudaMemcpy(&gpu_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        bool correct = (cpu_result == gpu_result);

        cout << "CPU time: " << cpuTime << " ms" << endl;
        cout << "GPU time: " << gpuTime << " ms" << endl;
        cout << "Acceleration: " << cpuTime / gpuTime << endl;
        cout << "Correctness: " << (correct ? "OK" : "FAIL") << endl;

         free(A);

        cudaFree(d_A);
        cudaFree(d_result);

    }

    return 0;
}
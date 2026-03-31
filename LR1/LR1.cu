#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void CPU_mmul(const int* A, const int* B, int* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i + k * N] * B[k + j * N];
            }
            C[i + j * N] = sum;
        }
    }
}

__global__ void GPU_mmul_kernel(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row + k * N] * B[k + col * N];
        }
        C[row + col * N] = sum;
    }
}


void GPU_mmul(const int* d_A, const int* d_B, int* d_C, int N) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    GPU_mmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}

bool checkCorrectness(const int* C1, const int* C2, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (C1[i] != C2[i]) return false;
    }
    return true;
}

void fillRandMatrix(int* M, int N, int maxVal = 10) {
    for (int i = 0; i < N * N; ++i) {
        M[i] = rand() % maxVal;
    }
}

double recordEvent(clock_t& t0) {
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
    t0 = clock();
    return elapsed;
}

int main() {
    srand(time(NULL));

    const int sizes[] = {1, 100, 512, 1024, 2000};

    for (int s = 0; s < 5; ++s) {
        int N = sizes[s];
        cout << "Matrix size: " << N << "x" << N << endl;

        size_t memSize = sizeof(int) * N * N;
        int* A = (int*)malloc(memSize);
        int* B = (int*)malloc(memSize);
        int* C_cpu = (int*)malloc(memSize);
        int* C_gpu = (int*)malloc(memSize);

        fillRandMatrix(A, N);
        fillRandMatrix(B, N);

        clock_t t0 = clock();
        CPU_mmul(A, B, C_cpu, N);
        double cpuTime = recordEvent(t0);

        int *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, memSize);
        cudaMalloc(&d_B, memSize);
        cudaMalloc(&d_C, memSize);

        cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, memSize, cudaMemcpyHostToDevice);

        t0 = clock();
        GPU_mmul(d_A, d_B, d_C, N);
        double gpuTime = recordEvent(t0);

        cudaMemcpy(C_gpu, d_C, memSize, cudaMemcpyDeviceToHost);

        bool correct = checkCorrectness(C_cpu, C_gpu, N);

        cout << "CPU time: " << cpuTime << " ms" << endl;
        cout << "GPU time: " << gpuTime << " ms" << endl;
        cout << "Acceleration: " << cpuTime / gpuTime << endl;
        cout << "Correctness: " << (correct ? "OK" : "FAIL") << endl;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        free(A); free(B); free(C_cpu); free(C_gpu);

    }
    return 0;
}
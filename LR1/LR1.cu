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

    int N = 100;
    cout << "Matrix size: " << N << "x" << N << endl;

    size_t memSize = sizeof(int) * N * N;
    int* A = (int*)malloc(memSize);
    int* B = (int*)malloc(memSize);
    int* C_cpu = (int*)malloc(memSize);

    fillRandMatrix(A, N);
    fillRandMatrix(B, N);

    clock_t t0 = clock();
    CPU_mmul(A, B, C_cpu, N);
    double cpuTime = recordEvent(t0);

    cout << "CPU time: " << cpuTime << " ms" << endl;

    free(A); free(B); 
    free(C_cpu);
    return 0;
}
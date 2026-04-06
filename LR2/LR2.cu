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
        1000,
        10000,
        100000,
        1000000,
    };

    for (int s = 0; s < size(vectorSizes); ++s) {

        int N = vectorSizes[s];
 
        cout << "Vector size: " << N << endl;

        size_t memSize = sizeof(int) * N;

        int* A = (int*)malloc(memSize);

        int cpu_result = 0;

        fillRandVector(A, N);
        clock_t t0 = clock();
        CPU_vsum(A, &cpu_result, N);
        double cpuTime = recordEvent(t0);

        cout << "CPU time: " << cpuTime << " ms" << endl;

         free(A);

    }

    return 0;
}
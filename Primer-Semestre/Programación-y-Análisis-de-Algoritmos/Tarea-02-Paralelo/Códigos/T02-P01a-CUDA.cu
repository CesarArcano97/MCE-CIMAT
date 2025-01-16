#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// Kernel para calcular S1
__global__ void calculateS1(const double* V, double* S1, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Índice global del hilo
    if (idx < N - 1) {
        S1[idx] = V[idx] + V[idx + 1];
    }
}

int main() {
    const int N = 50000;
    std::vector<double> V(N);

    // Semilla para la aleatoriedad
    std::srand(std::time(0));

    // Llenar el vector V con valores aleatorios
    for (int i = 0; i < N; i++) {
        V[i] = static_cast<double>(std::rand() % 100) + (std::rand() % 100 / 100.0);
    }

    // Espacio para el vector S1
    std::vector<double> S1(N - 1);

    // Medir tiempo de ejecución
    auto start = std::chrono::high_resolution_clock::now();

    // Reservar memoria en la GPU
    double* d_V;
    double* d_S1;
    cudaMalloc(&d_V, N * sizeof(double));
    cudaMalloc(&d_S1, (N - 1) * sizeof(double));

    // Copiar datos de la CPU a la GPU
    cudaMemcpy(d_V, V.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // Configurar dimensiones del grid y bloque
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Lanzar kernel
    calculateS1<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S1, N);

    // Esperar a que los cálculos terminen
    cudaDeviceSynchronize();

    // Copiar resultados de vuelta a la CPU
    cudaMemcpy(S1.data(), d_S1, (N - 1) * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar memoria en la GPU
    cudaFree(d_V);
    cudaFree(d_S1);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Imprimir los primeros diez elementos de V y S1
    std::cout << "Primeros diez elementos del vector V:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << V[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Primeros diez elementos del vector S1:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << S1[i] << " ";
    }
    std::cout << "\n";

    // Imprimir tiempo de ejecución
    std::cout << "Tiempo de ejecución: " << duration.count() << " ms\n";

    return 0;
}
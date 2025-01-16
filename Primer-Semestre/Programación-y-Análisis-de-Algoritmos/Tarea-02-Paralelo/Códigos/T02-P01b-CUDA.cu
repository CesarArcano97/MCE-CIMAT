#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// Kernel para S2
__global__ void calculateS2(const double* V, double* S2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= 1 && idx < N - 1) {
        S2[idx - 1] = (V[idx + 1] + V[idx - 1]) / 2.0;
    }
}

int main() {
    const int N = 50000;
    std::vector<double> V(N);

    // Semilla para aleatoriedad
    std::srand(std::time(0));

    // Construcción del vector V
    for (int i = 0; i < N; i++) {
        V[i] = static_cast<double>(std::rand() % 100) + (std::rand() % 100 / 100.0);
    }

    // Reservar espacio para S2
    std::vector<double> S2(N-2);

    // Inicializar cronómetro
    auto start = std::chrono::high_resolution_clock::now();

    // Reservar memoria de la GPU
    double* d_V;
    double* d_S2;
    cudaMalloc(&d_V, N * sizeof(double));
    cudaMalloc(&d_S2, (N-2) * sizeof(double));

    // Copiar datos del host (la CPU) a la GPU
    cudaMemcpy(d_V, V.data(), N * sizeof(double), cudaMemcpyHostToDevice); 

    // Configurar el grid y bloque
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Ejecutar kernel
    calculateS2<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_S2, N);

    // Sincronizar
    cudaDeviceSynchronize();

    // Copiar resutlados de la GPU al host (la CPU)
    cudaMemcpy(S2.data(), d_S2, (N-2) * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar memoria de la GPU
    cudaFree(d_V);
    cudaFree(d_S2);

    //Terminar cronómetro
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Imprimir primeros diez elementos de V y S2
    std::cout << "Primeros diez elementos del vector V:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << V[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Primeros diez elementos del vector S2:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << S2[i] << " ";
    }
    std::cout << "\n";

    // Mostrar tiempo de ejecución
    std::cout << "Tiempo de ejecución: " << duration.count() << " ms\n";

    return 0;
}
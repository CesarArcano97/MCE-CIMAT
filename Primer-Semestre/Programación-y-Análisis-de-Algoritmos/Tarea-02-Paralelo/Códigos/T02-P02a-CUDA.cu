#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// Kernel CUDA para calcular C1
__global__ void calculateC1(const int* A, const int* B, int* C1, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Índice de fila global
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Índice de columna global

    if (row < N && col < M) {
        C1[row * M + col] = A[row * M + col] + B[(N - row - 1) * M + (M - col - 1)];
    }
}

int main() {
    // Caso 1: Matrices pequeñas
    std::vector<std::vector<int>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> B = {{5, 6}, {7, 8}};
    const int N = 2;
    const int M = 2;

    std::vector<std::vector<int>> C1(N, std::vector<int>(M, 0));

    // Cálculo de C1 para matrices pequeñas
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C1[i][j] = A[i][j] + B[N - i - 1][M - j - 1];
        }
    }

    // Mostrar resultados para el caso pequeño
    std::cout << "Matriz C1 (Caso 1):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << C1[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Caso 2: Matrices grandes
    const int N_large = 1000;
    const int M_large = 1000;

    std::vector<int> A_large(N_large * M_large);
    std::vector<int> B_large(N_large * M_large);
    std::vector<int> C1_large(N_large * M_large);

    // Llenar matrices con valores aleatorios
    std::srand(std::time(0));
    for (int i = 0; i < N_large; i++) {
        for (int j = 0; j < M_large; j++) {
            A_large[i * M_large + j] = std::rand() % 10 + 1;
            B_large[i * M_large + j] = std::rand() % 10 + 1;
        }
    }

    // Cronómetro para ejecución en GPU
    auto start = std::chrono::high_resolution_clock::now();

    // Reservar memoria en GPU
    int* d_A;
    int* d_B;
    int* d_C1;
    cudaMalloc(&d_A, N_large * M_large * sizeof(int));
    cudaMalloc(&d_B, N_large * M_large * sizeof(int));
    cudaMalloc(&d_C1, N_large * M_large * sizeof(int));

    // Copiar datos del host (CPU) a la GPU
    cudaMemcpy(d_A, A_large.data(), N_large * M_large * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_large.data(), N_large * M_large * sizeof(int), cudaMemcpyHostToDevice);

    // Configurar dimensiones del grid y bloques
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M_large + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N_large + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lanzar kernel
    calculateC1<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C1, N_large, M_large);

    // Sincronizar GPU
    cudaDeviceSynchronize();

    // Copiar resultados de la GPU al host
    cudaMemcpy(C1_large.data(), d_C1, N_large * M_large * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria en GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);

    // Cronómetro termina
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Mostrar primeros 10x10 elementos de C1_large
    std::cout << "\nMatriz C1 (Caso 2) - Primeros 10x10 elementos:\n";
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C1_large[i * M_large + j] << " ";
        }
        std::cout << std::endl;
    }

    // Mostrar tiempo de ejecución
    std::cout << "Tiempo de ejecución para matrices grandes: " << duration.count() << " ms\n";

    return 0;
}

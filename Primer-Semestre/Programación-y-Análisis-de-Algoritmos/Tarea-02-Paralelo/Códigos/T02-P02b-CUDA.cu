#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// Función de CUDA para calcular la matriz C2
__global__ void calcular_C2_kernel(int* A, int* B, int* C2, double alpha, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M) {
        C2[i * M + j] = static_cast<int>(alpha * A[i * M + j] + (1 - alpha) * B[i * M + j]);
    }
}

// Función para asignar memoria en GPU y realizar el cálculo en paralelo
void calcular_C2_cuda(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B,
                      std::vector<std::vector<int>>& C2, double alpha) {
    int N = A.size();
    int M = A[0].size();

    int *d_A, *d_B, *d_C2;

    // Asignar memoria en GPU
    cudaMalloc(&d_A, N * M * sizeof(int));
    cudaMalloc(&d_B, N * M * sizeof(int));
    cudaMalloc(&d_C2, N * M * sizeof(int));

    // Copiar datos desde la memoria de host a la memoria de GPU
    cudaMemcpy(d_A, A[0].data(), N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B[0].data(), N * M * sizeof(int), cudaMemcpyHostToDevice);

    // Definir el tamaño de los bloques y la cuadrícula
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Ejecutar el kernel CUDA
    calcular_C2_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C2, alpha, N, M);

    // Esperar que todos los hilos terminen
    cudaDeviceSynchronize();

    // Copiar los resultados de la memoria de GPU a la memoria de host
    cudaMemcpy(C2[0].data(), d_C2, N * M * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memoria de la GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C2);
}

int main() {
    // Caso 2: Matrices grandes de tamaño N x M (1000x1000)
    const int N_large = 1000;
    const int M_large = 1000;
    double alpha = 0.7;  // Valor de alpha entre 0 y 1

    // Crear matrices grandes A y B
    std::vector<std::vector<int>> A_large(N_large, std::vector<int>(M_large));
    std::vector<std::vector<int>> B_large(N_large, std::vector<int>(M_large));

    // Llenar matrices con valores aleatorios
    std::srand(std::time(0));  // Semilla para números aleatorios
    for (int i = 0; i < N_large; i++) {
        for (int j = 0; j < M_large; j++) {
            A_large[i][j] = std::rand() % 10 + 1;  // Valores entre 1 y 10
            B_large[i][j] = std::rand() % 10 + 1;  // Valores entre 1 y 10
        }
    }

    // Construcción de C2 para matrices grandes
    std::vector<std::vector<int>> C2_large(N_large, std::vector<int>(M_large, 0));

    // Comienza cronómetro
    auto start = std::chrono::high_resolution_clock::now();

    // Cálculo de C2 para matrices grandes usando CUDA
    calcular_C2_cuda(A_large, B_large, C2_large, alpha);

    // Termina cronómetro
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Mostrar solo el bloque de los primeros 10x10 elementos de C2 para matrices grandes
    std::cout << "\nMatriz C2 (Caso 2) - Primeros 10x10 elementos:\n";
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C2_large[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Mostrar tiempo de ejecución
    std::cout << "Tiempo de ejecución para matrices grandes: " << duration.count() << " ms\n";

    return 0;
}



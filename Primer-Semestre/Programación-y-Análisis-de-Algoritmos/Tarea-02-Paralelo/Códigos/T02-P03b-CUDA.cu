#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>  // Incluir la librería chrono para medir el tiempo
#include <cuda_runtime.h>

// Función para generar números flotantes aleatorios
double generarNumeroAleatorio(double min, double max) {
    // Configurar la distribución aleatoria en el rango [min, max]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Función para llenar una matriz con valores aleatorios
void llenarMatrizConAleatorios(std::vector<double>& matriz, int filas, int columnas, double min, double max) {
    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < columnas; ++j) {
            matriz[i * columnas + j] = generarNumeroAleatorio(min, max);
        }
    }
}

// Función para realizar la multiplicación de matrices usando CUDA con memoria compartida
__global__ void multiplicarMatricesCUDA(const double* A, const double* B, double* C, int N, int K, int M, int BS) {
    // Índices de fila y columna de la matriz C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Definir memoria compartida para submatrices A y B
    __shared__ double s_A[32][32];  // Tamaño de submatriz A por bloque
    __shared__ double s_B[32][32];  // Tamaño de submatriz B por bloque

    double sum = 0.0;

    for (int k = 0; k < (K + BS - 1) / BS; ++k) {
        // Cargar los elementos de la matriz A y B en memoria compartida
        if (row < N && k * BS + threadIdx.x < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + k * BS + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < M && k * BS + threadIdx.y < K) {
            s_B[threadIdx.y][threadIdx.x] = B[(k * BS + threadIdx.y) * M + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Sincronizar los hilos para asegurar que la carga de datos en memoria compartida termine
        __syncthreads();

        // Realizar la multiplicación de submatrices
        for (int n = 0; n < BS; ++n) {
            sum += s_A[threadIdx.y][n] * s_B[n][threadIdx.x];
        }

        // Sincronizar antes de cargar los siguientes bloques
        __syncthreads();
    }

    // Almacenar el resultado en la matriz C
    if (row < N && col < M) {
        C[row * M + col] = sum;
    }
}

int main() {
    // Dimensiones de las matrices (modificar aquí según sea necesario)
    const int N = 2048;  // Filas de A
    const int K = 8192;  // Columnas de A y filas de B
    const int M = 2048;  // Columnas de B
    const int BS = 32;   // Tamaño del bloque (ajustar según el tamaño de la matriz)

    // Crear matrices A, B y C
    std::vector<double> A(N * K);
    std::vector<double> B(K * M);
    std::vector<double> C(N * M);

    // Llenar matrices A y B con valores aleatorios
    llenarMatrizConAleatorios(A, N, K, 0.0, 10.0);  // Valores aleatorios entre 0.0 y 10.0
    llenarMatrizConAleatorios(B, K, M, 0.0, 10.0);  // Valores aleatorios entre 0.0 y 10.0

    // Asignar memoria en la GPU
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * K * sizeof(double));
    cudaMalloc(&d_B, K * M * sizeof(double));
    cudaMalloc(&d_C, N * M * sizeof(double));

    // Copiar matrices A y B desde la memoria principal a la GPU
    cudaMemcpy(d_A, A.data(), N * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * M * sizeof(double), cudaMemcpyHostToDevice);

    // Definir el tamaño del bloque y la grilla
    dim3 blockDim(BS, BS);
    dim3 gridDim((M + BS - 1) / BS, (N + BS - 1) / BS);

    // Medir el tiempo de ejecución de la multiplicación
    auto start = std::chrono::high_resolution_clock::now();

    // Llamar al kernel de CUDA para multiplicar las matrices
    multiplicarMatricesCUDA<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, K, M, BS);

    // Esperar a que la GPU termine
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;  // Calcular la duración

    // Copiar el resultado de C desde la GPU a la memoria principal
    cudaMemcpy(C.data(), d_C, N * M * sizeof(double), cudaMemcpyDeviceToHost);

    // Mostrar el tiempo de ejecución
    std::cout << "Tiempo de ejecución en GPU: " << duration.count() << " segundos\n";

    // Imprimir algunos valores de la matriz resultado C
    std::cout << "Los primeros 10 valores de cada fila (Resultado de A x B):\n";
    for (int i = 0; i < std::min(10, N); i++) {  // Imprimir solo los primeros 10 valores de cada fila
        for (int j = 0; j < std::min(10, M); j++) {
            std::cout << std::fixed << std::setprecision(2) << C[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    // Liberar memoria de la GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

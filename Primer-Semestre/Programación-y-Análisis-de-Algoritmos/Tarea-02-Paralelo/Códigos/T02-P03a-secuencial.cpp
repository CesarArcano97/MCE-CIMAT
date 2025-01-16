#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>  // Incluir la librería chrono para medir el tiempo

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

// Función para realizar la multiplicación de matrices usando memoria global y bloques
void multiplicarMatricesBloques(const std::vector<double>& A, const std::vector<double>& B, 
                                std::vector<double>& C, int N, int K, int M, int BS) {
    // Inicializar matriz de resultado C con ceros
    std::fill(C.begin(), C.end(), 0.0);

    // Multiplicación de matrices en bloques
    for (int i = 0; i < N; i += BS) {
        for (int j = 0; j < M; j += BS) {
            for (int k = 0; k < K; k += BS) {
                // Multiplicación dentro de cada bloque
                for (int ii = i; ii < std::min(i + BS, N); ii++) {
                    for (int jj = j; jj < std::min(j + BS, M); jj++) {
                        for (int kk = k; kk < std::min(k + BS, K); kk++) {
                            C[ii * M + jj] += A[ii * K + kk] * B[kk * M + jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Dimensiones de las matrices (modificar aquí según sea necesario)
    const int N = 2048;  // Filas de A
    const int K = 8192;  // Columnas de A y filas de B
    const int M = 2048;  // Columnas de B
    const int BS = 64;   // Tamaño del bloque (ajustar según el tamaño de la matriz)

    // Crear matrices A, B y C
    std::vector<double> A(N * K);
    std::vector<double> B(K * M);
    std::vector<double> C(N * M);

    // Llenar matrices A y B con valores aleatorios
    llenarMatrizConAleatorios(A, N, K, 0.0, 10.0);  // Valores aleatorios entre 0.0 y 10.0
    llenarMatrizConAleatorios(B, K, M, 0.0, 10.0);  // Valores aleatorios entre 0.0 y 10.0

    // Medir el tiempo de ejecución de la multiplicación
    auto start = std::chrono::high_resolution_clock::now();

    // Realizar la multiplicación de matrices con bloques
    multiplicarMatricesBloques(A, B, C, N, K, M, BS);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;  // Calcular la duración

    // Mostrar el tiempo de ejecución
    std::cout << "Tiempo de ejecución: " << duration.count() << " segundos\n";

    // Imprimir algunos valores de la matriz resultado C
    //std::cout << "Los primeros 10 valores de cada fila (Resultado de A x B):\n";
    //for (int i = 0; i < std::min(10, N); i++) {  // Imprimir solo los primeros 10 valores de cada fila
    //    for (int j = 0; j < std::min(10, M); j++) {
    //        std::cout << std::fixed << std::setprecision(2) << C[i * M + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    return 0;
}



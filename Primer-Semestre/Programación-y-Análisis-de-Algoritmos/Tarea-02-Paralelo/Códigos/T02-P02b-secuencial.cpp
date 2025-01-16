#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

// Función para calcular la matriz C2
void calcular_C2(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B, 
                 std::vector<std::vector<int>>& C2, double alpha) {
    int N = A.size();
    int M = A[0].size();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C2[i][j] = static_cast<int>(alpha * A[i][j] + (1 - alpha) * B[i][j]);
        }
    }
}

int main() {
    // Caso 1: Matrices pequeñas
    std::vector<std::vector<int>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> B = {{5, 6}, {7, 8}};

    const int N = 2;
    const int M = 2;
    double alpha = 0.7;  // Valor de alpha entre 0 y 1

    // Construcción de C2
    std::vector<std::vector<int>> C2(N, std::vector<int>(M, 0));

    // Cálculo de C2 para matrices pequeñas
    calcular_C2(A, B, C2, alpha);

    // Mostrar matrices para caso 1
    std::cout << "\nCaso 1 - Matrices pequeñas:\n";
    std::cout << "Matriz A (Caso 1):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matriz B (Caso 1):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matriz C2 (Caso 1):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << C2[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Caso 2: Matrices grandes de tamaño N x M (1000x1000)
    const int N_large = 1000;
    const int M_large = 1000;

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

    // Cálculo de C2 para matrices grandes
    calcular_C2(A_large, B_large, C2_large, alpha);

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

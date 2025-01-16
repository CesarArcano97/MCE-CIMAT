#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main() {
    // Caso 1: Matrices pequeñas
    std::vector<std::vector<int>> A = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> B = {{5, 6}, {7, 8}};

    const int N = 2;
    const int M = 2;

    // Construcción de C1
    std::vector<std::vector<int>> C1(N, std::vector<int>(M, 0));

    // Cálculo de C1 para matrices pequeñas
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C1[i][j] = A[i][j] + B[N - i - 1][M - j - 1];

            std::cout << "Caso 1 - Calculando C1[" << i << "][" << j << "] = " 
                      << A[i][j] << " + " << B[N - i - 1][M - j - 1] 
                      << " = " << C1[i][j] << std::endl;
        }
    }

    // Mostrar matrices para caso 1
    std::cout << "\nMatriz A (Caso 1):\n";
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

    std::cout << "Matriz C1 (Caso 1):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << C1[i][j] << " ";
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

    // Construcción de C1 para matrices grandes
    std::vector<std::vector<int>> C1_large(N_large, std::vector<int>(M_large, 0));

    // Comienza cronómetro
    auto start = std::chrono::high_resolution_clock::now();

    // Cálculo de C1 para matrices grandes
    for (int i = 0; i < N_large; i++) {
        for (int j = 0; j < M_large; j++) {
            C1_large[i][j] = A_large[i][j] + B_large[N_large - i - 1][M_large - j - 1];
        }
    }

    // Termina cronómetro
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Mostrar solo el bloque de los primeros 10x10 elementos de C1 para matrices grandes
    std::cout << "\nMatriz C1 (Caso 2) - Primeros 10x10 elementos:\n";
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C1_large[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Mostrar tiempo de ejecución
    std::cout << "Tiempo de ejecución para matrices grandes: " << duration.count() << " ms\n";

    return 0;
}



#include <iostream>
#include <vector>

int main() {
    const int N = 2;
    const int M = 2;
    const double alpha = 0.5;

    std::vector<std::vector<double>> A = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    std::vector<std::vector<double>> B = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    std::cout << "Matriz A:\n";
    for (const auto& row : A) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Matriz B:\n";
    for (const auto& row : B) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    // Construcción de C2
    std::vector<std::vector<double>> C2(N, std::vector<double>(M, 0.0));

    // Cálculo de C2 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C2[i][j] = alpha * A[i][j] + (1 - alpha) * B[i][j];

            std::cout << "Calculando C2[" << i << "][" << j << "] = " 
                      << alpha << " * " << A[i][j] << " + " 
                      << "(1 - " << alpha << ") * " << B[i][j] << " = " 
                      << C2[i][j] << std::endl;
        }
    }

    std::cout << "Matriz C2:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << C2[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

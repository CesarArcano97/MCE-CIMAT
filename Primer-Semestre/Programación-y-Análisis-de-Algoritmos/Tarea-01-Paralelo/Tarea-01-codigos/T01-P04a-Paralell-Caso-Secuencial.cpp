#include <iostream>
#include <vector>

int main() {
    const int N = 2;
    const int M = 2;

    std::vector<std::vector<int>> A = {
        {1, 2},
        {3, 4}
    };

    std::vector<std::vector<int>> B = {
        {5, 6},
        {7, 8}
    };

    // Construcción de C1
    std::vector<std::vector<int>> C1(N, std::vector<int>(M, 0));

    // Cálculo de C1
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C1[i][j] = A[i][j] + B[N - i - 1][M - j - 1];

            std::cout << "Calculando C1[" << i << "][" << j << "] = " 
                      << A[i][j] << " + " << B[N - i - 1][M - j - 1] 
                      << " = " << C1[i][j] << std::endl;
        }
    }

    std::cout << "Matriz C1:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << C1[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main() {
    const int N = 50000;
    std::vector<double> V(N);

    // Semilla pa' aleatoriedad
    std::srand(std::time(0));

    // Construcción del vector V
    for (int i = 0; i < N; i++) {
        V[i] = static_cast<double>(std::rand() % 100) + (std::rand() % 100 / 100.0);
    }

    // Comienza cronómetro
    auto start = std::chrono::high_resolution_clock::now();

    // Construcción de S2
    std::vector<double> S2(N - 2);

    for (int i = 1; i < N - 1; i++) {  
        S2[i - 1] = ((V[i + 1] + V[i - 1]) / 2.0);
    }

    // Termina cronómetro
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Imprimir primeros diez elementos de V y S2 para verificar resultados
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
    std::cout << "Tiempo de ejecución: " << duration.count() << " ms\n";  // Corregir el uso de count()

    return 0;  
}

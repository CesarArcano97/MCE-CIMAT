#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main() {
    // Vamos a generar un vector de 5000 elementos reales
    const int N = 50000;
    std::vector<double> V(N);

    // Coloquemos una semilla para generar aleatoriedad
    std::srand(std::time(0));

    // Construyamos ahora el vector con valores que vayan del 0 al 100
    for (int i = 0; i < N; i++) {
        V[i] = static_cast<double>(std::rand() % 100) + (std::rand() % 100 / 100.0);
    }

    // Asignemos un cronómetro para el cálculo de nuestro S1
    auto start = std::chrono::high_resolution_clock::now();

    // Definamos la construcción de S1
    std::vector<double> S1(N - 1);

    // Calculemos S1[i] = V[i] + V[i+1]
    for (int i = 0; i < N - 1; i++) {
        S1[i] = V[i] + V[i + 1];
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Calculemos la duración en milisegundos
    std::chrono::duration<double, std::milli> duration = end - start;

    // Imprimir los primeros diez elementos de V y S1
    std::cout << "Primeros diez elementos del vector V:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << V[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Primeros diez elementos del vector S1:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << S1[i] << " ";
    }
    std::cout << "\n";

    // Imprimir el tiempo de ejecución
    std::cout << "Tiempo de ejecución: " << duration.count() << " ms\n";

    return 0;
}


#include <iostream>
#include <vector>
#include <stdexcept>

void imprimir_vector(const std::vector<int>& vec) {
    std::cout << "(";
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")";
}

std::vector<int> suma_vectores(const std::vector<std::vector<int>>& vectores) {
    size_t n = vectores[0].size();
    for (const auto& vec : vectores) {
        if (vec.size() != n) {
            throw std::invalid_argument("Los vectores no tienen el mismo tamaño.");
        }
    }
    std::cout << "Vectores a sumar:\n";
    for (const auto& vec : vectores) {
        imprimir_vector(vec);
        std::cout << "\n";
    }
    // Inicializar vector
    std::vector<int> resultado(n, 0);
    for (size_t i = 0; i < n; i++) {
        for (const auto& vec : vectores) {
            std::cout << "Sumando el elemento vec[" << i << "] = " << vec[i] << std::endl;
            resultado[i] += vec[i];
        }
    }
    return resultado;
}

int main() {
    // Ejemplo 1: tres vectores de tamaño 3
    std::vector<int> vec1 = {1, 0, 1, 1, 4, 0, 2, 1};
    std::vector<int> vec2 = {1, 1, 2, 0, 3, 1, 2, 1};
    std::vector<int> vec3 = {1, 0, 1, 0, 2, 1, 2, 3};

    try {
        std::vector<int> resultado = suma_vectores({vec1, vec2, vec3});
        std::cout << "Resultado de la suma: ";
        imprimir_vector(resultado);
        std::cout << "\n" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
    }

    // Ejemplo 2: vectores de diferentes tamaños
    std::vector<int> vec4 = {1, 2, 3};
    std::vector<int> vec5 = {4, 3, 2};
    std::vector<int> vec6 = {5, 5};

    try {
        std::vector<int> resultado = suma_vectores({vec4, vec5, vec6});
        std::cout << "Resultado de la suma: ";
        imprimir_vector(resultado);
        std::cout << "\n" << std::endl;
    } catch (const std::invalid_argument& e) {
        // Imprimir los vectores que no se pueden sumar
        std::cout << "\nVectores que no se pueden sumar:\n";
        imprimir_vector(vec4);
        std::cout << "\n";
        imprimir_vector(vec5);
        std::cout << "\n";
        imprimir_vector(vec6);
        std::cout << "\n" << e.what() << std::endl;
    }

    return 0;
}


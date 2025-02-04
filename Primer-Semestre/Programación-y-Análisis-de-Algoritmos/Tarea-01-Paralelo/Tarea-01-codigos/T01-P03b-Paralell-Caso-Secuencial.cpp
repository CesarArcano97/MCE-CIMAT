#include <iostream>
#include <vector>

int main() {
    const int N = 5;
    std::vector<double> V = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> S2(N - 2);

    for (int i = 1; i < N - 1; i++) {
        // Calcular S2[i - 1] como el promedio de los elementos V[i - 1] y V[i + 1]
        S2[i - 1] = (V[i - 1] + V[i + 1]) / 2.0;
        std::cout << "Calculando S2[" << (i - 1) << "] = (" 
                  << V[i - 1] << " + " << V[i + 1] << ") / 2 = " 
                  << S2[i - 1] << std::endl;
    }

    std::cout << "Vector S2: ";
    for (const auto& value : S2) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

#include <iostream>
#include <vector>

int main() {
    int N = 5;
    std::vector<double> V = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Sv(N - 1);

    for (int i = 0; i < N - 1; i++) {
        Sv[i] = V[i] + V[i + 1]; 
        std::cout << "Calculando Sv[" << i << "] = " 
                  << V[i] << " + " << V[i + 1] << " = " << Sv[i] << std::endl;
    }

    std::cout << "Vector Sv: ";
    for (const auto& s : Sv) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    int N = 5;
    std::vector<double> V = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> Sv(N - 1);
    
    #pragma omp parallel for
    for (int i = 0; i < N - 1; i++) {
        int thread_id = omp_get_thread_num();
        Sv[i] = V[i] + V[i + 1]; 
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " está calculando Sv[" << i << "] = " 
                      << V[i] << " + " << V[i + 1] << " = " << Sv[i] << std::endl;
        }
    }
    std::cout << "Vector Sv: ";
    for (const auto& s : Sv) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    return 0;
}



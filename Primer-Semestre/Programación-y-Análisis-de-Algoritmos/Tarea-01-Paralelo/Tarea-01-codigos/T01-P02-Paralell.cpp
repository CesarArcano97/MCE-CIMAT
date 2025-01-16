#include <iostream>
#include <omp.h>

const int N = 4; 
const int BS = 2;

void matmul_depend(int N, int BS, float** A, float** B, float** C) {
    int i, j, k, ii, jj, kk;

    // Inicializar la matriz resultado C en ceros
    #pragma omp parallel for collapse(2)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }

    // Multiplicación de matrices en bloques
    for (i = 0; i < N; i += BS) {
        for (j = 0; j < N; j += BS) {
            for (k = 0; k < N; k += BS) {
                #pragma omp task private(ii, jj, kk)
                {
                    for (ii = i; ii < i + BS; ii++) {
                        for (jj = j; jj < j + BS; jj++) {
                            for (kk = k; kk < k + BS; kk++) {
                                C[ii][jj] += A[ii][kk] * B[kk][jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

void print_matrix(float** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Declarar matrices A, B y C como punteros
    float** A = new float*[N];
    float** B = new float*[N];
    float** C = new float*[N];

    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
        B[i] = new float[N];
        C[i] = new float[N];
    }

    float val = 1.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = val++;
            B[i][j] = val++;
        }
    }

    std::cout << "Matriz A:\n";
    print_matrix(A, N);

    std::cout << "\nMatriz B:\n";
    print_matrix(B, N);

    matmul_depend(N, BS, A, B, C);
    
    std::cout << "\nMatriz resultado C:\n";
    print_matrix(C, N);

    // Liberar la memoria
    for (int i = 0; i < N; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}



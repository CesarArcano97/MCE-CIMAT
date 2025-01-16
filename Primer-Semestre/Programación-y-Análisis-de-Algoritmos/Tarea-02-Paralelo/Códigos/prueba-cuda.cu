#include <iostream>

// Kernel CUDA que realiza la suma de dos vectores
__global__ void add(int* a, int* b, int* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 16;  // Tamaño del vector
    int a[N], b[N], c[N];  // Vectores en la CPU

    // Inicialización de los vectores a y b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Reserva de memoria en la GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    // Copia de los datos de la CPU a la GPU
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Lanzamiento del kernel (1 bloque con 16 hilos)
    add<<<1, N>>>(d_a, d_b, d_c, N);

    // Copia del resultado de vuelta a la CPU
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Liberación de memoria en la GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Verificación de los resultados
    std::cout << "Resultados de la suma en GPU:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    return 0;
}


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// Kernel para combinar las imágenes
__global__ void blendImagesKernel(unsigned char* d_A, unsigned char* d_B, unsigned char* d_alpha, unsigned char* d_C, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Coordenada X
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Coordenada Y
    int idx = (y * width + x) * channels; // Índice en el array plano

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            d_C[idx + c] = d_alpha[y * width + x] / 255.0 * d_A[idx + c] + (1.0 - d_alpha[y * width + x] / 255.0) * d_B[idx + c];
        }
    }
}

int main() {
    // Carga de imágenes con OpenCV
    cv::Mat imgA = cv::imread("/home/cesar/Imágenes/Imagenes-Tarea-03-Programación-Paralelo/Figura-A.png", cv::IMREAD_COLOR); // Imagen A
    cv::Mat imgB = cv::imread("/home/cesar/Imágenes/Imagenes-Tarea-03-Programación-Paralelo/Figura-B.png", cv::IMREAD_COLOR); // Imagen B
    cv::Mat alpha = cv::imread("/home/cesar/Imágenes/Imagenes-Tarea-03-Programación-Paralelo/Mascara.png", cv::IMREAD_GRAYSCALE); // Máscara α

    if (imgA.empty() || imgB.empty() || alpha.empty()) {
        std::cerr << "Error: No se pudieron cargar las imágenes." << std::endl;
        return -1;
    }

    // Verifica que las dimensiones de las imágenes sean iguales
    if (imgA.size() != imgB.size() || imgA.size() != alpha.size()) {
        std::cerr << "Error: Las imágenes deben tener el mismo tamaño." << std::endl;
        return -1;
    }

    int width = imgA.cols;
    int height = imgA.rows;
    int channels = imgA.channels();

    // Crea la imagen de salida
    cv::Mat imgC(height, width, CV_8UC3);

    // Reservar memoria en la GPU
    unsigned char *d_A, *d_B, *d_alpha, *d_C;
    size_t imgSize = width * height * channels * sizeof(unsigned char);
    size_t alphaSize = width * height * sizeof(unsigned char);

    cudaMalloc(&d_A, imgSize);
    cudaMalloc(&d_B, imgSize);
    cudaMalloc(&d_alpha, alphaSize);
    cudaMalloc(&d_C, imgSize);

    // Copiar datos desde el host (CPU) a la GPU
    cudaMemcpy(d_A, imgA.data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, imgB.data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha.data, alphaSize, cudaMemcpyHostToDevice);

    // Configurar la cuadrícula y los bloques
    dim3 blockSize(16, 16); // Tamaño de bloque
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Lanzar el kernel
    blendImagesKernel<<<gridSize, blockSize>>>(d_A, d_B, d_alpha, d_C, width, height, channels);

    // Copiar la imagen resultante de la GPU al host (CPU)
    cudaMemcpy(imgC.data, d_C, imgSize, cudaMemcpyDeviceToHost);

    // Guardar la imagen resultante
    cv::imwrite("output.jpg", imgC);

    // Liberar memoria en la GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_alpha);
    cudaFree(d_C);

    std::cout << "Imagen combinada guardada como 'output.jpg'" << std::endl;

    return 0;
}

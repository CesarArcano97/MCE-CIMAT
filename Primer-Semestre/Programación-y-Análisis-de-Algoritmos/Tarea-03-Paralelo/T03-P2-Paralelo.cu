#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

// Kernel para aplicar la convolución
__global__ void convolutionKernel(const unsigned char* input, float* output, const float* kernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfKernel = kernelSize / 2;
    float sum = 0.0;

    if (x >= halfKernel && x < (width - halfKernel) && y >= halfKernel && y < (height - halfKernel)) {
        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int pixel = (y + ky) * width + (x + kx);
                float weight = kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                sum += input[pixel] * weight;
            }
        }
        output[y * width + x] = sum;
    }
}

// Kernel para calcular la magnitud del gradiente
__global__ void gradientMagnitudeKernel(const float* gradX, const float* gradY, float* magnitude, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float scale = 0.5; // Factor de escala para reducir la intensidad del gradiente
        magnitude[idx] = scale * sqrtf(gradX[idx] * gradX[idx] + gradY[idx] * gradY[idx]);
    }
}

// Kernel para aplicar el umbral
__global__ void thresholdKernel(const float* magnitude, unsigned char* output, int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = (magnitude[idx] > threshold) ? 255 : 0;
    }
}

int main() {
    // Cargar imagen de entrada en escala de grises
    cv::Mat img = cv::imread("/home/cesar/Descargas/Jinx-Tarea.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    // Aplicar suavizado previo para reducir el ruido
    cv::GaussianBlur(img, img, cv::Size(5, 5), 1.5);

    // Definir los kernels
    float h_K1[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; // Kernel K1
    float h_K2[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; // Kernel K2
    int kernelSize = 3;

    // Reservar memoria en la GPU
    unsigned char *d_input, *d_output;
    float *d_K1, *d_K2, *d_gradX, *d_gradY, *d_magnitude;

    size_t imgSize = width * height * sizeof(unsigned char);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_K1, kernelSizeBytes);
    cudaMalloc(&d_K2, kernelSizeBytes);
    cudaMalloc(&d_gradX, width * height * sizeof(float));
    cudaMalloc(&d_gradY, width * height * sizeof(float));
    cudaMalloc(&d_magnitude, width * height * sizeof(float));

    // Copiar datos a la GPU
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_input, img.data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K1, h_K1, kernelSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K2, h_K2, kernelSizeBytes, cudaMemcpyHostToDevice);

    // Configuración de bloques e hilos
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Aplicar convolución con K1 y K2
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_gradX, d_K1, width, height, kernelSize);
    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_gradY, d_K2, width, height, kernelSize);

    // Calcular la magnitud del gradiente
    gradientMagnitudeKernel<<<gridSize, blockSize>>>(d_gradX, d_gradY, d_magnitude, width, height);

    // Aplicar umbral
    float threshold = 50.0;
    thresholdKernel<<<gridSize, blockSize>>>(d_magnitude, d_output, width, height, threshold);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << "Tiempo total en GPU: " << duration.count() << " ms" << std::endl;

    // Copiar el resultado al host
    cv::Mat outputImg(height, width, CV_8UC1);
    cudaMemcpy(outputImg.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    // Crear una matriz para MG (magnitud del gradiente)
    cv::Mat mgImg(height, width, CV_32F);
    cudaMemcpy(mgImg.data, d_magnitude, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Normalizar MG para valores entre 0 y 255
    cv::Mat mgImgNormalized;
    cv::normalize(mgImg, mgImgNormalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Aplicar un mapa de calor para una imagen más estética
    cv::Mat colorMap;
    cv::applyColorMap(mgImgNormalized, colorMap, cv::COLORMAP_JET);
    cv::imwrite("/home/cesar/Documentos/Tareas-CIMAT/Jinx-mapa-color.jpg", colorMap);

    // Superponer los bordes sobre la imagen original
    cv::Mat colorOriginal;
    cv::cvtColor(img, colorOriginal, cv::COLOR_GRAY2BGR);
    cv::Mat overlay;
    cv::addWeighted(colorOriginal, 0.7, colorMap, 0.3, 0, overlay);
    cv::imwrite("/home/cesar/Documentos/Tareas-CIMAT/Jinx-Overlay.jpg", overlay);

    // Guardar la imagen con el umbral aplicado
    cv::imwrite("/home/cesar/Documentos/Tareas-CIMAT/Jinx-Umbral.jpg", outputImg);

    // Liberar memoria en la GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_K1);
    cudaFree(d_K2);
    cudaFree(d_gradX);
    cudaFree(d_gradY);
    cudaFree(d_magnitude);

    std::cout << "Detección de bordes completada." << std::endl;

    return 0;
}

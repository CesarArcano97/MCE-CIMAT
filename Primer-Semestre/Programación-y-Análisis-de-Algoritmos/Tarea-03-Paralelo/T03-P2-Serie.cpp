#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Función para aplicar la convolución en la CPU
void convolutionCPU(const cv::Mat& input, cv::Mat& output, const float* kernel, int kernelSize) {
    int halfKernel = kernelSize / 2;
    for (int y = halfKernel; y < input.rows - halfKernel; ++y) {
        for (int x = halfKernel; x < input.cols - halfKernel; ++x) {
            float sum = 0.0;
            for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
                for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                    int pixel = input.at<unsigned char>(y + ky, x + kx);
                    float weight = kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                    sum += pixel * weight;
                }
            }
            output.at<float>(y, x) = sum;
        }
    }
}

// Función para calcular la magnitud del gradiente en la CPU
void gradientMagnitudeCPU(const cv::Mat& gradX, const cv::Mat& gradY, cv::Mat& magnitude, float scale) {
    for (int y = 0; y < gradX.rows; ++y) {
        for (int x = 0; x < gradX.cols; ++x) {
            float gx = gradX.at<float>(y, x);
            float gy = gradY.at<float>(y, x);
            magnitude.at<float>(y, x) = scale * sqrtf(gx * gx + gy * gy);
        }
    }
}

// Función para aplicar un umbral en la CPU
void thresholdCPU(const cv::Mat& magnitude, cv::Mat& output, float threshold) {
    for (int y = 0; y < magnitude.rows; ++y) {
        for (int x = 0; x < magnitude.cols; ++x) {
            output.at<unsigned char>(y, x) = (magnitude.at<float>(y, x) > threshold) ? 255 : 0;
        }
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

    // Crear matrices para almacenar resultados
    cv::Mat gradX(height, width, CV_32F, cv::Scalar(0));
    cv::Mat gradY(height, width, CV_32F, cv::Scalar(0));
    cv::Mat magnitude(height, width, CV_32F, cv::Scalar(0));
    cv::Mat output(height, width, CV_8UC1, cv::Scalar(0));

    // Medir el tiempo de ejecución
    auto start = std::chrono::high_resolution_clock::now();

    // Aplicar convoluciones con K1 y K2
    convolutionCPU(img, gradX, h_K1, kernelSize);
    convolutionCPU(img, gradY, h_K2, kernelSize);

    // Calcular la magnitud del gradiente
    float scale = 0.5; // Factor de escala para reducir la intensidad del gradiente
    gradientMagnitudeCPU(gradX, gradY, magnitude, scale);

    // Aplicar umbral
    float threshold = 50.0; // Umbral menos agresivo
    thresholdCPU(magnitude, output, threshold);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Tiempo de ejecución en la CPU: " << duration.count() << " ms" << std::endl;

    // Normalizar la magnitud para guardar como imagen
    cv::Mat magnitudeNormalized;
    cv::normalize(magnitude, magnitudeNormalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Guardar la magnitud del gradiente
    cv::imwrite("/home/cesar/Documentos/Tareas-CIMAT/Jinx-Mapa.jpg", magnitudeNormalized);

    // Aplicar un mapa de calor
    cv::Mat colorMap;
    cv::applyColorMap(magnitudeNormalized, colorMap, cv::COLORMAP_JET);

    // Superponer los bordes sobre la imagen original
    cv::Mat colorOriginal;
    cv::cvtColor(img, colorOriginal, cv::COLOR_GRAY2BGR);
    cv::Mat overlay;
    cv::addWeighted(colorOriginal, 0.7, colorMap, 0.3, 0, overlay);
    cv::imwrite("/home/cesar/Documentos/Tareas-CIMAT/Jinx-Overlay-S.jpg", overlay);

    // Guardar la imagen con el umbral aplicado
    cv::imwrite("/home/cesar/Documentos/Tareas-CIMAT/Jinx-Umbral-S.jpg", output);

    std::cout << "Procesamiento en CPU completado." << std::endl;

    return 0;
}

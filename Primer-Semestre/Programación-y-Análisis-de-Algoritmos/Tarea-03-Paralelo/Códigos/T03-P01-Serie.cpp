#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// Función para combinar imágenes de forma secuencial
void blendImagesSequential(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Mat& alpha, cv::Mat& imgC) {
    int width = imgA.cols;
    int height = imgA.rows;
    int channels = imgA.channels();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float alphaValue = alpha.at<unsigned char>(y, x) / 255.0f; // Normalizar alpha entre 0 y 1
            for (int c = 0; c < channels; ++c) {
                imgC.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(
                    alphaValue * imgA.at<cv::Vec3b>(y, x)[c] +
                    (1.0f - alphaValue) * imgB.at<cv::Vec3b>(y, x)[c]
                );
            }
        }
    }
}

int main() {
    // Cargar imágenes
    auto startLoad = std::chrono::high_resolution_clock::now();
    cv::Mat imgA = cv::imread("/home/cesar/Imágenes/Imagenes-Tarea-03-Programación-Paralelo/Figura-A.png", cv::IMREAD_COLOR);
    cv::Mat imgB = cv::imread("/home/cesar/Imágenes/Imagenes-Tarea-03-Programación-Paralelo/Figura-B.png", cv::IMREAD_COLOR);
    cv::Mat alpha = cv::imread("/home/cesar/Imágenes/Imagenes-Tarea-03-Programación-Paralelo/Mascara.png", cv::IMREAD_GRAYSCALE);
    auto endLoad = std::chrono::high_resolution_clock::now();

    // Verificar que las imágenes se cargaron correctamente
    if (imgA.empty() || imgB.empty() || alpha.empty()) {
        std::cerr << "Error: No se pudieron cargar las imágenes." << std::endl;
        return -1;
    }

    // Verificar que las dimensiones de las imágenes sean iguales
    if (imgA.size() != imgB.size() || imgA.size() != alpha.size()) {
        std::cerr << "Error: Las imágenes deben tener el mismo tamaño." << std::endl;
        return -1;
    }

    // Crear la imagen de salida
    cv::Mat imgC(imgA.rows, imgA.cols, CV_8UC3);

    // Medir tiempo del procesamiento secuencial
    auto startProcessing = std::chrono::high_resolution_clock::now();
    blendImagesSequential(imgA, imgB, alpha, imgC);
    auto endProcessing = std::chrono::high_resolution_clock::now();

    // Guardar la imagen de salida
    auto startSave = std::chrono::high_resolution_clock::now();
    cv::imwrite("/home/cesar/Documentos/Tareas-CIMAT/Primer_Semestre/PyAA/Programacion-Paralelo/output_sequential.jpg", imgC);
    auto endSave = std::chrono::high_resolution_clock::now();

    // Calcular y mostrar tiempos
    std::chrono::duration<float, std::milli> loadTime = endLoad - startLoad;
    std::chrono::duration<float, std::milli> processingTime = endProcessing - startProcessing;
    std::chrono::duration<float, std::milli> saveTime = endSave - startSave;
    std::chrono::duration<float, std::milli> totalTime = endSave - startLoad;

    std::cout << "Tiempo de carga de imágenes: " << loadTime.count() << " ms" << std::endl;
    std::cout << "Tiempo de procesamiento secuencial: " << processingTime.count() << " ms" << std::endl;
    std::cout << "Tiempo de guardado de imágenes: " << saveTime.count() << " ms" << std::endl;
    std::cout << "Tiempo total: " << totalTime.count() << " ms" << std::endl;

    std::cout << "Imagen combinada guardada como 'output_sequential.jpg'" << std::endl;

    return 0;
}

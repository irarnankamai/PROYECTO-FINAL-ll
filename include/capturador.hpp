#pragma once

#include "detector_hog.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

class Capturador
{
public:
    /*
     * calidadJpeg debe estar entre 1 y 100.
     *
     * Un valor de 90 ofrece buena calidad sin generar
     * una imagen demasiado pesada.
     */
    explicit Capturador(
        int calidadJpeg = 90
    );

    /*
     * Valida y prepara el capturador.
     *
     * No crea carpetas ni archivos.
     */
    [[nodiscard]]
    bool inicializar();

    /*
     * Dibuja las detecciones sobre una copia del frame
     * y codifica la imagen como JPEG en memoria RAM.
     *
     * Devuelve un vector vacío si ocurre un error.
     */
    [[nodiscard]]
    std::vector<unsigned char> capturarEnMemoria(
        const cv::Mat& frame,
        const std::vector<TaxiDetectado>& detecciones
    );

    /*
     * Codifica el frame original como JPEG en memoria RAM,
     * sin dibujar detecciones.
     *
     * Devuelve un vector vacío si ocurre un error.
     */
    [[nodiscard]]
    std::vector<unsigned char> capturarEnMemoria(
        const cv::Mat& frame
    );

    [[nodiscard]]
    std::size_t obtenerTotalCapturas() const;

    [[nodiscard]]
    int obtenerCalidadJpeg() const;

private:
    int calidadJpeg_ = 90;

    std::size_t totalCapturas_ = 0;

    bool inicializado_ = false;

    /*
     * Crea una copia del frame y dibuja en ella
     * los cuadros de detección.
     */
    [[nodiscard]]
    cv::Mat dibujarDetecciones(
        const cv::Mat& frame,
        const std::vector<TaxiDetectado>& detecciones
    ) const;

    /*
     * Codifica una imagen OpenCV en formato JPEG
     * dentro de un vector de bytes.
     */
    [[nodiscard]]
    std::vector<unsigned char> codificarJpeg(
        const cv::Mat& imagen
    ) const;
};
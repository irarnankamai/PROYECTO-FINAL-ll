#pragma once

#include "detector_hog.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

class Capturador
{
public:
    explicit Capturador(
        const std::filesystem::path& directorioCapturas = "captures"
    );

    [[nodiscard]]
    bool inicializar();

    [[nodiscard]]
    std::string guardarCaptura(
        const cv::Mat& frame,
        const std::vector<TaxiDetectado>& detecciones
    );

    [[nodiscard]]
    std::string guardarCaptura(
        const cv::Mat& frame
    );

    [[nodiscard]]
    const std::filesystem::path& obtenerDirectorio() const;

    [[nodiscard]]
    std::size_t obtenerTotalCapturas() const;

private:
    std::filesystem::path directorioCapturas_;
    std::size_t totalCapturas_ = 0;
    bool inicializado_ = false;

    [[nodiscard]]
    std::string generarMarcaTiempo() const;

    [[nodiscard]]
    std::string generarFechaHoraLegible() const;

    [[nodiscard]]
    cv::Mat dibujarDetecciones(
        const cv::Mat& frame,
        const std::vector<TaxiDetectado>& detecciones
    ) const;
};
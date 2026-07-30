#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>

class GrabadorVideo
{
public:
    explicit GrabadorVideo(
        const std::filesystem::path& directorioVideos = "videos",
        double duracionSegundos = 5.0
    );

    ~GrabadorVideo();

    GrabadorVideo(const GrabadorVideo&) = delete;
    GrabadorVideo& operator=(const GrabadorVideo&) = delete;

    [[nodiscard]]
    bool inicializar();

    [[nodiscard]]
    bool iniciar(
        const cv::Size& tamanoFrame,
        double fps
    );

    [[nodiscard]]
    bool escribirFrame(
        const cv::Mat& frame
    );

    void finalizar();

    [[nodiscard]]
    bool estaGrabando() const;

    [[nodiscard]]
    bool grabacionCompletada() const;

    [[nodiscard]]
    double obtenerTiempoTranscurrido() const;

    [[nodiscard]]
    double obtenerTiempoRestante() const;

    [[nodiscard]]
    std::size_t obtenerFramesGrabados() const;

    [[nodiscard]]
    std::size_t obtenerTotalVideos() const;

    [[nodiscard]]
    const std::string& obtenerRutaVideoActual() const;

    [[nodiscard]]
    const std::string& obtenerUltimoVideoGuardado() const;

private:
    std::filesystem::path directorioVideos_;

    cv::VideoWriter escritor_;

    /*
     * Conserva el frame más reciente para completar los frames
     * que falten cuando la aplicación procese menos FPS que los
     * declarados en el archivo de video.
     */
    cv::Mat ultimoFrame_;

    double duracionSegundos_;
    double fpsGrabacion_ = 0.0;

    cv::Size tamanoGrabacion_{0, 0};

    bool inicializado_ = false;
    bool grabando_ = false;
    bool completada_ = false;

    std::size_t framesGrabados_ = 0;
    std::size_t totalVideos_ = 0;

    std::string rutaVideoActual_;
    std::string ultimoVideoGuardado_;

    std::chrono::steady_clock::time_point tiempoInicio_{};

    [[nodiscard]]
    std::string generarMarcaTiempo() const;

    [[nodiscard]]
    bool abrirEscritor(
        const std::filesystem::path& ruta,
        const cv::Size& tamanoFrame,
        double fps
    );
};
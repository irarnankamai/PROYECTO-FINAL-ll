#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

class GrabadorVideo
{
public:
    /*
     * duracionSegundos:
     * duración objetivo del video generado.
     *
     * bitrateBitsPorSegundo:
     * tasa de bits usada por el codificador.
     */
    explicit GrabadorVideo(
        double duracionSegundos = 5.0,
        int bitrateBitsPorSegundo = 2'000'000
    );

    ~GrabadorVideo();

    GrabadorVideo(
        const GrabadorVideo&
    ) = delete;

    GrabadorVideo& operator=(
        const GrabadorVideo&
    ) = delete;

    GrabadorVideo(
        GrabadorVideo&&
    ) noexcept;

    GrabadorVideo& operator=(
        GrabadorVideo&&
    ) noexcept;

    /*
     * Prepara internamente FFmpeg.
     *
     * No crea carpetas ni archivos.
     */
    [[nodiscard]]
    bool inicializar();

    /*
     * Inicia una nueva grabación MP4 en memoria RAM.
     *
     * tamanoFrame:
     * resolución de los frames recibidos.
     *
     * fps:
     * fotogramas por segundo del video.
     */
    [[nodiscard]]
    bool iniciar(
        const cv::Size& tamanoFrame,
        double fps
    );

    /*
     * Codifica un frame dentro del video actual.
     *
     * El frame puede ser BGR, BGRA o escala de grises.
     */
    [[nodiscard]]
    bool escribirFrame(
        const cv::Mat& frame
    );

    /*
     * Finaliza el codificador y construye el MP4 completo
     * en memoria RAM.
     */
    [[nodiscard]]
    bool finalizar();

    /*
     * Cancela la grabación actual y libera los datos
     * parcialmente generados.
     */
    void cancelar();

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
    std::size_t obtenerFramesObjetivo() const;

    [[nodiscard]]
    std::size_t obtenerTotalVideos() const;

    [[nodiscard]]
    double obtenerDuracionSegundos() const;

    [[nodiscard]]
    double obtenerFpsGrabacion() const;

    [[nodiscard]]
    const cv::Size& obtenerTamanoGrabacion() const;

    /*
     * Devuelve los bytes del último video MP4 finalizado.
     *
     * El vector permanece válido hasta iniciar una nueva
     * grabación, cancelar o destruir el objeto.
     */
    [[nodiscard]]
    const std::vector<unsigned char>& obtenerVideoEnMemoria() const;

    /*
     * Transfiere los bytes del video fuera del grabador
     * sin realizar una copia completa.
     *
     * Después de llamar a esta función, el vector interno
     * del grabador queda vacío.
     */
    [[nodiscard]]
    std::vector<unsigned char> extraerVideoEnMemoria();

    [[nodiscard]]
    std::size_t obtenerTamanoVideoBytes() const;

private:
    /*
     * Oculta los tipos internos de FFmpeg para evitar incluir
     * sus encabezados directamente en este archivo.
     */
    class Implementacion;

    std::unique_ptr<Implementacion> implementacion_;

    double duracionSegundos_ = 5.0;
    double fpsGrabacion_ = 0.0;

    int bitrateBitsPorSegundo_ = 2'000'000;

    cv::Size tamanoGrabacion_{0, 0};

    bool inicializado_ = false;
    bool grabando_ = false;
    bool completada_ = false;

    std::size_t framesGrabados_ = 0;
    std::size_t framesObjetivo_ = 0;
    std::size_t totalVideos_ = 0;

    std::vector<unsigned char> videoMp4_;

    std::chrono::steady_clock::time_point tiempoInicio_{};

    [[nodiscard]]
    bool parametrosValidos(
        const cv::Size& tamanoFrame,
        double fps
    ) const;

    void reiniciarEstadoGrabacion();
};
#include "grabador_video.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>

namespace
{
constexpr double FPS_PREDETERMINADO = 10.0;
constexpr double FPS_MINIMO = 5.0;
constexpr double FPS_MAXIMO = 60.0;
constexpr double DURACION_PREDETERMINADA = 5.0;
}

GrabadorVideo::GrabadorVideo(
    const std::filesystem::path& directorioVideos,
    double duracionSegundos
)
    : directorioVideos_(directorioVideos),
      duracionSegundos_(
          duracionSegundos > 0.0
              ? duracionSegundos
              : DURACION_PREDETERMINADA
      )
{
}

GrabadorVideo::~GrabadorVideo()
{
    finalizar();
}

bool GrabadorVideo::inicializar()
{
    if (inicializado_)
    {
        return true;
    }

    std::error_code error;

    const bool existe =
        std::filesystem::exists(
            directorioVideos_,
            error
        );

    if (error)
    {
        std::cerr
            << "Error al comprobar el directorio de videos: "
            << error.message()
            << '\n';

        return false;
    }

    if (existe)
    {
        const bool esDirectorio =
            std::filesystem::is_directory(
                directorioVideos_,
                error
            );

        if (error)
        {
            std::cerr
                << "Error al comprobar el tipo de la ruta de videos: "
                << error.message()
                << '\n';

            return false;
        }

        if (!esDirectorio)
        {
            std::cerr
                << "La ruta existe, pero no es un directorio: "
                << directorioVideos_
                << '\n';

            return false;
        }
    }
    else
    {
        const bool creado =
            std::filesystem::create_directories(
                directorioVideos_,
                error
            );

        if (error || !creado)
        {
            std::cerr
                << "No se pudo crear el directorio de videos: "
                << directorioVideos_
                << '\n';

            if (error)
            {
                std::cerr
                    << "Motivo: "
                    << error.message()
                    << '\n';
            }

            return false;
        }
    }

    inicializado_ = true;

    std::error_code errorRuta;

    const std::filesystem::path rutaAbsoluta =
        std::filesystem::absolute(
            directorioVideos_,
            errorRuta
        );

    std::cout
        << "Directorio de videos preparado: "
        << (
               errorRuta
                   ? directorioVideos_
                   : rutaAbsoluta
           )
        << '\n';

    return true;
}

std::string GrabadorVideo::generarMarcaTiempo() const
{
    const auto ahora =
        std::chrono::system_clock::now();

    const std::time_t tiempoActual =
        std::chrono::system_clock::to_time_t(
            ahora
        );

    const auto milisegundos =
        std::chrono::duration_cast<
            std::chrono::milliseconds
        >(
            ahora.time_since_epoch()
        ) % 1000;

    std::tm tiempoLocal{};

#if defined(_WIN32)

    localtime_s(
        &tiempoLocal,
        &tiempoActual
    );

#else

    localtime_r(
        &tiempoActual,
        &tiempoLocal
    );

#endif

    std::ostringstream marcaTiempo;

    marcaTiempo
        << std::put_time(
               &tiempoLocal,
               "%Y%m%d_%H%M%S"
           )
        << '_'
        << std::setw(3)
        << std::setfill('0')
        << milisegundos.count();

    return marcaTiempo.str();
}

bool GrabadorVideo::abrirEscritor(
    const std::filesystem::path& ruta,
    const cv::Size& tamanoFrame,
    double fps
)
{
    if (escritor_.isOpened())
    {
        escritor_.release();
    }

    const int codecMJPG =
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'
        );

    escritor_.open(
        ruta.string(),
        codecMJPG,
        fps,
        tamanoFrame,
        true
    );

    if (escritor_.isOpened())
    {
        std::cout
            << "Códec de video seleccionado: MJPG"
            << '\n';

        return true;
    }

    escritor_.release();

    std::cerr
        << "No se pudo abrir VideoWriter con MJPG."
        << '\n'
        << "Intentando con XVID..."
        << '\n';

    const int codecXVID =
        cv::VideoWriter::fourcc(
            'X',
            'V',
            'I',
            'D'
        );

    escritor_.open(
        ruta.string(),
        codecXVID,
        fps,
        tamanoFrame,
        true
    );

    if (escritor_.isOpened())
    {
        std::cout
            << "Códec de video seleccionado: XVID"
            << '\n';

        return true;
    }

    escritor_.release();

    return false;
}

bool GrabadorVideo::iniciar(
    const cv::Size& tamanoFrame,
    double fps
)
{
    if (grabando_)
    {
        std::cerr
            << "No se puede iniciar otra grabación: "
            << "ya existe una grabación en curso."
            << '\n';

        return false;
    }

    if (!inicializar())
    {
        return false;
    }

    if (
        tamanoFrame.width <= 0
        || tamanoFrame.height <= 0
    )
    {
        std::cerr
            << "Tamaño de frame inválido para grabar: "
            << tamanoFrame.width
            << 'x'
            << tamanoFrame.height
            << '\n';

        return false;
    }

    if (
        !std::isfinite(fps)
        || fps <= 1.0
    )
    {
        fps = FPS_PREDETERMINADO;
    }

    fps =
        std::clamp(
            fps,
            FPS_MINIMO,
            FPS_MAXIMO
        );

    const std::string nombreVideo =
        "taxi_"
        + generarMarcaTiempo()
        + ".avi";

    const std::filesystem::path ruta =
        directorioVideos_
        / nombreVideo;

    if (
        !abrirEscritor(
            ruta,
            tamanoFrame,
            fps
        )
    )
    {
        std::error_code errorRuta;

        const std::filesystem::path rutaAbsoluta =
            std::filesystem::absolute(
                ruta,
                errorRuta
            );

        std::cerr
            << "No se pudo iniciar la grabación."
            << '\n'
            << "Ruta intentada: "
            << (
                   errorRuta
                       ? ruta
                       : rutaAbsoluta
               )
            << '\n';

        return false;
    }

    fpsGrabacion_ = fps;
    tamanoGrabacion_ = tamanoFrame;

    framesGrabados_ = 0;

    /*
     * Elimina cualquier frame conservado por una grabación
     * anterior.
     */
    ultimoFrame_.release();

    grabando_ = true;
    completada_ = false;

    ultimoVideoGuardado_.clear();

    std::error_code errorRuta;

    const std::filesystem::path rutaAbsoluta =
        std::filesystem::absolute(
            ruta,
            errorRuta
        );

    rutaVideoActual_ =
        (
            errorRuta
                ? ruta
                : rutaAbsoluta
        ).string();

    tiempoInicio_ =
        std::chrono::steady_clock::now();

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "GRABACION DE VIDEO INICIADA"
        << '\n'
        << "============================================="
        << '\n'
        << "Ruta: "
        << rutaVideoActual_
        << '\n'
        << "Resolución: "
        << tamanoGrabacion_.width
        << 'x'
        << tamanoGrabacion_.height
        << '\n'
        << "FPS: "
        << fpsGrabacion_
        << '\n'
        << "Duración objetivo: "
        << duracionSegundos_
        << " segundos"
        << '\n'
        << "============================================="
        << '\n';

    return true;
}

bool GrabadorVideo::escribirFrame(
    const cv::Mat& frame
)
{
    if (!grabando_)
    {
        return false;
    }

    if (frame.empty())
    {
        std::cerr
            << "No se puede grabar un frame vacío."
            << '\n';

        return false;
    }

    cv::Mat frameSalida;

    if (frame.size() != tamanoGrabacion_)
    {
        cv::resize(
            frame,
            frameSalida,
            tamanoGrabacion_,
            0.0,
            0.0,
            cv::INTER_LINEAR
        );
    }
    else
    {
        frameSalida = frame;
    }

    if (frameSalida.channels() == 1)
    {
        cv::cvtColor(
            frameSalida,
            frameSalida,
            cv::COLOR_GRAY2BGR
        );
    }
    else if (frameSalida.channels() == 4)
    {
        cv::cvtColor(
            frameSalida,
            frameSalida,
            cv::COLOR_BGRA2BGR
        );
    }
    else if (frameSalida.channels() != 3)
    {
        std::cerr
            << "Cantidad de canales no compatible con VideoWriter: "
            << frameSalida.channels()
            << '\n';

        return false;
    }

    if (frameSalida.depth() != CV_8U)
    {
        std::cerr
            << "Profundidad de imagen no compatible con VideoWriter."
            << '\n';

        return false;
    }

    /*
     * Guardar una copia independiente del frame más reciente.
     * Esta copia se usa para rellenar intervalos en los que la
     * aplicación no alcanza a entregar 30 frames por segundo.
     */
    ultimoFrame_ =
        frameSalida.clone();

    const double tiempoTranscurrido =
        obtenerTiempoTranscurrido();

    const std::size_t framesObjetivo =
        static_cast<std::size_t>(
            std::lround(
                duracionSegundos_
                * fpsGrabacion_
            )
        );

    /*
     * Cantidad de frames que debería contener el video hasta
     * este instante según el reloj real.
     *
     * Se suma uno para que el primer frame se escriba incluso
     * cuando han pasado pocos milisegundos desde el inicio.
     */
    std::size_t framesEsperados =
        static_cast<std::size_t>(
            std::floor(
                tiempoTranscurrido
                * fpsGrabacion_
            )
        ) + 1;

    framesEsperados =
        std::min(
            framesEsperados,
            framesObjetivo
        );

    try
    {
        /*
         * Si la interfaz funciona a menos FPS que el archivo,
         * repetir el último frame hasta mantener sincronizada
         * la duración del AVI con el tiempo real.
         */
        while (
            framesGrabados_
            < framesEsperados
        )
        {
            escritor_.write(
                ultimoFrame_
            );

            ++framesGrabados_;
        }
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "Error de OpenCV al escribir el video: "
            << error.what()
            << '\n';

        finalizar();

        return false;
    }

    if (
        tiempoTranscurrido
        >= duracionSegundos_
    )
    {
        finalizar();
    }

    return true;
}

void GrabadorVideo::finalizar()
{
    if (!grabando_)
    {
        return;
    }

    const double tiempoGrabado =
        obtenerTiempoTranscurrido();

    const std::size_t framesObjetivo =
        static_cast<std::size_t>(
            std::lround(
                duracionSegundos_
                * fpsGrabacion_
            )
        );

    /*
     * Cuando la grabación alcanzó la duración solicitada,
     * completar los frames que falten antes de cerrar el AVI.
     *
     * Ejemplo:
     * 5 segundos × 30 FPS = 150 frames.
     */
    if (
        tiempoGrabado >= duracionSegundos_
        && !ultimoFrame_.empty()
        && escritor_.isOpened()
    )
    {
        try
        {
            while (
                framesGrabados_
                < framesObjetivo
            )
            {
                escritor_.write(
                    ultimoFrame_
                );

                ++framesGrabados_;
            }
        }
        catch (const cv::Exception& error)
        {
            std::cerr
                << "Error al completar los frames finales: "
                << error.what()
                << '\n';
        }
    }

    if (escritor_.isOpened())
    {
        escritor_.release();
    }

    grabando_ = false;
    completada_ = true;

    ultimoVideoGuardado_ =
        rutaVideoActual_;

    ++totalVideos_;

    const double duracionArchivo =
        fpsGrabacion_ > 0.0
            ? static_cast<double>(
                  framesGrabados_
              ) / fpsGrabacion_
            : 0.0;

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "GRABACION DE VIDEO FINALIZADA"
        << '\n'
        << "============================================="
        << '\n'
        << "Ruta: "
        << ultimoVideoGuardado_
        << '\n'
        << "Tiempo real transcurrido: "
        << std::fixed
        << std::setprecision(2)
        << tiempoGrabado
        << " segundos"
        << '\n'
        << "Duración estimada del archivo: "
        << duracionArchivo
        << " segundos"
        << '\n'
        << "Frames grabados: "
        << framesGrabados_
        << '\n'
        << "FPS del archivo: "
        << fpsGrabacion_
        << '\n'
        << "Total de videos: "
        << totalVideos_
        << '\n'
        << "============================================="
        << '\n';

    rutaVideoActual_.clear();

    /*
     * Liberar la copia del último frame después de cerrar
     * completamente el archivo.
     */
    ultimoFrame_.release();
}

bool GrabadorVideo::estaGrabando() const
{
    return grabando_;
}

bool GrabadorVideo::grabacionCompletada() const
{
    return completada_;
}

double GrabadorVideo::obtenerTiempoTranscurrido() const
{
    if (!grabando_)
    {
        return 0.0;
    }

    const auto ahora =
        std::chrono::steady_clock::now();

    return std::chrono::duration<double>(
        ahora - tiempoInicio_
    ).count();
}

double GrabadorVideo::obtenerTiempoRestante() const
{
    if (!grabando_)
    {
        return 0.0;
    }

    return std::max(
        0.0,
        duracionSegundos_
            - obtenerTiempoTranscurrido()
    );
}

std::size_t GrabadorVideo::obtenerFramesGrabados() const
{
    return framesGrabados_;
}

std::size_t GrabadorVideo::obtenerTotalVideos() const
{
    return totalVideos_;
}

const std::string&
GrabadorVideo::obtenerRutaVideoActual() const
{
    return rutaVideoActual_;
}

const std::string&
GrabadorVideo::obtenerUltimoVideoGuardado() const
{
    return ultimoVideoGuardado_;
}
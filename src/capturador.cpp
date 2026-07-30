#include "capturador.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace
{
constexpr int CALIDAD_JPEG = 95;
constexpr double ESCALA_TEXTO = 0.65;
constexpr int GROSOR_TEXTO = 2;
constexpr int GROSOR_CAJA = 3;

cv::Mat convertirABgr(
    const cv::Mat& imagen
)
{
    if (imagen.empty())
    {
        return {};
    }

    if (imagen.depth() != CV_8U)
    {
        return {};
    }

    cv::Mat resultado;

    if (imagen.channels() == 3)
    {
        resultado = imagen.clone();
    }
    else if (imagen.channels() == 1)
    {
        cv::cvtColor(
            imagen,
            resultado,
            cv::COLOR_GRAY2BGR
        );
    }
    else if (imagen.channels() == 4)
    {
        cv::cvtColor(
            imagen,
            resultado,
            cv::COLOR_BGRA2BGR
        );
    }

    return resultado;
}

bool archivoGuardadoValido(
    const std::filesystem::path& ruta
)
{
    std::error_code error;

    if (
        !std::filesystem::is_regular_file(
            ruta,
            error
        )
        || error
    )
    {
        return false;
    }

    const std::uintmax_t tamano =
        std::filesystem::file_size(
            ruta,
            error
        );

    return !error && tamano > 0;
}

std::filesystem::path generarRutaDisponible(
    const std::filesystem::path& directorio,
    const std::string& nombreBase,
    const std::string& extension
)
{
    std::filesystem::path ruta =
        directorio
        / (
            nombreBase
            + extension
        );

    std::error_code error;

    if (
        !std::filesystem::exists(
            ruta,
            error
        )
        && !error
    )
    {
        return ruta;
    }

    for (std::size_t indice = 1; ; ++indice)
    {
        ruta =
            directorio
            / (
                nombreBase
                + "_"
                + std::to_string(indice)
                + extension
            );

        error.clear();

        if (
            !std::filesystem::exists(
                ruta,
                error
            )
            && !error
        )
        {
            return ruta;
        }
    }
}
}

Capturador::Capturador(
    const std::filesystem::path& directorioCapturas
)
    : directorioCapturas_(
          directorioCapturas
      )
{
}

bool Capturador::inicializar()
{
    if (inicializado_)
    {
        return true;
    }

    if (directorioCapturas_.empty())
    {
        std::cerr
            << "El directorio de capturas está vacío."
            << '\n';

        return false;
    }

    std::error_code error;

    const bool existe =
        std::filesystem::exists(
            directorioCapturas_,
            error
        );

    if (error)
    {
        std::cerr
            << "Error al comprobar el directorio de capturas: "
            << error.message()
            << '\n';

        return false;
    }

    if (existe)
    {
        const bool esDirectorio =
            std::filesystem::is_directory(
                directorioCapturas_,
                error
            );

        if (error)
        {
            std::cerr
                << "Error al comprobar el tipo de ruta: "
                << error.message()
                << '\n';

            return false;
        }

        if (!esDirectorio)
        {
            std::cerr
                << "La ruta existe, pero no es un directorio: "
                << directorioCapturas_
                << '\n';

            return false;
        }
    }
    else
    {
        const bool creado =
            std::filesystem::create_directories(
                directorioCapturas_,
                error
            );

        if (error || !creado)
        {
            std::cerr
                << "No se pudo crear el directorio de capturas: "
                << directorioCapturas_
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
            directorioCapturas_,
            errorRuta
        );

    std::cout
        << "Directorio de capturas preparado: "
        << (
               errorRuta
                   ? directorioCapturas_
                   : rutaAbsoluta
           )
        << '\n';

    return true;
}

std::string Capturador::generarMarcaTiempo() const
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

    std::ostringstream nombre;

    nombre
        << std::put_time(
               &tiempoLocal,
               "%Y%m%d_%H%M%S"
           )
        << '_'
        << std::setw(3)
        << std::setfill('0')
        << milisegundos.count();

    return nombre.str();
}

std::string Capturador::generarFechaHoraLegible() const
{
    const auto ahora =
        std::chrono::system_clock::now();

    const std::time_t tiempoActual =
        std::chrono::system_clock::to_time_t(
            ahora
        );

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

    std::ostringstream texto;

    texto
        << "Fecha: "
        << std::put_time(
               &tiempoLocal,
               "%d/%m/%Y"
           )
        << " | Hora: "
        << std::put_time(
               &tiempoLocal,
               "%H:%M:%S"
           );

    return texto.str();
}

cv::Mat Capturador::dibujarDetecciones(
    const cv::Mat& frame,
    const std::vector<TaxiDetectado>& detecciones
) const
{
    cv::Mat resultado =
        convertirABgr(frame);

    if (resultado.empty())
    {
        return {};
    }

    const cv::Rect limitesImagen(
        0,
        0,
        resultado.cols,
        resultado.rows
    );

    std::size_t numeroTaxi = 0;

    for (
        const TaxiDetectado& deteccion :
        detecciones
    )
    {
        const cv::Rect cajaValida =
            deteccion.caja
            & limitesImagen;

        if (
            cajaValida.width <= 0
            || cajaValida.height <= 0
        )
        {
            continue;
        }

        ++numeroTaxi;

        cv::rectangle(
            resultado,
            cajaValida,
            cv::Scalar(0, 255, 0),
            GROSOR_CAJA,
            cv::LINE_AA
        );

        const float score =
            std::isfinite(deteccion.score)
                ? deteccion.score
                : 0.0F;

        std::ostringstream etiqueta;

        etiqueta
            << "Taxi "
            << numeroTaxi
            << " | score: "
            << std::fixed
            << std::setprecision(2)
            << score;

        int lineaBase = 0;

        const cv::Size tamanoTexto =
            cv::getTextSize(
                etiqueta.str(),
                cv::FONT_HERSHEY_SIMPLEX,
                ESCALA_TEXTO,
                GROSOR_TEXTO,
                &lineaBase
            );

        int textoX =
            cajaValida.x;

        int textoY =
            cajaValida.y - 10;

        if (
            textoY - tamanoTexto.height < 0
        )
        {
            textoY =
                cajaValida.y
                + tamanoTexto.height
                + 10;
        }

        textoX =
            std::clamp(
                textoX,
                0,
                std::max(
                    0,
                    resultado.cols
                        - tamanoTexto.width
                        - 12
                )
            );

        textoY =
            std::clamp(
                textoY,
                tamanoTexto.height + 6,
                std::max(
                    tamanoTexto.height + 6,
                    resultado.rows
                        - lineaBase
                        - 6
                )
            );

        const int fondoX =
            std::max(
                0,
                textoX
            );

        const int fondoY =
            std::max(
                0,
                textoY
                    - tamanoTexto.height
                    - 6
            );

        const int anchoFondo =
            std::min(
                tamanoTexto.width + 12,
                resultado.cols - fondoX
            );

        const int altoFondo =
            std::min(
                tamanoTexto.height
                    + lineaBase
                    + 10,
                resultado.rows - fondoY
            );

        if (
            anchoFondo > 0
            && altoFondo > 0
        )
        {
            cv::rectangle(
                resultado,
                cv::Rect(
                    fondoX,
                    fondoY,
                    anchoFondo,
                    altoFondo
                ),
                cv::Scalar(0, 120, 0),
                cv::FILLED
            );
        }

        cv::putText(
            resultado,
            etiqueta.str(),
            cv::Point(
                textoX + 5,
                textoY
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            ESCALA_TEXTO,
            cv::Scalar(255, 255, 255),
            GROSOR_TEXTO,
            cv::LINE_AA
        );
    }

    const std::string fechaHora =
        generarFechaHoraLegible();

    int lineaBaseFecha = 0;

    const cv::Size tamanoFecha =
        cv::getTextSize(
            fechaHora,
            cv::FONT_HERSHEY_SIMPLEX,
            ESCALA_TEXTO,
            GROSOR_TEXTO,
            &lineaBaseFecha
        );

    constexpr int MARGEN = 15;
    constexpr int RELLENO_X = 10;
    constexpr int RELLENO_Y = 8;

    const int textoFechaX =
        std::clamp(
            MARGEN,
            0,
            std::max(
                0,
                resultado.cols
                    - tamanoFecha.width
                    - 5
            )
        );

    const int minimoY =
        tamanoFecha.height
        + RELLENO_Y;

    const int maximoY =
        std::max(
            minimoY,
            resultado.rows
                - lineaBaseFecha
                - 1
        );

    const int textoFechaY =
        std::clamp(
            resultado.rows
                - MARGEN
                - lineaBaseFecha,
            minimoY,
            maximoY
        );

    const cv::Rect fondoFechaSinLimitar(
        textoFechaX - 5,
        textoFechaY
            - tamanoFecha.height
            - RELLENO_Y,
        tamanoFecha.width
            + RELLENO_X,
        tamanoFecha.height
            + lineaBaseFecha
            + RELLENO_Y
    );

    const cv::Rect fondoFecha =
        fondoFechaSinLimitar
        & limitesImagen;

    if (
        fondoFecha.width > 0
        && fondoFecha.height > 0
    )
    {
        cv::Mat region =
            resultado(fondoFecha);

        cv::Mat fondo(
            region.size(),
            region.type(),
            cv::Scalar(0, 0, 0)
        );

        cv::addWeighted(
            fondo,
            0.65,
            region,
            0.35,
            0.0,
            region
        );
    }

    cv::putText(
        resultado,
        fechaHora,
        cv::Point(
            textoFechaX,
            textoFechaY
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        ESCALA_TEXTO,
        cv::Scalar(255, 255, 255),
        GROSOR_TEXTO,
        cv::LINE_AA
    );

    return resultado;
}

std::string Capturador::guardarCaptura(
    const cv::Mat& frame,
    const std::vector<TaxiDetectado>& detecciones
)
{
    if (frame.empty())
    {
        std::cerr
            << "No se puede guardar la captura: "
            << "el frame está vacío."
            << '\n';

        return {};
    }

    if (frame.depth() != CV_8U)
    {
        std::cerr
            << "No se puede guardar la captura: "
            << "la profundidad del frame debe ser CV_8U."
            << '\n';

        return {};
    }

    if (
        frame.channels() != 1
        && frame.channels() != 3
        && frame.channels() != 4
    )
    {
        std::cerr
            << "No se puede guardar la captura: "
            << "cantidad de canales no compatible: "
            << frame.channels()
            << '\n';

        return {};
    }

    if (!inicializar())
    {
        return {};
    }

    const std::string marcaTiempo =
        generarMarcaTiempo();

    const std::filesystem::path rutaCaptura =
        generarRutaDisponible(
            directorioCapturas_,
            "taxi_" + marcaTiempo,
            ".jpg"
        );

    const cv::Mat imagenEvidencia =
        dibujarDetecciones(
            frame,
            detecciones
        );

    if (imagenEvidencia.empty())
    {
        std::cerr
            << "No se pudo preparar la imagen de evidencia."
            << '\n';

        return {};
    }

    const std::vector<int> parametrosJpeg{
        cv::IMWRITE_JPEG_QUALITY,
        CALIDAD_JPEG
    };

    try
    {
        const bool guardada =
            cv::imwrite(
                rutaCaptura.string(),
                imagenEvidencia,
                parametrosJpeg
            );

        if (!guardada)
        {
            std::cerr
                << "OpenCV no pudo guardar la captura: "
                << rutaCaptura
                << '\n';

            return {};
        }
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "Error de OpenCV al guardar la captura: "
            << error.what()
            << '\n';

        return {};
    }

    if (!archivoGuardadoValido(rutaCaptura))
    {
        std::cerr
            << "La captura fue creada, pero el archivo "
            << "resultante no es válido: "
            << rutaCaptura
            << '\n';

        std::error_code errorEliminacion;

        std::filesystem::remove(
            rutaCaptura,
            errorEliminacion
        );

        return {};
    }

    ++totalCapturas_;

    std::error_code errorRuta;

    const std::filesystem::path rutaAbsoluta =
        std::filesystem::absolute(
            rutaCaptura,
            errorRuta
        );

    const std::filesystem::path rutaResultado =
        errorRuta
            ? rutaCaptura
            : rutaAbsoluta;

    std::size_t deteccionesValidas = 0;

    const cv::Rect limitesImagen(
        0,
        0,
        frame.cols,
        frame.rows
    );

    for (
        const TaxiDetectado& deteccion :
        detecciones
    )
    {
        const cv::Rect cajaValida =
            deteccion.caja
            & limitesImagen;

        if (
            cajaValida.width > 0
            && cajaValida.height > 0
        )
        {
            ++deteccionesValidas;
        }
    }

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "CAPTURA GUARDADA"
        << '\n'
        << "============================================="
        << '\n'
        << "Ruta: "
        << rutaResultado
        << '\n'
        << "Taxis registrados: "
        << deteccionesValidas
        << '\n'
        << "Total de capturas: "
        << totalCapturas_
        << '\n'
        << "============================================="
        << '\n';

    return rutaResultado.string();
}

std::string Capturador::guardarCaptura(
    const cv::Mat& frame
)
{
    return guardarCaptura(
        frame,
        {}
    );
}

const std::filesystem::path&
Capturador::obtenerDirectorio() const
{
    return directorioCapturas_;
}

std::size_t Capturador::obtenerTotalCapturas() const
{
    return totalCapturas_;
}
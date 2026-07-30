#include "capturador.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
constexpr double ESCALA_TEXTO = 0.65;
constexpr int GROSOR_TEXTO = 2;
constexpr int GROSOR_CAJA = 3;

constexpr int CALIDAD_JPEG_MINIMA = 1;
constexpr int CALIDAD_JPEG_MAXIMA = 100;

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
        resultado =
            imagen.clone();
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

std::string generarFechaHoraLegible()
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

std::size_t contarDeteccionesValidas(
    const cv::Mat& frame,
    const std::vector<TaxiDetectado>& detecciones
)
{
    if (frame.empty())
    {
        return 0;
    }

    const cv::Rect limitesImagen(
        0,
        0,
        frame.cols,
        frame.rows
    );

    std::size_t totalValidas = 0;

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
            ++totalValidas;
        }
    }

    return totalValidas;
}

double convertirBytesAKilobytes(
    std::size_t bytes
)
{
    constexpr double BYTES_POR_KILOBYTE =
        1024.0;

    return
        static_cast<double>(bytes)
        / BYTES_POR_KILOBYTE;
}
}

Capturador::Capturador(
    int calidadJpeg
)
    : calidadJpeg_(
          std::clamp(
              calidadJpeg,
              CALIDAD_JPEG_MINIMA,
              CALIDAD_JPEG_MAXIMA
          )
      )
{
}

bool Capturador::inicializar()
{
    if (inicializado_)
    {
        return true;
    }

    if (
        calidadJpeg_ < CALIDAD_JPEG_MINIMA
        || calidadJpeg_ > CALIDAD_JPEG_MAXIMA
    )
    {
        std::cerr
            << "La calidad JPEG debe estar entre "
            << CALIDAD_JPEG_MINIMA
            << " y "
            << CALIDAD_JPEG_MAXIMA
            << "."
            << '\n';

        return false;
    }

    inicializado_ = true;

    std::cout
        << "Capturador en memoria RAM preparado."
        << '\n'
        << "Calidad JPEG: "
        << calidadJpeg_
        << "%"
        << '\n';

    return true;
}

cv::Mat Capturador::dibujarDetecciones(
    const cv::Mat& frame,
    const std::vector<TaxiDetectado>& detecciones
) const
{
    cv::Mat resultado =
        convertirABgr(
            frame
        );

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
            cv::Scalar(
                0,
                255,
                0
            ),
            GROSOR_CAJA,
            cv::LINE_AA
        );

        const float score =
            std::isfinite(
                deteccion.score
            )
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
                cv::Scalar(
                    0,
                    120,
                    0
                ),
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
            cv::Scalar(
                255,
                255,
                255
            ),
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
            resultado(
                fondoFecha
            );

        cv::Mat fondo(
            region.size(),
            region.type(),
            cv::Scalar(
                0,
                0,
                0
            )
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
        cv::Scalar(
            255,
            255,
            255
        ),
        GROSOR_TEXTO,
        cv::LINE_AA
    );

    return resultado;
}

std::vector<unsigned char>
Capturador::codificarJpeg(
    const cv::Mat& imagen
) const
{
    if (imagen.empty())
    {
        std::cerr
            << "No se puede codificar la imagen: "
            << "la imagen está vacía."
            << '\n';

        return {};
    }

    if (imagen.depth() != CV_8U)
    {
        std::cerr
            << "No se puede codificar la imagen: "
            << "la profundidad debe ser CV_8U."
            << '\n';

        return {};
    }

    if (
        imagen.channels() != 1
        && imagen.channels() != 3
        && imagen.channels() != 4
    )
    {
        std::cerr
            << "No se puede codificar la imagen: "
            << "cantidad de canales no compatible: "
            << imagen.channels()
            << '\n';

        return {};
    }

    std::vector<unsigned char> imagenJpeg;

    const std::vector<int> parametrosJpeg{
        cv::IMWRITE_JPEG_QUALITY,
        calidadJpeg_
    };

    try
    {
        const bool codificada =
            cv::imencode(
                ".jpg",
                imagen,
                imagenJpeg,
                parametrosJpeg
            );

        if (
            !codificada
            || imagenJpeg.empty()
        )
        {
            std::cerr
                << "OpenCV no pudo codificar "
                << "la imagen JPEG en memoria RAM."
                << '\n';

            return {};
        }
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "Error de OpenCV al codificar "
            << "la imagen JPEG: "
            << error.what()
            << '\n';

        return {};
    }

    return imagenJpeg;
}

std::vector<unsigned char>
Capturador::capturarEnMemoria(
    const cv::Mat& frame,
    const std::vector<TaxiDetectado>& detecciones
)
{
    if (!inicializado_)
    {
        if (!inicializar())
        {
            return {};
        }
    }

    if (frame.empty())
    {
        std::cerr
            << "No se puede realizar la captura: "
            << "el frame está vacío."
            << '\n';

        return {};
    }

    const cv::Mat imagenEvidencia =
        dibujarDetecciones(
            frame,
            detecciones
        );

    if (imagenEvidencia.empty())
    {
        std::cerr
            << "No se pudo preparar "
            << "la imagen de evidencia."
            << '\n';

        return {};
    }

    std::vector<unsigned char> imagenJpeg =
        codificarJpeg(
            imagenEvidencia
        );

    if (imagenJpeg.empty())
    {
        return {};
    }

    ++totalCapturas_;

    const std::size_t deteccionesValidas =
        contarDeteccionesValidas(
            frame,
            detecciones
        );

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "CAPTURA GENERADA EN MEMORIA RAM"
        << '\n'
        << "============================================="
        << '\n'
        << "Tamaño: "
        << imagenJpeg.size()
        << " bytes ("
        << std::fixed
        << std::setprecision(2)
        << convertirBytesAKilobytes(
               imagenJpeg.size()
           )
        << " KB)"
        << '\n'
        << "Taxis registrados: "
        << deteccionesValidas
        << '\n'
        << "Total de capturas: "
        << totalCapturas_
        << '\n'
        << "Almacenamiento en disco: NO"
        << '\n'
        << "============================================="
        << '\n';

    return imagenJpeg;
}

std::vector<unsigned char>
Capturador::capturarEnMemoria(
    const cv::Mat& frame
)
{
    if (!inicializado_)
    {
        if (!inicializar())
        {
            return {};
        }
    }

    if (frame.empty())
    {
        std::cerr
            << "No se puede realizar la captura: "
            << "el frame está vacío."
            << '\n';

        return {};
    }

    const cv::Mat imagenPreparada =
        convertirABgr(
            frame
        );

    if (imagenPreparada.empty())
    {
        std::cerr
            << "No se pudo preparar el frame "
            << "para codificarlo."
            << '\n';

        return {};
    }

    std::vector<unsigned char> imagenJpeg =
        codificarJpeg(
            imagenPreparada
        );

    if (imagenJpeg.empty())
    {
        return {};
    }

    ++totalCapturas_;

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "CAPTURA GENERADA EN MEMORIA RAM"
        << '\n'
        << "============================================="
        << '\n'
        << "Tamaño: "
        << imagenJpeg.size()
        << " bytes ("
        << std::fixed
        << std::setprecision(2)
        << convertirBytesAKilobytes(
               imagenJpeg.size()
           )
        << " KB)"
        << '\n'
        << "Total de capturas: "
        << totalCapturas_
        << '\n'
        << "Almacenamiento en disco: NO"
        << '\n'
        << "============================================="
        << '\n';

    return imagenJpeg;
}

std::size_t
Capturador::obtenerTotalCapturas() const
{
    return totalCapturas_;
}

int Capturador::obtenerCalidadJpeg() const
{
    return calidadJpeg_;
}
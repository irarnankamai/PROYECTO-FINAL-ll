#include "detector_hog.hpp"
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
constexpr int ANCHO_HOG = 128;
constexpr int ALTO_HOG = 64;

constexpr int LADO_MAXIMO_BUSQUEDA = 1600;

constexpr int PASO_X = 24;
constexpr int PASO_Y = 16;

constexpr double PORCENTAJE_MINIMO_AMARILLO = 0.07;

constexpr float SCORE_MINIMO_SVM = 0.8F;

struct DeteccionInterna
{
    cv::Rect caja;

    float scoreSvm = 0.0F;

    double porcentajeAmarillo = 0.0;

    double scoreCombinado = 0.0;

    int cantidadVentanas = 1;
};

struct ImagenTrabajo
{
    cv::Mat imagen;

    double escalaX = 1.0;

    double escalaY = 1.0;
};

std::vector<float> extraerDescriptor(
    const cv::Mat& imagen,
    cv::HOGDescriptor& hog
)
{
    if (imagen.empty())
    {
        throw std::runtime_error(
            "No se puede extraer HOG de una imagen vacía."
        );
    }

    cv::Mat redimensionada;

    const int interpolacion =
        imagen.cols > ANCHO_HOG
        || imagen.rows > ALTO_HOG
        ? cv::INTER_AREA
        : cv::INTER_CUBIC;

    cv::resize(
        imagen,
        redimensionada,
        cv::Size(ANCHO_HOG, ALTO_HOG),
        0.0,
        0.0,
        interpolacion
    );

    cv::Mat gris;

    if (redimensionada.channels() == 3)
    {
        cv::cvtColor(
            redimensionada,
            gris,
            cv::COLOR_BGR2GRAY
        );
    }
    else if (redimensionada.channels() == 4)
    {
        cv::cvtColor(
            redimensionada,
            gris,
            cv::COLOR_BGRA2GRAY
        );
    }
    else
    {
        gris = redimensionada;
    }

    std::vector<float> descriptor;

    hog.compute(
        gris,
        descriptor,
        cv::Size(8, 8),
        cv::Size(0, 0)
    );

    return descriptor;
}

cv::Mat descriptorAMatriz(
    const std::vector<float>& descriptor
)
{
    if (descriptor.empty())
    {
        throw std::runtime_error(
            "El descriptor HOG está vacío."
        );
    }

    cv::Mat fila(
        1,
        static_cast<int>(descriptor.size()),
        CV_32F
    );

    std::copy(
        descriptor.begin(),
        descriptor.end(),
        fila.ptr<float>(0)
    );

    return fila;
}

cv::Mat convertirABgr(
    const cv::Mat& imagen
)
{
    if (imagen.empty())
    {
        return {};
    }

    if (imagen.channels() == 3)
    {
        return imagen;
    }

    cv::Mat bgr;

    if (imagen.channels() == 4)
    {
        cv::cvtColor(
            imagen,
            bgr,
            cv::COLOR_BGRA2BGR
        );
    }
    else if (imagen.channels() == 1)
    {
        cv::cvtColor(
            imagen,
            bgr,
            cv::COLOR_GRAY2BGR
        );
    }
    else
    {
        throw std::runtime_error(
            "Formato de imagen no compatible para detectar color amarillo."
        );
    }

    return bgr;
}

const cv::Mat& obtenerElementoApertura3x3()
{
    static const cv::Mat elemento =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(3, 3)
        );

    return elemento;
}

const cv::Mat& obtenerElementoCierre5x5()
{
    static const cv::Mat elemento =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(5, 5)
        );

    return elemento;
}

const cv::Mat& obtenerElementoApertura5x5()
{
    static const cv::Mat elemento =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(5, 5)
        );

    return elemento;
}

const cv::Mat& obtenerElementoCierre11x11()
{
    static const cv::Mat elemento =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(11, 11)
        );

    return elemento;
}

cv::Mat crearMascaraAmarilla(
    const cv::Mat& imagen,
    bool refinamiento
)
{
    if (imagen.empty())
    {
        return {};
    }

    const cv::Mat imagenBgr =
        convertirABgr(imagen);

    cv::Mat hsv;

    cv::cvtColor(
        imagenBgr,
        hsv,
        cv::COLOR_BGR2HSV
    );

    cv::Mat mascara;

    cv::inRange(
        hsv,
        cv::Scalar(15, 65, 60),
        cv::Scalar(42, 255, 255),
        mascara
    );

    if (refinamiento)
    {
        cv::morphologyEx(
            mascara,
            mascara,
            cv::MORPH_CLOSE,
            obtenerElementoCierre11x11()
        );

        cv::morphologyEx(
            mascara,
            mascara,
            cv::MORPH_OPEN,
            obtenerElementoApertura5x5()
        );
    }
    else
    {
        cv::morphologyEx(
            mascara,
            mascara,
            cv::MORPH_OPEN,
            obtenerElementoApertura3x3()
        );

        cv::morphologyEx(
            mascara,
            mascara,
            cv::MORPH_CLOSE,
            obtenerElementoCierre5x5()
        );
    }

    return mascara;
}

double calcularPorcentajeAmarillo(
    const cv::Mat& integralMascara,
    const cv::Rect& caja
)
{
    if (
        integralMascara.empty()
        || caja.width <= 0
        || caja.height <= 0
    )
    {
        return 0.0;
    }

    const int x1 = caja.x;
    const int y1 = caja.y;
    const int x2 = caja.x + caja.width;
    const int y2 = caja.y + caja.height;

    const int pixelesAmarillos =
        integralMascara.at<int>(y2, x2)
        - integralMascara.at<int>(y1, x2)
        - integralMascara.at<int>(y2, x1)
        + integralMascara.at<int>(y1, x1);

    const double pixelesTotales =
        static_cast<double>(
            caja.area()
        );

    if (pixelesTotales <= 0.0)
    {
        return 0.0;
    }

    return static_cast<double>(
        pixelesAmarillos
    ) / pixelesTotales;
}

double calcularIoU(
    const cv::Rect& cajaA,
    const cv::Rect& cajaB
)
{
    const cv::Rect interseccion =
        cajaA & cajaB;

    const double areaInterseccion =
        static_cast<double>(
            interseccion.area()
        );

    const double areaUnion =
        static_cast<double>(
            cajaA.area()
            + cajaB.area()
            - interseccion.area()
        );

    if (areaUnion <= 0.0)
    {
        return 0.0;
    }

    return areaInterseccion / areaUnion;
}

double calcularSolapamientoMenor(
    const cv::Rect& cajaA,
    const cv::Rect& cajaB
)
{
    const cv::Rect interseccion =
        cajaA & cajaB;

    const double areaInterseccion =
        static_cast<double>(
            interseccion.area()
        );

    const double areaMenor =
        static_cast<double>(
            std::min(
                cajaA.area(),
                cajaB.area()
            )
        );

    if (areaMenor <= 0.0)
    {
        return 0.0;
    }

    return areaInterseccion / areaMenor;
}

double calcularSolapamientoHorizontal(
    const cv::Rect& cajaA,
    const cv::Rect& cajaB
)
{
    const int izquierda =
        std::max(
            cajaA.x,
            cajaB.x
        );

    const int derecha =
        std::min(
            cajaA.x + cajaA.width,
            cajaB.x + cajaB.width
        );

    const int anchoInterseccion =
        std::max(
            0,
            derecha - izquierda
        );

    const int anchoMenor =
        std::min(
            cajaA.width,
            cajaB.width
        );

    if (anchoMenor <= 0)
    {
        return 0.0;
    }

    return static_cast<double>(
        anchoInterseccion
    ) / anchoMenor;
}

cv::Point2d calcularCentro(
    const cv::Rect& caja
)
{
    return cv::Point2d(
        caja.x + caja.width / 2.0,
        caja.y + caja.height / 2.0
    );
}

cv::Rect limitarCaja(
    const cv::Rect& caja,
    const cv::Size& tamanoImagen
)
{
    const cv::Rect limites(
        0,
        0,
        tamanoImagen.width,
        tamanoImagen.height
    );

    return caja & limites;
}

bool centrosCercanos(
    const cv::Rect& cajaA,
    const cv::Rect& cajaB
)
{
    const cv::Point2d centroA =
        calcularCentro(cajaA);

    const cv::Point2d centroB =
        calcularCentro(cajaB);

    const double distanciaX =
        std::abs(
            centroA.x - centroB.x
        );

    const double distanciaY =
        std::abs(
            centroA.y - centroB.y
        );

    const double anchoReferencia =
        static_cast<double>(
            std::max(
                cajaA.width,
                cajaB.width
            )
        );

    const double altoReferencia =
        static_cast<double>(
            std::max(
                cajaA.height,
                cajaB.height
            )
        );

    return distanciaX
            <= anchoReferencia * 0.70
        && distanciaY
            <= altoReferencia * 0.65;
}

bool cajasRelacionadasParaAgrupar(
    const cv::Rect& cajaA,
    const cv::Rect& cajaB
)
{
    const double iou =
        calcularIoU(
            cajaA,
            cajaB
        );

    const double solapamientoMenor =
        calcularSolapamientoMenor(
            cajaA,
            cajaB
        );

    if (iou >= 0.15)
    {
        return true;
    }

    if (solapamientoMenor >= 0.40)
    {
        return true;
    }

    return centrosCercanos(
        cajaA,
        cajaB
    ) && solapamientoMenor >= 0.10;
}

void combinarDetecciones(
    DeteccionInterna& destino,
    const DeteccionInterna& origen,
    const cv::Size& tamanoImagen
)
{
    destino.caja =
        limitarCaja(
            destino.caja | origen.caja,
            tamanoImagen
        );

    destino.scoreSvm =
        std::max(
            destino.scoreSvm,
            origen.scoreSvm
        );

    destino.porcentajeAmarillo =
        std::max(
            destino.porcentajeAmarillo,
            origen.porcentajeAmarillo
        );

    destino.scoreCombinado =
        std::max(
            destino.scoreCombinado,
            origen.scoreCombinado
        );

    destino.cantidadVentanas +=
        origen.cantidadVentanas;
}

std::vector<DeteccionInterna> agruparDetecciones(
    std::vector<DeteccionInterna> candidatas,
    const cv::Size& tamanoImagen
)
{
    if (candidatas.empty())
    {
        return {};
    }

    std::sort(
        candidatas.begin(),
        candidatas.end(),
        [](
            const DeteccionInterna& a,
            const DeteccionInterna& b
        )
        {
            return a.scoreCombinado
                > b.scoreCombinado;
        }
    );

    std::vector<DeteccionInterna> grupos;

    for (
        const DeteccionInterna& candidata :
        candidatas
    )
    {
        int mejorGrupo = -1;
        double mejorRelacion = 0.0;

        for (
            std::size_t indice = 0;
            indice < grupos.size();
            ++indice
        )
        {
            if (
                !cajasRelacionadasParaAgrupar(
                    candidata.caja,
                    grupos[indice].caja
                )
            )
            {
                continue;
            }

            const double relacion =
                std::max(
                    calcularIoU(
                        candidata.caja,
                        grupos[indice].caja
                    ),
                    calcularSolapamientoMenor(
                        candidata.caja,
                        grupos[indice].caja
                    )
                );

            if (
                mejorGrupo < 0
                || relacion > mejorRelacion
            )
            {
                mejorGrupo =
                    static_cast<int>(indice);

                mejorRelacion = relacion;
            }
        }

        if (mejorGrupo >= 0)
        {
            combinarDetecciones(
                grupos[
                    static_cast<std::size_t>(
                        mejorGrupo
                    )
                ],
                candidata,
                tamanoImagen
            );
        }
        else
        {
            grupos.push_back(candidata);
        }
    }

    bool huboUnion = true;

    while (huboUnion)
    {
        huboUnion = false;

        for (
            std::size_t i = 0;
            i < grupos.size() && !huboUnion;
            ++i
        )
        {
            for (
                std::size_t j = i + 1;
                j < grupos.size();
                ++j
            )
            {
                if (
                    !cajasRelacionadasParaAgrupar(
                        grupos[i].caja,
                        grupos[j].caja
                    )
                )
                {
                    continue;
                }

                combinarDetecciones(
                    grupos[i],
                    grupos[j],
                    tamanoImagen
                );

                grupos.erase(
                    grupos.begin()
                    + static_cast<std::ptrdiff_t>(j)
                );

                huboUnion = true;
                break;
            }
        }
    }

    const int areaMinima =
        static_cast<int>(
            tamanoImagen.area() * 0.010
        );

    grupos.erase(
        std::remove_if(
            grupos.begin(),
            grupos.end(),
            [areaMinima](
                const DeteccionInterna& deteccion
            )
            {
                return deteccion.caja.area()
                        < areaMinima
                    || deteccion.cantidadVentanas
                        < 2;
            }
        ),
        grupos.end()
    );

    std::sort(
        grupos.begin(),
        grupos.end(),
        [](
            const DeteccionInterna& a,
            const DeteccionInterna& b
        )
        {
            if (
                a.cantidadVentanas
                != b.cantidadVentanas
            )
            {
                return a.cantidadVentanas
                    > b.cantidadVentanas;
            }

            if (
                std::abs(
                    a.scoreCombinado
                    - b.scoreCombinado
                ) > 0.001
            )
            {
                return a.scoreCombinado
                    > b.scoreCombinado;
            }

            return a.caja.area()
                > b.caja.area();
        }
    );

    return grupos;
}

cv::Rect refinarCajaConColorAmarillo(
    const cv::Mat& imagen,
    const cv::Rect& cajaInicial
)
{
    const cv::Rect cajaValida =
        limitarCaja(
            cajaInicial,
            imagen.size()
        );

    if (
        cajaValida.width <= 0
        || cajaValida.height <= 0
    )
    {
        return cajaInicial;
    }

    const cv::Mat region =
        imagen(cajaValida);

    cv::Mat mascara =
        crearMascaraAmarilla(
            region,
            true
        );

    std::vector<std::vector<cv::Point>> contornos;

    cv::findContours(
        mascara,
        contornos,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    if (contornos.empty())
    {
        return cajaInicial;
    }

    std::size_t indiceMayor = 0;
    double areaMayor = 0.0;

    for (
        std::size_t indice = 0;
        indice < contornos.size();
        ++indice
    )
    {
        const double area =
            cv::contourArea(
                contornos[indice]
            );

        if (area > areaMayor)
        {
            areaMayor = area;
            indiceMayor = indice;
        }
    }

    const double areaMinima =
        cajaValida.area() * 0.015;

    if (areaMayor < areaMinima)
    {
        return cajaInicial;
    }

    cv::Rect cajaAmarilla =
        cv::boundingRect(
            contornos[indiceMayor]
        );

    cajaAmarilla.x += cajaValida.x;
    cajaAmarilla.y += cajaValida.y;

    const int margenIzquierdo =
        static_cast<int>(
            cajaAmarilla.width * 0.05
        );

    const int margenDerecho =
        static_cast<int>(
            cajaAmarilla.width * 0.05
        );

    const int margenSuperior =
        static_cast<int>(
            cajaAmarilla.height * 0.06
        );

    const int margenInferior =
        static_cast<int>(
            cajaAmarilla.height * 0.22
        );

    const cv::Rect cajaRefinada(
        cajaAmarilla.x - margenIzquierdo,
        cajaAmarilla.y - margenSuperior,
        cajaAmarilla.width
            + margenIzquierdo
            + margenDerecho,
        cajaAmarilla.height
            + margenSuperior
            + margenInferior
    );

    return limitarCaja(
        cajaRefinada,
        imagen.size()
    );
}

std::vector<DeteccionInterna>
eliminarDuplicadosFinales(
    std::vector<DeteccionInterna> detecciones
)
{
    std::sort(
        detecciones.begin(),
        detecciones.end(),
        [](
            const DeteccionInterna& a,
            const DeteccionInterna& b
        )
        {
            if (
                a.cantidadVentanas
                != b.cantidadVentanas
            )
            {
                return a.cantidadVentanas
                    > b.cantidadVentanas;
            }

            return a.scoreCombinado
                > b.scoreCombinado;
        }
    );

    std::vector<DeteccionInterna> resultado;

    for (
        const DeteccionInterna& candidata :
        detecciones
    )
    {
        bool duplicada = false;

        for (
            const DeteccionInterna& aceptada :
            resultado
        )
        {
            const double iou =
                calcularIoU(
                    candidata.caja,
                    aceptada.caja
                );

            const double solapamientoMenor =
                calcularSolapamientoMenor(
                    candidata.caja,
                    aceptada.caja
                );

            const double solapamientoHorizontal =
                calcularSolapamientoHorizontal(
                    candidata.caja,
                    aceptada.caja
                );

            if (
                iou >= 0.20
                || solapamientoMenor >= 0.45
                || (
                    solapamientoHorizontal >= 0.70
                    && solapamientoMenor >= 0.22
                )
            )
            {
                duplicada = true;
                break;
            }
        }

        if (!duplicada)
        {
            resultado.push_back(candidata);
        }
    }

    return resultado;
}

ImagenTrabajo prepararImagenTrabajo(
    const cv::Mat& original
)
{
    const int ladoMayor =
        std::max(
            original.cols,
            original.rows
        );

    if (ladoMayor <= LADO_MAXIMO_BUSQUEDA)
    {
        return {
            original.clone(),
            1.0,
            1.0
        };
    }

    const double factor =
        static_cast<double>(
            LADO_MAXIMO_BUSQUEDA
        ) / ladoMayor;

    cv::Mat reducida;

    cv::resize(
        original,
        reducida,
        cv::Size(),
        factor,
        factor,
        cv::INTER_AREA
    );

    return {
        reducida,
        static_cast<double>(original.cols)
            / reducida.cols,
        static_cast<double>(original.rows)
            / reducida.rows
    };
}

cv::Rect proyectarCajaAOriginal(
    const cv::Rect& caja,
    double escalaX,
    double escalaY,
    const cv::Size& tamanoOriginal
)
{
    const cv::Rect proyectada(
        static_cast<int>(
            std::round(
                caja.x * escalaX
            )
        ),
        static_cast<int>(
            std::round(
                caja.y * escalaY
            )
        ),
        static_cast<int>(
            std::round(
                caja.width * escalaX
            )
        ),
        static_cast<int>(
            std::round(
                caja.height * escalaY
            )
        )
    );

    return limitarCaja(
        proyectada,
        tamanoOriginal
    );
}

std::vector<DeteccionInterna> buscarTaxis(
    const cv::Mat& imagen,
    const cv::Ptr<cv::ml::SVM>& svm,
    cv::HOGDescriptor& hog
)
{
    std::vector<DeteccionInterna> candidatas;

    const cv::Mat mascaraAmarilla =
        crearMascaraAmarilla(
            imagen,
            false
        );

    cv::Mat integralMascara;

    cv::integral(
        mascaraAmarilla / 255,
        integralMascara,
        CV_32S
    );

    const std::vector<double> escalas = {
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.25,
        3.50,
        3.75,
        4.00
    };

    for (double escala : escalas)
    {
        const int anchoVentana =
            static_cast<int>(
                std::round(
                    ANCHO_HOG * escala
                )
            );

        const int altoVentana =
            static_cast<int>(
                std::round(
                    ALTO_HOG * escala
                )
            );

        if (
            anchoVentana > imagen.cols
            || altoVentana > imagen.rows
        )
        {
            continue;
        }

        for (
            int y = 0;
            y <= imagen.rows - altoVentana;
            y += PASO_Y
        )
        {
            for (
                int x = 0;
                x <= imagen.cols - anchoVentana;
                x += PASO_X
            )
            {
                const cv::Rect caja(
                    x,
                    y,
                    anchoVentana,
                    altoVentana
                );

                const double porcentajeAmarillo =
                    calcularPorcentajeAmarillo(
                        integralMascara,
                        caja
                    );

                if (
                    porcentajeAmarillo
                    < PORCENTAJE_MINIMO_AMARILLO
                )
                {
                    continue;
                }

                const cv::Mat region =
                    imagen(caja);

                const std::vector<float> descriptor =
                    extraerDescriptor(
                        region,
                        hog
                    );

                const cv::Mat fila =
                    descriptorAMatriz(
                        descriptor
                    );

                const float salidaBruta =
                    svm->predict(
                        fila,
                        cv::noArray(),
                        cv::ml::StatModel::RAW_OUTPUT
                    );

                /*
                 * En este modelo, la clase positiva queda del lado
                 * negativo del hiperplano. Por eso se invierte el
                 * signo para expresar una confianza positiva.
                 */
                const float scoreSvm =
                    -salidaBruta;

                if (scoreSvm < SCORE_MINIMO_SVM)
                {
                    continue;
                }

                const double scoreCombinado =
                    static_cast<double>(
                        scoreSvm
                    )
                    + porcentajeAmarillo * 1.5;

                candidatas.push_back(
                    {
                        caja,
                        scoreSvm,
                        porcentajeAmarillo,
                        scoreCombinado,
                        1
                    }
                );
            }
        }
    }

    std::vector<DeteccionInterna> finales =
        agruparDetecciones(
            std::move(candidatas),
            imagen.size()
        );

    for (
        DeteccionInterna& deteccion :
        finales
    )
    {
        deteccion.caja =
            refinarCajaConColorAmarillo(
                imagen,
                deteccion.caja
            );
    }

    return eliminarDuplicadosFinales(
        std::move(finales)
    );
}
}

DetectorHOG::DetectorHOG()
    : hog_(
        cv::Size(ANCHO_HOG, ALTO_HOG),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    )
{
}

bool DetectorHOG::cargarModelo(
    const std::string& rutaModelo
)
{
    try
    {
        svm_ =
            cv::Algorithm::load<cv::ml::SVM>(
                rutaModelo
            );

        if (svm_.empty())
        {
            std::cerr
                << "ERROR: no se pudo cargar el modelo: "
                << rutaModelo
                << '\n';

            return false;
        }

        const int longitudHog =
            static_cast<int>(
                hog_.getDescriptorSize()
            );

        if (longitudHog != svm_->getVarCount())
        {
            std::cerr
                << "ERROR: el HOG genera "
                << longitudHog
                << " características, pero el modelo espera "
                << svm_->getVarCount()
                << ".\n";

            svm_.release();

            return false;
        }

        std::cout
            << "Modelo HOG + SVM cargado correctamente.\n"
            << "Ruta: "
            << rutaModelo
            << '\n'
            << "Descriptor HOG: "
            << longitudHog
            << " características\n"
            << "Score mínimo: "
            << SCORE_MINIMO_SVM
            << '\n';

        return true;
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "ERROR de OpenCV al cargar el modelo:\n"
            << error.what()
            << '\n';

        svm_.release();

        return false;
    }
}

bool DetectorHOG::modeloCargado() const
{
    return !svm_.empty();
}

std::vector<TaxiDetectado> DetectorHOG::detectar(
    const cv::Mat& frame
)
{
    if (frame.empty())
    {
        return {};
    }

    if (!modeloCargado())
    {
        throw std::runtime_error(
            "El modelo SVM no ha sido cargado."
        );
    }

    ImagenTrabajo trabajo =
        prepararImagenTrabajo(frame);

    std::vector<DeteccionInterna> internas =
        buscarTaxis(
            trabajo.imagen,
            svm_,
            hog_
        );

    std::vector<TaxiDetectado> resultado;

    resultado.reserve(
        internas.size()
    );

    for (
        const DeteccionInterna& deteccion :
        internas
    )
    {
        const cv::Rect cajaOriginal =
            proyectarCajaAOriginal(
                deteccion.caja,
                trabajo.escalaX,
                trabajo.escalaY,
                frame.size()
            );

        if (
            cajaOriginal.width <= 0
            || cajaOriginal.height <= 0
        )
        {
            continue;
        }

        resultado.push_back(
            {
                cajaOriginal,
                deteccion.scoreSvm
            }
        );
    }

    return resultado;
}
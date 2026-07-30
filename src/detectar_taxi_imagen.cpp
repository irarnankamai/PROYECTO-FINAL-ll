#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

constexpr int ANCHO_HOG = 128;
constexpr int ALTO_HOG = 64;

struct Deteccion
{
    cv::Rect caja;

    float scoreSvm = 0.0F;

    double porcentajeAmarillo = 0.0;

    double scoreCombinado = 0.0;

    int cantidadVentanas = 1;
};

struct Estadisticas
{
    std::size_t ventanasEvaluadas = 0;

    std::size_t ventanasConAmarillo = 0;

    std::size_t prediccionesPositivas = 0;

    std::size_t candidatasPorScore = 0;

    std::size_t gruposFinales = 0;
};

cv::HOGDescriptor crearHog()
{
    return cv::HOGDescriptor(
        cv::Size(ANCHO_HOG, ALTO_HOG),
        cv::Size(16, 16),
        cv::Size(8, 8),
        cv::Size(8, 8),
        9
    );
}

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

double calcularPorcentajeAmarillo(
    const cv::Mat& region
)
{
    if (region.empty())
    {
        return 0.0;
    }

    cv::Mat hsv;

    cv::cvtColor(
        region,
        hsv,
        cv::COLOR_BGR2HSV
    );

    /*
     * Rango aproximado del amarillo en OpenCV HSV.
     *
     * H: 15 a 42
     * S: 65 a 255
     * V: 60 a 255
     */
    cv::Mat mascaraAmarillo;

    cv::inRange(
        hsv,
        cv::Scalar(15, 65, 60),
        cv::Scalar(42, 255, 255),
        mascaraAmarillo
    );

    /*
     * Elimina pequeños puntos aislados.
     */
    const cv::Mat elementoApertura =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(3, 3)
        );

    cv::morphologyEx(
        mascaraAmarillo,
        mascaraAmarillo,
        cv::MORPH_OPEN,
        elementoApertura
    );

    /*
     * Une pequeñas zonas amarillas cercanas.
     */
    const cv::Mat elementoCierre =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(5, 5)
        );

    cv::morphologyEx(
        mascaraAmarillo,
        mascaraAmarillo,
        cv::MORPH_CLOSE,
        elementoCierre
    );

    const double pixelesAmarillos =
        static_cast<double>(
            cv::countNonZero(mascaraAmarillo)
        );

    const double totalPixeles =
        static_cast<double>(
            mascaraAmarillo.total()
        );

    if (totalPixeles <= 0.0)
    {
        return 0.0;
    }

    return pixelesAmarillos / totalPixeles;
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
    const cv::Rect limitesImagen(
        0,
        0,
        tamanoImagen.width,
        tamanoImagen.height
    );

    return caja & limitesImagen;
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
        std::abs(centroA.x - centroB.x);

    const double distanciaY =
        std::abs(centroA.y - centroB.y);

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

    /*
     * Casos en los que dos ventanas probablemente
     * pertenecen al mismo taxi:
     *
     * 1. Tienen una intersección IoU relevante.
     * 2. Una ventana está bastante contenida en la otra.
     * 3. Los centros están próximos y existe algo
     *    de solapamiento.
     */
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
    Deteccion& destino,
    const Deteccion& origen,
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

std::vector<Deteccion> agruparDetecciones(
    std::vector<Deteccion> candidatas,
    const cv::Size& tamanoImagen
)
{
    if (candidatas.empty())
    {
        return {};
    }

    /*
     * Procesamos primero las ventanas de mayor confianza.
     */
    std::sort(
        candidatas.begin(),
        candidatas.end(),
        [](const Deteccion& a, const Deteccion& b)
        {
            return a.scoreCombinado
                > b.scoreCombinado;
        }
    );

    std::vector<Deteccion> grupos;

    for (const Deteccion& candidata : candidatas)
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

    /*
     * Segunda pasada.
     *
     * Al crecer las cajas durante la primera pasada,
     * algunos grupos pueden terminar representando
     * el mismo vehículo.
     */
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

    /*
     * Eliminamos grupos que se formaron con pocas
     * ventanas o cuya caja final es muy pequeña.
     */
    const int areaImagen =
        tamanoImagen.area();

    const int areaMinima =
        static_cast<int>(
            areaImagen * 0.010
        );

    grupos.erase(
        std::remove_if(
            grupos.begin(),
            grupos.end(),
            [areaMinima](
                const Deteccion& deteccion
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

    /*
     * Ordenamos los grupos finales.
     *
     * Se da prioridad a:
     * 1. Cantidad de ventanas agrupadas.
     * 2. Score combinado.
     * 3. Área de la caja.
     */
    std::sort(
        grupos.begin(),
        grupos.end(),
        [](const Deteccion& a, const Deteccion& b)
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
    const cv::Rect limites(
        0,
        0,
        imagen.cols,
        imagen.rows
    );

    const cv::Rect cajaValida =
        cajaInicial & limites;

    if (
        cajaValida.width <= 0
        || cajaValida.height <= 0
    )
    {
        return cajaInicial;
    }

    const cv::Mat region =
        imagen(cajaValida);

    cv::Mat hsv;

    cv::cvtColor(
        region,
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

    const cv::Mat elementoCierre =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(11, 11)
        );

    cv::morphologyEx(
        mascara,
        mascara,
        cv::MORPH_CLOSE,
        elementoCierre
    );

    const cv::Mat elementoApertura =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(5, 5)
        );

    cv::morphologyEx(
        mascara,
        mascara,
        cv::MORPH_OPEN,
        elementoApertura
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

    /*
     * Convertir las coordenadas locales de la región
     * a coordenadas globales de la imagen.
     */
    cajaAmarilla.x += cajaValida.x;
    cajaAmarilla.y += cajaValida.y;

    /*
 * Márgenes pequeños para incluir las partes no amarillas:
 * ruedas, ventanas, parachoques y matrícula.
 */
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
    cv::Rect cajaRefinada(
        cajaAmarilla.x - margenIzquierdo,
        cajaAmarilla.y - margenSuperior,
        cajaAmarilla.width
            + margenIzquierdo
            + margenDerecho,
        cajaAmarilla.height
            + margenSuperior
            + margenInferior
    );

    return cajaRefinada & limites;
}

double calcularSolapamientoHorizontal(
    const cv::Rect& cajaA,
    const cv::Rect& cajaB
)
{
    const int izquierda =
        std::max(cajaA.x, cajaB.x);

    const int derecha =
        std::min(
            cajaA.x + cajaA.width,
            cajaB.x + cajaB.width
        );

    const int anchoInterseccion =
        std::max(0, derecha - izquierda);

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

std::vector<Deteccion> eliminarDuplicadosFinales(
    std::vector<Deteccion> detecciones
)
{
    /*
     * Primero se conservan las detecciones que fueron
     * respaldadas por más ventanas.
     */
    std::sort(
        detecciones.begin(),
        detecciones.end(),
        [](const Deteccion& a, const Deteccion& b)
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

    std::vector<Deteccion> resultado;

    for (const Deteccion& candidata : detecciones)
    {
        bool duplicada = false;

        for (const Deteccion& aceptada : resultado)
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

            /*
             * El segundo cuadro suele ser una parte del mismo
             * automóvil: parachoques, puerta o capó.
             *
             * Se elimina cuando:
             * - existe un IoU razonable; o
             * - comparten gran parte de su ancho y también
             *   una parte relevante de la caja pequeña.
             */
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

std::vector<Deteccion> detectarTaxis(
    const cv::Mat& imagen,
    const cv::Ptr<cv::ml::SVM>& svm,
    cv::HOGDescriptor& hog,
    float scoreMinimo,
    Estadisticas& estadisticas
)
{
    std::vector<Deteccion> candidatas;

    /*
     * Escalas de búsqueda respecto a la ventana
     * base HOG de 128 x 64.
     */
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

    constexpr int pasoX = 24;
    constexpr int pasoY = 16;

    /*
     * Una región debe contener al menos 7 %
     * de píxeles amarillos.
     */
    constexpr double minimoAmarillo = 0.07;

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
            y += pasoY
        )
        {
            for (
                int x = 0;
                x <= imagen.cols - anchoVentana;
                x += pasoX
            )
            {
                ++estadisticas.ventanasEvaluadas;

                const cv::Rect caja(
                    x,
                    y,
                    anchoVentana,
                    altoVentana
                );

                const cv::Mat region =
                    imagen(caja);

                const double porcentajeAmarillo =
                    calcularPorcentajeAmarillo(
                        region
                    );

                if (
                    porcentajeAmarillo
                    < minimoAmarillo
                )
                {
                    continue;
                }

                ++estadisticas.ventanasConAmarillo;

                const std::vector<float> descriptor =
                    extraerDescriptor(
                        region,
                        hog
                    );

                const cv::Mat fila =
                    descriptorAMatriz(
                        descriptor
                    );

                const float prediccion =
                    svm->predict(fila);

                if (prediccion != 1.0F)
                {
                    continue;
                }

                ++estadisticas.prediccionesPositivas;

                const float salidaBruta =
                    svm->predict(
                        fila,
                        cv::noArray(),
                        cv::ml::StatModel::RAW_OUTPUT
                    );

                /*
                 * En el modelo entrenado, la clase positiva
                 * utiliza el signo opuesto de RAW_OUTPUT.
                 */
                const float scoreSvm =
                    -salidaBruta;

                if (scoreSvm < scoreMinimo)
                {
                    continue;
                }

                ++estadisticas.candidatasPorScore;

                /*
                 * Combinamos la confianza del SVM con una
                 * pequeña bonificación por color amarillo.
                 */
                const double scoreCombinado =
                    static_cast<double>(scoreSvm)
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
std::vector<Deteccion> deteccionesFinales =
    agruparDetecciones(
        std::move(candidatas),
        imagen.size()
    );

for (Deteccion& deteccion : deteccionesFinales)
{
    deteccion.caja =
        refinarCajaConColorAmarillo(
            imagen,
            deteccion.caja
        );
}

deteccionesFinales =
    eliminarDuplicadosFinales(
        std::move(deteccionesFinales)
    );

estadisticas.gruposFinales =
    deteccionesFinales.size();

return deteccionesFinales;
}

void dibujarDetecciones(
    cv::Mat& imagen,
    const std::vector<Deteccion>& detecciones
)
{
    for (
        std::size_t indice = 0;
        indice < detecciones.size();
        ++indice
    )
    {
        const Deteccion& deteccion =
            detecciones[indice];

        cv::rectangle(
            imagen,
            deteccion.caja,
            cv::Scalar(0, 255, 0),
            3
        );

        std::ostringstream texto;

        texto
            << "Taxi "
            << indice + 1
            << " | SVM: "
            << std::fixed
            << std::setprecision(2)
            << deteccion.scoreSvm
            << " | Amarillo: "
            << std::setprecision(0)
            << deteccion.porcentajeAmarillo
                * 100.0
            << "%"
            << " | Ventanas: "
            << deteccion.cantidadVentanas;

        int lineaBase = 0;

        const cv::Size tamanoTexto =
            cv::getTextSize(
                texto.str(),
                cv::FONT_HERSHEY_SIMPLEX,
                0.50,
                2,
                &lineaBase
            );

        const int margen = 5;

        int textoX =
            deteccion.caja.x;

        int textoY =
            deteccion.caja.y - 8;

        if (
            textoY - tamanoTexto.height
            - margen * 2 < 0
        )
        {
            textoY =
                deteccion.caja.y
                + tamanoTexto.height
                + margen * 2;
        }

        textoX = std::clamp(
            textoX,
            0,
            std::max(
                0,
                imagen.cols
                - tamanoTexto.width
                - margen * 2
            )
        );

        const int fondoY =
            std::max(
                0,
                textoY
                - tamanoTexto.height
                - margen
            );

        const cv::Rect fondoTexto(
            textoX,
            fondoY,
            std::min(
                tamanoTexto.width
                    + margen * 2,
                imagen.cols - textoX
            ),
            std::min(
                tamanoTexto.height
                    + margen * 2,
                imagen.rows - fondoY
            )
        );

        cv::rectangle(
            imagen,
            fondoTexto,
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        cv::putText(
            imagen,
            texto.str(),
            cv::Point(
                textoX + margen,
                fondoY
                + tamanoTexto.height
                + margen / 2
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.50,
            cv::Scalar(0, 0, 0),
            2
        );
        
    }
}

struct ImagenTrabajo
{
    cv::Mat imagen;
    double escalaX = 1.0;
    double escalaY = 1.0;
};
ImagenTrabajo prepararImagenTrabajo(
    const cv::Mat& original
)
{
    constexpr int ladoMaximo = 1600;

    const int ladoMayor =
        std::max(
            original.cols,
            original.rows
        );

    if (ladoMayor <= ladoMaximo)
    {
        return {
            original.clone(),
            1.0,
            1.0
        };
    }

    const double factor =
        static_cast<double>(ladoMaximo)
        / ladoMayor;

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
    cv::Rect proyectada(
        static_cast<int>(
            std::round(caja.x * escalaX)
        ),
        static_cast<int>(
            std::round(caja.y * escalaY)
        ),
        static_cast<int>(
            std::round(caja.width * escalaX)
        ),
        static_cast<int>(
            std::round(caja.height * escalaY)
        )
    );

    return proyectada & cv::Rect(
        0,
        0,
        tamanoOriginal.width,
        tamanoOriginal.height
    );
}


int main(int argc, char* argv[])
{
    try
    {
        if (argc < 2 || argc > 3)
        {
            std::cerr
                << "Uso:\n"
                << argv[0]
                << " <ruta_imagen> [score_minimo]\n\n"
                << "Ejemplo:\n"
                << argv[0]
                << " imagen.jpg 1.0\n";

            return 1;
        }

        const fs::path rutaImagen =
            argv[1];

        float scoreMinimo = 1.0F;

        if (argc == 3)
        {
            scoreMinimo =
                std::stof(argv[2]);
        }

        if (!fs::exists(rutaImagen))
        {
            throw std::runtime_error(
                "No existe la imagen: "
                + rutaImagen.string()
            );
        }

        const fs::path raiz =
            fs::current_path();

        const fs::path rutaModelo =
            raiz
            / "models/hog_svm_taxi.yml";

        if (!fs::exists(rutaModelo))
        {
            throw std::runtime_error(
                "No existe el modelo: "
                + rutaModelo.string()
            );
        }

        cv::Mat imagen =
            cv::imread(
                rutaImagen.string()
            );

        if (imagen.empty())
        {
            throw std::runtime_error(
                "No se pudo leer la imagen."
            );
        }

        cv::Ptr<cv::ml::SVM> svm =
            cv::Algorithm::load<cv::ml::SVM>(
                rutaModelo.string()
            );

        if (svm.empty())
        {
            throw std::runtime_error(
                "No se pudo cargar el modelo SVM."
            );
        }

        cv::HOGDescriptor hog =
            crearHog();

        if (
            static_cast<int>(
                hog.getDescriptorSize()
            ) != svm->getVarCount()
        )
        {
            std::ostringstream mensaje;

            mensaje
                << "El descriptor HOG tiene "
                << hog.getDescriptorSize()
                << " características, pero el modelo espera "
                << svm->getVarCount()
                << ".";

            throw std::runtime_error(
                mensaje.str()
            );
        }

        ImagenTrabajo trabajo =
    prepararImagenTrabajo(imagen);

std::cout
    << "Imagen usada para búsqueda: "
    << trabajo.imagen.cols
    << "x"
    << trabajo.imagen.rows
    << '\n';

Estadisticas estadisticas;

const int64 inicio =
    cv::getTickCount();

std::vector<Deteccion> detecciones =
    detectarTaxis(
        trabajo.imagen,
        svm,
        hog,
        scoreMinimo,
        estadisticas
    );

/*
 * Regresar las cajas a las coordenadas de la
 * fotografía original.
 */
for (Deteccion& deteccion : detecciones)
{
    deteccion.caja =
        proyectarCajaAOriginal(
            deteccion.caja,
            trabajo.escalaX,
            trabajo.escalaY,
            imagen.size()
        );
}

        const double tiempoSegundos =
            (
                cv::getTickCount()
                - inicio
            ) / cv::getTickFrequency();

        cv::Mat resultado =
            imagen.clone();

        dibujarDetecciones(
            resultado,
            detecciones
        );

        const fs::path carpetaResultados =
            raiz / "resultados";

        fs::create_directories(
            carpetaResultados
        );

        const fs::path rutaSalida =
            carpetaResultados
            / "deteccion_taxi_v2.jpg";

        if (
            !cv::imwrite(
                rutaSalida.string(),
                resultado
            )
        )
        {
            throw std::runtime_error(
                "No se pudo guardar la imagen de resultado."
            );
        }

        std::cout
            << "\n=============================================\n"
            << "DETECTOR DE TAXIS AMARILLOS V2\n"
            << "=============================================\n"
            << "Imagen:                  "
            << rutaImagen
            << '\n'
            << "Dimensiones:             "
            << imagen.cols
            << "x"
            << imagen.rows
            << '\n'
            << "Score mínimo:            "
            << scoreMinimo
            << '\n'
            << "Ventanas evaluadas:      "
            << estadisticas.ventanasEvaluadas
            << '\n'
            << "Ventanas con amarillo:   "
            << estadisticas.ventanasConAmarillo
            << '\n'
            << "Positivas según SVM:     "
            << estadisticas.prediccionesPositivas
            << '\n'
            << "Candidatas por score:    "
            << estadisticas.candidatasPorScore
            << '\n'
            << "Detecciones finales:     "
            << detecciones.size()
            << '\n'
            << "Tiempo de procesamiento: "
            << std::fixed
            << std::setprecision(3)
            << tiempoSegundos
            << " segundos\n";

        if (tiempoSegundos > 0.0)
        {
            std::cout
                << "FPS equivalente:         "
                << std::setprecision(2)
                << 1.0 / tiempoSegundos
                << '\n';
        }

        std::cout
            << "Resultado guardado en:\n"
            << rutaSalida
            << "\n\n";

        for (
            std::size_t indice = 0;
            indice < detecciones.size();
            ++indice
        )
        {
            const Deteccion& deteccion =
                detecciones[indice];

            std::cout
                << "Taxi "
                << indice + 1
                << ": x="
                << deteccion.caja.x
                << ", y="
                << deteccion.caja.y
                << ", ancho="
                << deteccion.caja.width
                << ", alto="
                << deteccion.caja.height
                << ", score="
                << deteccion.scoreSvm
                << ", amarillo="
                << deteccion.porcentajeAmarillo
                    * 100.0
                << "%, ventanas="
                << deteccion.cantidadVentanas
                << '\n';
        }

        cv::namedWindow(
            "Detector de taxis amarillos V2",
            cv::WINDOW_NORMAL
        );

        cv::imshow(
            "Detector de taxis amarillos V2",
            resultado
        );

        std::cout
            << "\nPresiona cualquier tecla "
            << "sobre la ventana para cerrar.\n";

        cv::waitKey(0);

        cv::destroyAllWindows();

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "\nERROR: "
            << error.what()
            << '\n';

        return 1;
    }
}
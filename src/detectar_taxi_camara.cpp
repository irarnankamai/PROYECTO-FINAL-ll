#include "capturador.hpp"
#include "cliente_api.hpp"
#include "detector_hog.hpp"
#include "grabador_video.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{
constexpr int ANCHO_CAMARA = 640;
constexpr int ALTO_CAMARA = 360;

constexpr double FPS_CAMARA_DESEADO = 30.0;
constexpr double DURACION_VIDEO_SEGUNDOS = 5.0;

constexpr int BITRATE_VIDEO_BITS_SEGUNDO = 2'000'000;
constexpr int CALIDAD_JPEG = 90;

constexpr int INTERVALO_DETECCION_PREDETERMINADO = 10;

constexpr int DETECCIONES_POSITIVAS_PARA_CONFIRMAR = 2;
constexpr float SCORE_MINIMO_PARA_EVENTO = 1.0F;

constexpr int RESULTADOS_VACIOS_PARA_REACTIVAR = 5;
constexpr double SEGUNDOS_SIN_TAXI_PARA_REACTIVAR = 8.0;

constexpr int LECTURAS_FALLIDAS_PARA_RECONECTAR = 5;
constexpr int INTENTOS_MAXIMOS_RECONEXION = 3;
constexpr int ESPERA_RECONEXION_MS = 500;

constexpr int ALTO_PANEL = 185;

/*
 * Conserva toda la evidencia del evento en memoria RAM.
 *
 * imagenJpeg:
 * imagen JPEG codificada por Capturador.
 *
 * videoMp4:
 * video MP4 codificado por GrabadorVideo.
 */
struct DatosEventoActual
{
    std::vector<unsigned char> imagenJpeg;
    std::vector<unsigned char> videoMp4;

    std::string fechaHora;

    float scoreSvm = 0.0F;
    float confianzaNormalizada = 0.0F;

    bool pendienteEnvio = false;

    void limpiar()
    {
        imagenJpeg.clear();
        imagenJpeg.shrink_to_fit();

        videoMp4.clear();
        videoMp4.shrink_to_fit();

        fechaHora.clear();

        scoreSvm = 0.0F;
        confianzaNormalizada = 0.0F;

        pendienteEnvio = false;
    }
};

int convertirEntero(
    const char* texto,
    int valorPredeterminado,
    int minimo
)
{
    if (texto == nullptr)
    {
        return valorPredeterminado;
    }

    try
    {
        const int valor =
            std::stoi(texto);

        return std::max(
            valor,
            minimo
        );
    }
    catch (const std::exception&)
    {
        return valorPredeterminado;
    }
}

std::string generarFechaHoraEvento()
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
        << std::put_time(
               &tiempoLocal,
               "%Y-%m-%dT%H:%M:%S"
           );

    return texto.str();
}

float obtenerMejorScore(
    const std::vector<TaxiDetectado>& detecciones
)
{
    if (detecciones.empty())
    {
        return 0.0F;
    }

    const auto mejor =
        std::max_element(
            detecciones.begin(),
            detecciones.end(),
            [](
                const TaxiDetectado& izquierda,
                const TaxiDetectado& derecha
            )
            {
                return
                    izquierda.score
                    < derecha.score;
            }
        );

    if (
        mejor == detecciones.end()
        || !std::isfinite(mejor->score)
    )
    {
        return 0.0F;
    }

    return mejor->score;
}

double obtenerMemoriaRAMMegabytes()
{
#if defined(__linux__)

    std::ifstream estado(
        "/proc/self/status"
    );

    if (!estado.is_open())
    {
        return 0.0;
    }

    std::string clave;
    long valorKilobytes = 0;
    std::string unidad;

    while (estado >> clave)
    {
        if (clave == "VmRSS:")
        {
            estado
                >> valorKilobytes
                >> unidad;

            return
                static_cast<double>(
                    valorKilobytes
                ) / 1024.0;
        }

        std::string restoLinea;

        std::getline(
            estado,
            restoLinea
        );
    }

#endif

    return 0.0;
}

double convertirBytesAMegabytes(
    std::size_t bytes
)
{
    constexpr double BYTES_POR_MEGABYTE =
        1024.0 * 1024.0;

    return
        static_cast<double>(bytes)
        / BYTES_POR_MEGABYTE;
}

void configurarCamara(
    cv::VideoCapture& camara
)
{
#if defined(__linux__)

    camara.set(
        cv::CAP_PROP_FOURCC,
        cv::VideoWriter::fourcc(
            'Y',
            'U',
            'Y',
            'V'
        )
    );

#endif

    camara.set(
        cv::CAP_PROP_FRAME_WIDTH,
        ANCHO_CAMARA
    );

    camara.set(
        cv::CAP_PROP_FRAME_HEIGHT,
        ALTO_CAMARA
    );

    camara.set(
        cv::CAP_PROP_FPS,
        FPS_CAMARA_DESEADO
    );

    camara.set(
        cv::CAP_PROP_BUFFERSIZE,
        1
    );
}

bool abrirCamara(
    cv::VideoCapture& camara,
    int indiceCamara
)
{
    camara.release();

#if defined(__linux__)

    if (
        !camara.open(
            indiceCamara,
            cv::CAP_V4L2
        )
    )
    {
        std::cerr
            << "No se pudo abrir la cámara con V4L2."
            << '\n'
            << "Intentando apertura automática..."
            << '\n';

        camara.release();

        camara.open(
            indiceCamara
        );
    }

#elif defined(_WIN32)

    if (
        !camara.open(
            indiceCamara,
            cv::CAP_DSHOW
        )
    )
    {
        std::cerr
            << "No se pudo abrir la cámara con DirectShow."
            << '\n'
            << "Intentando apertura automática..."
            << '\n';

        camara.release();

        camara.open(
            indiceCamara
        );
    }

#else

    camara.open(
        indiceCamara
    );

#endif

    if (!camara.isOpened())
    {
        return false;
    }

    configurarCamara(
        camara
    );

    return true;
}

bool reconectarCamara(
    cv::VideoCapture& camara,
    int indiceCamara,
    cv::Mat& primerFrame
)
{
    std::cerr
        << "Intentando reconectar la cámara "
        << indiceCamara
        << "..."
        << '\n';

    for (
        int intento = 1;
        intento <= INTENTOS_MAXIMOS_RECONEXION;
        ++intento
    )
    {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                ESPERA_RECONEXION_MS
            )
        );

        if (
            !abrirCamara(
                camara,
                indiceCamara
            )
        )
        {
            std::cerr
                << "Intento de reconexión "
                << intento
                << '/'
                << INTENTOS_MAXIMOS_RECONEXION
                << ": no se pudo abrir la cámara."
                << '\n';

            continue;
        }

        primerFrame.release();

        for (
            int descarte = 0;
            descarte < 5;
            ++descarte
        )
        {
            if (
                camara.read(
                    primerFrame
                )
                && !primerFrame.empty()
            )
            {
                std::cout
                    << "Cámara reconectada correctamente."
                    << '\n';

                return true;
            }

            std::this_thread::sleep_for(
                std::chrono::milliseconds(50)
            );
        }

        std::cerr
            << "Intento de reconexión "
            << intento
            << '/'
            << INTENTOS_MAXIMOS_RECONEXION
            << ": la cámara abrió, pero no entregó frames."
            << '\n';
    }

    return false;
}

void dibujarDetecciones(
    cv::Mat& frame,
    const std::vector<TaxiDetectado>& detecciones
)
{
    if (frame.empty())
    {
        return;
    }

    const cv::Rect limitesImagen(
        0,
        0,
        frame.cols,
        frame.rows
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
            frame,
            cajaValida,
            cv::Scalar(
                0,
                255,
                0
            ),
            3,
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
                0.65,
                2,
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
                    frame.cols
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
                    frame.rows
                        - lineaBase
                        - 6
                )
            );

        const int fondoX =
            textoX;

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
                frame.cols - fondoX
            );

        const int altoFondo =
            std::min(
                tamanoTexto.height
                    + lineaBase
                    + 10,
                frame.rows - fondoY
            );

        if (
            anchoFondo > 0
            && altoFondo > 0
        )
        {
            cv::rectangle(
                frame,
                cv::Rect(
                    fondoX,
                    fondoY,
                    anchoFondo,
                    altoFondo
                ),
                cv::Scalar(
                    0,
                    130,
                    0
                ),
                cv::FILLED
            );
        }

        cv::putText(
            frame,
            etiqueta.str(),
            cv::Point(
                textoX + 5,
                textoY
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(
                255,
                255,
                255
            ),
            2,
            cv::LINE_AA
        );
    }
}

void dibujarPanelEstado(
    cv::Mat& frame,
    double fpsAplicacion,
    bool deteccionEnCurso,
    double ultimoTiempoDeteccion,
    const std::vector<TaxiDetectado>& detecciones,
    bool eventoActivo,
    const GrabadorVideo& grabador,
    bool apiConfigurada,
    bool envioApiEnCurso,
    const std::string& estadoApi,
    std::size_t totalEventos,
    std::size_t totalCapturas,
    std::size_t totalVideos,
    std::size_t totalEnviosExitosos,
    double ultimaLatenciaApi,
    double memoriaRAM
)
{
    if (frame.empty())
    {
        return;
    }

    const int altoPanel =
        std::min(
            ALTO_PANEL,
            frame.rows
        );

    const cv::Rect regionPanel(
        0,
        0,
        frame.cols,
        altoPanel
    );

    cv::Mat zonaFrame =
        frame(
            regionPanel
        );

    cv::Mat capa =
        zonaFrame.clone();

    cv::rectangle(
        capa,
        cv::Rect(
            0,
            0,
            capa.cols,
            capa.rows
        ),
        cv::Scalar(
            0,
            0,
            0
        ),
        cv::FILLED
    );

    cv::addWeighted(
        capa,
        0.60,
        zonaFrame,
        0.40,
        0.0,
        zonaFrame
    );

    std::ostringstream linea1;

    linea1
        << "FPS interfaz: "
        << std::fixed
        << std::setprecision(1)
        << fpsAplicacion
        << " | Taxis: "
        << detecciones.size()
        << " | RAM: "
        << memoriaRAM
        << " MB";

    cv::putText(
        frame,
        linea1.str(),
        cv::Point(
            20,
            28
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(
            255,
            255,
            255
        ),
        2,
        cv::LINE_AA
    );

    const double fpsDetector =
        ultimoTiempoDeteccion > 0.0
            ? 1.0 / ultimoTiempoDeteccion
            : 0.0;

    std::ostringstream linea2;

    linea2
        << "Detector: "
        << (
               deteccionEnCurso
                   ? "PROCESANDO"
                   : "DISPONIBLE"
           )
        << " | Tiempo: "
        << std::fixed
        << std::setprecision(2)
        << ultimoTiempoDeteccion
        << " s | FPS detector: "
        << fpsDetector;

    cv::putText(
        frame,
        linea2.str(),
        cv::Point(
            20,
            58
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.62,
        deteccionEnCurso
            ? cv::Scalar(
                  0,
                  200,
                  255
              )
            : cv::Scalar(
                  0,
                  255,
                  0
              ),
        2,
        cv::LINE_AA
    );

    std::ostringstream linea3;

    if (grabador.estaGrabando())
    {
        linea3
            << "Estado: GRABANDO EN RAM | Restante: "
            << std::fixed
            << std::setprecision(1)
            << grabador.obtenerTiempoRestante()
            << " s | Frames: "
            << grabador.obtenerFramesGrabados()
            << '/'
            << grabador.obtenerFramesObjetivo();
    }
    else if (envioApiEnCurso)
    {
        linea3
            << "Estado: ENVIANDO RAM A FASTAPI";
    }
    else if (eventoActivo)
    {
        linea3
            << "Estado: EVENTO ACTIVO";
    }
    else
    {
        linea3
            << "Estado: ESPERANDO TAXI";
    }

    cv::putText(
        frame,
        linea3.str(),
        cv::Point(
            20,
            88
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.62,
        grabador.estaGrabando()
            ? cv::Scalar(
                  0,
                  0,
                  255
              )
            : cv::Scalar(
                  255,
                  255,
                  255
              ),
        2,
        cv::LINE_AA
    );

    std::ostringstream linea4;

    linea4
        << "Eventos: "
        << totalEventos
        << " | Capturas RAM: "
        << totalCapturas
        << " | Videos RAM: "
        << totalVideos
        << " | Enviados: "
        << totalEnviosExitosos;

    cv::putText(
        frame,
        linea4.str(),
        cv::Point(
            20,
            118
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.60,
        cv::Scalar(
            255,
            255,
            255
        ),
        2,
        cv::LINE_AA
    );

    std::ostringstream linea5;

    linea5
        << "API: ";

    if (!apiConfigurada)
    {
        linea5
            << "NO CONFIGURADA";
    }
    else if (envioApiEnCurso)
    {
        linea5
            << "ENVIANDO";
    }
    else
    {
        linea5
            << estadoApi;
    }

    linea5
        << " | Ultima latencia: "
        << std::fixed
        << std::setprecision(1)
        << ultimaLatenciaApi
        << " ms";

    cv::putText(
        frame,
        linea5.str(),
        cv::Point(
            20,
            148
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.60,
        apiConfigurada
            ? cv::Scalar(
                  255,
                  255,
                  255
              )
            : cv::Scalar(
                  0,
                  180,
                  255
              ),
        2,
        cv::LINE_AA
    );

    cv::putText(
        frame,
        "Q o ESC: salir",
        cv::Point(
            20,
            176
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(
            200,
            200,
            200
        ),
        1,
        cv::LINE_AA
    );

    if (grabador.estaGrabando())
    {
        const int radio =
            (
                static_cast<int>(
                    grabador.obtenerTiempoTranscurrido()
                    * 2.0
                ) % 2 == 0
            )
                ? 9
                : 6;

        cv::circle(
            frame,
            cv::Point(
                frame.cols - 35,
                35
            ),
            radio,
            cv::Scalar(
                0,
                0,
                255
            ),
            cv::FILLED,
            cv::LINE_AA
        );
    }
}
}

int main(
    int argc,
    char* argv[]
)
{
    int indiceCamara = 0;

    int intervaloDeteccion =
        INTERVALO_DETECCION_PREDETERMINADO;

    if (argc >= 2)
    {
        indiceCamara =
            convertirEntero(
                argv[1],
                0,
                0
            );
    }

    if (argc >= 3)
    {
        intervaloDeteccion =
            convertirEntero(
                argv[2],
                INTERVALO_DETECCION_PREDETERMINADO,
                1
            );
    }

    /*
     * filesystem se utiliza solamente para localizar el modelo.
     * Las evidencias de imagen y video no se escriben en disco.
     */
    std::error_code errorRuta;

    const std::filesystem::path rutaModelo =
        std::filesystem::absolute(
            "models/hog_svm_taxi.yml",
            errorRuta
        );

    if (
        errorRuta
        || !std::filesystem::is_regular_file(
               rutaModelo
           )
    )
    {
        std::cerr
            << "No se encontró el modelo:"
            << '\n'
            << rutaModelo
            << '\n'
            << "Ejecuta el programa desde la raíz del proyecto."
            << '\n';

        return 1;
    }

    DetectorHOG detector;

    if (
        !detector.cargarModelo(
            rutaModelo.string()
        )
    )
    {
        std::cerr
            << "No se pudo cargar el detector HOG + SVM."
            << '\n';

        return 1;
    }

    /*
     * El capturador codifica JPEG directamente en RAM.
     */
    Capturador capturador(
        CALIDAD_JPEG
    );

    if (!capturador.inicializar())
    {
        std::cerr
            << "No se pudo inicializar el capturador."
            << '\n';

        return 1;
    }

    /*
     * El grabador genera MP4 directamente en RAM mediante
     * FFmpeg y un AVIOContext personalizado.
     */
    GrabadorVideo grabador(
        DURACION_VIDEO_SEGUNDOS,
        BITRATE_VIDEO_BITS_SEGUNDO
    );

    if (!grabador.inicializar())
    {
        std::cerr
            << "No se pudo inicializar el grabador de video."
            << '\n';

        return 1;
    }

    ClienteAPI clienteApi =
        ClienteAPI::desdeVariablesEntorno();

    const bool apiConfigurada =
        clienteApi.configurado();

    std::string estadoApi =
        apiConfigurada
            ? "CONFIGURADA"
            : "NO CONFIGURADA";

    if (apiConfigurada)
    {
        std::cout
            << "API configurada: "
            << clienteApi.obtenerUrlBase()
            << '\n';

        const RespuestaAPI prueba =
            clienteApi.probarConexion();

        if (prueba.exitoso)
        {
            estadoApi =
                "CONECTADA";

            std::cout
                << "Conexión con FastAPI correcta."
                << '\n'
                << "Latencia health: "
                << std::fixed
                << std::setprecision(2)
                << prueba.latenciaMilisegundos
                << " ms"
                << '\n';
        }
        else
        {
            estadoApi =
                "SIN CONEXION";

            std::cerr
                << "No se pudo comprobar FastAPI: "
                << prueba.mensajeError
                << '\n';
        }
    }
    else
    {
        std::cerr
            << '\n'
            << "ADVERTENCIA: la API no está configurada."
            << '\n'
            << "Define las variables:"
            << '\n'
            << "  TAXI_API_URL"
            << '\n'
            << "  TAXI_API_KEY"
            << '\n'
            << "La detección local continuará funcionando."
            << '\n';
    }

    cv::VideoCapture camara;

    if (
        !abrirCamara(
            camara,
            indiceCamara
        )
    )
    {
        std::cerr
            << "No se pudo abrir la cámara "
            << indiceCamara
            << '.'
            << '\n';

        return 1;
    }

    cv::Mat frame;

    bool primerFrameLeido = false;

    for (
        int intento = 0;
        intento < 10;
        ++intento
    )
    {
        if (
            camara.read(frame)
            && !frame.empty()
        )
        {
            primerFrameLeido = true;

            break;
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(50)
        );
    }

    if (!primerFrameLeido)
    {
        std::cerr
            << "No se pudo leer el primer frame de la cámara."
            << '\n';

        return 1;
    }

    const int anchoReal =
        frame.cols;

    const int altoReal =
        frame.rows;

    double fpsCamara =
        camara.get(
            cv::CAP_PROP_FPS
        );

    if (
        !std::isfinite(fpsCamara)
        || fpsCamara <= 1.0
    )
    {
        fpsCamara =
            FPS_CAMARA_DESEADO;
    }

    fpsCamara =
        std::clamp(
            fpsCamara,
            5.0,
            60.0
        );

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "DETECTOR DE TAXIS AMARILLOS"
        << '\n'
        << "============================================="
        << '\n'
        << "Cámara:                   "
        << indiceCamara
        << '\n'
        << "Resolución solicitada:    "
        << ANCHO_CAMARA
        << 'x'
        << ALTO_CAMARA
        << '\n'
        << "Resolución real:          "
        << anchoReal
        << 'x'
        << altoReal
        << '\n'
        << "FPS solicitado:           "
        << FPS_CAMARA_DESEADO
        << '\n'
        << "FPS reportado:            "
        << fpsCamara
        << '\n'
        << "Detección:                cada "
        << intervaloDeteccion
        << " frames"
        << '\n'
        << "Duración de video:        "
        << DURACION_VIDEO_SEGUNDOS
        << " segundos"
        << '\n'
        << "Calidad JPEG:             "
        << CALIDAD_JPEG
        << "%"
        << '\n'
        << "Bitrate MP4:              "
        << BITRATE_VIDEO_BITS_SEGUNDO
        << " bits/s"
        << '\n'
        << "Evidencias en disco:      NO"
        << '\n'
        << "Modelo:                   "
        << rutaModelo
        << '\n'
        << "API:                      "
        << estadoApi
        << '\n'
        << "============================================="
        << '\n';

    std::future<std::vector<TaxiDetectado>>
        tareaDeteccion;

    bool deteccionEnCurso = false;

    std::future<RespuestaAPI>
        tareaEnvioApi;

    bool envioApiEnCurso = false;

    cv::Mat frameAsociadoDeteccion;

    std::vector<TaxiDetectado>
        ultimasDetecciones;

    DatosEventoActual eventoActual;

    bool eventoActivo = false;

    int deteccionesPositivasConsecutivas = 0;
    int resultadosVaciosConsecutivos = 0;

    float mejorScoreConfirmacion = 0.0F;

    auto ultimaDeteccionPositiva =
        std::chrono::steady_clock::now();

    std::size_t totalEventos = 0;
    std::size_t totalEnviosExitosos = 0;
    std::size_t totalEnviosFallidos = 0;

    double ultimoTiempoDeteccion = 0.0;
    double ultimaLatenciaApi = 0.0;

    std::chrono::steady_clock::time_point
        inicioDeteccion;

    std::size_t numeroFrame = 0;

    auto tiempoFPS =
        std::chrono::steady_clock::now();

    std::size_t framesParaFPS = 0;

    double fpsAplicacion = 0.0;
    double memoriaRAM = 0.0;

    int lecturasFallidasConsecutivas = 0;

    bool ejecutando = true;

    while (ejecutando)
    {
        if (
            !camara.read(frame)
            || frame.empty()
        )
        {
            ++lecturasFallidasConsecutivas;

            std::cerr
                << "No se pudo leer un frame de la cámara. "
                << "Fallo "
                << lecturasFallidasConsecutivas
                << '/'
                << LECTURAS_FALLIDAS_PARA_RECONECTAR
                << '\n';

            if (
                lecturasFallidasConsecutivas
                < LECTURAS_FALLIDAS_PARA_RECONECTAR
            )
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(20)
                );

                continue;
            }

            cv::Mat frameReconectado;

            if (
                !reconectarCamara(
                    camara,
                    indiceCamara,
                    frameReconectado
                )
            )
            {
                std::cerr
                    << "No fue posible recuperar la cámara."
                    << '\n';

                break;
            }

            frame =
                std::move(
                    frameReconectado
                );

            lecturasFallidasConsecutivas = 0;
        }
        else
        {
            lecturasFallidasConsecutivas = 0;
        }

        ++numeroFrame;
        ++framesParaFPS;

        /*
         * Recuperar el resultado HTTP sin bloquear
         * la interfaz de cámara.
         */
        if (envioApiEnCurso)
        {
            const auto estado =
                tareaEnvioApi.wait_for(
                    std::chrono::milliseconds(0)
                );

            if (
                estado
                == std::future_status::ready
            )
            {
                try
                {
                    const RespuestaAPI respuesta =
                        tareaEnvioApi.get();

                    ultimaLatenciaApi =
                        respuesta.latenciaMilisegundos;

                    if (respuesta.exitoso)
                    {
                        ++totalEnviosExitosos;

                        estadoApi =
                            "ULTIMO ENVIO CORRECTO";
                    }
                    else
                    {
                        ++totalEnviosFallidos;

                        estadoApi =
                            "ERROR DE ENVIO";

                        std::cerr
                            << "Fallo en el envío del evento: "
                            << respuesta.mensajeError
                            << '\n';
                    }
                }
                catch (const std::exception& error)
                {
                    ++totalEnviosFallidos;

                    estadoApi =
                        "ERROR DE ENVIO";

                    std::cerr
                        << "Excepción en la tarea de API: "
                        << error.what()
                        << '\n';
                }

                envioApiEnCurso = false;
            }
        }

        /*
         * Recuperar el resultado de la detección asíncrona.
         */
        if (deteccionEnCurso)
        {
            const auto estado =
                tareaDeteccion.wait_for(
                    std::chrono::milliseconds(0)
                );

            if (
                estado
                == std::future_status::ready
            )
            {
                try
                {
                    ultimasDetecciones =
                        tareaDeteccion.get();
                }
                catch (const std::exception& error)
                {
                    std::cerr
                        << "Error durante la detección: "
                        << error.what()
                        << '\n';

                    ultimasDetecciones.clear();
                }

                deteccionEnCurso = false;

                const auto finDeteccion =
                    std::chrono::steady_clock::now();

                ultimoTiempoDeteccion =
                    std::chrono::duration<double>(
                        finDeteccion
                        - inicioDeteccion
                    ).count();

                std::cout
                    << "Detección completada: "
                    << ultimasDetecciones.size()
                    << " taxi(s), tiempo "
                    << std::fixed
                    << std::setprecision(3)
                    << ultimoTiempoDeteccion
                    << " s"
                    << '\n';

                if (!ultimasDetecciones.empty())
                {
                    resultadosVaciosConsecutivos = 0;

                    ultimaDeteccionPositiva =
                        std::chrono::steady_clock::now();

                    if (!eventoActivo)
                    {
                        const float mejorScore =
                            obtenerMejorScore(
                                ultimasDetecciones
                            );

                        if (
                            mejorScore
                            >= SCORE_MINIMO_PARA_EVENTO
                        )
                        {
                            ++deteccionesPositivasConsecutivas;

                            mejorScoreConfirmacion =
                                std::max(
                                    mejorScoreConfirmacion,
                                    mejorScore
                                );

                            std::cout
                                << "Confirmación de taxi: "
                                << deteccionesPositivasConsecutivas
                                << '/'
                                << DETECCIONES_POSITIVAS_PARA_CONFIRMAR
                                << " | score: "
                                << std::fixed
                                << std::setprecision(3)
                                << mejorScore
                                << '\n';
                        }
                        else
                        {
                            deteccionesPositivasConsecutivas = 0;
                            mejorScoreConfirmacion = 0.0F;

                            std::cout
                                << "Detección descartada para evento. "
                                << "Score "
                                << std::fixed
                                << std::setprecision(3)
                                << mejorScore
                                << " < "
                                << SCORE_MINIMO_PARA_EVENTO
                                << '\n';
                        }

                        if (
                            deteccionesPositivasConsecutivas
                            >= DETECCIONES_POSITIVAS_PARA_CONFIRMAR
                        )
                        {
                            deteccionesPositivasConsecutivas = 0;
                            resultadosVaciosConsecutivos = 0;

                            eventoActivo = true;

                            ++totalEventos;

                            eventoActual.limpiar();

                            eventoActual.fechaHora =
                                generarFechaHoraEvento();

                            eventoActual.scoreSvm =
                                mejorScoreConfirmacion;

                            eventoActual.confianzaNormalizada =
                                ClienteAPI::normalizarScoreSVM(
                                    eventoActual.scoreSvm
                                );

                            std::cout
                                << '\n'
                                << "============================================="
                                << '\n'
                                << "NUEVO EVENTO DE TAXI DETECTADO"
                                << '\n'
                                << "============================================="
                                << '\n'
                                << "Evento número: "
                                << totalEventos
                                << '\n'
                                << "Score máximo: "
                                << eventoActual.scoreSvm
                                << '\n'
                                << "Confianza enviada: "
                                << eventoActual.confianzaNormalizada
                                << '\n';

                            /*
                             * La captura se codifica como JPEG y
                             * permanece dentro del vector.
                             */
                            eventoActual.imagenJpeg =
                                capturador.capturarEnMemoria(
                                    frameAsociadoDeteccion,
                                    ultimasDetecciones
                                );

                            if (eventoActual.imagenJpeg.empty())
                            {
                                std::cerr
                                    << "No se pudo generar la "
                                    << "captura JPEG en RAM."
                                    << '\n';
                            }
                            else
                            {
                                std::cout
                                    << "Imagen JPEG en RAM: "
                                    << eventoActual.imagenJpeg.size()
                                    << " bytes ("
                                    << std::fixed
                                    << std::setprecision(2)
                                    << convertirBytesAMegabytes(
                                           eventoActual.imagenJpeg.size()
                                       )
                                    << " MB)"
                                    << '\n';
                            }

                            const bool videoIniciado =
                                grabador.iniciar(
                                    frame.size(),
                                    fpsCamara
                                );

                            if (!videoIniciado)
                            {
                                std::cerr
                                    << "No se pudo iniciar "
                                    << "el video MP4 en RAM."
                                    << '\n';
                            }

                            eventoActual.pendienteEnvio =
                                apiConfigurada
                                && !eventoActual.imagenJpeg.empty()
                                && videoIniciado;

                            mejorScoreConfirmacion = 0.0F;
                        }
                    }
                    else
                    {
                        deteccionesPositivasConsecutivas = 0;
                        mejorScoreConfirmacion = 0.0F;
                    }
                }
                else
                {
                    deteccionesPositivasConsecutivas = 0;

                    if (!eventoActivo)
                    {
                        mejorScoreConfirmacion = 0.0F;
                    }

                    if (eventoActivo)
                    {
                        ++resultadosVaciosConsecutivos;

                        std::cout
                            << "Resultados consecutivos sin taxi: "
                            << std::min(
                                   resultadosVaciosConsecutivos,
                                   RESULTADOS_VACIOS_PARA_REACTIVAR
                               )
                            << '/'
                            << RESULTADOS_VACIOS_PARA_REACTIVAR
                            << '\n';
                    }
                }
            }
        }

        /*
         * Iniciar una nueva detección solamente cuando el
         * detector se encuentre disponible.
         */
        if (
            !deteccionEnCurso
            && numeroFrame
                    % static_cast<std::size_t>(
                          intervaloDeteccion
                      )
                == 0
        )
        {
            frameAsociadoDeteccion =
                frame.clone();

            const cv::Mat frameParaDetector =
                frameAsociadoDeteccion;

            inicioDeteccion =
                std::chrono::steady_clock::now();

            deteccionEnCurso = true;

            tareaDeteccion =
                std::async(
                    std::launch::async,
                    [&detector, frameParaDetector]()
                    {
                        return detector.detectar(
                            frameParaDetector
                        );
                    }
                );
        }

        cv::Mat frameSalida =
            frame.clone();

        dibujarDetecciones(
            frameSalida,
            ultimasDetecciones
        );

        /*
         * Los frames mostrados se codifican dentro del MP4
         * mantenido por GrabadorVideo en memoria RAM.
         */
        if (grabador.estaGrabando())
        {
            const bool escrito =
                grabador.escribirFrame(
                    frameSalida
                );

            if (
                !escrito
                && grabador.estaGrabando()
            )
            {
                std::cerr
                    << "No se pudo codificar "
                    << "un frame del video."
                    << '\n';
            }
        }

        /*
         * Al completarse el video, se mueve el vector MP4
         * desde GrabadorVideo hacia el evento.
         */
        if (
            eventoActual.pendienteEnvio
            && !grabador.estaGrabando()
            && grabador.grabacionCompletada()
            && !envioApiEnCurso
        )
        {
            eventoActual.videoMp4 =
                grabador.extraerVideoEnMemoria();

            if (
                eventoActual.videoMp4.empty()
                || eventoActual.imagenJpeg.empty()
            )
            {
                std::cerr
                    << "El evento no contiene ambas "
                    << "evidencias en memoria RAM."
                    << '\n';

                eventoActual.pendienteEnvio = false;
            }
            else
            {
                std::cout
                    << '\n'
                    << "EVIDENCIAS PREPARADAS EN RAM"
                    << '\n'
                    << "Imagen: "
                    << eventoActual.imagenJpeg.size()
                    << " bytes"
                    << '\n'
                    << "Video: "
                    << eventoActual.videoMp4.size()
                    << " bytes"
                    << '\n';

                DatosEventoTaxi datosEnvio;

                datosEnvio.imagenDatos =
                    std::move(
                        eventoActual.imagenJpeg
                    );

                datosEnvio.videoDatos =
                    std::move(
                        eventoActual.videoMp4
                    );

                datosEnvio.nombreImagen =
                    "captura_taxi.jpg";

                datosEnvio.nombreVideo =
                    "evidencia_taxi.mp4";

                datosEnvio.vehiculo =
                    "Taxi amarillo";

                datosEnvio.camara =
                    "Cámara "
                    + std::to_string(
                          indiceCamara
                      );

                datosEnvio.fechaHora =
                    eventoActual.fechaHora;

                datosEnvio.confianzaCpp =
                    eventoActual.confianzaNormalizada;

                eventoActual.pendienteEnvio = false;

                envioApiEnCurso = true;

                estadoApi =
                    "ENVIANDO";

                /*
                 * La tarea asíncrona toma posesión de los
                 * vectores y los conserva durante toda la
                 * petición HTTP.
                 */
                tareaEnvioApi =
                    std::async(
                        std::launch::async,
                        [
                            &clienteApi,
                            datosEnvio =
                                std::move(datosEnvio)
                        ]() mutable
                        {
                            return clienteApi.enviarEvento(
                                datosEnvio
                            );
                        }
                    );
            }
        }

        const double segundosSinTaxi =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now()
                - ultimaDeteccionPositiva
            ).count();

        if (
            eventoActivo
            && resultadosVaciosConsecutivos
                    >= RESULTADOS_VACIOS_PARA_REACTIVAR
            && segundosSinTaxi
                    >= SEGUNDOS_SIN_TAXI_PARA_REACTIVAR
            && !grabador.estaGrabando()
            && !eventoActual.pendienteEnvio
            && !envioApiEnCurso
        )
        {
            eventoActivo = false;

            deteccionesPositivasConsecutivas = 0;
            resultadosVaciosConsecutivos = 0;

            mejorScoreConfirmacion = 0.0F;

            ultimasDetecciones.clear();
            eventoActual.limpiar();

            std::cout
                << '\n'
                << "Evento finalizado."
                << '\n'
                << "Tiempo sin taxi: "
                << std::fixed
                << std::setprecision(2)
                << segundosSinTaxi
                << " segundos"
                << '\n'
                << "Sistema listo para detectar un nuevo taxi."
                << '\n';
        }

        const auto ahora =
            std::chrono::steady_clock::now();

        const double segundosFPS =
            std::chrono::duration<double>(
                ahora - tiempoFPS
            ).count();

        if (segundosFPS >= 1.0)
        {
            fpsAplicacion =
                static_cast<double>(
                    framesParaFPS
                ) / segundosFPS;

            framesParaFPS = 0;
            tiempoFPS = ahora;

            memoriaRAM =
                obtenerMemoriaRAMMegabytes();
        }

        dibujarPanelEstado(
            frameSalida,
            fpsAplicacion,
            deteccionEnCurso,
            ultimoTiempoDeteccion,
            ultimasDetecciones,
            eventoActivo,
            grabador,
            apiConfigurada,
            envioApiEnCurso,
            estadoApi,
            totalEventos,
            capturador.obtenerTotalCapturas(),
            grabador.obtenerTotalVideos(),
            totalEnviosExitosos,
            ultimaLatenciaApi,
            memoriaRAM
        );

        cv::imshow(
            "Detector de taxis amarillos",
            frameSalida
        );

        const int tecla =
            cv::waitKey(1)
            & 0xFF;

        if (
            tecla == 'q'
            || tecla == 'Q'
            || tecla == 27
        )
        {
            ejecutando = false;
        }
    }

    /*
     * Completar correctamente cualquier video activo.
     */
    if (grabador.estaGrabando())
    {
        std::cout
            << "Finalizando grabación en RAM..."
            << '\n';

        if (!grabador.finalizar())
        {
            std::cerr
                << "No se pudo finalizar el video MP4."
                << '\n';
        }
    }

    /*
     * Si el usuario cerró justo cuando terminó el video,
     * realizar el último envío antes de salir.
     */
    if (
        eventoActual.pendienteEnvio
        && grabador.grabacionCompletada()
        && !envioApiEnCurso
    )
    {
        eventoActual.videoMp4 =
            grabador.extraerVideoEnMemoria();

        if (
            !eventoActual.imagenJpeg.empty()
            && !eventoActual.videoMp4.empty()
        )
        {
            DatosEventoTaxi datosEnvio;

            datosEnvio.imagenDatos =
                std::move(
                    eventoActual.imagenJpeg
                );

            datosEnvio.videoDatos =
                std::move(
                    eventoActual.videoMp4
                );

            datosEnvio.nombreImagen =
                "captura_taxi.jpg";

            datosEnvio.nombreVideo =
                "evidencia_taxi.mp4";

            datosEnvio.vehiculo =
                "Taxi amarillo";

            datosEnvio.camara =
                "Cámara "
                + std::to_string(
                      indiceCamara
                  );

            datosEnvio.fechaHora =
                eventoActual.fechaHora;

            datosEnvio.confianzaCpp =
                eventoActual.confianzaNormalizada;

            std::cout
                << "Enviando el último evento pendiente..."
                << '\n';

            const RespuestaAPI respuesta =
                clienteApi.enviarEvento(
                    datosEnvio
                );

            ultimaLatenciaApi =
                respuesta.latenciaMilisegundos;

            if (respuesta.exitoso)
            {
                ++totalEnviosExitosos;
            }
            else
            {
                ++totalEnviosFallidos;

                std::cerr
                    << "El último envío no fue exitoso: "
                    << respuesta.mensajeError
                    << '\n';
            }

            eventoActual.pendienteEnvio = false;
        }
    }

    if (deteccionEnCurso)
    {
        std::cout
            << "Esperando la detección en curso..."
            << '\n';

        try
        {
            (void)tareaDeteccion.get();
        }
        catch (const std::exception& error)
        {
            std::cerr
                << "Error al finalizar la detección: "
                << error.what()
                << '\n';
        }
    }

    if (envioApiEnCurso)
    {
        std::cout
            << "Esperando el envío a la API..."
            << '\n';

        try
        {
            const RespuestaAPI respuesta =
                tareaEnvioApi.get();

            ultimaLatenciaApi =
                respuesta.latenciaMilisegundos;

            if (respuesta.exitoso)
            {
                ++totalEnviosExitosos;
            }
            else
            {
                ++totalEnviosFallidos;

                std::cerr
                    << "El último envío no fue exitoso: "
                    << respuesta.mensajeError
                    << '\n';
            }
        }
        catch (const std::exception& error)
        {
            ++totalEnviosFallidos;

            std::cerr
                << "Error esperando la tarea de API: "
                << error.what()
                << '\n';
        }
    }

    camara.release();

    cv::destroyAllWindows();

    eventoActual.limpiar();

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "RESUMEN DE LA EJECUCION"
        << '\n'
        << "============================================="
        << '\n'
        << "Eventos detectados:       "
        << totalEventos
        << '\n'
        << "Capturas generadas RAM:   "
        << capturador.obtenerTotalCapturas()
        << '\n'
        << "Videos generados RAM:     "
        << grabador.obtenerTotalVideos()
        << '\n'
        << "Envíos exitosos:          "
        << totalEnviosExitosos
        << '\n'
        << "Envíos fallidos:          "
        << totalEnviosFallidos
        << '\n'
        << "Última latencia API:      "
        << std::fixed
        << std::setprecision(2)
        << ultimaLatenciaApi
        << " ms"
        << '\n'
        << "Memoria RAM final:        "
        << obtenerMemoriaRAMMegabytes()
        << " MB"
        << '\n'
        << "Archivos de evidencia:    0"
        << '\n'
        << "============================================="
        << '\n'
        << "Detector finalizado correctamente."
        << '\n';

    return 0;
}

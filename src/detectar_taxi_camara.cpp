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
#include <vector>

namespace
{
/*
 * Resolución optimizada para la cámara y el detector.
 *
 * 640x360 reduce a una cuarta parte la cantidad de píxeles
 * respecto a 1280x720, sin cambiar el modelo ni las clases
 * usadas por la integración con FastAPI y Telegram.
 */
constexpr int ANCHO_CAMARA = 640;
constexpr int ALTO_CAMARA = 360;

constexpr double FPS_CAMARA_DESEADO = 30.0;
constexpr double DURACION_VIDEO_SEGUNDOS = 5.0;

constexpr int INTERVALO_DETECCION_PREDETERMINADO = 10;

/*
 * Cantidad de resultados consecutivos sin taxis necesarios
 * para permitir un nuevo evento.
 */
constexpr int DETECCIONES_POSITIVAS_PARA_CONFIRMAR = 2;
constexpr float SCORE_MINIMO_PARA_EVENTO = 1.0F;

constexpr int RESULTADOS_VACIOS_PARA_REACTIVAR = 5;
constexpr double SEGUNDOS_SIN_TAXI_PARA_REACTIVAR = 8.0;

/*
 * Recuperación básica cuando una cámara USB deja de entregar
 * frames temporalmente.
 */
constexpr int LECTURAS_FALLIDAS_PARA_RECONECTAR = 5;
constexpr int INTENTOS_MAXIMOS_RECONEXION = 3;
constexpr int ESPERA_RECONEXION_MS = 500;

constexpr int ALTO_PANEL = 185;

struct DatosEventoActual
{
    std::filesystem::path rutaImagen;
    std::filesystem::path rutaVideo;

    std::string fechaHora;

    float scoreSvm = 0.0F;
    float confianzaNormalizada = 0.0F;

    bool pendienteEnvio = false;

    void limpiar()
    {
        rutaImagen.clear();
        rutaVideo.clear();

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
        const int valor = std::stoi(texto);

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
            [](const TaxiDetectado& izquierda,
               const TaxiDetectado& derecha)
            {
                return izquierda.score < derecha.score;
            }
        );

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
                ) /
                1024.0;
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

/*
 * Aplica únicamente propiedades de VideoCapture.
 *
 * No modifica DetectorHOG, Capturador, GrabadorVideo ni
 * ClienteAPI, por lo que no cambia el contrato con FastAPI.
 */
void configurarCamara(
    cv::VideoCapture& camara
)
{
#if defined(__linux__)

    /*
     * La cámara Fullhan confirmó soporte para YUYV.
     * Si otra cámara no acepta esta propiedad, OpenCV
     * conservará el formato que el dispositivo permita.
     */
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

    /*
     * Reduce la posibilidad de procesar frames antiguos
     * acumulados en el búfer de captura.
     *
     * Algunos backends pueden ignorar esta propiedad.
     */
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

        /*
         * Se descartan unos pocos frames iniciales porque
         * algunas webcams entregan frames vacíos al abrirse.
         */
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

    for (
        std::size_t indice = 0;
        indice < detecciones.size();
        ++indice
    )
    {
        const TaxiDetectado& deteccion =
            detecciones[indice];

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

        cv::rectangle(
            frame,
            cajaValida,
            cv::Scalar(0, 255, 0),
            3,
            cv::LINE_AA
        );

        std::ostringstream etiqueta;

        etiqueta
            << "Taxi "
            << indice + 1
            << " | score: "
            << std::fixed
            << std::setprecision(2)
            << deteccion.score;

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
                cv::Scalar(0, 130, 0),
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
            cv::Scalar(255, 255, 255),
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

    /*
     * Antes se clonaba el frame completo para crear el panel.
     * Ahora solo se copia la región ocupada por el panel.
     */
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
        cv::Scalar(0, 0, 0),
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
        << std::setprecision(1)
        << memoriaRAM
        << " MB";

    cv::putText(
        frame,
        linea1.str(),
        cv::Point(20, 28),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    std::ostringstream linea2;

    const double fpsDetector =
        ultimoTiempoDeteccion > 0.0
            ? 1.0 / ultimoTiempoDeteccion
            : 0.0;

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
        cv::Point(20, 58),
        cv::FONT_HERSHEY_SIMPLEX,
        0.62,
        deteccionEnCurso
            ? cv::Scalar(0, 200, 255)
            : cv::Scalar(0, 255, 0),
        2,
        cv::LINE_AA
    );

    std::ostringstream linea3;

    if (grabador.estaGrabando())
    {
        linea3
            << "Estado: GRABANDO | Restante: "
            << std::fixed
            << std::setprecision(1)
            << grabador.obtenerTiempoRestante()
            << " s | Frames: "
            << grabador.obtenerFramesGrabados();
    }
    else if (envioApiEnCurso)
    {
        linea3
            << "Estado: ENVIANDO EVENTO A FASTAPI";
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
        cv::Point(20, 88),
        cv::FONT_HERSHEY_SIMPLEX,
        0.62,
        grabador.estaGrabando()
            ? cv::Scalar(0, 0, 255)
            : cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    std::ostringstream linea4;

    linea4
        << "Eventos: "
        << totalEventos
        << " | Capturas: "
        << totalCapturas
        << " | Videos: "
        << totalVideos
        << " | Enviados: "
        << totalEnviosExitosos;

    cv::putText(
        frame,
        linea4.str(),
        cv::Point(20, 118),
        cv::FONT_HERSHEY_SIMPLEX,
        0.62,
        cv::Scalar(255, 255, 255),
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
        cv::Point(20, 148),
        cv::FONT_HERSHEY_SIMPLEX,
        0.60,
        apiConfigurada
            ? cv::Scalar(255, 255, 255)
            : cv::Scalar(0, 180, 255),
        2,
        cv::LINE_AA
    );

    cv::putText(
        frame,
        "Q o ESC: salir",
        cv::Point(20, 176),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(200, 200, 200),
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
            cv::Scalar(0, 0, 255),
            cv::FILLED,
            cv::LINE_AA
        );
    }
}
}

int main(int argc, char* argv[])
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

    std::error_code errorRuta;

    const std::filesystem::path rutaModelo =
        std::filesystem::absolute(
            "models/hog_svm_taxi.yml",
            errorRuta
        );

    if (
        errorRuta
        || !std::filesystem::exists(
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

    Capturador capturador(
        "captures"
    );

    if (!capturador.inicializar())
    {
        std::cerr
            << "No se pudo inicializar el capturador."
            << '\n';

        return 1;
    }

    GrabadorVideo grabador(
        "videos",
        DURACION_VIDEO_SEGUNDOS
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

    /*
     * Algunas webcams necesitan varios intentos al iniciar.
     */
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

    /*
     * Conserva el mejor score obtenido durante todas las
     * detecciones positivas usadas para confirmar un evento.
     */
    float mejorScoreConfirmacion = 0.0F;

    /*
     * Guarda el momento de la detección positiva más reciente.
     * Se usa para evitar que el mismo taxi genere varios eventos.
     */
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
                frameReconectado;

            lecturasFallidasConsecutivas = 0;
        }
        else
        {
            lecturasFallidasConsecutivas = 0;
        }

        ++numeroFrame;
        ++framesParaFPS;

        /*
         * Recoger el resultado de la petición HTTP sin
         * bloquear la interfaz.
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
         * Recoger el resultado del detector.
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
                            obtenerMejorScore(ultimasDetecciones);

                        if (mejorScore >= SCORE_MINIMO_PARA_EVENTO)
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
                            eventoActivo = true;
                            ++totalEventos;

                            eventoActual.limpiar();
                            eventoActual.fechaHora = generarFechaHoraEvento();
                            eventoActual.scoreSvm =
                                mejorScoreConfirmacion;
                            eventoActual.confianzaNormalizada =
                                ClienteAPI::normalizarScoreSVM(
                                    eventoActual.scoreSvm
                                );

                            std::cout
                                << '\n'
                                << "NUEVO EVENTO DE TAXI DETECTADO"
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

                            const std::string rutaCaptura =
                                capturador.guardarCaptura(
                                    frameAsociadoDeteccion,
                                    ultimasDetecciones
                                );

                            if (rutaCaptura.empty())
                            {
                                std::cerr
                                    << "No se pudo guardar la captura del evento."
                                    << '\n';
                            }
                            else
                            {
                                eventoActual.rutaImagen = rutaCaptura;
                            }

                            const bool videoIniciado =
                                grabador.iniciar(frame.size(), fpsCamara);

                            if (!videoIniciado)
                            {
                                std::cerr
                                    << "No se pudo iniciar el video del evento."
                                    << '\n';
                            }

                            eventoActual.pendienteEnvio =
                                apiConfigurada
                                && !eventoActual.rutaImagen.empty()
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
         * Iniciar otra detección solamente si el detector
         * está libre.
         *
         * Se realiza una sola copia profunda del frame.
         * El cv::Mat capturado por la tarea comparte esa memoria
         * mediante conteo de referencias y permanece válido hasta
         * que la detección finaliza.
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

        /*
         * Esta copia es necesaria porque sobre frameSalida se
         * dibujan cajas y paneles sin alterar el frame original
         * capturado por la cámara.
         */
        cv::Mat frameSalida =
            frame.clone();

        dibujarDetecciones(
            frameSalida,
            ultimasDetecciones
        );

        /*
         * El video registra los frames mostrados, incluyendo
         * las cajas de detección.
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
                    << "No se pudo escribir un frame del video."
                    << '\n';
            }
        }

        /*
         * Cuando GrabadorVideo finaliza automáticamente,
         * recuperamos la ruta y enviamos el evento.
         */
        if (
            eventoActual.pendienteEnvio
            && !grabador.estaGrabando()
            && grabador.grabacionCompletada()
            && !envioApiEnCurso
        )
        {
            eventoActual.rutaVideo =
                grabador.obtenerUltimoVideoGuardado();

            if (
                eventoActual.rutaVideo.empty()
                || eventoActual.rutaImagen.empty()
            )
            {
                std::cerr
                    << "El evento no contiene ambas evidencias."
                    << '\n';

                eventoActual.pendienteEnvio = false;
            }
            else
            {
                DatosEventoTaxi datosEnvio;

                datosEnvio.rutaImagen =
                    eventoActual.rutaImagen;

                datosEnvio.rutaVideo =
                    eventoActual.rutaVideo;

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

                tareaEnvioApi =
                    std::async(
                        std::launch::async,
                        [&clienteApi, datosEnvio]()
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
                ) /
                segundosFPS;

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

    if (grabador.estaGrabando())
    {
        std::cout
            << "Finalizando grabación en curso..."
            << '\n';

        grabador.finalizar();
    }

    if (deteccionEnCurso)
    {
        std::cout
            << "Esperando la detección en curso..."
            << '\n';

        try
        {
            tareaDeteccion.get();
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

    std::cout
        << '\n'<< "============================================="
        << '\n'
        << "RESUMEN DE LA EJECUCION"
        << '\n'
        << "============================================="
        << '\n'
        << "Eventos detectados:       "
        << totalEventos
        << '\n'
        << "Capturas guardadas:       "
        << capturador.obtenerTotalCapturas()
        << '\n'
        << "Videos guardados:         "
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
        << "============================================="
        << '\n'
        << "Detector finalizado correctamente."
        << '\n';

    return 0;
}

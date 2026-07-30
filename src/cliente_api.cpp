#include "cliente_api.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr long TIMEOUT_HEALTH_SEGUNDOS = 30L;
constexpr long LIMITE_BAJA_VELOCIDAD = 1024L;
constexpr long TIEMPO_BAJA_VELOCIDAD = 30L;
constexpr long MAXIMAS_REDIRECCIONES = 5L;

std::size_t recibirRespuesta(
    char* datos,
    std::size_t tamano,
    std::size_t cantidad,
    void* destino
)
{
    if (
        datos == nullptr
        || destino == nullptr
    )
    {
        return 0;
    }

    const std::size_t total =
        tamano * cantidad;

    auto* texto =
        static_cast<std::string*>(
            destino
        );

    texto->append(
        datos,
        total
    );

    return total;
}

class InicializadorCurl
{
public:
    InicializadorCurl()
        : correcto_(
              curl_global_init(
                  CURL_GLOBAL_DEFAULT
              ) == CURLE_OK
          )
    {
    }

    ~InicializadorCurl()
    {
        if (correcto_)
        {
            curl_global_cleanup();
        }
    }

    InicializadorCurl(
        const InicializadorCurl&
    ) = delete;

    InicializadorCurl& operator=(
        const InicializadorCurl&
    ) = delete;

    [[nodiscard]]
    bool correcto() const
    {
        return correcto_;
    }

private:
    bool correcto_ = false;
};

InicializadorCurl& obtenerCurlGlobal()
{
    static InicializadorCurl instancia;

    return instancia;
}

std::string quitarEspacios(
    const std::string& texto
)
{
    const auto noEsEspacio =
        [](unsigned char caracter)
        {
            return
                std::isspace(
                    caracter
                ) == 0;
        };

    const auto inicio =
        std::find_if(
            texto.begin(),
            texto.end(),
            noEsEspacio
        );

    if (inicio == texto.end())
    {
        return {};
    }

    const auto fin =
        std::find_if(
            texto.rbegin(),
            texto.rend(),
            noEsEspacio
        ).base();

    return std::string(
        inicio,
        fin
    );
}

std::string quitarEspaciosJson(
    const std::string& texto
)
{
    std::string resultado;

    resultado.reserve(
        texto.size()
    );

    for (
        const unsigned char caracter
        : texto
    )
    {
        if (
            std::isspace(
                caracter
            ) == 0
        )
        {
            resultado.push_back(
                static_cast<char>(
                    caracter
                )
            );
        }
    }

    return resultado;
}

bool jsonConfirmaOperacion(
    const std::string& cuerpo
)
{
    const std::string cuerpoCompacto =
        quitarEspaciosJson(
            cuerpo
        );

    return
        cuerpoCompacto.find(
            "\"ok\":true"
        ) != std::string::npos;
}

bool agregarTexto(
    curl_mime* formulario,
    const char* nombre,
    const std::string& valor
)
{
    if (
        formulario == nullptr
        || nombre == nullptr
    )
    {
        return false;
    }

    curl_mimepart* parte =
        curl_mime_addpart(
            formulario
        );

    if (parte == nullptr)
    {
        return false;
    }

    if (
        curl_mime_name(
            parte,
            nombre
        ) != CURLE_OK
    )
    {
        return false;
    }

    return
        curl_mime_data(
            parte,
            valor.c_str(),
            CURL_ZERO_TERMINATED
        ) == CURLE_OK;
}

/*
 * Agrega un archivo al formulario multipart directamente
 * desde un vector de bytes almacenado en memoria RAM.
 *
 * No abre ni crea archivos en el sistema.
 */
bool agregarArchivoMemoria(
    curl_mime* formulario,
    const char* nombreCampo,
    const std::vector<unsigned char>& datos,
    const std::string& nombreArchivo,
    const char* tipoMime
)
{
    if (
        formulario == nullptr
        || nombreCampo == nullptr
        || datos.empty()
        || nombreArchivo.empty()
    )
    {
        return false;
    }

    curl_mimepart* parte =
        curl_mime_addpart(
            formulario
        );

    if (parte == nullptr)
    {
        return false;
    }

    if (
        curl_mime_name(
            parte,
            nombreCampo
        ) != CURLE_OK
    )
    {
        return false;
    }

    /*
     * Se entrega a libcurl la dirección del vector y
     * su tamaño exacto.
     *
     * El tamaño explícito permite enviar datos binarios
     * que contienen bytes con valor cero.
     */
    if (
        curl_mime_data(
            parte,
            reinterpret_cast<const char*>(
                datos.data()
            ),
            datos.size()
        ) != CURLE_OK
    )
    {
        return false;
    }

    /*
     * Este nombre se incluye en multipart/form-data,
     * pero no representa una ruta física en el disco.
     */
    if (
        curl_mime_filename(
            parte,
            nombreArchivo.c_str()
        ) != CURLE_OK
    )
    {
        return false;
    }

    if (
        tipoMime != nullptr
        && curl_mime_type(
               parte,
               tipoMime
           ) != CURLE_OK
    )
    {
        return false;
    }

    return true;
}

bool agregarEncabezado(
    curl_slist*& encabezados,
    const std::string& encabezado
)
{
    curl_slist* nuevaLista =
        curl_slist_append(
            encabezados,
            encabezado.c_str()
        );

    if (nuevaLista == nullptr)
    {
        return false;
    }

    encabezados =
        nuevaLista;

    return true;
}

std::string convertirMinusculas(
    std::string texto
)
{
    std::transform(
        texto.begin(),
        texto.end(),
        texto.begin(),
        [](unsigned char caracter)
        {
            return static_cast<char>(
                std::tolower(
                    caracter
                )
            );
        }
    );

    return texto;
}

/*
 * Determina el tipo MIME usando solamente el nombre lógico
 * enviado en multipart. No inspecciona ningún archivo físico.
 */
std::string detectarMimeVideo(
    const std::string& nombreVideo
)
{
    const std::string nombre =
        convertirMinusculas(
            nombreVideo
        );

    if (
        nombre.ends_with(".mp4")
        || nombre.ends_with(".m4v")
    )
    {
        return "video/mp4";
    }

    if (nombre.ends_with(".avi"))
    {
        return "video/x-msvideo";
    }

    if (nombre.ends_with(".mov"))
    {
        return "video/quicktime";
    }

    if (nombre.ends_with(".webm"))
    {
        return "video/webm";
    }

    if (nombre.ends_with(".mkv"))
    {
        return "video/x-matroska";
    }

    return "application/octet-stream";
}

void obtenerInformacionRespuesta(
    CURL* curl,
    RespuestaAPI& respuesta
)
{
    if (curl == nullptr)
    {
        return;
    }

    curl_easy_getinfo(
        curl,
        CURLINFO_RESPONSE_CODE,
        &respuesta.codigoHttp
    );

    double tiempoTotalSegundos = 0.0;

    if (
        curl_easy_getinfo(
            curl,
            CURLINFO_TOTAL_TIME,
            &tiempoTotalSegundos
        ) == CURLE_OK
    )
    {
        respuesta.latenciaMilisegundos =
            tiempoTotalSegundos
            * 1000.0;
    }
}

double convertirBytesAMegabytes(
    std::size_t bytes
)
{
    constexpr double BYTES_POR_MEGABYTE =
        1024.0 * 1024.0;

    return
        static_cast<double>(
            bytes
        ) / BYTES_POR_MEGABYTE;
}
}

ClienteAPI::ClienteAPI(
    std::string urlBase,
    std::string apiKey
)
    : urlBase_(
          limpiarUrl(
              urlBase
          )
      ),
      apiKey_(
          quitarEspacios(
              apiKey
          )
      )
{
}

ClienteAPI ClienteAPI::desdeVariablesEntorno()
{
    const char* url =
        std::getenv(
            "TAXI_API_URL"
        );

    const char* clave =
        std::getenv(
            "TAXI_API_KEY"
        );

    return ClienteAPI(
        url != nullptr
            ? url
            : "",
        clave != nullptr
            ? clave
            : ""
    );
}

bool ClienteAPI::configurado() const
{
    return
        !urlBase_.empty()
        && !apiKey_.empty();
}

const std::string&
ClienteAPI::obtenerUrlBase() const
{
    return urlBase_;
}

std::string ClienteAPI::limpiarUrl(
    const std::string& url
)
{
    std::string resultado =
        quitarEspacios(
            url
        );

    while (
        resultado.size() > 1
        && resultado.back() == '/'
    )
    {
        resultado.pop_back();
    }

    return resultado;
}

std::string ClienteAPI::convertirDecimal(
    float valor
)
{
    std::ostringstream texto;

    texto
        << std::fixed
        << std::setprecision(6)
        << valor;

    return texto.str();
}

float ClienteAPI::normalizarScoreSVM(
    float scoreSvm
)
{
    if (!std::isfinite(scoreSvm))
    {
        return 0.0F;
    }

    /*
     * Conversión logística provisional:
     *
     * score 0.0 -> 0.50
     * score 1.0 -> 0.73
     * score 2.0 -> 0.88
     * score 3.0 -> 0.95
     *
     * Este valor no representa una probabilidad
     * estadísticamente calibrada.
     */
    const float confianza =
        1.0F
        / (
            1.0F
            + std::exp(
                  -scoreSvm
              )
        );

    return std::clamp(
        confianza,
        0.0F,
        1.0F
    );
}

bool ClienteAPI::verificarEvento(
    const DatosEventoTaxi& evento,
    std::string& error
) const
{
    if (!configurado())
    {
        error =
            "Faltan TAXI_API_URL o TAXI_API_KEY.";

        return false;
    }

    if (evento.imagenDatos.empty())
    {
        error =
            "La imagen almacenada en RAM está vacía.";

        return false;
    }

    if (evento.videoDatos.empty())
    {
        error =
            "El video almacenado en RAM está vacío.";

        return false;
    }

    const std::string nombreImagen =
        quitarEspacios(
            evento.nombreImagen
        );

    const std::string nombreVideo =
        quitarEspacios(
            evento.nombreVideo
        );

    if (nombreImagen.empty())
    {
        error =
            "El nombre lógico de la imagen está vacío.";

        return false;
    }

    if (nombreVideo.empty())
    {
        error =
            "El nombre lógico del video está vacío.";

        return false;
    }

    const std::string vehiculo =
        quitarEspacios(
            evento.vehiculo
        );

    const std::string camara =
        quitarEspacios(
            evento.camara
        );

    if (
        vehiculo.size() < 2
        || vehiculo.size() > 100
    )
    {
        error =
            "El campo vehiculo debe contener "
            "entre 2 y 100 caracteres.";

        return false;
    }

    if (camara.size() > 100)
    {
        error =
            "El campo camara no puede superar "
            "100 caracteres.";

        return false;
    }

    if (
        !std::isfinite(
            evento.confianzaCpp
        )
        || evento.confianzaCpp < 0.0F
        || evento.confianzaCpp > 1.0F
    )
    {
        error =
            "confianzaCpp debe ser un número "
            "finito entre 0.0 y 1.0.";

        return false;
    }

    return true;
}

RespuestaAPI ClienteAPI::ejecutarPeticionGet(
    const std::string& url
) const
{
    RespuestaAPI respuesta;

    if (!obtenerCurlGlobal().correcto())
    {
        respuesta.mensajeError =
            "No se pudo inicializar libcurl.";

        return respuesta;
    }

    CURL* curl =
        curl_easy_init();

    if (curl == nullptr)
    {
        respuesta.mensajeError =
            "No se pudo crear la sesión CURL.";

        return respuesta;
    }

    curl_easy_setopt(
        curl,
        CURLOPT_URL,
        url.c_str()
    );

    curl_easy_setopt(
        curl,
        CURLOPT_HTTPGET,
        1L
    );

    curl_easy_setopt(
        curl,
        CURLOPT_CONNECTTIMEOUT,
        timeoutConexionSegundos_
    );

    curl_easy_setopt(
        curl,
        CURLOPT_TIMEOUT,
        TIMEOUT_HEALTH_SEGUNDOS
    );

    curl_easy_setopt(
        curl,
        CURLOPT_FOLLOWLOCATION,
        1L
    );

    curl_easy_setopt(
        curl,
        CURLOPT_MAXREDIRS,
        MAXIMAS_REDIRECCIONES
    );

    curl_easy_setopt(
        curl,
        CURLOPT_WRITEFUNCTION,
        recibirRespuesta
    );

    curl_easy_setopt(
        curl,
        CURLOPT_WRITEDATA,
        &respuesta.cuerpo
    );

    curl_easy_setopt(
        curl,
        CURLOPT_USERAGENT,
        "ProyectoTaxiDetector/2.0"
    );

    const CURLcode resultado =
        curl_easy_perform(
            curl
        );

    obtenerInformacionRespuesta(
        curl,
        respuesta
    );

    if (resultado != CURLE_OK)
    {
        respuesta.mensajeError =
            std::string(
                "Error de conexión: "
            )
            + curl_easy_strerror(
                  resultado
              );

        curl_easy_cleanup(
            curl
        );

        return respuesta;
    }

    const bool codigoCorrecto =
        respuesta.codigoHttp >= 200
        && respuesta.codigoHttp < 300;

    respuesta.exitoso =
        codigoCorrecto
        && jsonConfirmaOperacion(
               respuesta.cuerpo
           );

    if (!codigoCorrecto)
    {
        respuesta.mensajeError =
            "La comprobación de salud respondió con HTTP "
            + std::to_string(
                  respuesta.codigoHttp
              );
    }
    else if (!respuesta.exitoso)
    {
        respuesta.mensajeError =
            "La respuesta de salud no contiene ok=true.";
    }

    curl_easy_cleanup(
        curl
    );

    return respuesta;
}

RespuestaAPI ClienteAPI::probarConexion() const
{
    if (!configurado())
    {
        RespuestaAPI respuesta;

        respuesta.mensajeError =
            "Configura TAXI_API_URL y TAXI_API_KEY.";

        return respuesta;
    }

    return ejecutarPeticionGet(
        urlBase_ + "/health"
    );
}

RespuestaAPI ClienteAPI::enviarEvento(
    const DatosEventoTaxi& evento
) const
{
    std::string error;

    if (
        !verificarEvento(
            evento,
            error
        )
    )
    {
        RespuestaAPI respuesta;

        respuesta.mensajeError =
            std::move(
                error
            );

        return respuesta;
    }

    return ejecutarPeticionEvento(
        evento
    );
}

RespuestaAPI ClienteAPI::ejecutarPeticionEvento(
    const DatosEventoTaxi& evento
) const
{
    RespuestaAPI respuesta;

    if (!obtenerCurlGlobal().correcto())
    {
        respuesta.mensajeError =
            "No se pudo inicializar libcurl.";

        return respuesta;
    }

    CURL* curl =
        curl_easy_init();

    if (curl == nullptr)
    {
        respuesta.mensajeError =
            "No se pudo crear la sesión CURL.";

        return respuesta;
    }

    curl_mime* formulario =
        curl_mime_init(
            curl
        );

    if (formulario == nullptr)
    {
        respuesta.mensajeError =
            "No se pudo crear el formulario multipart.";

        curl_easy_cleanup(
            curl
        );

        return respuesta;
    }

    const std::string tipoMimeVideo =
        detectarMimeVideo(
            evento.nombreVideo
        );

    bool formularioCorrecto = true;

    formularioCorrecto =
        formularioCorrecto
        && agregarTexto(
               formulario,
               "vehiculo",
               evento.vehiculo
           );

    formularioCorrecto =
        formularioCorrecto
        && agregarTexto(
               formulario,
               "confianza_cpp",
               convertirDecimal(
                   evento.confianzaCpp
               )
           );

    formularioCorrecto =
        formularioCorrecto
        && agregarArchivoMemoria(
               formulario,
               "imagen",
               evento.imagenDatos,
               evento.nombreImagen,
               "image/jpeg"
           );

    formularioCorrecto =
        formularioCorrecto
        && agregarArchivoMemoria(
               formulario,
               "video",
               evento.videoDatos,
               evento.nombreVideo,
               tipoMimeVideo.c_str()
           );

    formularioCorrecto =
        formularioCorrecto
        && agregarTexto(
               formulario,
               "camara",
               evento.camara
           );

    if (!evento.fechaHora.empty())
    {
        formularioCorrecto =
            formularioCorrecto
            && agregarTexto(
                   formulario,
                   "fecha_hora",
                   evento.fechaHora
               );
    }

    if (!formularioCorrecto)
    {
        respuesta.mensajeError =
            "No se pudieron construir todos "
            "los campos multipart.";

        curl_mime_free(
            formulario
        );

        curl_easy_cleanup(
            curl
        );

        return respuesta;
    }

    curl_slist* encabezados =
        nullptr;

    bool encabezadosCorrectos = true;

    encabezadosCorrectos =
        encabezadosCorrectos
        && agregarEncabezado(
               encabezados,
               "Accept: application/json"
           );

    /*
     * Evita esperar una respuesta provisional
     * HTTP 100 Continue antes de enviar los datos.
     */
    encabezadosCorrectos =
        encabezadosCorrectos
        && agregarEncabezado(
               encabezados,
               "Expect:"
           );

    encabezadosCorrectos =
        encabezadosCorrectos
        && agregarEncabezado(
               encabezados,
               "X-API-Key: "
                   + apiKey_
           );

    if (!encabezadosCorrectos)
    {
        respuesta.mensajeError =
            "No se pudieron construir los encabezados HTTP.";

        curl_slist_free_all(
            encabezados
        );

        curl_mime_free(
            formulario
        );

        curl_easy_cleanup(
            curl
        );

        return respuesta;
    }

    const std::string urlAlerta =
        urlBase_
        + "/api/v1/alerta";

    curl_easy_setopt(
        curl,
        CURLOPT_URL,
        urlAlerta.c_str()
    );

    curl_easy_setopt(
        curl,
        CURLOPT_MIMEPOST,
        formulario
    );

    curl_easy_setopt(
        curl,
        CURLOPT_HTTPHEADER,
        encabezados
    );

    curl_easy_setopt(
        curl,
        CURLOPT_CONNECTTIMEOUT,
        timeoutConexionSegundos_
    );

    curl_easy_setopt(
        curl,
        CURLOPT_TIMEOUT,
        timeoutEnvioSegundos_
    );

    curl_easy_setopt(
        curl,
        CURLOPT_LOW_SPEED_LIMIT,
        LIMITE_BAJA_VELOCIDAD
    );

    curl_easy_setopt(
        curl,
        CURLOPT_LOW_SPEED_TIME,
        TIEMPO_BAJA_VELOCIDAD
    );

    curl_easy_setopt(
        curl,
        CURLOPT_FOLLOWLOCATION,
        1L
    );

    curl_easy_setopt(
        curl,
        CURLOPT_MAXREDIRS,
        MAXIMAS_REDIRECCIONES
    );

    curl_easy_setopt(
        curl,
        CURLOPT_WRITEFUNCTION,
        recibirRespuesta
    );

    curl_easy_setopt(
        curl,
        CURLOPT_WRITEDATA,
        &respuesta.cuerpo
    );

    curl_easy_setopt(
        curl,
        CURLOPT_USERAGENT,
        "ProyectoTaxiDetector/2.0"
    );

    std::cout
        << '\n'
        << "========================================"
        << '\n'
        << "ENVIANDO ALERTA DESDE MEMORIA RAM"
        << '\n'
        << "========================================"
        << '\n'
        << "URL: "
        << urlAlerta
        << '\n'
        << "Vehículo: "
        << evento.vehiculo
        << '\n'
        << "Confianza C++: "
        << convertirDecimal(
               evento.confianzaCpp
           )
        << '\n'
        << "Cámara: "
        << evento.camara
        << '\n'
        << "Fecha y hora: "
        << evento.fechaHora
        << '\n'
        << "Imagen en RAM: "
        << evento.imagenDatos.size()
        << " bytes ("
        << std::fixed
        << std::setprecision(2)
        << convertirBytesAMegabytes(
               evento.imagenDatos.size()
           )
        << " MB)"
        << '\n'
        << "Video en RAM: "
        << evento.videoDatos.size()
        << " bytes ("
        << convertirBytesAMegabytes(
               evento.videoDatos.size()
           )
        << " MB)"
        << '\n'
        << "Nombre imagen: "
        << evento.nombreImagen
        << '\n'
        << "Nombre video: "
        << evento.nombreVideo
        << '\n'
        << "========================================"
        << '\n';

    const CURLcode resultado =
        curl_easy_perform(
            curl
        );

    obtenerInformacionRespuesta(
        curl,
        respuesta
    );

    if (resultado != CURLE_OK)
    {
        respuesta.mensajeError =
            std::string(
                "Error de libcurl: "
            )
            + curl_easy_strerror(
                  resultado
              );
    }
    else
    {
        const bool codigoCorrecto =
            respuesta.codigoHttp >= 200
            && respuesta.codigoHttp < 300;

        respuesta.exitoso =
            codigoCorrecto
            && jsonConfirmaOperacion(
                   respuesta.cuerpo
               );

        if (!codigoCorrecto)
        {
            respuesta.mensajeError =
                "La API respondió con HTTP "
                + std::to_string(
                      respuesta.codigoHttp
                  );
        }
        else if (!respuesta.exitoso)
        {
            respuesta.mensajeError =
                "La API no devolvió ok=true.";
        }
    }

    curl_slist_free_all(
        encabezados
    );

    curl_mime_free(
        formulario
    );

    curl_easy_cleanup(
        curl
    );

    if (respuesta.exitoso)
    {
        std::cout
            << '\n'
            << "ALERTA ACEPTADA POR LA API"
            << '\n'
            << "HTTP: "
            << respuesta.codigoHttp
            << '\n'
            << "Latencia: "
            << std::fixed
            << std::setprecision(2)
            << respuesta.latenciaMilisegundos
            << " ms"
            << '\n'
            << "Respuesta: "
            << respuesta.cuerpo
            << '\n';
    }
    else
    {
        std::cerr
            << '\n'
            << "ERROR ENVIANDO ALERTA"
            << '\n'
            << "HTTP: "
            << respuesta.codigoHttp
            << '\n'
            << "Latencia: "
            << std::fixed
            << std::setprecision(2)
            << respuesta.latenciaMilisegundos
            << " ms"
            << '\n'
            << "Detalle: "
            << respuesta.mensajeError
            << '\n';

        if (!respuesta.cuerpo.empty())
        {
            std::cerr
                << "Respuesta del servidor: "
                << respuesta.cuerpo
                << '\n';
        }
    }

    return respuesta;
}
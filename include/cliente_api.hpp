#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct DatosEventoTaxi
{
    /*
     * Datos binarios almacenados en memoria RAM.
     *
     * imagenDatos debe contener una imagen JPEG válida.
     * videoDatos debe contener un video MP4 válido.
     */
    std::vector<unsigned char> imagenDatos;
    std::vector<unsigned char> videoDatos;

    /*
     * Nombres enviados dentro del formulario multipart.
     *
     * No representan rutas físicas en el disco.
     */
    std::string nombreImagen = "captura_taxi.jpg";
    std::string nombreVideo = "evidencia_taxi.mp4";

    std::string vehiculo = "Taxi amarillo";
    std::string camara = "Cámara principal";
    std::string fechaHora;

    /*
     * Debe encontrarse entre 0.0 y 1.0 porque
     * FastAPI valida Form(ge=0.0, le=1.0).
     *
     * Este valor no representa necesariamente
     * una probabilidad calibrada.
     */
    float confianzaCpp = 0.0F;
};

struct RespuestaAPI
{
    bool exitoso = false;

    long codigoHttp = 0;

    /*
     * Tiempo total de la petición medido por libcurl.
     */
    double latenciaMilisegundos = 0.0;

    std::string cuerpo;
    std::string mensajeError;
};

class ClienteAPI
{
public:
    ClienteAPI(
        std::string urlBase,
        std::string apiKey
    );

    [[nodiscard]]
    static ClienteAPI desdeVariablesEntorno();

    [[nodiscard]]
    bool configurado() const;

    [[nodiscard]]
    RespuestaAPI probarConexion() const;

    [[nodiscard]]
    RespuestaAPI enviarEvento(
        const DatosEventoTaxi& evento
    ) const;

    [[nodiscard]]
    const std::string& obtenerUrlBase() const;

    /*
     * Convierte provisionalmente el score bruto del SVM
     * en un valor entre 0 y 1 mediante una función logística.
     *
     * No representa una probabilidad calibrada.
     */
    [[nodiscard]]
    static float normalizarScoreSVM(
        float scoreSvm
    );

private:
    std::string urlBase_;
    std::string apiKey_;

    long timeoutConexionSegundos_ = 15L;
    long timeoutEnvioSegundos_ = 180L;

    [[nodiscard]]
    bool verificarEvento(
        const DatosEventoTaxi& evento,
        std::string& error
    ) const;

    [[nodiscard]]
    static std::string limpiarUrl(
        const std::string& url
    );

    [[nodiscard]]
    static std::string convertirDecimal(
        float valor
    );

    [[nodiscard]]
    RespuestaAPI ejecutarPeticionGet(
        const std::string& url
    ) const;

    [[nodiscard]]
    RespuestaAPI ejecutarPeticionEvento(
        const DatosEventoTaxi& evento
    ) const;
};
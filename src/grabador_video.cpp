#include "grabador_video.hpp"

#include <opencv2/imgproc.hpp>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace
{
constexpr double FPS_PREDETERMINADO = 10.0;
constexpr double FPS_MINIMO = 5.0;
constexpr double FPS_MAXIMO = 60.0;

constexpr double DURACION_PREDETERMINADA = 5.0;

constexpr int BITRATE_PREDETERMINADO = 2'000'000;
constexpr int BITRATE_MINIMO = 100'000;
constexpr int BITRATE_MAXIMO = 20'000'000;

constexpr int TAMANO_BUFFER_AVIO = 32 * 1024;

std::string convertirErrorFFmpeg(
    int codigoError
)
{
    char textoError[AV_ERROR_MAX_STRING_SIZE]{};

    const int resultado =
        av_strerror(
            codigoError,
            textoError,
            sizeof(textoError)
        );

    if (resultado < 0)
    {
        return
            "Error FFmpeg desconocido: "
            + std::to_string(codigoError);
    }

    return std::string(textoError);
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

cv::Mat prepararFrameBgr(
    const cv::Mat& frame,
    const cv::Size& tamanoDestino
)
{
    if (frame.empty())
    {
        return {};
    }

    if (frame.depth() != CV_8U)
    {
        return {};
    }

    cv::Mat frameConvertido;

    if (frame.channels() == 3)
    {
        frameConvertido = frame;
    }
    else if (frame.channels() == 1)
    {
        cv::cvtColor(
            frame,
            frameConvertido,
            cv::COLOR_GRAY2BGR
        );
    }
    else if (frame.channels() == 4)
    {
        cv::cvtColor(
            frame,
            frameConvertido,
            cv::COLOR_BGRA2BGR
        );
    }
    else
    {
        return {};
    }

    if (frameConvertido.size() == tamanoDestino)
    {
        /*
         * El codificador consume los datos durante esta llamada,
         * por lo que no es necesario clonar el frame.
         */
        return frameConvertido;
    }

    cv::Mat frameRedimensionado;

    cv::resize(
        frameConvertido,
        frameRedimensionado,
        tamanoDestino,
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    return frameRedimensionado;
}
}

/*
 * Implementación privada que contiene todas las estructuras
 * pertenecientes a FFmpeg.
 */
class GrabadorVideo::Implementacion
{
public:
    AVFormatContext* formato = nullptr;
    AVCodecContext* codificador = nullptr;
    AVStream* flujoVideo = nullptr;

    AVFrame* frameYuv = nullptr;
    AVPacket* paquete = nullptr;

    SwsContext* conversorColor = nullptr;

    AVIOContext* contextoIo = nullptr;
    unsigned char* bufferIo = nullptr;

    std::vector<unsigned char>* destinoVideo = nullptr;

    bool cabeceraEscrita = false;
    bool trailerEscrito = false;

    ~Implementacion()
    {
        liberar();
    }

    Implementacion() = default;

    Implementacion(
        const Implementacion&
    ) = delete;

    Implementacion& operator=(
        const Implementacion&
    ) = delete;

    static int escribirEnMemoria(
        void* opaco,
        std::uint8_t* datos,
        int tamano
    )
    {
        if (
            opaco == nullptr
            || datos == nullptr
            || tamano <= 0
        )
        {
            return AVERROR(EINVAL);
        }

        auto* implementacion =
            static_cast<Implementacion*>(
                opaco
            );

        if (
            implementacion->destinoVideo
            == nullptr
        )
        {
            return AVERROR(EINVAL);
        }

        std::vector<unsigned char>& destino =
            *implementacion->destinoVideo;

        const std::size_t cantidad =
            static_cast<std::size_t>(
                tamano
            );

        if (
            cantidad
            > destino.max_size() - destino.size()
        )
        {
            return AVERROR(ENOMEM);
        }

        try
        {
            destino.insert(
                destino.end(),
                datos,
                datos + cantidad
            );
        }
        catch (const std::bad_alloc&)
        {
            return AVERROR(ENOMEM);
        }
        catch (...)
        {
            return AVERROR(EIO);
        }

        return tamano;
    }

    void liberar()
    {
        if (paquete != nullptr)
        {
            av_packet_free(
                &paquete
            );
        }

        if (frameYuv != nullptr)
        {
            av_frame_free(
                &frameYuv
            );
        }

        if (conversorColor != nullptr)
        {
            sws_freeContext(
                conversorColor
            );

            conversorColor = nullptr;
        }

        if (codificador != nullptr)
        {
            avcodec_free_context(
                &codificador
            );
        }

        /*
         * El AVIOContext es personalizado, por eso se libera
         * manualmente después de desvincularlo del formato.
         */
        if (formato != nullptr)
        {
            formato->pb = nullptr;

            avformat_free_context(
                formato
            );

            formato = nullptr;
        }

        if (contextoIo != nullptr)
        {
            /*
             * avio_context_free libera también el buffer
             * asociado al contexto.
             */
            avio_context_free(
                &contextoIo
            );

            bufferIo = nullptr;
        }
        else if (bufferIo != nullptr)
        {
            av_free(
                bufferIo
            );

            bufferIo = nullptr;
        }

        flujoVideo = nullptr;
        destinoVideo = nullptr;

        cabeceraEscrita = false;
        trailerEscrito = false;
    }
};

GrabadorVideo::GrabadorVideo(
    double duracionSegundos,
    int bitrateBitsPorSegundo
)
    : implementacion_(
          std::make_unique<Implementacion>()
      ),
      duracionSegundos_(
          std::isfinite(duracionSegundos)
          && duracionSegundos > 0.0
              ? duracionSegundos
              : DURACION_PREDETERMINADA
      ),
      bitrateBitsPorSegundo_(
          std::clamp(
              bitrateBitsPorSegundo,
              BITRATE_MINIMO,
              BITRATE_MAXIMO
          )
      )
{
}

GrabadorVideo::~GrabadorVideo()
{
    if (grabando_)
    {
        /*
         * Se intenta cerrar correctamente el MP4 antes
         * de destruir el objeto.
         */
        (void)finalizar();
    }

    if (implementacion_ != nullptr)
    {
        implementacion_->liberar();
    }
}

GrabadorVideo::GrabadorVideo(
    GrabadorVideo&&
) noexcept = default;

GrabadorVideo& GrabadorVideo::operator=(
    GrabadorVideo&&
) noexcept = default;

bool GrabadorVideo::inicializar()
{
    if (inicializado_)
    {
        return true;
    }

    if (
        !std::isfinite(duracionSegundos_)
        || duracionSegundos_ <= 0.0
    )
    {
        std::cerr
            << "La duración del video no es válida."
            << '\n';

        return false;
    }

    if (
        bitrateBitsPorSegundo_ < BITRATE_MINIMO
        || bitrateBitsPorSegundo_ > BITRATE_MAXIMO
    )
    {
        std::cerr
            << "El bitrate del video no es válido."
            << '\n';

        return false;
    }

    if (implementacion_ == nullptr)
    {
        implementacion_ =
            std::make_unique<Implementacion>();
    }

    inicializado_ = true;

    std::cout
        << "Grabador de video en memoria RAM preparado."
        << '\n'
        << "Duración configurada: "
        << duracionSegundos_
        << " segundos"
        << '\n'
        << "Bitrate configurado: "
        << bitrateBitsPorSegundo_
        << " bits/s"
        << '\n';

    return true;
}

bool GrabadorVideo::parametrosValidos(
    const cv::Size& tamanoFrame,
    double fps
) const
{
    if (
        tamanoFrame.width <= 0
        || tamanoFrame.height <= 0
    )
    {
        return false;
    }

    /*
     * YUV420P necesita dimensiones pares.
     */
    if (
        tamanoFrame.width % 2 != 0
        || tamanoFrame.height % 2 != 0
    )
    {
        return false;
    }

    return
        std::isfinite(fps)
        && fps > 0.0;
}

void GrabadorVideo::reiniciarEstadoGrabacion()
{
    fpsGrabacion_ = 0.0;

    tamanoGrabacion_ =
        cv::Size(
            0,
            0
        );

    grabando_ = false;
    completada_ = false;

    framesGrabados_ = 0;
    framesObjetivo_ = 0;

    videoMp4_.clear();

    tiempoInicio_ = {};
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
            << "ya existe una grabación activa."
            << '\n';

        return false;
    }

    if (!inicializar())
    {
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

    if (
        !parametrosValidos(
            tamanoFrame,
            fps
        )
    )
    {
        std::cerr
            << "Parámetros de grabación inválidos."
            << '\n'
            << "Resolución recibida: "
            << tamanoFrame.width
            << 'x'
            << tamanoFrame.height
            << '\n'
            << "La resolución debe tener ancho y alto pares."
            << '\n'
            << "FPS recibido: "
            << fps
            << '\n';

        return false;
    }

    if (implementacion_ == nullptr)
    {
        implementacion_ =
            std::make_unique<Implementacion>();
    }
    else
    {
        implementacion_->liberar();
    }

    reiniciarEstadoGrabacion();

    fpsGrabacion_ = fps;
    tamanoGrabacion_ = tamanoFrame;

    framesObjetivo_ =
        std::max<std::size_t>(
            1,
            static_cast<std::size_t>(
                std::lround(
                    duracionSegundos_
                    * fpsGrabacion_
                )
            )
        );

    videoMp4_.reserve(
        static_cast<std::size_t>(
            std::max(
                bitrateBitsPorSegundo_,
                BITRATE_MINIMO
            )
        )
        * static_cast<std::size_t>(
              std::ceil(
                  duracionSegundos_
              )
          )
        / 8U
        + 256U * 1024U
    );

    implementacion_->destinoVideo =
        &videoMp4_;

    int resultado =
        avformat_alloc_output_context2(
            &implementacion_->formato,
            nullptr,
            "mp4",
            nullptr
        );

    if (
        resultado < 0
        || implementacion_->formato == nullptr
    )
    {
        std::cerr
            << "No se pudo crear el contenedor MP4: "
            << convertirErrorFFmpeg(resultado)
            << '\n';

        cancelar();

        return false;
    }

    /*
     * Primero se intenta usar H.264. Si no está disponible,
     * se utiliza MPEG-4 Parte 2 como alternativa compatible
     * con el contenedor MP4.
     */
    const AVCodec* codec =
        avcodec_find_encoder(
            AV_CODEC_ID_H264
        );

    if (codec == nullptr)
    {
        codec =
            avcodec_find_encoder(
                AV_CODEC_ID_MPEG4
            );
    }

    if (codec == nullptr)
    {
        std::cerr
            << "FFmpeg no dispone de un codificador "
            << "H.264 ni MPEG-4."
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->flujoVideo =
        avformat_new_stream(
            implementacion_->formato,
            nullptr
        );

    if (implementacion_->flujoVideo == nullptr)
    {
        std::cerr
            << "No se pudo crear el flujo de video."
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->codificador =
        avcodec_alloc_context3(
            codec
        );

    if (implementacion_->codificador == nullptr)
    {
        std::cerr
            << "No se pudo crear el contexto del codificador."
            << '\n';

        cancelar();

        return false;
    }

    AVCodecContext* contextoCodec =
        implementacion_->codificador;

    const int fpsEntero =
        std::clamp(
            static_cast<int>(
                std::lround(fpsGrabacion_)
            ),
            static_cast<int>(FPS_MINIMO),
            static_cast<int>(FPS_MAXIMO)
        );

    contextoCodec->codec_id =
        codec->id;

    contextoCodec->codec_type =
        AVMEDIA_TYPE_VIDEO;

    contextoCodec->width =
        tamanoGrabacion_.width;

    contextoCodec->height =
        tamanoGrabacion_.height;

    contextoCodec->pix_fmt =
        AV_PIX_FMT_YUV420P;

    contextoCodec->bit_rate =
        bitrateBitsPorSegundo_;

    contextoCodec->time_base =
        AVRational{
            1,
            fpsEntero
        };

    contextoCodec->framerate =
        AVRational{
            fpsEntero,
            1
        };

    contextoCodec->gop_size =
        fpsEntero;

    contextoCodec->max_b_frames = 0;

    if (
        (
            implementacion_->formato
                ->oformat
                ->flags
            & AVFMT_GLOBALHEADER
        ) != 0
    )
    {
        contextoCodec->flags |=
            AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (codec->id == AV_CODEC_ID_H264)
    {
        /*
         * Estas opciones son aceptadas por libx264.
         * Si el codificador H.264 disponible no las admite,
         * FFmpeg simplemente puede ignorarlas.
         */
        av_opt_set(
            contextoCodec->priv_data,
            "preset",
            "veryfast",
            0
        );

        av_opt_set(
            contextoCodec->priv_data,
            "tune",
            "zerolatency",
            0
        );
    }

    resultado =
        avcodec_open2(
            contextoCodec,
            codec,
            nullptr
        );

    if (resultado < 0)
    {
        std::cerr
            << "No se pudo abrir el codificador "
            << codec->name
            << ": "
            << convertirErrorFFmpeg(resultado)
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->flujoVideo->time_base =
        contextoCodec->time_base;

    implementacion_->flujoVideo->avg_frame_rate =
        contextoCodec->framerate;

    resultado =
        avcodec_parameters_from_context(
            implementacion_
                ->flujoVideo
                ->codecpar,
            contextoCodec
        );

    if (resultado < 0)
    {
        std::cerr
            << "No se pudieron copiar los parámetros "
            << "del codificador: "
            << convertirErrorFFmpeg(resultado)
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->bufferIo =
        static_cast<unsigned char*>(
            av_malloc(
                TAMANO_BUFFER_AVIO
            )
        );

    if (implementacion_->bufferIo == nullptr)
    {
        std::cerr
            << "No se pudo reservar el buffer AVIO."
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->contextoIo =
        avio_alloc_context(
            implementacion_->bufferIo,
            TAMANO_BUFFER_AVIO,
            1,
            implementacion_.get(),
            nullptr,
            &Implementacion::escribirEnMemoria,
            nullptr
        );

    if (implementacion_->contextoIo == nullptr)
    {
        std::cerr
            << "No se pudo crear el contexto AVIO de memoria."
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->formato->pb =
        implementacion_->contextoIo;

    implementacion_->formato->flags |=
        AVFMT_FLAG_CUSTOM_IO;

    implementacion_->frameYuv =
        av_frame_alloc();

    if (implementacion_->frameYuv == nullptr)
    {
        std::cerr
            << "No se pudo crear el frame YUV."
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->frameYuv->format =
        contextoCodec->pix_fmt;

    implementacion_->frameYuv->width =
        contextoCodec->width;

    implementacion_->frameYuv->height =
        contextoCodec->height;

    resultado =
        av_frame_get_buffer(
            implementacion_->frameYuv,
            32
        );

    if (resultado < 0)
    {
        std::cerr
            << "No se pudo reservar memoria para el frame YUV: "
            << convertirErrorFFmpeg(resultado)
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->paquete =
        av_packet_alloc();

    if (implementacion_->paquete == nullptr)
    {
        std::cerr
            << "No se pudo crear el paquete de video."
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->conversorColor =
        sws_getContext(
            tamanoGrabacion_.width,
            tamanoGrabacion_.height,
            AV_PIX_FMT_BGR24,
            tamanoGrabacion_.width,
            tamanoGrabacion_.height,
            AV_PIX_FMT_YUV420P,
            SWS_BILINEAR,
            nullptr,
            nullptr,
            nullptr
        );

    if (implementacion_->conversorColor == nullptr)
    {
        std::cerr
            << "No se pudo crear el conversor BGR a YUV420P."
            << '\n';

        cancelar();

        return false;
    }

    AVDictionary* opcionesMuxer = nullptr;

    /*
     * El MP4 fragmentado permite escribir en una salida
     * de memoria que no implementa búsqueda o retroceso.
     */
    av_dict_set(
        &opcionesMuxer,
        "movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        0
    );

    resultado =
        avformat_write_header(
            implementacion_->formato,
            &opcionesMuxer
        );

    av_dict_free(
        &opcionesMuxer
    );

    if (resultado < 0)
    {
        std::cerr
            << "No se pudo escribir la cabecera MP4: "
            << convertirErrorFFmpeg(resultado)
            << '\n';

        cancelar();

        return false;
    }

    implementacion_->cabeceraEscrita = true;

    grabando_ = true;
    completada_ = false;

    tiempoInicio_ =
        std::chrono::steady_clock::now();

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << "GRABACION DE VIDEO INICIADA EN RAM"
        << '\n'
        << "============================================="
        << '\n'
        << "Contenedor: MP4 fragmentado"
        << '\n'
        << "Códec: "
        << codec->name
        << '\n'
        << "Resolución: "
        << tamanoGrabacion_.width
        << 'x'
        << tamanoGrabacion_.height
        << '\n'
        << "FPS: "
        << fpsGrabacion_
        << '\n'
        << "Frames objetivo: "
        << framesObjetivo_
        << '\n'
        << "Duración objetivo: "
        << duracionSegundos_
        << " segundos"
        << '\n'
        << "Almacenamiento en disco: NO"
        << '\n'
        << "============================================="
        << '\n';

    return true;
}

bool GrabadorVideo::escribirFrame(
    const cv::Mat& frame
)
{
    if (
        !grabando_
        || implementacion_ == nullptr
        || implementacion_->codificador == nullptr
        || implementacion_->frameYuv == nullptr
        || implementacion_->paquete == nullptr
    )
    {
        return false;
    }

    if (frame.empty())
    {
        std::cerr
            << "No se puede codificar un frame vacío."
            << '\n';

        return false;
    }

    cv::Mat frameBgr;

    try
    {
        frameBgr =
            prepararFrameBgr(
                frame,
                tamanoGrabacion_
            );
    }
    catch (const cv::Exception& error)
    {
        std::cerr
            << "Error preparando el frame: "
            << error.what()
            << '\n';

        return false;
    }

    if (frameBgr.empty())
    {
        std::cerr
            << "El formato del frame no es compatible."
            << '\n';

        return false;
    }

    const double tiempoTranscurrido =
        obtenerTiempoTranscurrido();

    std::size_t framesEsperados =
        static_cast<std::size_t>(
            std::floor(
                tiempoTranscurrido
                * fpsGrabacion_
            )
        ) + 1U;

    framesEsperados =
        std::clamp<std::size_t>(
            framesEsperados,
            1U,
            framesObjetivo_
        );

    auto codificarFrame =
        [this, &frameBgr]() -> bool
        {
            AVFrame* frameYuv =
                implementacion_->frameYuv;

            AVCodecContext* codec =
                implementacion_->codificador;

            int resultado =
                av_frame_make_writable(
                    frameYuv
                );

            if (resultado < 0)
            {
                std::cerr
                    << "No se pudo preparar el frame YUV: "
                    << convertirErrorFFmpeg(resultado)
                    << '\n';

                return false;
            }

            const std::uint8_t* datosOrigen[4]{
                frameBgr.data,
                nullptr,
                nullptr,
                nullptr
            };

            const int lineasOrigen[4]{
                static_cast<int>(
                    frameBgr.step[0]
                ),
                0,
                0,
                0
            };

            const int lineasConvertidas =
                sws_scale(
                    implementacion_
                        ->conversorColor,
                    datosOrigen,
                    lineasOrigen,
                    0,
                    tamanoGrabacion_.height,
                    frameYuv->data,
                    frameYuv->linesize
                );

            if (
                lineasConvertidas
                != tamanoGrabacion_.height
            )
            {
                std::cerr
                    << "FFmpeg no convirtió todas las líneas "
                    << "del frame."
                    << '\n';

                return false;
            }

            frameYuv->pts =
                static_cast<std::int64_t>(
                    framesGrabados_
                );

            resultado =
                avcodec_send_frame(
                    codec,
                    frameYuv
                );

            if (resultado < 0)
            {
                std::cerr
                    << "No se pudo enviar el frame al codificador: "
                    << convertirErrorFFmpeg(resultado)
                    << '\n';

                return false;
            }

            while (true)
            {
                resultado =
                    avcodec_receive_packet(
                        codec,
                        implementacion_->paquete
                    );

                if (
                    resultado == AVERROR(EAGAIN)
                    || resultado == AVERROR_EOF
                )
                {
                    break;
                }

                if (resultado < 0)
                {
                    std::cerr
                        << "No se pudo recibir el paquete codificado: "
                        << convertirErrorFFmpeg(resultado)
                        << '\n';

                    return false;
                }

                av_packet_rescale_ts(
                    implementacion_->paquete,
                    codec->time_base,
                    implementacion_
                        ->flujoVideo
                        ->time_base
                );

                implementacion_
                    ->paquete
                    ->stream_index =
                    implementacion_
                        ->flujoVideo
                        ->index;

                resultado =
                    av_interleaved_write_frame(
                        implementacion_->formato,
                        implementacion_->paquete
                    );

                av_packet_unref(
                    implementacion_->paquete
                );

                if (resultado < 0)
                {
                    std::cerr
                        << "No se pudo escribir el paquete MP4: "
                        << convertirErrorFFmpeg(resultado)
                        << '\n';

                    return false;
                }
            }

            ++framesGrabados_;

            return true;
        };

    /*
     * Si el procesamiento de detección reduce la frecuencia
     * real, se repite el frame actual hasta alcanzar la
     * cantidad correspondiente al tiempo transcurrido.
     */
    while (
        framesGrabados_ < framesEsperados
    )
    {
        if (!codificarFrame())
        {
            cancelar();

            return false;
        }
    }

    if (
        tiempoTranscurrido >= duracionSegundos_
        || framesGrabados_ >= framesObjetivo_
    )
    {
        return finalizar();
    }

    return true;
}

bool GrabadorVideo::finalizar()
{
    if (!grabando_)
    {
        return completada_;
    }

    if (
        implementacion_ == nullptr
        || implementacion_->codificador == nullptr
        || implementacion_->formato == nullptr
    )
    {
        cancelar();

        return false;
    }

    const double tiempoReal =
        obtenerTiempoTranscurrido();

    bool correcto = true;

    int resultado =
        avcodec_send_frame(
            implementacion_->codificador,
            nullptr
        );

    if (
        resultado < 0
        && resultado != AVERROR_EOF
    )
    {
        std::cerr
            << "No se pudo vaciar el codificador: "
            << convertirErrorFFmpeg(resultado)
            << '\n';

        correcto = false;
    }

    if (correcto)
    {
        while (true)
        {
            resultado =
                avcodec_receive_packet(
                    implementacion_->codificador,
                    implementacion_->paquete
                );

            if (
                resultado == AVERROR_EOF
                || resultado == AVERROR(EAGAIN)
            )
            {
                break;
            }

            if (resultado < 0)
            {
                std::cerr
                    << "Error recibiendo los paquetes finales: "
                    << convertirErrorFFmpeg(resultado)
                    << '\n';

                correcto = false;

                break;
            }

            av_packet_rescale_ts(
                implementacion_->paquete,
                implementacion_
                    ->codificador
                    ->time_base,
                implementacion_
                    ->flujoVideo
                    ->time_base
            );

            implementacion_
                ->paquete
                ->stream_index =
                implementacion_
                    ->flujoVideo
                    ->index;

            resultado =
                av_interleaved_write_frame(
                    implementacion_->formato,
                    implementacion_->paquete
                );

            av_packet_unref(
                implementacion_->paquete
            );

            if (resultado < 0)
            {
                std::cerr
                    << "No se pudo escribir un paquete final: "
                    << convertirErrorFFmpeg(resultado)
                    << '\n';

                correcto = false;

                break;
            }
        }
    }

    if (
        correcto
        && implementacion_->cabeceraEscrita
    )
    {
        resultado =
            av_write_trailer(
                implementacion_->formato
            );

        if (resultado < 0)
        {
            std::cerr
                << "No se pudo escribir el tráiler MP4: "
                << convertirErrorFFmpeg(resultado)
                << '\n';

            correcto = false;
        }
        else
        {
            implementacion_->trailerEscrito = true;
        }
    }

    if (
        implementacion_->contextoIo != nullptr
    )
    {
        avio_flush(
            implementacion_->contextoIo
        );
    }

    grabando_ = false;

    completada_ =
        correcto
        && !videoMp4_.empty();

    if (completada_)
    {
        ++totalVideos_;
    }

    const double duracionVideo =
        fpsGrabacion_ > 0.0
            ? static_cast<double>(
                  framesGrabados_
              ) / fpsGrabacion_
            : 0.0;

    std::cout
        << '\n'
        << "============================================="
        << '\n'
        << (
               completada_
                   ? "VIDEO MP4 FINALIZADO EN MEMORIA RAM"
                   : "ERROR FINALIZANDO EL VIDEO MP4"
           )
        << '\n'
        << "============================================="
        << '\n'
        << "Tiempo real transcurrido: "
        << std::fixed
        << std::setprecision(2)
        << tiempoReal
        << " segundos"
        << '\n'
        << "Duración estimada del video: "
        << duracionVideo
        << " segundos"
        << '\n'
        << "Frames codificados: "
        << framesGrabados_
        << '\n'
        << "FPS: "
        << fpsGrabacion_
        << '\n'
        << "Tamaño en RAM: "
        << videoMp4_.size()
        << " bytes ("
        << convertirBytesAMegabytes(
               videoMp4_.size()
           )
        << " MB)"
        << '\n'
        << "Total de videos: "
        << totalVideos_
        << '\n'
        << "Almacenamiento en disco: NO"
        << '\n'
        << "============================================="
        << '\n';

    implementacion_->liberar();

    return completada_;
}

void GrabadorVideo::cancelar()
{
    if (implementacion_ != nullptr)
    {
        implementacion_->liberar();
    }

    videoMp4_.clear();

    fpsGrabacion_ = 0.0;

    tamanoGrabacion_ =
        cv::Size(
            0,
            0
        );

    grabando_ = false;
    completada_ = false;

    framesGrabados_ = 0;
    framesObjetivo_ = 0;

    tiempoInicio_ = {};
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

    return
        std::chrono::duration<double>(
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

std::size_t
GrabadorVideo::obtenerFramesGrabados() const
{
    return framesGrabados_;
}

std::size_t
GrabadorVideo::obtenerFramesObjetivo() const
{
    return framesObjetivo_;
}

std::size_t
GrabadorVideo::obtenerTotalVideos() const
{
    return totalVideos_;
}

double GrabadorVideo::obtenerDuracionSegundos() const
{
    return duracionSegundos_;
}

double GrabadorVideo::obtenerFpsGrabacion() const
{
    return fpsGrabacion_;
}

const cv::Size&
GrabadorVideo::obtenerTamanoGrabacion() const
{
    return tamanoGrabacion_;
}

const std::vector<unsigned char>&
GrabadorVideo::obtenerVideoEnMemoria() const
{
    return videoMp4_;
}

std::vector<unsigned char>
GrabadorVideo::extraerVideoEnMemoria()
{
    if (
        grabando_
        || !completada_
    )
    {
        std::cerr
            << "El video todavía no está disponible "
            << "para ser extraído."
            << '\n';

        return {};
    }

    std::vector<unsigned char> resultado =
        std::move(
            videoMp4_
        );

    videoMp4_.clear();

    return resultado;
}

std::size_t
GrabadorVideo::obtenerTamanoVideoBytes() const
{
    return videoMp4_.size();
}
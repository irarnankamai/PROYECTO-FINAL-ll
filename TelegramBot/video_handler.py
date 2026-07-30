from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from telegram import Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import (
    VIDEO_CLIP_SECONDS,
    VIDEO_MAX_OUTPUT_FPS,
    VIDEO_MAX_SIDE,
)
from video_processor import YOLOVideoProcessor
from yolo_processor import YOLOSegmenter


# =========================================================
# LOGS
# =========================================================

logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURACIÓN
# =========================================================

MAX_DOWNLOAD_BYTES = 19 * 1024 * 1024
MAX_UPLOAD_BYTES = 49 * 1024 * 1024

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}

MIME_EXTENSION_MAP = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
    "video/x-m4v": ".m4v",
}


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def create_event_id() -> str:
    """
    Genera un identificador único para el evento.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    random_id = uuid4().hex[:8]

    return f"{timestamp}_{random_id}"


def obtain_video_extension(
    file_name: str | None,
    mime_type: str | None,
) -> str:
    """
    Obtiene una extensión válida para el video.
    """

    if file_name:
        extension = Path(
            file_name
        ).suffix.lower()

        if extension in SUPPORTED_VIDEO_EXTENSIONS:
            return extension

    if mime_type:
        mapped_extension = MIME_EXTENSION_MAP.get(
            mime_type.lower()
        )

        if mapped_extension:
            return mapped_extension

    return ".mp4"


def sanitize_filename(
    file_name: str | None,
    default_name: str,
) -> str:
    """
    Elimina rutas o caracteres de directorio
    incluidos en el nombre recibido.
    """

    if not file_name:
        return default_name

    clean_name = Path(
        file_name
    ).name.strip()

    return clean_name or default_name


def shorten_text(
    text: str,
    maximum_length: int = 500,
) -> str:
    """
    Reduce textos demasiado largos.
    """

    if len(text) <= maximum_length:
        return text

    return (
        text[: maximum_length - 3]
        + "..."
    )


def create_memory_buffer(
    data: bytes,
    filename: str,
) -> BytesIO:
    """
    Crea un archivo virtual compatible con Telegram.
    """

    if not data:
        raise ValueError(
            f"El archivo {filename} está vacío."
        )

    buffer = BytesIO(
        data
    )

    buffer.name = filename
    buffer.seek(0)

    return buffer


def build_video_caption(
    metrics: dict[str, Any],
    event_id: str,
) -> str:
    """
    Construye el mensaje con las métricas del video.
    """

    classes_summary = shorten_text(
        str(
            metrics.get(
                "classes_summary",
                "Sin información",
            )
        ),
        maximum_length=300,
    )

    return (
        "🎬 VIDEO SEGMENTADO CON YOLO\n\n"
        f"Evento: {event_id}\n"
        f"Modelo: "
        f"{metrics.get('model', 'Sin información')}\n"
        f"Dispositivo: "
        f"{metrics.get('device', 'Sin información')}\n"
        f"Duración: "
        f"{float(metrics.get('clip_duration_seconds', 0.0)):.2f} s\n"
        f"Frames procesados: "
        f"{metrics.get('frames_processed', 0)}\n"
        f"Frames con detecciones: "
        f"{metrics.get('frames_with_detections', 0)}\n"
        f"FPS de procesamiento: "
        f"{float(metrics.get('processing_fps', 0.0)):.2f}\n"
        f"FPS equivalente de inferencia: "
        f"{float(metrics.get('inference_fps', 0.0)):.2f}\n"
        f"Inferencia promedio: "
        f"{float(metrics.get('average_inference_ms', 0.0)):.2f} ms\n"
        f"Confianza promedio: "
        f"{float(metrics.get('confidence_average', 0.0)) * 100:.2f} %\n"
        f"Confianza máxima: "
        f"{float(metrics.get('confidence_maximum', 0.0)) * 100:.2f} %\n"
        f"RAM máxima: "
        f"{float(metrics.get('ram_peak_mb', 0.0)):.2f} MB\n"
        f"Detecciones acumuladas: "
        f"{metrics.get('detections_accumulated', 0)}\n"
        f"Máscaras acumuladas: "
        f"{metrics.get('masks_accumulated', 0)}\n"
        f"Tamaño del resultado: "
        f"{float(metrics.get('output_size_mb', 0.0)):.2f} MB\n\n"
        f"Clases detectadas:\n"
        f"{classes_summary}"
    )


# =========================================================
# ENVIAR VIDEO DESDE RAM
# =========================================================

async def send_video_from_memory(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    video_data: bytes,
    metrics: dict[str, Any],
    caption: str,
) -> None:
    """
    Envía el video desde memoria RAM.

    Si Telegram no puede reproducirlo como video,
    se envía como documento MP4.
    """

    if not video_data:
        raise ValueError(
            "El video segmentado está vacío."
        )

    if len(video_data) > MAX_UPLOAD_BYTES:
        raise RuntimeError(
            "El video segmentado supera 49 MB. "
            "Reduce VIDEO_MAX_SIDE o "
            "VIDEO_MAX_OUTPUT_FPS en config.py."
        )

    duration = max(
        1,
        int(
            round(
                float(
                    metrics.get(
                        "clip_duration_seconds",
                        VIDEO_CLIP_SECONDS,
                    )
                )
            )
        ),
    )

    width = max(
        1,
        int(
            metrics.get(
                "output_width",
                1,
            )
        ),
    )

    height = max(
        1,
        int(
            metrics.get(
                "output_height",
                1,
            )
        ),
    )

    video_buffer = create_memory_buffer(
        video_data,
        "video_segmentado.mp4",
    )

    try:
        await context.bot.send_video(
            chat_id=chat_id,
            video=video_buffer,
            caption=caption,
            duration=duration,
            width=width,
            height=height,
            supports_streaming=True,
            connect_timeout=30,
            read_timeout=180,
            write_timeout=180,
        )

    except TelegramError as send_error:
        logger.warning(
            "Telegram no pudo enviar el resultado "
            "como video: %s. Se enviará como documento.",
            send_error,
        )

        if not video_buffer.closed:
            video_buffer.close()

        document_buffer = create_memory_buffer(
            video_data,
            "video_segmentado.mp4",
        )

        try:
            await context.bot.send_document(
                chat_id=chat_id,
                document=document_buffer,
                caption=(
                    caption
                    + "\n\n"
                    "⚠️ Telegram recibió el resultado "
                    "como archivo MP4."
                ),
                connect_timeout=30,
                read_timeout=180,
                write_timeout=180,
            )

        finally:
            document_buffer.close()

    finally:
        if not video_buffer.closed:
            video_buffer.close()


# =========================================================
# RECIBIR VIDEO
# =========================================================

async def receive_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Recibe un video desde Telegram, lo descarga en RAM,
    lo procesa con YOLO y devuelve el resultado.
    """

    message = update.effective_message
    chat = update.effective_chat

    if message is None or chat is None:
        return

    attachment = None
    file_name: str | None = None
    mime_type: str | None = None
    duration = 0.0
    source = ""

    # =====================================================
    # VIDEO ENVIADO NORMALMENTE
    # =====================================================

    if message.video:
        attachment = message.video

        file_name = getattr(
            attachment,
            "file_name",
            None,
        )

        mime_type = attachment.mime_type

        duration = float(
            attachment.duration or 0.0
        )

        source = "video enviado desde Telegram"

    # =====================================================
    # VIDEO ENVIADO COMO DOCUMENTO
    # =====================================================

    elif message.document:
        attachment = message.document
        file_name = attachment.file_name
        mime_type = attachment.mime_type
        source = "video enviado como documento"

    if attachment is None:
        await message.reply_text(
            "❌ No se recibió un video válido."
        )
        return

    # =====================================================
    # VALIDAR FORMATO
    # =====================================================

    extension = obtain_video_extension(
        file_name=file_name,
        mime_type=mime_type,
    )

    if (
        file_name
        and Path(file_name).suffix
        and Path(file_name).suffix.lower()
        not in SUPPORTED_VIDEO_EXTENSIONS
    ):
        await message.reply_text(
            "❌ Formato de video no compatible.\n\n"
            "Formatos permitidos:\n"
            "MP4, MOV, AVI, MKV, WEBM y M4V."
        )
        return

    # =====================================================
    # VALIDAR TAMAÑO
    # =====================================================

    file_size = attachment.file_size

    if (
        file_size is not None
        and file_size > MAX_DOWNLOAD_BYTES
    ):
        size_mb = (
            file_size
            / (1024 * 1024)
        )

        await message.reply_text(
            "❌ El video supera el límite "
            "permitido para esta prueba.\n\n"
            f"Tamaño recibido: {size_mb:.2f} MB\n"
            "Tamaño máximo configurado: 19 MB."
        )
        return

    # =====================================================
    # VALIDAR DURACIÓN
    # =====================================================

    if (
        duration > 0.0
        and (
            duration + 0.20
            < VIDEO_CLIP_SECONDS
        )
    ):
        await message.reply_text(
            "❌ El video es demasiado corto.\n\n"
            f"Duración recibida: {duration:.2f} s\n"
            f"Duración mínima: "
            f"{VIDEO_CLIP_SECONDS:.2f} s."
        )
        return

    processor: YOLOVideoProcessor = (
        context.application.bot_data[
            "video_processor"
        ]
    )

    yolo_lock: asyncio.Lock = (
        context.application.bot_data[
            "yolo_lock"
        ]
    )

    event_id = create_event_id()

    original_filename = sanitize_filename(
        file_name,
        f"video_{event_id}{extension}",
    )

    status_message = await message.reply_text(
        "📥 Video recibido.\n\n"
        f"Evento: {event_id}\n"
        f"Fuente: {source}\n"
        "Descargando directamente a memoria RAM..."
    )

    try:
        # =================================================
        # DESCARGAR VIDEO EN RAM
        # =================================================

        telegram_file = await context.bot.get_file(
            attachment.file_id
        )

        downloaded_data = (
            await telegram_file.download_as_bytearray()
        )

        video_data = bytes(
            downloaded_data
        )

        if not video_data:
            raise RuntimeError(
                "Telegram devolvió un video vacío."
            )

        if len(video_data) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(
                "El video descargado supera "
                "el tamaño permitido."
            )

        logger.info(
            "Video recibido en RAM. "
            "Evento: %s. Archivo: %s. Tamaño: %.2f MB.",
            event_id,
            original_filename,
            len(video_data) / (1024 * 1024),
        )

        await status_message.edit_text(
            "🧠 Procesando video frame a frame "
            "con YOLO...\n\n"
            f"Evento: {event_id}\n"
            f"Duración objetivo: "
            f"{VIDEO_CLIP_SECONDS:.0f} segundos\n"
            "Procesamiento: memoria RAM."
        )

        # =================================================
        # PROCESAR VIDEO EN RAM
        # =================================================

        async with yolo_lock:
            (
                segmented_video_data,
                metrics,
            ) = await asyncio.to_thread(
                processor.process_video,
                video_data,
            )

        if not segmented_video_data:
            raise RuntimeError(
                "YOLO generó un video segmentado vacío."
            )

        if (
            len(segmented_video_data)
            > MAX_UPLOAD_BYTES
        ):
            raise RuntimeError(
                "El video segmentado supera 49 MB. "
                "Reduce VIDEO_MAX_SIDE o "
                "VIDEO_MAX_OUTPUT_FPS en config.py."
            )

        caption = build_video_caption(
            metrics=metrics,
            event_id=event_id,
        )

        # =================================================
        # ENVIAR RESULTADO DESDE RAM
        # =================================================

        await send_video_from_memory(
            context=context,
            chat_id=chat.id,
            video_data=segmented_video_data,
            metrics=metrics,
            caption=caption,
        )

        await status_message.edit_text(
            "✅ Video segmentado y enviado "
            "correctamente.\n\n"
            f"Evento: {event_id}\n"
            "Almacenamiento permanente: desactivado."
        )

        logger.info(
            "Video %s procesado y enviado correctamente.",
            event_id,
        )

    except Exception as error:
        logger.exception(
            "Error procesando el video recibido."
        )

        await status_message.edit_text(
            "❌ No fue posible procesar el video.\n\n"
            f"Evento: {event_id}\n"
            f"Detalle: "
            f"{shorten_text(str(error), 500)}"
        )


# =========================================================
# REGISTRAR MÓDULO EN EL BOT
# =========================================================

def register_video_module(
    app: Application,
    segmenter: YOLOSegmenter,
    base_dir: Path | None = None,
) -> None:
    """
    Registra el procesador y el manejador de videos.

    base_dir se conserva para mantener compatibilidad
    con la llamada realizada desde bot.py, pero ya no
    se crean carpetas ni archivos.
    """

    del base_dir

    if not isinstance(
        segmenter,
        YOLOSegmenter,
    ):
        raise TypeError(
            "segmenter debe ser una instancia "
            "de YOLOSegmenter."
        )

    app.bot_data["video_processor"] = (
        YOLOVideoProcessor(
            segmenter=segmenter,
            clip_seconds=VIDEO_CLIP_SECONDS,
            max_side=VIDEO_MAX_SIDE,
            max_output_fps=(
                VIDEO_MAX_OUTPUT_FPS
            ),
        )
    )

    # El bloqueo normalmente ya se crea en bot.py.
    # Se incluye esta validación por seguridad.
    if "yolo_lock" not in app.bot_data:
        app.bot_data["yolo_lock"] = (
            asyncio.Lock()
        )

    app.add_handler(
        MessageHandler(
            filters.VIDEO
            | filters.Document.VIDEO,
            receive_video,
        )
    )

    logger.info(
        "Módulo de video registrado. "
        "Procesamiento configurado en memoria RAM."
    )
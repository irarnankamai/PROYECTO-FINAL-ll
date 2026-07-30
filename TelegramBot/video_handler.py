from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
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

from video_processor import (
    YOLOVideoProcessor,
)

from yolo_processor import YOLOSegmenter


logger = logging.getLogger(__name__)


# Margen respecto a los límites de Telegram.
MAX_DOWNLOAD_BYTES = (
    19 * 1024 * 1024
)

MAX_UPLOAD_BYTES = (
    49 * 1024 * 1024
)


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
# DETERMINAR EXTENSIÓN
# =========================================================

def _video_extension(
    file_name: str | None,
    mime_type: str | None,
) -> str:

    if file_name:

        extension = (
            Path(file_name)
            .suffix
            .lower()
        )

        if (
            extension
            in SUPPORTED_VIDEO_EXTENSIONS
        ):
            return extension

    if mime_type:

        mapped = MIME_EXTENSION_MAP.get(
            mime_type.lower()
        )

        if mapped:
            return mapped

    return ".mp4"


# =========================================================
# CREAR RUTAS ÚNICAS
# =========================================================

def _create_video_paths(
    received_dir: Path,
    output_dir: Path,
    extension: str,
) -> tuple[str, Path, Path]:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    event_id = (
        f"{timestamp}_"
        f"{uuid4().hex[:8]}"
    )

    input_path = (
        received_dir
        / f"video_entrada_{event_id}"
        f"{extension}"
    )

    output_path = (
        output_dir
        / f"video_segmentado_{event_id}.mp4"
    )

    return (
        event_id,
        input_path,
        output_path,
    )


# =========================================================
# MENSAJE DE RESULTADOS
# =========================================================

def _build_caption(
    metrics: dict,
    event_id: str,
) -> str:

    return (
        "🎬 VIDEO SEGMENTADO CON YOLO\n\n"
        f"Evento: {event_id}\n"
        f"Modelo: {metrics['model']}\n"
        f"Dispositivo: {metrics['device']}\n"
        f"Duración: "
        f"{metrics['clip_duration_seconds']:.2f} s\n"
        f"Frames procesados: "
        f"{metrics['frames_processed']}\n"
        f"FPS de procesamiento: "
        f"{metrics['processing_fps']:.2f}\n"
        f"FPS equivalente de inferencia: "
        f"{metrics['inference_fps']:.2f}\n"
        f"Confianza promedio: "
        f"{metrics['confidence_average'] * 100:.2f} %\n"
        f"Confianza máxima: "
        f"{metrics['confidence_maximum'] * 100:.2f} %\n"
        f"RAM máxima: "
        f"{metrics['ram_peak_mb']:.2f} MB\n"
        f"Detecciones acumuladas: "
        f"{metrics['detections_accumulated']}\n"
        f"Máscaras acumuladas: "
        f"{metrics['masks_accumulated']}\n\n"
        f"Clases: "
        f"{metrics['classes_summary']}"
    )


# =========================================================
# RECIBIR VIDEO
# =========================================================

async def receive_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    message = update.effective_message
    chat_id = update.effective_chat.id

    attachment = None
    file_name = None
    mime_type = None
    duration = 0.0

    # Video enviado normalmente.
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

    # Video enviado como archivo.
    elif message.document:

        attachment = message.document
        file_name = attachment.file_name
        mime_type = attachment.mime_type

    if attachment is None:

        await message.reply_text(
            "❌ No se recibió un video válido."
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

        await message.reply_text(
            "❌ El video supera el límite "
            "permitido para esta prueba.\n\n"
            f"Tamaño: "
            f"{file_size / (1024 * 1024):.2f} MB\n"
            "Máximo configurado: 19 MB."
        )

        return

    # =====================================================
    # VALIDAR DURACIÓN
    # =====================================================

    if (
        duration > 0
        and duration + 0.20
        < VIDEO_CLIP_SECONDS
    ):

        await message.reply_text(
            "❌ El video es demasiado corto.\n\n"
            f"Duración recibida: "
            f"{duration:.2f} s\n"
            f"Duración mínima: "
            f"{VIDEO_CLIP_SECONDS:.2f} s."
        )

        return

    received_dir: Path = (
        context.application.bot_data[
            "received_videos_dir"
        ]
    )

    output_dir: Path = (
        context.application.bot_data[
            "output_videos_dir"
        ]
    )

    processor: YOLOVideoProcessor = (
        context.application.bot_data[
            "video_processor"
        ]
    )

    extension = _video_extension(
        file_name,
        mime_type,
    )

    (
        event_id,
        input_path,
        output_path,
    ) = _create_video_paths(
        received_dir,
        output_dir,
        extension,
    )

    status_message = (
        await message.reply_text(
            "📥 Video recibido.\n\n"
            f"Evento: {event_id}\n"
            "Descargando el archivo..."
        )
    )

    try:

        # =================================================
        # DESCARGAR VIDEO
        # =================================================

        telegram_file = (
            await context.bot.get_file(
                attachment.file_id
            )
        )

        await telegram_file.download_to_drive(
            custom_path=input_path
        )

        if (
            not input_path.is_file()
            or input_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "El video no se descargó "
                "correctamente."
            )

        await status_message.edit_text(
            "🧠 Procesando video frame a frame "
            "con YOLO...\n\n"
            f"Evento: {event_id}\n"
            f"Se generará un clip de "
            f"{VIDEO_CLIP_SECONDS:.0f} segundos."
        )

        # =================================================
        # PROCESAR EN OTRO HILO
        # =================================================

        metrics = await asyncio.to_thread(
            processor.process_video,
            input_path,
            output_path,
        )

        # =================================================
        # VALIDAR ARCHIVO RESULTANTE
        # =================================================

        if (
            output_path.stat().st_size
            > MAX_UPLOAD_BYTES
        ):
            raise RuntimeError(
                "El video segmentado supera "
                "49 MB. Reduce VIDEO_MAX_SIDE "
                "o VIDEO_MAX_OUTPUT_FPS "
                "en config.py."
            )

        caption = _build_caption(
            metrics,
            event_id,
        )

        # =================================================
        # ENVIAR COMO VIDEO
        # =================================================

        try:

            with output_path.open(
                "rb"
            ) as video_file:

                await context.bot.send_video(
                    chat_id=chat_id,
                    video=video_file,
                    caption=caption,
                    duration=int(
                        round(
                            metrics[
                                "clip_duration_seconds"
                            ]
                        )
                    ),
                    width=int(
                        metrics["output_width"]
                    ),
                    height=int(
                        metrics["output_height"]
                    ),
                    supports_streaming=True,
                )

        # Si Telegram no reproduce el codec,
        # se envía el MP4 como documento.
        except TelegramError as send_error:

            logger.warning(
                "Telegram no pudo mostrar el "
                "MP4 como video. Se enviará "
                "como documento: %s",
                send_error,
            )

            with output_path.open(
                "rb"
            ) as video_file:

                await context.bot.send_document(
                    chat_id=chat_id,
                    document=video_file,
                    caption=(
                        caption
                        + "\n\n"
                        "⚠️ Telegram recibió "
                        "el resultado como archivo MP4."
                    ),
                )

        await status_message.edit_text(
            "✅ Video segmentado y enviado "
            "correctamente.\n\n"
            f"Evento: {event_id}"
        )

        logger.info(
            "Video %s procesado y enviado.",
            event_id,
        )

    except Exception as error:

        logger.exception(
            "Error procesando el video recibido."
        )

        await status_message.edit_text(
            "❌ No fue posible procesar "
            "el video.\n\n"
            f"Evento: {event_id}\n"
            f"Detalle: {error}"
        )


# =========================================================
# REGISTRAR MÓDULO EN EL BOT
# =========================================================

def register_video_module(
    app: Application,
    segmenter: YOLOSegmenter,
    base_dir: Path,
) -> None:

    received_videos_dir = (
        Path(base_dir)
        / "input"
        / "received"
        / "videos"
    )

    output_videos_dir = (
        Path(base_dir)
        / "output"
        / "videos"
    )

    received_videos_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_videos_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    app.bot_data[
        "received_videos_dir"
    ] = received_videos_dir

    app.bot_data[
        "output_videos_dir"
    ] = output_videos_dir

    app.bot_data[
        "video_processor"
    ] = YOLOVideoProcessor(
        segmenter=segmenter,
        clip_seconds=(
            VIDEO_CLIP_SECONDS
        ),
        max_side=VIDEO_MAX_SIDE,
        max_output_fps=(
            VIDEO_MAX_OUTPUT_FPS
        ),
    )

    app.add_handler(
        MessageHandler(
            filters.VIDEO
            | filters.Document.VIDEO,
            receive_video,
        )
    )
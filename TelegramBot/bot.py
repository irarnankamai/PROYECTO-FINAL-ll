from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import (
    TARGET_VEHICLE,
    TOKEN,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_FILTER_ONLY_TARGET,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL,
    YOLO_TARGET_CLASS,
)
from video_handler import register_video_module
from yolo_processor import YOLOSegmenter


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parent

TEST_IMAGE_PATH = (
    BASE_DIR
    / "input"
    / "images"
    / "matriz.png"
)


# =========================================================
# CONFIGURACIÓN DE ARCHIVOS
# =========================================================

MAX_DOWNLOAD_BYTES = (
    19
    * 1024
    * 1024
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

MIME_EXTENSION_MAP = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}


# =========================================================
# CONFIGURACIÓN DE LOGS
# =========================================================

logging.basicConfig(
    format=(
        "%(asctime)s - "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    ),
    level=logging.INFO,
)

logger = logging.getLogger(
    __name__
)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def create_event_id() -> str:
    """
    Genera un identificador único para cada evento.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    random_id = uuid4().hex[:8]

    return f"{timestamp}_{random_id}"


def get_document_extension(
    file_name: str | None,
    mime_type: str | None,
) -> str | None:
    """
    Determina la extensión de una imagen enviada
    como documento.
    """

    if file_name:
        extension = Path(
            file_name
        ).suffix.lower()

        if extension in SUPPORTED_IMAGE_EXTENSIONS:
            return extension

    if mime_type:
        return MIME_EXTENSION_MAP.get(
            mime_type.lower()
        )

    return None


def sanitize_filename(
    file_name: str | None,
    default_name: str,
) -> str:
    """
    Limpia el nombre de un archivo recibido.
    """

    if not file_name:
        return default_name

    clean_name = Path(
        file_name
    ).name.strip()

    return clean_name or default_name


def shorten_text(
    text: str,
    maximum_length: int = 300,
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
    Crea un archivo virtual en memoria RAM.
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


def build_metrics_caption(
    metrics: dict[str, Any],
    event_id: str,
) -> str:
    """
    Construye el mensaje con las métricas YOLO.
    """

    classes_summary = shorten_text(
        str(
            metrics.get(
                "classes_summary",
                "Sin información",
            )
        ),
        maximum_length=250,
    )

    return (
        "🧠 SEGMENTACIÓN DE INSTANCIAS YOLO\n\n"
        f"Evento: {event_id}\n"
        f"Modelo: "
        f"{metrics.get('model', YOLO_MODEL)}\n"
        f"Dispositivo: "
        f"{metrics.get('device', YOLO_DEVICE)}\n\n"
        f"Objetos detectados: "
        f"{metrics.get('detections', 0)}\n"
        f"Máscaras generadas: "
        f"{metrics.get('masks', 0)}\n"
        f"Confianza promedio: "
        f"{float(metrics.get('confidence_average', 0.0)) * 100:.2f} %\n"
        f"Confianza máxima: "
        f"{float(metrics.get('confidence_maximum', 0.0)) * 100:.2f} %\n\n"
        f"FPS de inferencia: "
        f"{float(metrics.get('inference_fps', 0.0)):.2f}\n"
        f"Tiempo de inferencia: "
        f"{float(metrics.get('inference_ms', 0.0)):.2f} ms\n"
        f"Tiempo total: "
        f"{float(metrics.get('total_seconds', 0.0)):.3f} s\n\n"
        f"RAM del proceso: "
        f"{float(metrics.get('ram_after_mb', 0.0)):.2f} MB\n"
        f"RAM total utilizada: "
        f"{float(metrics.get('system_ram_percent', 0.0)):.2f} %\n\n"
        f"Clases detectadas:\n"
        f"{classes_summary}"
    )


# =========================================================
# ENVÍO DE IMÁGENES DESDE RAM
# =========================================================

async def send_image_from_memory(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    image_data: bytes,
    filename: str,
    caption: str,
) -> None:
    """
    Envía una imagen almacenada en memoria RAM.
    """

    image_buffer = create_memory_buffer(
        image_data,
        filename,
    )

    try:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=image_buffer,
            caption=caption,
            connect_timeout=30,
            read_timeout=120,
            write_timeout=120,
        )

    finally:
        image_buffer.close()


# =========================================================
# PROCESAMIENTO YOLO EN RAM
# =========================================================

async def process_and_send_image(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    image_data: bytes,
    image_filename: str,
    event_id: str,
    status_message,
    source: str,
) -> dict[str, Any]:
    """
    Procesa una imagen completamente en memoria RAM.

    Flujo:
        1. Recibe bytes.
        2. Ejecuta YOLO.
        3. Obtiene la imagen segmentada como bytes.
        4. Envía ambas imágenes a Telegram.
    """

    segmenter: YOLOSegmenter = (
        context.application.bot_data[
            "segmenter"
        ]
    )

    yolo_lock: asyncio.Lock = (
        context.application.bot_data[
            "yolo_lock"
        ]
    )

    await status_message.edit_text(
        "🧠 Ejecutando YOLO11-seg...\n\n"
        f"Evento: {event_id}\n"
        "Procesamiento: memoria RAM\n\n"
        "La primera inferencia puede tardar más."
    )

    # Evita que varias imágenes utilicen el mismo
    # modelo YOLO simultáneamente.
    async with yolo_lock:
        (
            segmented_image_data,
            metrics,
        ) = await asyncio.to_thread(
            segmenter.process_image,
            image_data,
        )

    if not segmented_image_data:
        raise RuntimeError(
            "YOLO generó una imagen segmentada vacía."
        )

    # =====================================================
    # ENVIAR IMAGEN ORIGINAL
    # =====================================================

    await send_image_from_memory(
        context=context,
        chat_id=chat_id,
        image_data=image_data,
        filename=image_filename,
        caption=(
            "🚨 PRUEBA DE ALERTA DE TRÁFICO\n\n"
            "Vehículo objetivo configurado:\n"
            f"{TARGET_VEHICLE}\n\n"
            f"Fuente: {source}\n"
            f"Evento: {event_id}\n\n"
            "Estado: imagen recibida correctamente.\n"
            "Procesamiento: memoria RAM."
        ),
    )

    # =====================================================
    # ENVIAR IMAGEN SEGMENTADA
    # =====================================================

    await send_image_from_memory(
        context=context,
        chat_id=chat_id,
        image_data=segmented_image_data,
        filename="imagen_segmentada.jpg",
        caption=build_metrics_caption(
            metrics,
            event_id,
        ),
    )

    await status_message.edit_text(
        "✅ Procesamiento terminado correctamente.\n\n"
        f"Evento: {event_id}\n"
        "Almacenamiento permanente: desactivado."
    )

    return metrics


# =========================================================
# COMANDO /start
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Muestra la información principal del bot.
    """

    message = (
        "🚗 BOT DE MONITOREO DE TRÁFICO\n\n"
        "El sistema está funcionando correctamente.\n\n"
        "Puedes enviarme directamente una fotografía "
        "o una imagen como archivo y ejecutaré la "
        "segmentación de instancias con YOLO.\n\n"
        "Las imágenes se procesan en memoria RAM "
        "sin almacenamiento permanente.\n\n"
        "Comandos disponibles:\n\n"
        "/start - Mostrar información\n"
        "/id - Mostrar identificador del chat\n"
        "/foto - Enviar imagen local de prueba\n"
        "/segmentar - Procesar matriz.png\n\n"
        "📷 También puedes enviar cualquier fotografía "
        "sin utilizar comandos."
    )

    if update.effective_message:
        await update.effective_message.reply_text(
            message
        )


# =========================================================
# COMANDO /id
# =========================================================

async def show_chat_id(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Muestra el identificador del chat actual.
    """

    if (
        update.effective_chat is None
        or update.effective_message is None
    ):
        return

    chat_id = update.effective_chat.id

    await update.effective_message.reply_text(
        "🆔 Identificador del chat:\n\n"
        f"{chat_id}"
    )

    logger.info(
        "Chat ID detectado: %s",
        chat_id,
    )


# =========================================================
# COMANDO /foto
# =========================================================

async def send_test_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Envía matriz.png como fotografía de prueba.
    """

    if (
        update.effective_chat is None
        or update.effective_message is None
    ):
        return

    if not TEST_IMAGE_PATH.is_file():
        await update.effective_message.reply_text(
            "❌ No se encontró matriz.png.\n\n"
            f"Ruta buscada:\n{TEST_IMAGE_PATH}"
        )
        return

    try:
        image_data = await asyncio.to_thread(
            TEST_IMAGE_PATH.read_bytes
        )

        await send_image_from_memory(
            context=context,
            chat_id=update.effective_chat.id,
            image_data=image_data,
            filename=TEST_IMAGE_PATH.name,
            caption=(
                "📸 Imagen local de prueba.\n\n"
                f"Archivo: {TEST_IMAGE_PATH.name}"
            ),
        )

    except Exception:
        logger.exception(
            "Error al enviar la imagen local."
        )

        await update.effective_message.reply_text(
            "❌ No se pudo enviar la imagen local."
        )


# =========================================================
# COMANDO /segmentar
# =========================================================

async def segment_test_image(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Procesa matriz.png con YOLO en memoria RAM.
    """

    if (
        update.effective_chat is None
        or update.effective_message is None
    ):
        return

    if not TEST_IMAGE_PATH.is_file():
        await update.effective_message.reply_text(
            "❌ No se encontró matriz.png.\n\n"
            f"Ruta buscada:\n{TEST_IMAGE_PATH}"
        )
        return

    event_id = (
        "prueba_"
        + create_event_id()
    )

    status_message = (
        await update.effective_message.reply_text(
            "📥 Imagen local encontrada.\n"
            "Preparando segmentación en RAM..."
        )
    )

    try:
        image_data = await asyncio.to_thread(
            TEST_IMAGE_PATH.read_bytes
        )

        if not image_data:
            raise ValueError(
                "matriz.png está vacía."
            )

        await process_and_send_image(
            context=context,
            chat_id=update.effective_chat.id,
            image_data=image_data,
            image_filename=TEST_IMAGE_PATH.name,
            event_id=event_id,
            status_message=status_message,
            source="comando /segmentar",
        )

    except Exception as error:
        logger.exception(
            "Error procesando matriz.png."
        )

        await status_message.edit_text(
            "❌ Error durante la segmentación.\n\n"
            f"Detalle: "
            f"{shorten_text(str(error), 500)}"
        )


# =========================================================
# RECIBIR IMÁGENES DESDE TELEGRAM
# =========================================================

async def receive_image(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Descarga una imagen desde Telegram directamente
    a memoria RAM y la procesa con YOLO.
    """

    message = update.effective_message
    chat = update.effective_chat

    if message is None or chat is None:
        return

    attachment = None
    extension: str | None = None
    file_size: int | None = None
    source = ""
    original_filename: str | None = None

    # =====================================================
    # IMAGEN ENVIADA COMO FOTOGRAFÍA
    # =====================================================

    if message.photo:
        attachment = message.photo[-1]
        extension = ".jpg"
        file_size = attachment.file_size
        source = "fotografía enviada por Telegram"

    # =====================================================
    # IMAGEN ENVIADA COMO DOCUMENTO
    # =====================================================

    elif message.document:
        attachment = message.document
        file_size = attachment.file_size
        source = "imagen enviada como archivo"
        original_filename = attachment.file_name

        extension = get_document_extension(
            file_name=attachment.file_name,
            mime_type=attachment.mime_type,
        )

        if extension is None:
            await message.reply_text(
                "❌ Formato de imagen no compatible.\n\n"
                "Formatos permitidos:\n"
                "JPG, JPEG, PNG, BMP y WEBP."
            )
            return

    else:
        await message.reply_text(
            "❌ El mensaje recibido no contiene "
            "una imagen válida."
        )
        return

    # =====================================================
    # VALIDAR TAMAÑO
    # =====================================================

    if (
        file_size is not None
        and file_size > MAX_DOWNLOAD_BYTES
    ):
        size_mb = (
            file_size
            / (1024 * 1024)
        )

        await message.reply_text(
            "❌ La imagen es demasiado grande.\n\n"
            f"Tamaño recibido: {size_mb:.2f} MB\n"
            "Tamaño máximo aceptado: 19 MB."
        )
        return

    event_id = create_event_id()

    image_filename = sanitize_filename(
        original_filename,
        f"imagen_{event_id}{extension}",
    )

    status_message = await message.reply_text(
        "📥 Imagen recibida.\n\n"
        f"Evento: {event_id}\n"
        "Descargando directamente a memoria RAM..."
    )

    try:
        # Obtener referencia al archivo de Telegram.
        telegram_file = await context.bot.get_file(
            attachment.file_id
        )

        # Descargar directamente como bytearray.
        downloaded_data = (
            await telegram_file.download_as_bytearray()
        )

        image_data = bytes(
            downloaded_data
        )

        if not image_data:
            raise ValueError(
                "Telegram devolvió una imagen vacía."
            )

        if len(image_data) > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                "La imagen descargada supera "
                "el tamaño permitido."
            )

        logger.info(
            "Imagen recibida en RAM. "
            "Evento: %s. Tamaño: %.2f MB.",
            event_id,
            len(image_data) / (1024 * 1024),
        )

        await process_and_send_image(
            context=context,
            chat_id=chat.id,
            image_data=image_data,
            image_filename=image_filename,
            event_id=event_id,
            status_message=status_message,
            source=source,
        )

        logger.info(
            "Evento %s procesado correctamente.",
            event_id,
        )

    except Exception as error:
        logger.exception(
            "Error procesando la imagen recibida."
        )

        await status_message.edit_text(
            "❌ No fue posible procesar la imagen.\n\n"
            f"Evento: {event_id}\n"
            f"Detalle: "
            f"{shorten_text(str(error), 500)}"
        )


# =========================================================
# MENSAJES DE TEXTO
# =========================================================

async def reply_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Responde a mensajes de texto normales.
    """

    if update.effective_message is None:
        return

    await update.effective_message.reply_text(
        "📨 Mensaje recibido.\n\n"
        "Para ejecutar YOLO, envía una fotografía "
        "o una imagen como archivo."
    )


# =========================================================
# MANEJADOR GENERAL DE ERRORES
# =========================================================

async def handle_error(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Registra errores no controlados del bot.
    """

    logger.error(
        "Error no controlado procesando actualización.",
        exc_info=(
            context.error
            if isinstance(
                context.error,
                BaseException,
            )
            else None
        ),
    )


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def main() -> None:
    """
    Configura e inicia el bot de Telegram.
    """

    if not TOKEN:
        raise ValueError(
            "El TOKEN de Telegram no está configurado."
        )

    # Cargar YOLO una sola vez.
    segmenter = YOLOSegmenter(
        model_name=YOLO_MODEL,
        confidence=YOLO_CONFIDENCE,
        device=YOLO_DEVICE,
        image_size=YOLO_IMAGE_SIZE,
        target_class_name=YOLO_TARGET_CLASS,
        filter_only_target=(
            YOLO_FILTER_ONLY_TARGET
        ),
    )

    app: Application = (
        ApplicationBuilder()
        .token(TOKEN)
        .build()
    )

    app.bot_data["segmenter"] = segmenter

    # Evita inferencias simultáneas sobre
    # la misma instancia del modelo.
    app.bot_data["yolo_lock"] = (
        asyncio.Lock()
    )

    # Registrar el módulo encargado de videos.
    register_video_module(
        app=app,
        segmenter=segmenter,
        base_dir=BASE_DIR,
    )

    # =====================================================
    # COMANDOS
    # =====================================================

    app.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    app.add_handler(
        CommandHandler(
            "id",
            show_chat_id,
        )
    )

    app.add_handler(
        CommandHandler(
            "foto",
            send_test_photo,
        )
    )

    app.add_handler(
        CommandHandler(
            "segmentar",
            segment_test_image,
        )
    )

    # =====================================================
    # IMÁGENES
    # =====================================================

    app.add_handler(
        MessageHandler(
            filters.PHOTO
            | filters.Document.IMAGE,
            receive_image,
        )
    )

    # =====================================================
    # TEXTO NORMAL
    # =====================================================

    app.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            reply_text,
        )
    )

    app.add_error_handler(
        handle_error
    )

    print()
    print("=" * 65)
    print("BOT DE MONITOREO DE TRÁFICO INICIADO")
    print("=" * 65)
    print(f"Modelo YOLO       : {YOLO_MODEL}")
    print(f"Dispositivo       : {YOLO_DEVICE}")
    print(f"Confianza mínima  : {YOLO_CONFIDENCE}")
    print("Procesamiento      : memoria RAM")
    print("Archivos permanentes: desactivados")
    print()
    print("Envía una fotografía directamente al bot.")
    print("Presiona Ctrl + C para detenerlo.")
    print("=" * 65)
    print()

    app.run_polling(
        drop_pending_updates=False
    )


if __name__ == "__main__":
    main()
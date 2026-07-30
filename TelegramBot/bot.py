import asyncio
import logging
from datetime import datetime
from pathlib import Path
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
    YOLO_IMAGE_SIZE,
    YOLO_MODEL,
    YOLO_TARGET_CLASS,
)

from yolo_processor import YOLOSegmenter
from video_handler import register_video_module


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_IMAGE_PATH = (
    BASE_DIR
    / "input"
    / "images"
    / "matriz.png"
)

RECEIVED_IMAGES_DIR = (
    BASE_DIR
    / "input"
    / "received"
    / "images"
)

OUTPUT_IMAGES_DIR = (
    BASE_DIR
    / "output"
    / "images"
)


# =========================================================
# CONFIGURACIÓN DE ARCHIVOS
# =========================================================

# Telegram permite descargar archivos de hasta 20 MB
# mediante la API pública. Dejamos un pequeño margen.
MAX_DOWNLOAD_BYTES = 19 * 1024 * 1024

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
# LOGS
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

logger = logging.getLogger(__name__)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def create_event_paths(
    extension: str,
) -> tuple[str, Path, Path]:
    """
    Genera un identificador y rutas únicas para cada evento.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    random_id = uuid4().hex[:8]

    event_id = f"{timestamp}_{random_id}"

    input_path = (
        RECEIVED_IMAGES_DIR
        / f"entrada_{event_id}{extension}"
    )

    output_path = (
        OUTPUT_IMAGES_DIR
        / f"segmentada_{event_id}.jpg"
    )

    return event_id, input_path, output_path


def get_document_extension(
    file_name: str | None,
    mime_type: str | None,
) -> str | None:
    """
    Determina si una imagen enviada como documento
    tiene un formato compatible.
    """

    if file_name:
        extension = Path(file_name).suffix.lower()

        if extension in SUPPORTED_IMAGE_EXTENSIONS:
            return extension

    if mime_type:
        return MIME_EXTENSION_MAP.get(
            mime_type.lower()
        )

    return None


def build_metrics_caption(
    metrics: dict,
    event_id: str,
) -> str:
    """
    Crea el texto que acompañará a la imagen segmentada.
    """

    return (
        "🧠 SEGMENTACIÓN DE INSTANCIAS YOLO\n\n"
        f"Evento: {event_id}\n"
        f"Modelo: {metrics['model']}\n"
        f"Dispositivo: {metrics['device']}\n\n"
        f"Objetos detectados: "
        f"{metrics['detections']}\n"
        f"Máscaras generadas: "
        f"{metrics['masks']}\n"
        f"Confianza promedio: "
        f"{metrics['confidence_average'] * 100:.2f} %\n"
        f"Confianza máxima: "
        f"{metrics['confidence_maximum'] * 100:.2f} %\n\n"
        f"FPS de inferencia: "
        f"{metrics['inference_fps']:.2f}\n"
        f"Tiempo de inferencia: "
        f"{metrics['inference_ms']:.2f} ms\n"
        f"Tiempo total: "
        f"{metrics['total_seconds']:.3f} s\n\n"
        f"RAM del proceso: "
        f"{metrics['ram_after_mb']:.2f} MB\n"
        f"RAM total utilizada: "
        f"{metrics['system_ram_percent']:.2f} %\n\n"
        f"Clases detectadas:\n"
        f"{metrics['classes_summary']}"
    )


async def process_and_send_image(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    input_path: Path,
    output_path: Path,
    event_id: str,
    status_message,
    source: str,
) -> dict:
    """
    Ejecuta YOLO y envía la imagen original y la
    imagen segmentada al usuario.
    """

    segmenter: YOLOSegmenter = (
        context.application.bot_data["segmenter"]
    )

    await status_message.edit_text(
        "🧠 Ejecutando YOLO11-seg...\n\n"
        f"Evento: {event_id}\n"
        "La primera inferencia puede tardar más."
    )

    # La inferencia es bloqueante, por eso se ejecuta
    # en un hilo separado.
    metrics = await asyncio.to_thread(
        segmenter.process_image,
        input_path,
        output_path,
    )

    # =====================================================
    # ENVIAR IMAGEN ORIGINAL
    # =====================================================

    with input_path.open("rb") as original_image:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=original_image,
            caption=(
                "🚨 PRUEBA DE ALERTA DE TRÁFICO\n\n"
                f"Vehículo objetivo configurado:\n"
                f"{TARGET_VEHICLE}\n\n"
                f"Fuente: {source}\n"
                f"Evento: {event_id}\n\n"
                "Estado: imagen recibida correctamente.\n"
                "Esta prueba todavía no proviene del "
                "detector clásico en C++."
            ),
        )

    # =====================================================
    # ENVIAR IMAGEN SEGMENTADA
    # =====================================================

    caption = build_metrics_caption(
        metrics,
        event_id,
    )

    with output_path.open("rb") as segmented_image:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=segmented_image,
            caption=caption,
        )

    await status_message.edit_text(
        "✅ Procesamiento terminado correctamente.\n\n"
        f"Evento: {event_id}"
    )

    return metrics


# =========================================================
# COMANDO /start
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    message = (
        "🚗 BOT DE MONITOREO DE TRÁFICO\n\n"
        "El sistema está funcionando correctamente.\n\n"
        "Puedes enviarme directamente una fotografía "
        "o una imagen como archivo y ejecutaré la "
        "segmentación de instancias.\n\n"
        "Comandos disponibles:\n\n"
        "/start - Mostrar información\n"
        "/id - Mostrar identificador del chat\n"
        "/foto - Enviar imagen local de prueba\n"
        "/segmentar - Procesar matriz.png\n\n"
        "📷 También puedes enviar cualquier fotografía "
        "sin utilizar comandos."
    )

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

    if not TEST_IMAGE_PATH.is_file():
        await update.effective_message.reply_text(
            "❌ No se encontró matriz.png.\n\n"
            f"Ruta buscada:\n{TEST_IMAGE_PATH}"
        )
        return

    try:
        with TEST_IMAGE_PATH.open("rb") as photo:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=photo,
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

    if not TEST_IMAGE_PATH.is_file():
        await update.effective_message.reply_text(
            "❌ No se encontró matriz.png.\n\n"
            f"Ruta buscada:\n{TEST_IMAGE_PATH}"
        )
        return

    event_id = (
        "prueba_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_path = (
        OUTPUT_IMAGES_DIR
        / f"segmentada_{event_id}.jpg"
    )

    status_message = (
        await update.effective_message.reply_text(
            "📥 Imagen local encontrada.\n"
            "Preparando segmentación..."
        )
    )

    try:
        await process_and_send_image(
            context=context,
            chat_id=update.effective_chat.id,
            input_path=TEST_IMAGE_PATH,
            output_path=output_path,
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
            f"Detalle: {error}"
        )


# =========================================================
# RECIBIR IMÁGENES DESDE TELEGRAM
# =========================================================

async def receive_image(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    message = update.effective_message
    chat_id = update.effective_chat.id

    attachment = None
    extension = None
    file_size = None
    source = None

    # =====================================================
    # IMAGEN ENVIADA COMO FOTOGRAFÍA
    # =====================================================

    if message.photo:
        # Telegram entrega varias resoluciones.
        # La última normalmente es la de mayor tamaño.
        attachment = message.photo[-1]
        extension = ".jpg"
        file_size = attachment.file_size
        source = "fotografía enviada por Telegram"

    # =====================================================
    # IMAGEN ENVIADA COMO ARCHIVO
    # =====================================================

    elif message.document:
        attachment = message.document
        file_size = attachment.file_size
        source = "imagen enviada como archivo"

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
        size_mb = file_size / (1024 * 1024)

        await message.reply_text(
            "❌ La imagen es demasiado grande.\n\n"
            f"Tamaño recibido: {size_mb:.2f} MB\n"
            "Tamaño máximo aceptado: 19 MB."
        )
        return

    event_id, input_path, output_path = (
        create_event_paths(extension)
    )

    status_message = await message.reply_text(
        "📥 Imagen recibida.\n\n"
        f"Evento: {event_id}\n"
        "Descargando archivo..."
    )

    try:
        # Obtener el archivo desde Telegram.
        telegram_file = await context.bot.get_file(
            attachment.file_id
        )

        # Descargarlo en la carpeta del proyecto.
        await telegram_file.download_to_drive(
            custom_path=input_path
        )

        logger.info(
            "Imagen descargada: %s",
            input_path,
        )

        if not input_path.is_file():
            raise FileNotFoundError(
                "Telegram informó una descarga correcta, "
                "pero el archivo no existe en disco."
            )

        await process_and_send_image(
            context=context,
            chat_id=chat_id,
            input_path=input_path,
            output_path=output_path,
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
            f"Detalle: {error}"
        )


# =========================================================
# MENSAJES DE TEXTO
# =========================================================

async def reply_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

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

    logger.exception(
        "Error no controlado procesando actualización.",
        exc_info=context.error,
    )


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def main() -> None:

    if not TOKEN:
        raise ValueError(
            "El TOKEN de Telegram no está configurado."
        )

    RECEIVED_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Cargar YOLO una sola vez.
    segmenter = YOLOSegmenter(
        model_name=YOLO_MODEL,
        confidence=YOLO_CONFIDENCE,
        device=YOLO_DEVICE,
        image_size=YOLO_IMAGE_SIZE,
        target_class_name=YOLO_TARGET_CLASS,
        filter_only_target=YOLO_FILTER_ONLY_TARGET,
    )

    app: Application = (
        ApplicationBuilder()
        .token(TOKEN)
        .build()
    )

    app.bot_data["segmenter"] = segmenter

    register_video_module(
    app=app,
    segmenter=segmenter,
    base_dir=BASE_DIR,
)

    # Comandos
    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        CommandHandler("id", show_chat_id)
    )

    app.add_handler(
        CommandHandler("foto", send_test_photo)
    )

    app.add_handler(
        CommandHandler(
            "segmentar",
            segment_test_image,
        )
    )

    # Fotografías e imágenes enviadas como archivo
    app.add_handler(
        MessageHandler(
            filters.PHOTO
            | filters.Document.IMAGE,
            receive_image,
        )
    )

    # Texto normal
    app.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            reply_text,
        )
    )

    app.add_error_handler(handle_error)

    print()
    print("=" * 65)
    print("BOT DE MONITOREO DE TRÁFICO INICIADO")
    print("=" * 65)
    print(f"Modelo YOLO       : {YOLO_MODEL}")
    print(f"Dispositivo       : {YOLO_DEVICE}")
    print(f"Confianza mínima  : {YOLO_CONFIDENCE}")
    print(f"Imágenes recibidas: {RECEIVED_IMAGES_DIR}")
    print(f"Resultados        : {OUTPUT_IMAGES_DIR}")
    print()
    print("Envía una fotografía directamente al bot.")
    print("Presiona Ctrl + C para detenerlo.")
    print("=" * 65)
    print()

    app.run_polling()


if __name__ == "__main__":
    main()
from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from telegram import Bot
from telegram.error import TelegramError

from config import (
    API_HOST,
    API_KEY,
    API_MAX_IMAGE_MB,
    API_MAX_VIDEO_MB,
    API_PORT,
    CHAT_ID,
    TARGET_VEHICLE,
    TOKEN,
    VIDEO_CLIP_SECONDS,
    VIDEO_MAX_OUTPUT_FPS,
    VIDEO_MAX_SIDE,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    YOLO_FILTER_ONLY_TARGET,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL,
    YOLO_TARGET_CLASS,
)
from video_processor import YOLOVideoProcessor
from yolo_processor import YOLOSegmenter


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

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTES
# =========================================================

MAX_TELEGRAM_UPLOAD_BYTES = 49 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


# =========================================================
# INFORMACIÓN DEL EVENTO
# =========================================================

@dataclass(frozen=True)
class TrafficEvent:
    """
    Contiene toda la información del evento en memoria RAM.
    """

    event_id: str
    vehicle: str
    cpp_confidence: float
    camera: str
    event_datetime: str

    image_data: bytes
    video_data: bytes

    image_filename: str
    video_filename: str


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

    random_part = uuid4().hex[:8]

    return f"{timestamp}_{random_part}"


def obtain_extension(
    file_name: str | None,
    permitted_extensions: set[str],
    default_extension: str,
) -> str:
    """
    Obtiene una extensión válida del archivo recibido.
    """

    if file_name:
        extension = Path(
            file_name
        ).suffix.lower()

        if extension in permitted_extensions:
            return extension

    return default_extension


def sanitize_filename(
    file_name: str | None,
    default_name: str,
) -> str:
    """
    Elimina cualquier ruta incluida en el nombre recibido.
    """

    if not file_name:
        return default_name

    clean_name = Path(
        file_name
    ).name.strip()

    return clean_name or default_name


def shorten_text(
    text: str,
    maximum_length: int = 250,
) -> str:
    """
    Reduce un texto para evitar mensajes demasiado largos.
    """

    if len(text) <= maximum_length:
        return text

    return (
        text[: maximum_length - 3]
        + "..."
    )


def validate_api_key(
    received_key: str | None,
) -> None:
    """
    Verifica la API Key enviada por la aplicación C++.
    """

    if not API_KEY:
        raise RuntimeError(
            "API_KEY no está configurada."
        )

    valid = secrets.compare_digest(
        received_key or "",
        API_KEY,
    )

    if not valid:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="API Key inválida.",
        )


async def read_uploaded_file_in_memory(
    upload: UploadFile,
    maximum_bytes: int,
    file_description: str,
) -> bytes:
    """
    Lee un archivo recibido por FastAPI directamente
    en memoria RAM.

    El archivo se lee por bloques para comprobar que
    no supere el tamaño máximo configurado.
    """

    data = bytearray()
    total_bytes = 0

    try:
        while True:
            chunk = await upload.read(
                CHUNK_SIZE
            )

            if not chunk:
                break

            total_bytes += len(
                chunk
            )

            if total_bytes > maximum_bytes:
                raise HTTPException(
                    status_code=(
                        status
                        .HTTP_413_REQUEST_ENTITY_TOO_LARGE
                    ),
                    detail=(
                        f"El archivo de {file_description} "
                        "supera el tamaño permitido."
                    ),
                )

            data.extend(
                chunk
            )

    finally:
        await upload.close()

    if total_bytes == 0:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                f"El archivo de {file_description} "
                "está vacío."
            ),
        )

    return bytes(
        data
    )


def create_telegram_buffer(
    data: bytes,
    filename: str,
) -> BytesIO:
    """
    Crea un archivo virtual en RAM compatible
    con Telegram.
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


# =========================================================
# CREACIÓN DE MENSAJES
# =========================================================

def build_image_caption(
    metrics: dict[str, Any],
    event: TrafficEvent,
) -> str:
    """
    Construye el mensaje de la imagen segmentada.
    """

    classes = shorten_text(
        str(
            metrics.get(
                "classes_summary",
                "Sin información",
            )
        )
    )

    return (
        "🧠 SEGMENTACIÓN DE IMAGEN YOLO\n\n"
        f"Evento: {event.event_id}\n"
        f"Vehículo objetivo: {event.vehicle}\n"
        f"Modelo: "
        f"{metrics.get('model', YOLO_MODEL)}\n"
        f"Dispositivo: "
        f"{metrics.get('device', YOLO_DEVICE)}\n\n"
        f"Objetos detectados: "
        f"{metrics.get('detections', 0)}\n"
        f"Máscaras: "
        f"{metrics.get('masks', 0)}\n"
        f"Confianza promedio: "
        f"{float(metrics.get('confidence_average', 0.0)) * 100:.2f} %\n"
        f"Confianza máxima: "
        f"{float(metrics.get('confidence_maximum', 0.0)) * 100:.2f} %\n"
        f"FPS de inferencia: "
        f"{float(metrics.get('inference_fps', 0.0)):.2f}\n"
        f"RAM del proceso: "
        f"{float(metrics.get('ram_after_mb', 0.0)):.2f} MB\n\n"
        f"Clases: {classes}"
    )


def build_video_caption(
    metrics: dict[str, Any],
    event: TrafficEvent,
) -> str:
    """
    Construye el mensaje del video segmentado.
    """

    classes = shorten_text(
        str(
            metrics.get(
                "classes_summary",
                "Sin información",
            )
        )
    )

    return (
        "🎬 VIDEO SEGMENTADO CON YOLO\n\n"
        f"Evento: {event.event_id}\n"
        f"Vehículo objetivo: {event.vehicle}\n"
        f"Duración: "
        f"{float(metrics.get('clip_duration_seconds', 0.0)):.2f} s\n"
        f"Frames procesados: "
        f"{metrics.get('frames_processed', 0)}\n"
        f"FPS de procesamiento: "
        f"{float(metrics.get('processing_fps', 0.0)):.2f}\n"
        f"FPS equivalente de inferencia: "
        f"{float(metrics.get('inference_fps', 0.0)):.2f}\n"
        f"Confianza promedio: "
        f"{float(metrics.get('confidence_average', 0.0)) * 100:.2f} %\n"
        f"Confianza máxima: "
        f"{float(metrics.get('confidence_maximum', 0.0)) * 100:.2f} %\n"
        f"RAM máxima: "
        f"{float(metrics.get('ram_peak_mb', 0.0)):.2f} MB\n\n"
        f"Clases: {classes}"
    )


# =========================================================
# ENVÍO DE IMAGEN A TELEGRAM
# =========================================================

async def send_photo_from_memory(
    telegram_bot: Bot,
    image_data: bytes,
    filename: str,
    caption: str,
) -> None:
    """
    Envía una imagen almacenada en RAM a Telegram.
    """

    image_buffer = create_telegram_buffer(
        image_data,
        filename,
    )

    try:
        await telegram_bot.send_photo(
            chat_id=CHAT_ID,
            photo=image_buffer,
            caption=caption,
            connect_timeout=30,
            read_timeout=120,
            write_timeout=120,
        )

    finally:
        image_buffer.close()


# =========================================================
# ENVÍO DE VIDEO A TELEGRAM
# =========================================================

async def send_video_from_memory(
    telegram_bot: Bot,
    video_data: bytes,
    video_metrics: dict[str, Any],
    caption: str,
) -> None:
    """
    Intenta enviar el resultado como video.

    Si Telegram rechaza el video, lo vuelve a enviar
    como documento MP4.
    """

    if not video_data:
        raise ValueError(
            "El video segmentado está vacío."
        )

    if (
        len(video_data)
        > MAX_TELEGRAM_UPLOAD_BYTES
    ):
        raise RuntimeError(
            "El video segmentado supera 49 MB. "
            "Reduce VIDEO_MAX_SIDE o "
            "VIDEO_MAX_OUTPUT_FPS."
        )

    duration = int(
        round(
            float(
                video_metrics.get(
                    "clip_duration_seconds",
                    VIDEO_CLIP_SECONDS,
                )
            )
        )
    )

    width = int(
        video_metrics.get(
            "output_width",
            0,
        )
    )

    height = int(
        video_metrics.get(
            "output_height",
            0,
        )
    )

    video_buffer = create_telegram_buffer(
        video_data,
        "video_segmentado.mp4",
    )

    try:
        await telegram_bot.send_video(
            chat_id=CHAT_ID,
            video=video_buffer,
            caption=caption,
            duration=max(
                1,
                duration,
            ),
            width=max(
                1,
                width,
            ),
            height=max(
                1,
                height,
            ),
            supports_streaming=True,
            connect_timeout=30,
            read_timeout=180,
            write_timeout=180,
        )

    except TelegramError as video_error:
        logger.warning(
            "No se pudo enviar el resultado como video: %s. "
            "Se intentará enviar como documento.",
            video_error,
        )

        if not video_buffer.closed:
            video_buffer.close()

        document_buffer = create_telegram_buffer(
            video_data,
            "video_segmentado.mp4",
        )

        try:
            await telegram_bot.send_document(
                chat_id=CHAT_ID,
                document=document_buffer,
                caption=(
                    caption
                    + "\n\n"
                    "⚠️ Resultado enviado como "
                    "archivo MP4."
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
# PROCESAMIENTO DEL EVENTO
# =========================================================

async def process_traffic_event(
    app: FastAPI,
    event: TrafficEvent,
) -> None:
    """
    Procesa la imagen y el video completamente en RAM.

    Flujo:
        1. Envía el mensaje de alerta.
        2. Envía la imagen original.
        3. Segmenta la imagen con YOLO.
        4. Segmenta el video con YOLO.
        5. Envía los resultados a Telegram.
    """

    telegram_bot: Bot = (
        app.state.telegram_bot
    )

    segmenter: YOLOSegmenter = (
        app.state.segmenter
    )

    video_processor: YOLOVideoProcessor = (
        app.state.video_processor
    )

    yolo_lock: asyncio.Lock = (
        app.state.yolo_lock
    )

    try:
        logger.info(
            "Procesando evento %s completamente en RAM.",
            event.event_id,
        )

        # =================================================
        # 1. MENSAJE DE ALERTA
        # =================================================

        alert_text = (
            "🚨 ALERTA DE TRÁFICO\n\n"
            f"Vehículo objetivo: {event.vehicle}\n"
            "Estado: detectado en la escena\n"
            f"Confianza detector C++: "
            f"{event.cpp_confidence * 100:.2f} %\n"
            f"Cámara: {event.camera}\n"
            f"Fecha y hora: {event.event_datetime}\n"
            f"Evento: {event.event_id}"
        )

        await telegram_bot.send_message(
            chat_id=CHAT_ID,
            text=alert_text,
            connect_timeout=30,
            read_timeout=120,
            write_timeout=120,
        )

        # =================================================
        # 2. IMAGEN ORIGINAL
        # =================================================

        await send_photo_from_memory(
            telegram_bot=telegram_bot,
            image_data=event.image_data,
            filename=event.image_filename,
            caption=(
                "📸 Imagen original del evento.\n\n"
                f"Vehículo objetivo: "
                f"{event.vehicle}\n"
                f"Evento: {event.event_id}"
            ),
        )

        # =================================================
        # 3. SEGMENTAR IMAGEN EN RAM
        # =================================================

        async with yolo_lock:
            (
                segmented_image_data,
                image_metrics,
            ) = await asyncio.to_thread(
                segmenter.process_image,
                event.image_data,
            )

        if not segmented_image_data:
            raise RuntimeError(
                "YOLO generó una imagen segmentada vacía."
            )

        await send_photo_from_memory(
            telegram_bot=telegram_bot,
            image_data=segmented_image_data,
            filename="imagen_segmentada.jpg",
            caption=build_image_caption(
                image_metrics,
                event,
            ),
        )

        # =================================================
        # 4. SEGMENTAR VIDEO EN RAM
        # =================================================

        async with yolo_lock:
            (
                segmented_video_data,
                video_metrics,
            ) = await asyncio.to_thread(
                video_processor.process_video,
                event.video_data,
            )

        if not segmented_video_data:
            raise RuntimeError(
                "YOLO generó un video segmentado vacío."
            )

        # =================================================
        # 5. ENVIAR VIDEO SEGMENTADO
        # =================================================

        await send_video_from_memory(
            telegram_bot=telegram_bot,
            video_data=segmented_video_data,
            video_metrics=video_metrics,
            caption=build_video_caption(
                video_metrics,
                event,
            ),
        )

        # =================================================
        # 6. MENSAJE FINAL
        # =================================================

        await telegram_bot.send_message(
            chat_id=CHAT_ID,
            text=(
                "✅ Evento procesado completamente.\n\n"
                f"Evento: {event.event_id}\n"
                "Procesamiento: memoria RAM"
            ),
            connect_timeout=30,
            read_timeout=120,
            write_timeout=120,
        )

        logger.info(
            "Evento %s procesado correctamente en RAM.",
            event.event_id,
        )

    except Exception as error:
        logger.exception(
            "Error procesando el evento %s.",
            event.event_id,
        )

        try:
            error_detail = shorten_text(
                str(error),
                maximum_length=500,
            )

            await telegram_bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    "❌ Error procesando una alerta.\n\n"
                    f"Evento: {event.event_id}\n"
                    f"Detalle: {error_detail}"
                ),
                connect_timeout=30,
                read_timeout=120,
                write_timeout=120,
            )

        except Exception:
            logger.exception(
                "Tampoco se pudo informar "
                "el error mediante Telegram."
            )


# =========================================================
# CICLO DE VIDA DE FASTAPI
# =========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Carga YOLO y Telegram una sola vez al iniciar la API.
    """

    if not TOKEN:
        raise RuntimeError(
            "TOKEN no está configurado."
        )

    if not CHAT_ID:
        raise RuntimeError(
            "CHAT_ID no está configurado."
        )

    if not API_KEY:
        raise RuntimeError(
            "API_KEY no está configurada."
        )

    print()
    print("=" * 65)
    print("CARGANDO SERVICIO YOLO + TELEGRAM")
    print("=" * 65)

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

    video_processor = YOLOVideoProcessor(
        segmenter=segmenter,
        clip_seconds=VIDEO_CLIP_SECONDS,
        max_side=VIDEO_MAX_SIDE,
        max_output_fps=(
            VIDEO_MAX_OUTPUT_FPS
        ),
    )

    telegram_bot = Bot(
        token=TOKEN
    )

    await telegram_bot.initialize()

    app.state.segmenter = segmenter
    app.state.video_processor = (
        video_processor
    )
    app.state.telegram_bot = telegram_bot

    # Evita que dos eventos utilicen el mismo
    # modelo YOLO al mismo tiempo.
    app.state.yolo_lock = asyncio.Lock()

    print("Servicio cargado correctamente.")
    print("Procesamiento de imagen: RAM")
    print("Procesamiento de video : RAM")
    print("Almacenamiento permanente: desactivado")
    print("=" * 65)
    print()

    try:
        yield

    finally:
        await telegram_bot.shutdown()

        logger.info(
            "Servicio YOLO + Telegram finalizado."
        )


# =========================================================
# CREACIÓN DE LA APLICACIÓN
# =========================================================

app = FastAPI(
    title="API de Monitoreo de Tráfico",
    description=(
        "Recibe evidencias desde C++, procesa imagen "
        "y video con YOLO en RAM y envía los resultados "
        "al bot de Telegram."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# =========================================================
# ENDPOINT DE ESTADO
# =========================================================

@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Comprueba que la API esté funcionando.
    """

    return {
        "ok": True,
        "service": "YOLO Telegram API",
        "version": "3.0.0",
        "model": YOLO_MODEL,
        "device": YOLO_DEVICE,
        "target_vehicle": TARGET_VEHICLE,
        "image_processing": "memory",
        "video_processing": "memory",
        "persistent_storage": False,
    }


# =========================================================
# ENDPOINT UTILIZADO POR C++
# =========================================================

@app.post(
    "/api/v1/alerta",
    status_code=status.HTTP_202_ACCEPTED,
)
async def receive_traffic_alert(
    request: Request,
    background_tasks: BackgroundTasks,

    vehiculo: Annotated[
        str,
        Form(
            min_length=2,
            max_length=100,
        ),
    ],

    confianza_cpp: Annotated[
        float,
        Form(
            ge=0.0,
            le=1.0,
        ),
    ],

    imagen: Annotated[
        UploadFile,
        File(),
    ],

    video: Annotated[
        UploadFile,
        File(),
    ],

    camara: Annotated[
        str,
        Form(
            max_length=100,
        ),
    ] = "Cámara principal",

    fecha_hora: Annotated[
        str | None,
        Form(),
    ] = None,

    x_api_key: Annotated[
        str | None,
        Header(
            alias="X-API-Key"
        ),
    ] = None,
) -> dict[str, Any]:
    """
    Recibe la imagen, el video y los datos enviados
    desde la aplicación C++.
    """

    validate_api_key(
        x_api_key
    )

    event_id = create_event_id()

    image_extension = obtain_extension(
        imagen.filename,
        SUPPORTED_IMAGE_EXTENSIONS,
        ".jpg",
    )

    video_extension = obtain_extension(
        video.filename,
        SUPPORTED_VIDEO_EXTENSIONS,
        ".mp4",
    )

    image_filename = sanitize_filename(
        imagen.filename,
        f"imagen{image_extension}",
    )

    video_filename = sanitize_filename(
        video.filename,
        f"video{video_extension}",
    )

    # =====================================================
    # LEER IMAGEN EN RAM
    # =====================================================

    image_data = await read_uploaded_file_in_memory(
        upload=imagen,
        maximum_bytes=(
            API_MAX_IMAGE_MB
            * 1024
            * 1024
        ),
        file_description="imagen",
    )

    # =====================================================
    # LEER VIDEO EN RAM
    # =====================================================

    video_data = await read_uploaded_file_in_memory(
        upload=video,
        maximum_bytes=(
            API_MAX_VIDEO_MB
            * 1024
            * 1024
        ),
        file_description="video",
    )

    clean_vehicle = (
        vehiculo.strip()
    )

    clean_camera = (
        camara.strip()
        or "Cámara principal"
    )

    clean_datetime = (
        fecha_hora.strip()
        if (
            fecha_hora
            and fecha_hora.strip()
        )
        else datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    event = TrafficEvent(
        event_id=event_id,
        vehicle=clean_vehicle,
        cpp_confidence=float(
            confianza_cpp
        ),
        camera=clean_camera,
        event_datetime=clean_datetime,
        image_data=image_data,
        video_data=video_data,
        image_filename=image_filename,
        video_filename=video_filename,
    )

    # La API devuelve primero la respuesta a C++.
    # Después procesa YOLO y Telegram.
    background_tasks.add_task(
        process_traffic_event,
        request.app,
        event,
    )

    image_mb = (
        len(image_data)
        / (1024 * 1024)
    )

    video_mb = (
        len(video_data)
        / (1024 * 1024)
    )

    logger.info(
        "Alerta %s recibida en RAM. "
        "Imagen: %.2f MB. Video: %.2f MB.",
        event_id,
        image_mb,
        video_mb,
    )

    return {
        "ok": True,
        "status": "accepted",
        "event_id": event_id,
        "message": (
            "Imagen y video recibidos correctamente. "
            "El procesamiento en RAM continuará "
            "en segundo plano."
        ),
        "image_mb": round(
            image_mb,
            2,
        ),
        "video_mb": round(
            video_mb,
            2,
        ),
        "image_filename": (
            image_filename
        ),
        "video_filename": (
            video_filename
        ),
        "processing_mode": "memory",
        "persistent_storage": False,
    }


# =========================================================
# EJECUCIÓN DIRECTA
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
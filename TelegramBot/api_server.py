from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated
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
    YOLO_FILTER_ONLY_TARGET,
    YOLO_TARGET_CLASS,
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
    YOLO_IMAGE_SIZE,
    YOLO_MODEL,
)
from video_processor import YOLOVideoProcessor
from yolo_processor import YOLOSegmenter


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
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

API_INPUT_IMAGES_DIR = (
    BASE_DIR / "input" / "api" / "images"
)

API_INPUT_VIDEOS_DIR = (
    BASE_DIR / "input" / "api" / "videos"
)

API_OUTPUT_IMAGES_DIR = (
    BASE_DIR / "output" / "api" / "images"
)

API_OUTPUT_VIDEOS_DIR = (
    BASE_DIR / "output" / "api" / "videos"
)


# La API pública de Telegram admite hasta 50 MB
# cuando el bot sube un video directamente.
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
# INFORMACIÓN DE UN EVENTO
# =========================================================

@dataclass(frozen=True)
class TrafficEvent:
    event_id: str
    vehicle: str
    cpp_confidence: float
    camera: str
    event_datetime: str

    input_image_path: Path
    input_video_path: Path

    segmented_image_path: Path
    segmented_video_path: Path


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def create_event_id() -> str:
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

    if file_name:
        extension = Path(file_name).suffix.lower()

        if extension in permitted_extensions:
            return extension

    return default_extension


def shorten_text(
    text: str,
    maximum_length: int = 250,
) -> str:

    if len(text) <= maximum_length:
        return text

    return text[: maximum_length - 3] + "..."


def validate_api_key(
    received_key: str | None,
) -> None:

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
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida.",
        )


async def save_uploaded_file(
    upload: UploadFile,
    destination: Path,
    maximum_bytes: int,
) -> int:
    """
    Guarda el archivo por bloques para no cargarlo
    completamente en memoria RAM.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_bytes = 0

    try:
        with destination.open("wb") as output_file:

            while True:
                chunk = await upload.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > maximum_bytes:
                    raise HTTPException(
                        status_code=(
                                   status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                                 ),
                        detail=(
                            f"El archivo {upload.filename} "
                            "supera el tamaño permitido."
                        ),
                    )

                output_file.write(chunk)

    except Exception:
        destination.unlink(
            missing_ok=True
        )
        raise

    finally:
        await upload.close()

    if total_bytes == 0:
        destination.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"El archivo {upload.filename} "
                "está vacío."
            ),
        )

    return total_bytes


def build_image_caption(
    metrics: dict,
    event: TrafficEvent,
) -> str:

    classes = shorten_text(
        metrics["classes_summary"]
    )

    return (
        "🧠 SEGMENTACIÓN DE IMAGEN YOLO\n\n"
        f"Evento: {event.event_id}\n"
        f"Vehículo objetivo: {event.vehicle}\n"
        f"Modelo: {metrics['model']}\n"
        f"Dispositivo: {metrics['device']}\n\n"
        f"Objetos detectados: "
        f"{metrics['detections']}\n"
        f"Máscaras: {metrics['masks']}\n"
        f"Confianza promedio: "
        f"{metrics['confidence_average'] * 100:.2f} %\n"
        f"Confianza máxima: "
        f"{metrics['confidence_maximum'] * 100:.2f} %\n"
        f"FPS de inferencia: "
        f"{metrics['inference_fps']:.2f}\n"
        f"RAM del proceso: "
        f"{metrics['ram_after_mb']:.2f} MB\n\n"
        f"Clases: {classes}"
    )


def build_video_caption(
    metrics: dict,
    event: TrafficEvent,
) -> str:

    classes = shorten_text(
        metrics["classes_summary"]
    )

    return (
        "🎬 VIDEO SEGMENTADO CON YOLO\n\n"
        f"Evento: {event.event_id}\n"
        f"Vehículo objetivo: {event.vehicle}\n"
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
        f"{metrics['ram_peak_mb']:.2f} MB\n\n"
        f"Clases: {classes}"
    )


# =========================================================
# PROCESAR EVENTO Y ENVIAR A TELEGRAM
# =========================================================

async def process_traffic_event(
    app: FastAPI,
    event: TrafficEvent,
) -> None:

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
            "Procesando evento %s",
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
        )

        # =================================================
        # 2. IMAGEN ORIGINAL
        # =================================================

        with event.input_image_path.open(
            "rb"
        ) as original_image:

            await telegram_bot.send_photo(
                chat_id=CHAT_ID,
                photo=original_image,
                caption=(
                    "📸 Imagen original del evento.\n\n"
                    f"Vehículo objetivo: "
                    f"{event.vehicle}\n"
                    f"Evento: {event.event_id}"
                ),
                write_timeout=120,
            )

        # =================================================
        # 3. SEGMENTAR IMAGEN
        # =================================================

        async with yolo_lock:
            image_metrics = await asyncio.to_thread(
                segmenter.process_image,
                event.input_image_path,
                event.segmented_image_path,
            )

        with event.segmented_image_path.open(
            "rb"
        ) as segmented_image:

            await telegram_bot.send_photo(
                chat_id=CHAT_ID,
                photo=segmented_image,
                caption=build_image_caption(
                    image_metrics,
                    event,
                ),
                write_timeout=120,
            )

        # =================================================
        # 4. SEGMENTAR VIDEO
        # =================================================

        async with yolo_lock:
            video_metrics = await asyncio.to_thread(
                video_processor.process_video,
                event.input_video_path,
                event.segmented_video_path,
            )

        output_size = (
            event.segmented_video_path.stat().st_size
        )

        if output_size > MAX_TELEGRAM_UPLOAD_BYTES:
            raise RuntimeError(
                "El video segmentado supera 49 MB. "
                "Reduce VIDEO_MAX_SIDE o "
                "VIDEO_MAX_OUTPUT_FPS."
            )

        video_caption = build_video_caption(
            video_metrics,
            event,
        )

        # =================================================
        # 5. ENVIAR VIDEO
        # =================================================

        try:
            with event.segmented_video_path.open(
                "rb"
            ) as segmented_video:

                await telegram_bot.send_video(
                    chat_id=CHAT_ID,
                    video=segmented_video,
                    caption=video_caption,
                    duration=int(
                        round(
                            video_metrics[
                                "clip_duration_seconds"
                            ]
                        )
                    ),
                    width=int(
                        video_metrics["output_width"]
                    ),
                    height=int(
                        video_metrics["output_height"]
                    ),
                    supports_streaming=True,
                    connect_timeout=30,
                    read_timeout=180,
                    write_timeout=180,
                )

        except TelegramError as video_error:
            logger.warning(
                "No se pudo enviar como video: %s. "
                "Se intentará como documento.",
                video_error,
            )

            with event.segmented_video_path.open(
                "rb"
            ) as segmented_video:

                await telegram_bot.send_document(
                    chat_id=CHAT_ID,
                    document=segmented_video,
                    caption=(
                        video_caption
                        + "\n\n"
                        "⚠️ Resultado enviado como "
                        "archivo MP4."
                    ),
                    connect_timeout=30,
                    read_timeout=180,
                    write_timeout=180,
                )

        await telegram_bot.send_message(
            chat_id=CHAT_ID,
            text=(
                "✅ Evento procesado completamente.\n\n"
                f"Evento: {event.event_id}"
            ),
        )

        logger.info(
            "Evento %s terminado correctamente.",
            event.event_id,
        )

    except Exception as error:
        logger.exception(
            "Error procesando evento %s",
            event.event_id,
        )

        try:
            await telegram_bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    "❌ Error procesando una alerta.\n\n"
                    f"Evento: {event.event_id}\n"
                    f"Detalle: {error}"
                ),
            )

        except Exception:
            logger.exception(
                "Tampoco se pudo informar "
                "el error por Telegram."
            )


# =========================================================
# CICLO DE VIDA DE LA API
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    if not TOKEN:
        raise RuntimeError(
            "TOKEN no está configurado."
        )

    if not CHAT_ID:
        raise RuntimeError(
            "CHAT_ID no está configurado."
        )

    API_INPUT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    API_INPUT_VIDEOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    API_OUTPUT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    API_OUTPUT_VIDEOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
        filter_only_target=YOLO_FILTER_ONLY_TARGET,
    )

    video_processor = YOLOVideoProcessor(
        segmenter=segmenter,
        clip_seconds=VIDEO_CLIP_SECONDS,
        max_side=VIDEO_MAX_SIDE,
        max_output_fps=VIDEO_MAX_OUTPUT_FPS,
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

    # Impide que dos eventos utilicen el mismo
    # modelo YOLO simultáneamente.
    app.state.yolo_lock = asyncio.Lock()

    print("Servicio cargado correctamente.")
    print("=" * 65)
    print()

    try:
        yield

    finally:
        await telegram_bot.shutdown()


# =========================================================
# CREAR APLICACIÓN FASTAPI
# =========================================================

app = FastAPI(
    title="API de Monitoreo de Tráfico",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# ENDPOINT DE ESTADO
# =========================================================

@app.get("/health")
async def health() -> dict:

    return {
        "ok": True,
        "service": "YOLO Telegram API",
        "model": YOLO_MODEL,
        "device": YOLO_DEVICE,
        "target_vehicle": TARGET_VEHICLE,
    }


# =========================================================
# ENDPOINT QUE USARÁ C++
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
        Form(min_length=2, max_length=100),
    ],

    confianza_cpp: Annotated[
        float,
        Form(ge=0.0, le=1.0),
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
        Form(max_length=100),
    ] = "Cámara principal",

    fecha_hora: Annotated[
        str | None,
        Form(),
    ] = None,

    x_api_key: Annotated[
        str | None,
        Header(alias="X-API-Key"),
    ] = None,
) -> dict:

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

    input_image_path = (
        API_INPUT_IMAGES_DIR
        / f"imagen_{event_id}{image_extension}"
    )

    input_video_path = (
        API_INPUT_VIDEOS_DIR
        / f"video_{event_id}{video_extension}"
    )

    segmented_image_path = (
        API_OUTPUT_IMAGES_DIR
        / f"imagen_segmentada_{event_id}.jpg"
    )

    segmented_video_path = (
        API_OUTPUT_VIDEOS_DIR
        / f"video_segmentado_{event_id}.mp4"
    )

    try:
        image_bytes = await save_uploaded_file(
            upload=imagen,
            destination=input_image_path,
            maximum_bytes=(
                API_MAX_IMAGE_MB
                * 1024
                * 1024
            ),
        )

        video_bytes = await save_uploaded_file(
            upload=video,
            destination=input_video_path,
            maximum_bytes=(
                API_MAX_VIDEO_MB
                * 1024
                * 1024
            ),
        )

    except Exception:
        input_image_path.unlink(
            missing_ok=True
        )

        input_video_path.unlink(
            missing_ok=True
        )

        raise

    event = TrafficEvent(
        event_id=event_id,
        vehicle=vehiculo.strip(),
        cpp_confidence=float(
            confianza_cpp
        ),
        camera=camara.strip(),
        event_datetime=(
            fecha_hora
            or datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),
        input_image_path=input_image_path,
        input_video_path=input_video_path,
        segmented_image_path=(
            segmented_image_path
        ),
        segmented_video_path=(
            segmented_video_path
        ),
    )

    # La respuesta HTTP se devuelve primero.
    # Después se ejecuta YOLO y Telegram.
    background_tasks.add_task(
        process_traffic_event,
        request.app,
        event,
    )

    logger.info(
        "Alerta %s recibida. Imagen: %.2f MB, "
        "video: %.2f MB",
        event_id,
        image_bytes / (1024 * 1024),
        video_bytes / (1024 * 1024),
    )

    return {
        "ok": True,
        "status": "accepted",
        "event_id": event_id,
        "message": (
            "Imagen y video recibidos. "
            "El procesamiento continuará "
            "en segundo plano."
        ),
        "image_mb": round(
            image_bytes / (1024 * 1024),
            2,
        ),
        "video_mb": round(
            video_bytes / (1024 * 1024),
            2,
        ),
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
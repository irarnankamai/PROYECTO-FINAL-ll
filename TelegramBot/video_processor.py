from __future__ import annotations

import math
import os
import time
from collections import Counter
from fractions import Fraction
from io import BytesIO
from typing import Any

import av
import cv2
import numpy as np
import psutil
from numpy.typing import NDArray

from yolo_processor import YOLOSegmenter


class YOLOVideoProcessor:
    """
    Procesa videos con YOLO completamente en memoria RAM.

    Entrada:
        bytes del video recibido por FastAPI.

    Salida:
        bytes del video MP4 segmentado y sus métricas.

    No utiliza:
        - rutas Path;
        - archivos temporales;
        - cv2.VideoCapture;
        - cv2.VideoWriter;
        - almacenamiento permanente.
    """

    MINIMUM_CLIP_SECONDS = 5.0
    MINIMUM_MAX_SIDE = 320
    DURATION_TOLERANCE_SECONDS = 0.20
    DEFAULT_SOURCE_FPS = 25.0
    OUTPUT_VIDEO_FORMAT = "mp4"
    OUTPUT_CODEC = "libx264"

    def __init__(
        self,
        segmenter: YOLOSegmenter,
        clip_seconds: float = 5.0,
        max_side: int = 1280,
        max_output_fps: float = 30.0,
    ) -> None:
        """
        Inicializa el procesador de video.
        """

        self._validate_configuration(
            segmenter=segmenter,
            clip_seconds=clip_seconds,
            max_side=max_side,
            max_output_fps=max_output_fps,
        )

        self.segmenter = segmenter
        self.clip_seconds = float(
            clip_seconds
        )
        self.max_side = int(
            max_side
        )
        self.max_output_fps = float(
            max_output_fps
        )

    # =====================================================
    # VALIDACIONES
    # =====================================================

    @classmethod
    def _validate_configuration(
        cls,
        segmenter: YOLOSegmenter,
        clip_seconds: float,
        max_side: int,
        max_output_fps: float,
    ) -> None:
        """
        Valida los parámetros principales.
        """

        if not isinstance(
            segmenter,
            YOLOSegmenter,
        ):
            raise TypeError(
                "segmenter debe ser una instancia "
                "de YOLOSegmenter."
            )

        if not math.isfinite(
            clip_seconds
        ):
            raise ValueError(
                "clip_seconds debe ser "
                "un número válido."
            )

        if (
            clip_seconds
            < cls.MINIMUM_CLIP_SECONDS
        ):
            raise ValueError(
                "El clip debe durar al menos "
                f"{cls.MINIMUM_CLIP_SECONDS:.1f} "
                "segundos."
            )

        if max_side < cls.MINIMUM_MAX_SIDE:
            raise ValueError(
                "max_side debe ser de al menos "
                f"{cls.MINIMUM_MAX_SIDE} píxeles."
            )

        if (
            not math.isfinite(
                max_output_fps
            )
            or max_output_fps <= 0.0
        ):
            raise ValueError(
                "max_output_fps debe ser "
                "mayor que cero."
            )

    @staticmethod
    def _validate_video_data(
        video_data: bytes,
    ) -> None:
        """
        Verifica que los bytes del video sean válidos.
        """

        if not isinstance(
            video_data,
            bytes,
        ):
            raise TypeError(
                "El video debe recibirse como bytes."
            )

        if not video_data:
            raise ValueError(
                "El video recibido está vacío."
            )

    @staticmethod
    def _validate_frame(
        frame: NDArray[np.uint8] | None,
        description: str,
    ) -> None:
        """
        Verifica que un frame sea válido.
        """

        if frame is None:
            raise RuntimeError(
                f"{description} es None."
            )

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                f"{description} no es "
                "una matriz NumPy."
            )

        if frame.size == 0:
            raise RuntimeError(
                f"{description} está vacío."
            )

        if frame.ndim != 3:
            raise RuntimeError(
                f"{description} debe tener "
                "tres dimensiones."
            )

        if frame.shape[2] != 3:
            raise RuntimeError(
                f"{description} debe tener "
                "tres canales BGR."
            )

    # =====================================================
    # FUNCIONES AUXILIARES
    # =====================================================

    @staticmethod
    def _calculate_output_size(
        width: int,
        height: int,
        max_side: int,
    ) -> tuple[int, int]:
        """
        Calcula la resolución de salida conservando
        la relación de aspecto.

        Las dimensiones se ajustan a números pares
        para mejorar la compatibilidad con H.264.
        """

        if width <= 0 or height <= 0:
            raise ValueError(
                "El video tiene dimensiones inválidas."
            )

        largest_side = max(
            width,
            height,
        )

        scale = min(
            1.0,
            max_side / float(
                largest_side
            ),
        )

        output_width = max(
            2,
            int(
                round(
                    width * scale
                )
            ),
        )

        output_height = max(
            2,
            int(
                round(
                    height * scale
                )
            ),
        )

        output_width -= (
            output_width % 2
        )

        output_height -= (
            output_height % 2
        )

        return (
            max(
                2,
                output_width,
            ),
            max(
                2,
                output_height,
            ),
        )

    @staticmethod
    def _get_process_ram_mb(
        process: psutil.Process,
    ) -> float:
        """
        Devuelve la RAM del proceso en MB.
        """

        return (
            process.memory_info().rss
            / (1024 * 1024)
        )

    @staticmethod
    def _obtain_stream_fps(
        stream: av.video.stream.VideoStream,
    ) -> float:
        """
        Obtiene el FPS del flujo de video.
        """

        fps_candidates = [
            stream.average_rate,
            stream.base_rate,
            stream.guessed_rate,
        ]

        for candidate in fps_candidates:
            if candidate is None:
                continue

            try:
                fps = float(
                    candidate
                )
            except (
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                continue

            if (
                math.isfinite(fps)
                and fps > 0.0
            ):
                return fps

        return (
            YOLOVideoProcessor
            .DEFAULT_SOURCE_FPS
        )

    @staticmethod
    def _obtain_source_duration(
        input_container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
        source_fps: float,
    ) -> float:
        """
        Calcula la duración aproximada del video.
        """

        if (
            stream.duration is not None
            and stream.time_base is not None
        ):
            duration = float(
                stream.duration
                * stream.time_base
            )

            if (
                math.isfinite(duration)
                and duration > 0.0
            ):
                return duration

        if (
            input_container.duration
            is not None
        ):
            duration = (
                float(
                    input_container.duration
                )
                / av.time_base
            )

            if (
                math.isfinite(duration)
                and duration > 0.0
            ):
                return duration

        if (
            stream.frames is not None
            and stream.frames > 0
            and source_fps > 0.0
        ):
            return (
                float(stream.frames)
                / source_fps
            )

        return 0.0

    @staticmethod
    def _extract_result_data(
        metrics: dict[str, Any],
    ) -> tuple[
        list[float],
        int,
        int,
        Counter[str],
    ]:
        """
        Extrae información de las métricas producidas
        por YOLOSegmenter.process_frame().
        """

        confidence_average = float(
            metrics.get(
                "confidence_average",
                0.0,
            )
        )

        detection_count = int(
            metrics.get(
                "detections",
                0,
            )
        )

        mask_count = int(
            metrics.get(
                "masks",
                0,
            )
        )

        confidences: list[float] = []

        if detection_count > 0:
            confidences.extend(
                [
                    confidence_average
                ]
                * detection_count
            )

        class_counter: Counter[str] = (
            Counter()
        )

        classes_summary = str(
            metrics.get(
                "classes_summary",
                "",
            )
        )

        if (
            classes_summary
            and classes_summary
            not in {
                "Ningún objeto detectado",
                "Ninguna clase detectada",
            }
        ):
            for class_item in (
                classes_summary.split(",")
            ):
                class_item = (
                    class_item.strip()
                )

                if not class_item:
                    continue

                try:
                    class_name, quantity = (
                        class_item.rsplit(
                            ":",
                            maxsplit=1,
                        )
                    )

                    class_counter[
                        class_name.strip()
                    ] += int(
                        quantity.strip()
                    )

                except (
                    ValueError,
                    TypeError,
                ):
                    class_counter[
                        class_item
                    ] += 1

        return (
            confidences,
            detection_count,
            mask_count,
            class_counter,
        )

    # =====================================================
    # PANEL DE MÉTRICAS
    # =====================================================

    @staticmethod
    def _draw_overlay(
        frame: NDArray[np.uint8],
        processing_fps: float,
        ram_mb: float,
        detections: int,
        average_confidence: float,
        frame_number: int,
    ) -> None:
        """
        Dibuja las métricas principales sobre el frame.
        """

        lines = [
            (
                "YOLO-SEG | "
                f"FPS PROCESO: "
                f"{processing_fps:.2f}"
            ),
            (
                f"RAM: {ram_mb:.1f} MB | "
                f"OBJETOS: {detections}"
            ),
            (
                "CONF. PROM.: "
                f"{average_confidence * 100:.1f}% | "
                f"FRAME: {frame_number}"
            ),
        ]

        panel_width = min(
            frame.shape[1],
            580,
        )

        panel_height = min(
            frame.shape[0],
            92,
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (
                panel_width,
                panel_height,
            ),
            (0, 0, 0),
            thickness=-1,
        )

        for index, line in enumerate(
            lines
        ):
            y_position = (
                28 + index * 27
            )

            if y_position >= frame.shape[0]:
                break

            cv2.putText(
                frame,
                line,
                (
                    12,
                    y_position,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # =====================================================
    # PROCESAMIENTO EN RAM
    # =====================================================

    def process_video(
        self,
        video_data: bytes,
    ) -> tuple[
        bytes,
        dict[str, Any],
    ]:
        """
        Procesa un video completamente en RAM.

        Parámetros:
            video_data:
                Video original como bytes.

        Retorna:
            video_segmentado_bytes, métricas

        El método:
            1. Lee el MP4 desde BytesIO.
            2. Decodifica cada frame con PyAV.
            3. Procesa cada frame con YOLO.
            4. Codifica el resultado con H.264.
            5. Devuelve el MP4 segmentado como bytes.
        """

        self._validate_video_data(
            video_data
        )

        input_buffer = BytesIO(
            video_data
        )

        output_buffer = BytesIO()

        input_container: (
            av.container.InputContainer
            | None
        ) = None

        output_container: (
            av.container.OutputContainer
            | None
        ) = None

        try:
            input_container = av.open(
                input_buffer,
                mode="r",
            )

            video_streams = [
                stream
                for stream
                in input_container.streams
                if stream.type == "video"
            ]

            if not video_streams:
                raise ValueError(
                    "El archivo recibido no contiene "
                    "un flujo de video."
                )

            input_stream = (
                video_streams[0]
            )

            source_width = int(
                input_stream.codec_context.width
                or input_stream.width
                or 0
            )

            source_height = int(
                input_stream.codec_context.height
                or input_stream.height
                or 0
            )

            if (
                source_width <= 0
                or source_height <= 0
            ):
                raise ValueError(
                    "No se pudieron determinar "
                    "las dimensiones del video."
                )

            source_fps = (
                self._obtain_stream_fps(
                    input_stream
                )
            )

            source_duration = (
                self._obtain_source_duration(
                    input_container,
                    input_stream,
                    source_fps,
                )
            )

            if (
                source_duration > 0.0
                and (
                    source_duration
                    + self.DURATION_TOLERANCE_SECONDS
                    < self.clip_seconds
                )
            ):
                raise ValueError(
                    f"El video dura "
                    f"{source_duration:.2f} s. "
                    f"Se requieren al menos "
                    f"{self.clip_seconds:.2f} s."
                )

            (
                output_width,
                output_height,
            ) = self._calculate_output_size(
                width=source_width,
                height=source_height,
                max_side=self.max_side,
            )

            output_fps = min(
                source_fps,
                self.max_output_fps,
            )

            if (
                not math.isfinite(
                    output_fps
                )
                or output_fps <= 0.0
            ):
                raise ValueError(
                    "No se pudo determinar "
                    "un FPS de salida válido."
                )

            output_rate = Fraction(
                max(
                    1,
                    int(
                        round(
                            output_fps
                        )
                    ),
                ),
                1,
            )

            target_output_frames = max(
                1,
                int(
                    round(
                        self.clip_seconds
                        * output_fps
                    )
                ),
            )

            sample_step = (
                source_fps
                / output_fps
            )

            # =================================================
            # CREAR CONTENEDOR MP4 EN MEMORIA
            # =================================================

            output_container = av.open(
                output_buffer,
                mode="w",
                format=self.OUTPUT_VIDEO_FORMAT,
                options={
                    "movflags": (
                        "frag_keyframe+"
                        "empty_moov+"
                        "default_base_moof"
                    ),
                },
            )

            try:
                output_stream = (
                    output_container.add_stream(
                        self.OUTPUT_CODEC,
                        rate=output_rate,
                    )
                )

            except av.error.FFmpegError:
                # Alternativa para sistemas donde
                # libx264 no está disponible.
                output_stream = (
                    output_container.add_stream(
                        "mpeg4",
                        rate=output_rate,
                    )
                )

            output_stream.width = (
                output_width
            )

            output_stream.height = (
                output_height
            )

            output_stream.pix_fmt = (
                "yuv420p"
            )

            output_stream.options = {
                "preset": "veryfast",
                "crf": "23",
            }

            # =================================================
            # VARIABLES DE MÉTRICAS
            # =================================================

            process = psutil.Process(
                os.getpid()
            )

            ram_before_mb = (
                self._get_process_ram_mb(
                    process
                )
            )

            ram_peak_mb = ram_before_mb

            all_confidences: list[float] = []

            class_counter: Counter[str] = (
                Counter()
            )

            total_detections = 0
            total_masks = 0
            frames_with_detections = 0
            inference_ms_total = 0.0

            frames_read = 0
            frames_processed = 0
            next_sample_index = 0.0

            processing_start = (
                time.perf_counter()
            )

            # =================================================
            # DECODIFICAR Y PROCESAR FRAMES
            # =================================================

            for decoded_frame in (
                input_container.decode(
                    input_stream
                )
            ):
                if (
                    frames_processed
                    >= target_output_frames
                ):
                    break

                current_index = (
                    frames_read
                )

                frames_read += 1

                if (
                    current_index + 1e-9
                    < next_sample_index
                ):
                    continue

                next_sample_index += (
                    sample_step
                )

                frame = (
                    decoded_frame.to_ndarray(
                        format="bgr24"
                    )
                )

                self._validate_frame(
                    frame,
                    "El frame decodificado",
                )

                if (
                    frame.shape[1]
                    != output_width
                    or frame.shape[0]
                    != output_height
                ):
                    frame = cv2.resize(
                        frame,
                        (
                            output_width,
                            output_height,
                        ),
                        interpolation=(
                            cv2.INTER_AREA
                        ),
                    )

                # =============================================
                # YOLO COMPLETAMENTE EN RAM
                # =============================================

                (
                    annotated_frame,
                    frame_metrics,
                ) = self.segmenter.process_frame(
                    frame
                )

                self._validate_frame(
                    annotated_frame,
                    "El frame segmentado",
                )

                if (
                    annotated_frame.shape[1]
                    != output_width
                    or annotated_frame.shape[0]
                    != output_height
                ):
                    annotated_frame = cv2.resize(
                        annotated_frame,
                        (
                            output_width,
                            output_height,
                        ),
                        interpolation=(
                            cv2.INTER_AREA
                        ),
                    )

                (
                    confidences,
                    frame_detection_count,
                    frame_mask_count,
                    frame_class_counter,
                ) = self._extract_result_data(
                    frame_metrics
                )

                frame_average_confidence = (
                    float(
                        frame_metrics.get(
                            "confidence_average",
                            0.0,
                        )
                    )
                )

                if (
                    frame_detection_count
                    > 0
                ):
                    frames_with_detections += 1

                total_detections += (
                    frame_detection_count
                )

                total_masks += (
                    frame_mask_count
                )

                all_confidences.extend(
                    confidences
                )

                class_counter.update(
                    frame_class_counter
                )

                inference_ms_total += float(
                    frame_metrics.get(
                        "inference_ms",
                        0.0,
                    )
                )

                frames_processed += 1

                elapsed_seconds = (
                    time.perf_counter()
                    - processing_start
                )

                current_processing_fps = (
                    frames_processed
                    / elapsed_seconds
                    if elapsed_seconds > 0.0
                    else 0.0
                )

                ram_current_mb = (
                    self._get_process_ram_mb(
                        process
                    )
                )

                ram_peak_mb = max(
                    ram_peak_mb,
                    ram_current_mb,
                )

                self._draw_overlay(
                    frame=annotated_frame,
                    processing_fps=(
                        current_processing_fps
                    ),
                    ram_mb=ram_current_mb,
                    detections=(
                        frame_detection_count
                    ),
                    average_confidence=(
                        frame_average_confidence
                    ),
                    frame_number=(
                        frames_processed
                    ),
                )

                # =============================================
                # CODIFICAR FRAME EN MEMORIA
                # =============================================

                output_frame = (
                    av.VideoFrame.from_ndarray(
                        annotated_frame,
                        format="bgr24",
                    )
                )

                output_frame.pts = (
                    frames_processed - 1
                )

                output_frame.time_base = (
                    Fraction(
                        1,
                        output_rate.numerator,
                    )
                )

                for packet in (
                    output_stream.encode(
                        output_frame
                    )
                ):
                    output_container.mux(
                        packet
                    )

                if (
                    frames_processed
                    % 10 == 0
                ):
                    print(
                        "\rFrames segmentados: "
                        f"{frames_processed}/"
                        f"{target_output_frames} | "
                        "FPS proceso: "
                        f"{current_processing_fps:.2f}",
                        end="",
                        flush=True,
                    )

            print()

            if frames_processed == 0:
                raise RuntimeError(
                    "No se pudo procesar ningún "
                    "frame del video."
                )

            # Vaciar los frames pendientes del codec.
            for packet in (
                output_stream.encode()
            ):
                output_container.mux(
                    packet
                )

            output_container.close()
            output_container = None

            processing_seconds = (
                time.perf_counter()
                - processing_start
            )

            output_duration = (
                frames_processed
                / output_fps
            )

            if (
                output_duration
                + self.DURATION_TOLERANCE_SECONDS
                < self.clip_seconds
            ):
                raise ValueError(
                    "Solo se pudieron generar "
                    f"{output_duration:.2f} s "
                    "de video. Se requieren "
                    f"{self.clip_seconds:.2f} s."
                )

            output_video_data = (
                output_buffer.getvalue()
            )

            if not output_video_data:
                raise RuntimeError(
                    "El video segmentado generado "
                    "en RAM está vacío."
                )

            # =================================================
            # MÉTRICAS FINALES
            # =================================================

            average_inference_ms = (
                inference_ms_total
                / frames_processed
                if frames_processed > 0
                else 0.0
            )

            inference_fps = (
                1000.0
                / average_inference_ms
                if average_inference_ms > 0.0
                else 0.0
            )

            processing_fps = (
                frames_processed
                / processing_seconds
                if processing_seconds > 0.0
                else 0.0
            )

            average_confidence = (
                sum(all_confidences)
                / len(all_confidences)
                if all_confidences
                else 0.0
            )

            maximum_confidence = (
                max(all_confidences)
                if all_confidences
                else 0.0
            )

            classes_summary = ", ".join(
                (
                    f"{name}: {quantity}"
                )
                for name, quantity
                in class_counter.most_common()
            )

            if not classes_summary:
                classes_summary = (
                    "Ninguna clase detectada"
                )

            ram_after_mb = (
                self._get_process_ram_mb(
                    process
                )
            )

            source_total_frames = int(
                input_stream.frames or 0
            )

            metrics: dict[str, Any] = {
                "storage_mode": "memory",
                "input_path": None,
                "output_path": None,
                "model": (
                    self.segmenter.model_name
                ),
                "device": str(
                    self.segmenter.device
                ),
                "target_class": (
                    self.segmenter.target_class_name
                    if (
                        self.segmenter
                        .filter_only_target
                    )
                    else None
                ),
                "source_fps": source_fps,
                "output_fps": output_fps,
                "source_width": (
                    source_width
                ),
                "source_height": (
                    source_height
                ),
                "source_total_frames": (
                    source_total_frames
                ),
                "source_duration_seconds": (
                    source_duration
                ),
                "clip_duration_seconds": (
                    output_duration
                ),
                "target_output_frames": (
                    target_output_frames
                ),
                "frames_read": (
                    frames_read
                ),
                "frames_processed": (
                    frames_processed
                ),
                "frames_with_detections": (
                    frames_with_detections
                ),
                "detections_accumulated": (
                    total_detections
                ),
                "masks_accumulated": (
                    total_masks
                ),
                "confidence_average": (
                    average_confidence
                ),
                "confidence_maximum": (
                    maximum_confidence
                ),
                "processing_fps": (
                    processing_fps
                ),
                "inference_fps": (
                    inference_fps
                ),
                "average_inference_ms": (
                    average_inference_ms
                ),
                "processing_seconds": (
                    processing_seconds
                ),
                "ram_before_mb": (
                    ram_before_mb
                ),
                "ram_after_mb": (
                    ram_after_mb
                ),
                "ram_difference_mb": (
                    ram_after_mb
                    - ram_before_mb
                ),
                "ram_peak_mb": (
                    ram_peak_mb
                ),
                "system_ram_percent": float(
                    psutil.virtual_memory()
                    .percent
                ),
                "classes_summary": (
                    classes_summary
                ),
                "output_width": (
                    output_width
                ),
                "output_height": (
                    output_height
                ),
                "input_size_bytes": (
                    len(video_data)
                ),
                "output_size_bytes": (
                    len(output_video_data)
                ),
                "input_size_mb": (
                    len(video_data)
                    / (1024 * 1024)
                ),
                "output_size_mb": (
                    len(output_video_data)
                    / (1024 * 1024)
                ),
                "output_format": "mp4",
                "output_codec": (
                    output_stream.codec_context.name
                ),
            }

            self.print_metrics(
                metrics
            )

            return (
                output_video_data,
                metrics,
            )

        except av.error.FFmpegError as error:
            raise RuntimeError(
                "PyAV no pudo decodificar o codificar "
                f"el video: {error}"
            ) from error

        finally:
            if input_container is not None:
                input_container.close()

            if output_container is not None:
                output_container.close()

            input_buffer.close()
            output_buffer.close()

    # =====================================================
    # MOSTRAR MÉTRICAS
    # =====================================================

    @staticmethod
    def print_metrics(
        metrics: dict[str, Any],
    ) -> None:
        """
        Imprime las métricas generales del video.
        """

        print()
        print("=" * 68)
        print("RESULTADO DE SEGMENTACIÓN DE VIDEO")
        print("=" * 68)
        print(
            f"Modelo                    : "
            f"{metrics['model']}"
        )
        print(
            f"Dispositivo               : "
            f"{metrics['device']}"
        )
        print(
            f"Clase objetivo            : "
            f"{metrics['target_class'] or 'Todas'}"
        )
        print(
            f"Modo de almacenamiento    : "
            f"{metrics['storage_mode']}"
        )
        print(
            f"Resolución original       : "
            f"{metrics['source_width']} x "
            f"{metrics['source_height']}"
        )
        print(
            f"Resolución de salida      : "
            f"{metrics['output_width']} x "
            f"{metrics['output_height']}"
        )
        print(
            f"FPS del video original    : "
            f"{metrics['source_fps']:.2f}"
        )
        print(
            f"FPS del video de salida   : "
            f"{metrics['output_fps']:.2f}"
        )
        print(
            f"Frames leídos             : "
            f"{metrics['frames_read']}"
        )
        print(
            f"Frames procesados         : "
            f"{metrics['frames_processed']}"
        )
        print(
            f"Frames con detecciones    : "
            f"{metrics['frames_with_detections']}"
        )
        print(
            f"Duración generada         : "
            f"{metrics['clip_duration_seconds']:.2f} s"
        )
        print(
            f"FPS real de procesamiento : "
            f"{metrics['processing_fps']:.2f}"
        )
        print(
            f"FPS equivalente inferencia: "
            f"{metrics['inference_fps']:.2f}"
        )
        print(
            f"Inferencia promedio       : "
            f"{metrics['average_inference_ms']:.2f} ms"
        )
        print(
            f"Detecciones acumuladas    : "
            f"{metrics['detections_accumulated']}"
        )
        print(
            f"Máscaras acumuladas       : "
            f"{metrics['masks_accumulated']}"
        )
        print(
            f"Confianza promedio        : "
            f"{metrics['confidence_average'] * 100:.2f} %"
        )
        print(
            f"Confianza máxima          : "
            f"{metrics['confidence_maximum'] * 100:.2f} %"
        )
        print(
            f"RAM antes                 : "
            f"{metrics['ram_before_mb']:.2f} MB"
        )
        print(
            f"RAM después               : "
            f"{metrics['ram_after_mb']:.2f} MB"
        )
        print(
            f"Diferencia de RAM         : "
            f"{metrics['ram_difference_mb']:.2f} MB"
        )
        print(
            f"RAM máxima del proceso    : "
            f"{metrics['ram_peak_mb']:.2f} MB"
        )
        print(
            f"RAM total del sistema     : "
            f"{metrics['system_ram_percent']:.2f} %"
        )
        print(
            f"Tamaño del video original : "
            f"{metrics['input_size_mb']:.2f} MB"
        )
        print(
            f"Tamaño del video generado : "
            f"{metrics['output_size_mb']:.2f} MB"
        )
        print(
            f"Codec de salida           : "
            f"{metrics['output_codec']}"
        )
        print(
            f"Clases detectadas         : "
            f"{metrics['classes_summary']}"
        )
        print(
            "Resultado                 : "
            "conservado en memoria RAM"
        )
        print("=" * 68)
        print()
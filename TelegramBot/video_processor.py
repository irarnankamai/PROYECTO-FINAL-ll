from __future__ import annotations

import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import psutil

from yolo_processor import YOLOSegmenter


class YOLOVideoProcessor:
    """
    Procesa un clip de video cuadro por cuadro utilizando
    un modelo YOLO de segmentación previamente cargado.

    Genera un archivo MP4 anotado y devuelve métricas de:
    - detecciones;
    - máscaras;
    - confianza;
    - FPS;
    - duración;
    - uso de memoria RAM.
    """

    MINIMUM_CLIP_SECONDS = 5.0
    MINIMUM_MAX_SIDE = 320
    DURATION_TOLERANCE_SECONDS = 0.20

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
        self.clip_seconds = float(clip_seconds)
        self.max_side = int(max_side)
        self.max_output_fps = float(max_output_fps)

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

        if not isinstance(segmenter, YOLOSegmenter):
            raise TypeError(
                "segmenter debe ser una instancia "
                "de YOLOSegmenter."
            )

        if not math.isfinite(clip_seconds):
            raise ValueError(
                "clip_seconds debe ser un número válido."
            )

        if clip_seconds < cls.MINIMUM_CLIP_SECONDS:
            raise ValueError(
                "El clip debe durar al menos "
                f"{cls.MINIMUM_CLIP_SECONDS:.1f} segundos."
            )

        if max_side < cls.MINIMUM_MAX_SIDE:
            raise ValueError(
                "max_side debe ser de al menos "
                f"{cls.MINIMUM_MAX_SIDE} píxeles."
            )

        if (
            not math.isfinite(max_output_fps)
            or max_output_fps <= 0.0
        ):
            raise ValueError(
                "max_output_fps debe ser mayor que cero."
            )

    @staticmethod
    def _calculate_output_size(
        width: int,
        height: int,
        max_side: int,
    ) -> tuple[int, int]:
        """
        Calcula la resolución de salida manteniendo
        la relación de aspecto.

        Las dimensiones finales se ajustan a números pares
        para mejorar la compatibilidad con codecs de video.
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
            max_side / float(largest_side),
        )

        output_width = max(
            2,
            int(round(width * scale)),
        )

        output_height = max(
            2,
            int(round(height * scale)),
        )

        output_width -= output_width % 2
        output_height -= output_height % 2

        return (
            max(2, output_width),
            max(2, output_height),
        )

    @staticmethod
    def _get_process_ram_mb(
        process: psutil.Process,
    ) -> float:
        """
        Devuelve la memoria RAM utilizada por el proceso.
        """

        return (
            process.memory_info().rss
            / (1024 * 1024)
        )

    @staticmethod
    def _extract_detections(
        result: Any,
    ) -> tuple[list[float], list[int], int, int]:
        """
        Extrae las confianzas, clases, cantidad de cajas
        y cantidad de máscaras de un resultado YOLO.
        """

        confidences: list[float] = []
        class_ids: list[int] = []

        detection_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        if result.boxes is not None:
            if result.boxes.conf is not None:
                confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(float)
                    .tolist()
                )

            if result.boxes.cls is not None:
                class_ids = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )

        mask_count = 0

        if (
            result.masks is not None
            and result.masks.data is not None
        ):
            mask_count = int(
                result.masks.data.shape[0]
            )

        return (
            confidences,
            class_ids,
            detection_count,
            mask_count,
        )

    @staticmethod
    def _get_class_name(
        result: Any,
        class_id: int,
    ) -> str:
        """
        Convierte un identificador de clase
        en un nombre legible.
        """

        names = result.names

        if isinstance(names, dict):
            return str(
                names.get(
                    class_id,
                    f"clase_{class_id}",
                )
            )

        if (
            isinstance(names, (list, tuple))
            and 0 <= class_id < len(names)
        ):
            return str(
                names[class_id]
            )

        return f"clase_{class_id}"

    @staticmethod
    def _draw_overlay(
        frame: Any,
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
                f"FPS PROCESO: {processing_fps:.2f}"
            ),
            (
                f"RAM: {ram_mb:.1f} MB | "
                f"OBJETOS: {detections}"
            ),
            (
                f"CONF. PROM.: "
                f"{average_confidence * 100:.1f}% | "
                f"FRAME: {frame_number}"
            ),
        ]

        panel_width = min(
            frame.shape[1],
            580,
        )

        panel_height = 92

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

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (
                    12,
                    28 + index * 27,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _validate_annotated_frame(
        annotated_frame: Any,
    ) -> None:
        """
        Comprueba que YOLO haya generado un frame válido.
        """

        if annotated_frame is None:
            raise RuntimeError(
                "YOLO no generó un frame anotado."
            )

        if annotated_frame.size == 0:
            raise RuntimeError(
                "YOLO generó un frame anotado vacío."
            )

        if len(annotated_frame.shape) != 3:
            raise RuntimeError(
                "El frame anotado tiene un formato inválido."
            )

    @staticmethod
    def _create_video_writer(
        output_path: Path,
        output_fps: float,
        output_width: int,
        output_height: int,
    ) -> cv2.VideoWriter:
        """
        Crea el escritor MP4 utilizando el codec mp4v.
        """

        codec = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            codec,
            output_fps,
            (
                output_width,
                output_height,
            ),
        )

        if not writer.isOpened():
            writer.release()

            raise RuntimeError(
                "No se pudo crear el video MP4 "
                "con el codec mp4v."
            )

        return writer

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        """
        Procesa el video de entrada, genera un MP4
        segmentado y devuelve sus métricas.
        """

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.is_file():
            raise FileNotFoundError(
                "No se encontró el video de entrada: "
                f"{input_path}"
            )

        if input_path.stat().st_size == 0:
            raise ValueError(
                "El video de entrada está vacío: "
                f"{input_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.unlink(
            missing_ok=True
        )

        capture = cv2.VideoCapture(
            str(input_path)
        )

        if not capture.isOpened():
            capture.release()

            raise RuntimeError(
                "OpenCV no pudo abrir el video. "
                "El archivo puede estar dañado o "
                "utilizar un codec no compatible."
            )

        writer: cv2.VideoWriter | None = None
        processing_completed = False

        try:
            # =================================================
            # INFORMACIÓN DEL VIDEO ORIGINAL
            # =================================================

            source_fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            if (
                not math.isfinite(source_fps)
                or source_fps <= 0.0
            ):
                source_fps = 25.0

            source_width = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            source_height = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            source_total_frames = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            if (
                source_width <= 0
                or source_height <= 0
            ):
                raise ValueError(
                    "No se pudieron obtener las dimensiones "
                    "del video de entrada."
                )

            source_duration = (
                source_total_frames / source_fps
                if source_total_frames > 0
                else 0.0
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
                    f"El video dura {source_duration:.2f} s. "
                    f"Se requieren al menos "
                    f"{self.clip_seconds:.2f} s."
                )

            # =================================================
            # RESOLUCIÓN Y FPS DE SALIDA
            # =================================================

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
                not math.isfinite(output_fps)
                or output_fps <= 0.0
            ):
                raise ValueError(
                    "No se pudo calcular un FPS "
                    "de salida válido."
                )

            sample_step = (
                source_fps / output_fps
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

            max_input_frames = max(
                1,
                int(
                    math.ceil(
                        self.clip_seconds
                        * source_fps
                    )
                ),
            )

            if source_total_frames > 0:
                max_input_frames = min(
                    max_input_frames,
                    source_total_frames,
                )

            # =================================================
            # CREAR VIDEO DE SALIDA
            # =================================================

            writer = self._create_video_writer(
                output_path=output_path,
                output_fps=output_fps,
                output_width=output_width,
                output_height=output_height,
            )

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

            class_counter: Counter[str] = Counter()

            total_detections = 0
            total_masks = 0
            frames_with_detections = 0
            inference_ms_total = 0.0

            read_frames = 0
            processed_frames = 0
            next_sample_index = 0.0

            processing_start = (
                time.perf_counter()
            )

            # =================================================
            # PROCESAMIENTO CUADRO POR CUADRO
            # =================================================

            while (
                read_frames < max_input_frames
                and processed_frames
                < target_output_frames
            ):
                ok, frame = capture.read()

                if not ok or frame is None:
                    break

                current_index = read_frames
                read_frames += 1

                if (
                    current_index + 1e-9
                    < next_sample_index
                ):
                    continue

                next_sample_index += sample_step

                if frame.size == 0:
                    raise RuntimeError(
                        "OpenCV devolvió un frame vacío."
                    )

                if (
                    frame.shape[1] != output_width
                    or frame.shape[0] != output_height
                ):
                    frame = cv2.resize(
                        frame,
                        (
                            output_width,
                            output_height,
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

                # =============================================
                # INFERENCIA YOLO
                # =============================================

                results = self.segmenter.model.predict(
                    source=frame,
                    conf=self.segmenter.confidence,
                    device=self.segmenter.device,
                    imgsz=self.segmenter.image_size,
                    retina_masks=True,
                    classes=(
                        self.segmenter.target_class_ids
                    ),
                    verbose=False,
                )

                if not results:
                    raise RuntimeError(
                        "YOLO no devolvió resultados "
                        "para un frame del video."
                    )

                result = results[0]

                annotated_frame = result.plot()

                self._validate_annotated_frame(
                    annotated_frame
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
                        interpolation=cv2.INTER_AREA,
                    )

                (
                    confidences,
                    class_ids,
                    frame_detection_count,
                    mask_count,
                ) = self._extract_detections(
                    result
                )

                frame_average_confidence = (
                    sum(confidences)
                    / len(confidences)
                    if confidences
                    else 0.0
                )

                if frame_detection_count > 0:
                    frames_with_detections += 1

                total_detections += (
                    frame_detection_count
                )

                total_masks += mask_count

                all_confidences.extend(
                    confidences
                )

                for class_id in class_ids:
                    class_name = (
                        self._get_class_name(
                            result,
                            class_id,
                        )
                    )

                    class_counter[class_name] += 1

                inference_ms_total += float(
                    result.speed.get(
                        "inference",
                        0.0,
                    )
                )

                processed_frames += 1

                elapsed_seconds = (
                    time.perf_counter()
                    - processing_start
                )

                current_processing_fps = (
                    processed_frames
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

                # =============================================
                # MÉTRICAS VISIBLES EN EL VIDEO
                # =============================================

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
                    frame_number=processed_frames,
                )

                writer.write(
                    annotated_frame
                )

                if processed_frames % 10 == 0:
                    print(
                        "\rFrames segmentados: "
                        f"{processed_frames}/"
                        f"{target_output_frames} | "
                        "FPS proceso: "
                        f"{current_processing_fps:.2f}",
                        end="",
                        flush=True,
                    )

            processing_seconds = (
                time.perf_counter()
                - processing_start
            )

            print()

            if processed_frames == 0:
                raise RuntimeError(
                    "No se pudo procesar ningún "
                    "frame del video."
                )

            output_duration = (
                processed_frames / output_fps
            )

            if (
                output_duration
                + self.DURATION_TOLERANCE_SECONDS
                < self.clip_seconds
            ):
                raise ValueError(
                    "Solo se pudieron generar "
                    f"{output_duration:.2f} s de video. "
                    f"Se requieren "
                    f"{self.clip_seconds:.2f} s."
                )

            writer.release()
            writer = None

            if (
                not output_path.is_file()
                or output_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "El video segmentado no fue "
                    "creado correctamente."
                )

            # =================================================
            # MÉTRICAS GENERALES
            # =================================================

            average_inference_ms = (
                inference_ms_total
                / processed_frames
                if processed_frames > 0
                else 0.0
            )

            inference_fps = (
                1000.0 / average_inference_ms
                if average_inference_ms > 0.0
                else 0.0
            )

            processing_fps = (
                processed_frames
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
                f"{name}: {quantity}"
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

            metrics: dict[str, Any] = {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "model": self.segmenter.model_name,
                "device": str(
                    self.segmenter.device
                ),
                "target_class": (
                    self.segmenter.target_class_name
                    if self.segmenter.filter_only_target
                    else None
                ),
                "source_fps": source_fps,
                "output_fps": output_fps,
                "source_width": source_width,
                "source_height": source_height,
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
                "frames_read": read_frames,
                "frames_processed": (
                    processed_frames
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
                "inference_fps": inference_fps,
                "average_inference_ms": (
                    average_inference_ms
                ),
                "processing_seconds": (
                    processing_seconds
                ),
                "ram_before_mb": ram_before_mb,
                "ram_after_mb": ram_after_mb,
                "ram_difference_mb": (
                    ram_after_mb - ram_before_mb
                ),
                "ram_peak_mb": ram_peak_mb,
                "system_ram_percent": float(
                    psutil.virtual_memory().percent
                ),
                "classes_summary": (
                    classes_summary
                ),
                "output_width": output_width,
                "output_height": output_height,
                "output_size_mb": (
                    output_path.stat().st_size
                    / (1024 * 1024)
                ),
            }

            processing_completed = True

            self.print_metrics(
                metrics
            )

            return metrics

        except Exception:
            output_path.unlink(
                missing_ok=True
            )
            raise

        finally:
            capture.release()

            if writer is not None:
                writer.release()

            if not processing_completed:
                output_path.unlink(
                    missing_ok=True
                )

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
            f"Tamaño del MP4            : "
            f"{metrics['output_size_mb']:.2f} MB"
        )
        print(
            f"Clases detectadas         : "
            f"{metrics['classes_summary']}"
        )
        print(
            f"Resultado                 : "
            f"{metrics['output_path']}"
        )
        print("=" * 68)
        print()


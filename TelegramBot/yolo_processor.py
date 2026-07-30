from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any

import cv2
import numpy as np
import psutil
from numpy.typing import NDArray
from ultralytics import YOLO


class YOLOSegmenter:
    """
    Procesa imágenes con YOLO completamente en memoria RAM.

    Entrada:
        bytes de una imagen recibida por FastAPI.

    Salida:
        bytes de la imagen segmentada y métricas del proceso.

    No lee ni escribe archivos en el disco.
    """

    def __init__(
        self,
        model_name: str,
        confidence: float = 0.25,
        device: str = "cpu",
        image_size: int = 640,
        target_class_name: str | None = None,
        filter_only_target: bool = False,
        output_extension: str = ".jpg",
        jpeg_quality: int = 95,
    ) -> None:
        """
        Inicializa y carga el modelo YOLO una sola vez.
        """

        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self.image_size = image_size
        self.target_class_name = target_class_name
        self.filter_only_target = filter_only_target
        self.output_extension = output_extension.lower()
        self.jpeg_quality = jpeg_quality

        self.target_class_ids: list[int] | None = None

        self._validate_configuration()
        self._print_loading_information()

        self.model = YOLO(
            self.model_name
        )

        self._configure_target_class_filter()

        print("Modelo YOLO cargado correctamente.")
        print("Modo de procesamiento: 100 % en RAM")
        print("=" * 60)
        print()

    # =====================================================
    # CONFIGURACIÓN
    # =====================================================

    def _validate_configuration(self) -> None:
        """
        Valida la configuración del procesador.
        """

        if not self.model_name:
            raise ValueError(
                "El nombre o la ruta del modelo YOLO "
                "no puede estar vacío."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "La confianza YOLO debe estar "
                "entre 0.0 y 1.0."
            )

        if self.image_size <= 0:
            raise ValueError(
                "El tamaño de imagen YOLO debe ser "
                "mayor que cero."
            )

        if (
            self.filter_only_target
            and not self.target_class_name
        ):
            raise ValueError(
                "Debe configurar target_class_name "
                "cuando filter_only_target está activado."
            )

        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }

        if self.output_extension not in supported_extensions:
            raise ValueError(
                "Formato de salida no compatible. "
                "Use .jpg, .jpeg, .png o .webp."
            )

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                "jpeg_quality debe estar entre 1 y 100."
            )

    def _print_loading_information(self) -> None:
        """
        Muestra la configuración inicial.
        """

        print()
        print("=" * 60)
        print("CARGANDO MODELO YOLO")
        print("=" * 60)
        print(
            f"Modelo           : {self.model_name}"
        )
        print(
            f"Dispositivo      : {self.device}"
        )
        print(
            f"Confianza mínima : {self.confidence}"
        )
        print(
            f"Tamaño de imagen : {self.image_size}"
        )
        print(
            f"Formato de salida: {self.output_extension}"
        )
        print(
            "Almacenamiento   : memoria RAM"
        )
        print("=" * 60)

    def _configure_target_class_filter(self) -> None:
        """
        Busca el ID correspondiente a la clase objetivo.
        """

        if (
            not self.filter_only_target
            or not self.target_class_name
        ):
            self.target_class_ids = None

            print(
                "Filtro de clase  : desactivado"
            )

            return

        model_names = self.model.names

        if isinstance(model_names, dict):
            class_items = model_names.items()
        else:
            class_items = enumerate(model_names)

        normalized_target_name = (
            self.target_class_name
            .strip()
            .casefold()
        )

        target_id = next(
            (
                int(class_id)
                for class_id, class_name in class_items
                if (
                    str(class_name)
                    .strip()
                    .casefold()
                    == normalized_target_name
                )
            ),
            None,
        )

        if target_id is None:
            if isinstance(model_names, dict):
                available_names = model_names.values()
            else:
                available_names = model_names

            available_classes = ", ".join(
                str(class_name)
                for class_name in available_names
            )

            raise ValueError(
                "La clase objetivo no existe en el modelo: "
                f"{self.target_class_name}. "
                f"Clases disponibles: {available_classes}"
            )

        self.target_class_ids = [
            target_id
        ]

        print(
            "Filtro de clase  : "
            f"{self.target_class_name} "
            f"(ID {target_id})"
        )

    # =====================================================
    # PROCESAMIENTO PRINCIPAL EN RAM
    # =====================================================

    def process_image(
        self,
        image_data: bytes,
    ) -> tuple[bytes, dict[str, Any]]:
        """
        Procesa una imagen completamente en RAM.

        Parámetros:
            image_data:
                Contenido binario de la imagen original.

        Retorna:
            Una tupla formada por:

            1. Bytes de la imagen segmentada.
            2. Diccionario de métricas.

        No crea archivos temporales ni permanentes.
        """

        input_image = self.decode_image(
            image_data
        )

        process = psutil.Process(
            os.getpid()
        )

        ram_before_mb = self._get_process_ram_mb(
            process
        )

        total_start = time.perf_counter()

        results = self.model.predict(
            source=input_image,
            conf=self.confidence,
            device=self.device,
            imgsz=self.image_size,
            retina_masks=True,
            classes=self.target_class_ids,
            verbose=False,
        )

        total_elapsed_seconds = (
            time.perf_counter()
            - total_start
        )

        if not results:
            raise RuntimeError(
                "YOLO no devolvió ningún resultado."
            )

        result = results[0]

        segmented_image = result.plot()

        self._validate_decoded_image(
            segmented_image,
            description="imagen segmentada",
        )

        segmented_image_data = self.encode_image(
            segmented_image
        )

        metrics = self._build_metrics(
            result=result,
            input_image=input_image,
            segmented_image=segmented_image,
            input_bytes=len(image_data),
            output_bytes=len(segmented_image_data),
            total_elapsed_seconds=(
                total_elapsed_seconds
            ),
            process=process,
            ram_before_mb=ram_before_mb,
        )

        self.print_metrics(
            metrics
        )

        return (
            segmented_image_data,
            metrics,
        )

    def process_frame(
        self,
        frame: NDArray[np.uint8],
    ) -> tuple[
        NDArray[np.uint8],
        dict[str, Any],
    ]:
        """
        Procesa directamente un frame de OpenCV en RAM.

        Este método será útil para procesar los frames
        del video sin guardar imágenes en el disco.
        """

        self._validate_decoded_image(
            frame,
            description="frame de entrada",
        )

        process = psutil.Process(
            os.getpid()
        )

        ram_before_mb = self._get_process_ram_mb(
            process
        )

        total_start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            imgsz=self.image_size,
            retina_masks=True,
            classes=self.target_class_ids,
            verbose=False,
        )

        total_elapsed_seconds = (
            time.perf_counter()
            - total_start
        )

        if not results:
            raise RuntimeError(
                "YOLO no devolvió resultados "
                "para el frame."
            )

        result = results[0]

        segmented_frame = result.plot()

        self._validate_decoded_image(
            segmented_frame,
            description="frame segmentado",
        )

        metrics = self._build_metrics(
            result=result,
            input_image=frame,
            segmented_image=segmented_frame,
            input_bytes=int(frame.nbytes),
            output_bytes=int(
                segmented_frame.nbytes
            ),
            total_elapsed_seconds=(
                total_elapsed_seconds
            ),
            process=process,
            ram_before_mb=ram_before_mb,
        )

        return (
            segmented_frame,
            metrics,
        )

    # =====================================================
    # DECODIFICACIÓN Y CODIFICACIÓN
    # =====================================================

    @staticmethod
    def decode_image(
        image_data: bytes,
    ) -> NDArray[np.uint8]:
        """
        Convierte bytes en una matriz de OpenCV.

        La operación se realiza completamente en RAM.
        """

        if not isinstance(
            image_data,
            bytes,
        ):
            raise TypeError(
                "La imagen debe recibirse como bytes."
            )

        if not image_data:
            raise ValueError(
                "La imagen recibida está vacía."
            )

        encoded_array = np.frombuffer(
            image_data,
            dtype=np.uint8,
        )

        if encoded_array.size == 0:
            raise ValueError(
                "No se encontraron datos válidos "
                "en la imagen."
            )

        decoded_image = cv2.imdecode(
            encoded_array,
            cv2.IMREAD_COLOR,
        )

        YOLOSegmenter._validate_decoded_image(
            decoded_image,
            description="imagen recibida",
        )

        return decoded_image

    def encode_image(
        self,
        image: NDArray[np.uint8],
    ) -> bytes:
        """
        Convierte una matriz de OpenCV en bytes.

        El resultado queda disponible para enviarse
        directamente a Telegram.
        """

        self._validate_decoded_image(
            image,
            description="imagen para codificar",
        )

        encoding_parameters: list[int] = []

        if self.output_extension in {
            ".jpg",
            ".jpeg",
        }:
            encoding_parameters = [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ]

        elif self.output_extension == ".png":
            encoding_parameters = [
                cv2.IMWRITE_PNG_COMPRESSION,
                3,
            ]

        success, encoded_image = cv2.imencode(
            self.output_extension,
            image,
            encoding_parameters,
        )

        if not success:
            raise RuntimeError(
                "OpenCV no pudo codificar "
                "la imagen segmentada."
            )

        if encoded_image is None:
            raise RuntimeError(
                "OpenCV devolvió un resultado "
                "de codificación vacío."
            )

        image_bytes = encoded_image.tobytes()

        if not image_bytes:
            raise RuntimeError(
                "La imagen segmentada codificada "
                "está vacía."
            )

        return image_bytes

    @staticmethod
    def _validate_decoded_image(
        image: NDArray[np.uint8] | None,
        description: str,
    ) -> None:
        """
        Verifica que una matriz de OpenCV sea válida.
        """

        if image is None:
            raise ValueError(
                f"No se pudo decodificar la {description}."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                f"La {description} no es "
                "una matriz NumPy válida."
            )

        if image.size == 0:
            raise ValueError(
                f"La {description} está vacía."
            )

        if image.ndim not in {
            2,
            3,
        }:
            raise ValueError(
                f"La {description} tiene una "
                "cantidad de dimensiones inválida."
            )

    # =====================================================
    # MÉTRICAS
    # =====================================================

    def _build_metrics(
        self,
        result: Any,
        input_image: NDArray[np.uint8],
        segmented_image: NDArray[np.uint8],
        input_bytes: int,
        output_bytes: int,
        total_elapsed_seconds: float,
        process: psutil.Process,
        ram_before_mb: float,
    ) -> dict[str, Any]:
        """
        Construye las métricas del procesamiento.
        """

        inference_ms = float(
            result.speed.get(
                "inference",
                total_elapsed_seconds * 1000.0,
            )
        )

        preprocessing_ms = float(
            result.speed.get(
                "preprocess",
                0.0,
            )
        )

        postprocessing_ms = float(
            result.speed.get(
                "postprocess",
                0.0,
            )
        )

        inference_fps = (
            1000.0 / inference_ms
            if inference_ms > 0.0
            else 0.0
        )

        confidences = self._extract_confidences(
            result
        )

        class_ids = self._extract_class_ids(
            result
        )

        detected_classes = (
            self._get_detected_classes(
                result,
                class_ids,
            )
        )

        class_counter = Counter(
            detected_classes
        )

        classes_summary = ", ".join(
            (
                f"{class_name}: {quantity}"
            )
            for class_name, quantity
            in sorted(
                class_counter.items()
            )
        )

        if not classes_summary:
            classes_summary = (
                "Ningún objeto detectado"
            )

        detection_count = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        mask_count = (
            len(result.masks.data)
            if (
                result.masks is not None
                and result.masks.data is not None
            )
            else 0
        )

        confidence_average = (
            sum(confidences)
            / len(confidences)
            if confidences
            else 0.0
        )

        confidence_maximum = (
            max(confidences)
            if confidences
            else 0.0
        )

        ram_after_mb = self._get_process_ram_mb(
            process
        )

        system_ram_percent = float(
            psutil.virtual_memory().percent
        )

        input_height = int(
            input_image.shape[0]
        )

        input_width = int(
            input_image.shape[1]
        )

        output_height = int(
            segmented_image.shape[0]
        )

        output_width = int(
            segmented_image.shape[1]
        )

        return {
            "storage_mode": "memory",
            "input_path": None,
            "output_path": None,
            "model": self.model_name,
            "device": str(self.device),
            "target_class": (
                self.target_class_name
                if self.filter_only_target
                else None
            ),
            "detections": detection_count,
            "masks": mask_count,
            "confidence_average": (
                confidence_average
            ),
            "confidence_maximum": (
                confidence_maximum
            ),
            "inference_fps": inference_fps,
            "inference_ms": inference_ms,
            "preprocessing_ms": (
                preprocessing_ms
            ),
            "postprocessing_ms": (
                postprocessing_ms
            ),
            "total_seconds": (
                total_elapsed_seconds
            ),
            "ram_before_mb": ram_before_mb,
            "ram_after_mb": ram_after_mb,
            "ram_difference_mb": (
                ram_after_mb
                - ram_before_mb
            ),
            "system_ram_percent": (
                system_ram_percent
            ),
            "classes_summary": (
                classes_summary
            ),
            "input_width": input_width,
            "input_height": input_height,
            "output_width": output_width,
            "output_height": output_height,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "input_mb": (
                input_bytes
                / (1024 * 1024)
            ),
            "output_mb": (
                output_bytes
                / (1024 * 1024)
            ),
            "output_extension": (
                self.output_extension
            ),
        }

    @staticmethod
    def _extract_confidences(
        result: Any,
    ) -> list[float]:
        """
        Extrae las confianzas de las detecciones.
        """

        if (
            result.boxes is None
            or result.boxes.conf is None
        ):
            return []

        return (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

    @staticmethod
    def _extract_class_ids(
        result: Any,
    ) -> list[int]:
        """
        Extrae los identificadores de las clases.
        """

        if (
            result.boxes is None
            or result.boxes.cls is None
        ):
            return []

        return (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

    @staticmethod
    def _get_detected_classes(
        result: Any,
        class_ids: list[int],
    ) -> list[str]:
        """
        Convierte los identificadores de clase
        en nombres legibles.
        """

        detected_classes: list[str] = []

        for class_id in class_ids:
            if isinstance(
                result.names,
                dict,
            ):
                class_name = result.names.get(
                    class_id,
                    f"clase_{class_id}",
                )

            elif 0 <= class_id < len(
                result.names
            ):
                class_name = result.names[
                    class_id
                ]

            else:
                class_name = (
                    f"clase_{class_id}"
                )

            detected_classes.append(
                str(class_name)
            )

        return detected_classes

    @staticmethod
    def _get_process_ram_mb(
        process: psutil.Process,
    ) -> float:
        """
        Devuelve la RAM utilizada por el proceso.
        """

        return (
            process.memory_info().rss
            / (1024 * 1024)
        )

    # =====================================================
    # PRESENTACIÓN DE RESULTADOS
    # =====================================================

    @staticmethod
    def print_metrics(
        metrics: dict[str, Any],
    ) -> None:
        """
        Imprime las métricas del procesamiento.
        """

        print()
        print("=" * 62)
        print("RESULTADO DE SEGMENTACIÓN YOLO")
        print("=" * 62)
        print(
            f"Modelo                  : "
            f"{metrics['model']}"
        )
        print(
            f"Dispositivo             : "
            f"{metrics['device']}"
        )
        print(
            f"Clase objetivo          : "
            f"{metrics['target_class'] or 'Todas'}"
        )
        print(
            f"Modo de almacenamiento  : "
            f"{metrics['storage_mode']}"
        )
        print(
            f"Objetos detectados      : "
            f"{metrics['detections']}"
        )
        print(
            f"Máscaras generadas      : "
            f"{metrics['masks']}"
        )
        print(
            f"Confianza promedio      : "
            f"{metrics['confidence_average'] * 100:.2f} %"
        )
        print(
            f"Confianza máxima        : "
            f"{metrics['confidence_maximum'] * 100:.2f} %"
        )
        print(
            f"FPS de inferencia       : "
            f"{metrics['inference_fps']:.2f}"
        )
        print(
            f"Preprocesamiento        : "
            f"{metrics['preprocessing_ms']:.2f} ms"
        )
        print(
            f"Inferencia              : "
            f"{metrics['inference_ms']:.2f} ms"
        )
        print(
            f"Postprocesamiento       : "
            f"{metrics['postprocessing_ms']:.2f} ms"
        )
        print(
            f"Tiempo total            : "
            f"{metrics['total_seconds']:.3f} s"
        )
        print(
            f"Resolución de entrada   : "
            f"{metrics['input_width']} x "
            f"{metrics['input_height']}"
        )
        print(
            f"Resolución de salida    : "
            f"{metrics['output_width']} x "
            f"{metrics['output_height']}"
        )
        print(
            f"Tamaño de entrada       : "
            f"{metrics['input_mb']:.2f} MB"
        )
        print(
            f"Tamaño de salida        : "
            f"{metrics['output_mb']:.2f} MB"
        )
        print(
            f"RAM antes               : "
            f"{metrics['ram_before_mb']:.2f} MB"
        )
        print(
            f"RAM después             : "
            f"{metrics['ram_after_mb']:.2f} MB"
        )
        print(
            f"Diferencia de RAM       : "
            f"{metrics['ram_difference_mb']:.2f} MB"
        )
        print(
            f"RAM total del sistema   : "
            f"{metrics['system_ram_percent']:.2f} %"
        )
        print(
            f"Clases detectadas       : "
            f"{metrics['classes_summary']}"
        )
        print(
            "Resultado                : "
            "conservado en memoria RAM"
        )
        print("=" * 62)
        print()
        
        
        
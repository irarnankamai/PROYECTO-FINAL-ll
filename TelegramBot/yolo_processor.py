import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import psutil
from ultralytics import YOLO


class YOLOSegmenter:
    """
    Procesa imágenes con un modelo YOLO de segmentación
    de instancias, guarda la imagen anotada y devuelve
    métricas de rendimiento.
    """

    def __init__(
        self,
        model_name: str,
        confidence: float = 0.25,
        device: str = "cpu",
        image_size: int = 640,
        target_class_name: str | None = None,
        filter_only_target: bool = False,
    ) -> None:
        """
        Inicializa el modelo YOLO y configura el filtro
        opcional de la clase objetivo.
        """

        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self.image_size = image_size
        self.target_class_name = target_class_name
        self.filter_only_target = filter_only_target

        # None permite que YOLO procese todas las clases.
        self.target_class_ids: list[int] | None = None

        self._validate_configuration()
        self._print_loading_information()

        # El modelo se carga una sola vez.
        self.model = YOLO(self.model_name)

        self._configure_target_class_filter()

        print("Modelo YOLO cargado correctamente.")
        print("=" * 60)
        print()

    def _validate_configuration(self) -> None:
        """
        Valida los valores principales recibidos.
        """

        if not self.model_name:
            raise ValueError(
                "El nombre o ruta del modelo YOLO está vacío."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "La confianza YOLO debe estar entre 0.0 y 1.0."
            )

        if self.image_size <= 0:
            raise ValueError(
                "El tamaño de imagen YOLO debe ser mayor que cero."
            )

        if self.filter_only_target and not self.target_class_name:
            raise ValueError(
                "Debe configurar target_class_name cuando "
                "filter_only_target está activado."
            )

    def _print_loading_information(self) -> None:
        """
        Muestra la configuración inicial del modelo.
        """

        print()
        print("=" * 60)
        print("CARGANDO MODELO YOLO")
        print("=" * 60)
        print(f"Modelo           : {self.model_name}")
        print(f"Dispositivo      : {self.device}")
        print(f"Confianza mínima : {self.confidence}")
        print(f"Tamaño de imagen : {self.image_size}")
        print("=" * 60)

    def _configure_target_class_filter(self) -> None:
        """
        Busca el identificador de la clase objetivo
        dentro de las clases disponibles en el modelo.
        """

        if (
            not self.filter_only_target
            or not self.target_class_name
        ):
            self.target_class_ids = None
            print("Filtro de clase  : desactivado")
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
                if str(class_name).strip().casefold()
                == normalized_target_name
            ),
            None,
        )

        if target_id is None:
            available_classes = ", ".join(
                str(class_name)
                for class_name in self.model.names.values()
            )

            raise ValueError(
                "La clase configurada no existe en el modelo: "
                f"{self.target_class_name}. "
                f"Clases disponibles: {available_classes}"
            )

        self.target_class_ids = [target_id]

        print(
            f"Filtro de clase  : "
            f"{self.target_class_name} "
            f"(ID {target_id})"
        )

    def process_image(
        self,
        input_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        """
        Procesa una imagen con YOLO, guarda la imagen
        segmentada y devuelve las métricas obtenidas.
        """

        input_path = Path(input_path)
        output_path = Path(output_path)

        self._validate_input_image(
            input_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        process = psutil.Process(
            os.getpid()
        )

        ram_before_mb = self._get_process_ram_mb(
            process
        )

        total_start = time.perf_counter()

        results = self.model.predict(
            source=str(input_path),
            conf=self.confidence,
            device=self.device,
            imgsz=self.image_size,
            retina_masks=True,
            classes=self.target_class_ids,
            verbose=False,
        )

        total_elapsed_seconds = (
            time.perf_counter() - total_start
        )

        if not results:
            raise RuntimeError(
                "YOLO no devolvió ningún resultado."
            )

        result = results[0]

        segmented_image = result.plot()

        if (
            segmented_image is None
            or segmented_image.size == 0
        ):
            raise RuntimeError(
                "YOLO no generó una imagen anotada válida."
            )

        saved = cv2.imwrite(
            str(output_path),
            segmented_image,
        )

        if not saved:
            raise RuntimeError(
                "No se pudo guardar la imagen procesada en: "
                f"{output_path}"
            )

        metrics = self._build_metrics(
            result=result,
            input_path=input_path,
            output_path=output_path,
            total_elapsed_seconds=total_elapsed_seconds,
            process=process,
            ram_before_mb=ram_before_mb,
        )

        self.print_metrics(
            metrics
        )

        return metrics

    @staticmethod
    def _validate_input_image(
        input_path: Path,
    ) -> None:
        """
        Comprueba que la imagen exista y pueda abrirse.
        """

        if not input_path.is_file():
            raise FileNotFoundError(
                "No se encontró la imagen de entrada: "
                f"{input_path}"
            )

        input_image = cv2.imread(
            str(input_path)
        )

        if input_image is None:
            raise ValueError(
                "La imagen está dañada o no tiene un formato válido: "
                f"{input_path}"
            )

        if input_image.size == 0:
            raise ValueError(
                "La imagen de entrada está vacía: "
                f"{input_path}"
            )

    def _build_metrics(
        self,
        result: Any,
        input_path: Path,
        output_path: Path,
        total_elapsed_seconds: float,
        process: psutil.Process,
        ram_before_mb: float,
    ) -> dict[str, Any]:
        """
        Construye el diccionario con las métricas
        de detección, inferencia y uso de memoria.
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

        detected_classes = self._get_detected_classes(
            result,
            class_ids,
        )

        class_counter = Counter(
            detected_classes
        )

        classes_summary = ", ".join(
            f"{class_name}: {quantity}"
            for class_name, quantity
            in sorted(class_counter.items())
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
            sum(confidences) / len(confidences)
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

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "model": self.model_name,
            "device": str(self.device),
            "target_class": (
                self.target_class_name
                if self.filter_only_target
                else None
            ),
            "detections": detection_count,
            "masks": mask_count,
            "confidence_average": confidence_average,
            "confidence_maximum": confidence_maximum,
            "inference_fps": inference_fps,
            "inference_ms": inference_ms,
            "preprocessing_ms": preprocessing_ms,
            "postprocessing_ms": postprocessing_ms,
            "total_seconds": total_elapsed_seconds,
            "ram_before_mb": ram_before_mb,
            "ram_after_mb": ram_after_mb,
            "ram_difference_mb": (
                ram_after_mb - ram_before_mb
            ),
            "system_ram_percent": system_ram_percent,
            "classes_summary": classes_summary,
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
        Extrae los identificadores de clase.
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
            if isinstance(result.names, dict):
                class_name = result.names.get(
                    class_id,
                    f"clase_{class_id}",
                )
            else:
                if 0 <= class_id < len(result.names):
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
        Devuelve la memoria RAM utilizada por
        el proceso actual en megabytes.
        """

        return (
            process.memory_info().rss
            / (1024 * 1024)
        )

    @staticmethod
    def print_metrics(
        metrics: dict[str, Any],
    ) -> None:
        """
        Imprime las métricas del procesamiento
        de la imagen.
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
            f"Resultado guardado en   : "
            f"{metrics['output_path']}"
        )
        print("=" * 62)
        print()


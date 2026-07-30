from pathlib import Path
import csv
import random
import shutil
import sys

import albumentations as A
import cv2


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

CARPETA_TRAIN = (
    RAIZ_PROYECTO
    / "dataset"
    / "positivos"
    / "recortes"
    / "train"
)

CARPETA_AUMENTADAS = (
    RAIZ_PROYECTO
    / "dataset"
    / "positivos"
    / "recortes"
    / "augmented"
)

ARCHIVO_CSV = (
    RAIZ_PROYECTO
    / "resultados"
    / "aumentacion_recortes_positivos.csv"
)

EXTENSIONES_VALIDAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

CANTIDAD_TRAIN_ESPERADA = 543
CANTIDAD_A_GENERAR = 3223
SEMILLA = 42


def obtener_imagenes(carpeta: Path) -> list[Path]:
    return sorted(
        archivo
        for archivo in carpeta.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in EXTENSIONES_VALIDAS
    )


def limpiar_carpeta(carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)

    for elemento in carpeta.iterdir():
        if elemento.is_file():
            elemento.unlink()
        elif elemento.is_dir():
            shutil.rmtree(elemento)


def crear_transformacion() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.40),

            A.Affine(
                scale=(0.92, 1.08),
                translate_percent=(-0.035, 0.035),
                rotate=(-6, 6),
                shear=(-2, 2),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.75,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.16,
                contrast_limit=0.16,
                p=0.60,
            ),

            A.HueSaturationValue(
                hue_shift_limit=4,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.25,
            ),

            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=(3, 5),
                        p=1.0,
                    ),
                    A.MotionBlur(
                        blur_limit=(3, 5),
                        p=1.0,
                    ),
                ],
                p=0.15,
            ),

            A.GaussNoise(
                std_range=(0.01, 0.025),
                p=0.12,
            ),
        ]
    )


def guardar_imagen(imagen, ruta: Path) -> None:
    guardada = cv2.imwrite(
        str(ruta),
        imagen,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not guardada:
        raise RuntimeError(
            f"No se pudo guardar la imagen: {ruta}"
        )


def main() -> None:
    random.seed(SEMILLA)

    if not CARPETA_TRAIN.exists():
        print("ERROR: no existe la carpeta:")
        print(CARPETA_TRAIN)
        sys.exit(1)

    imagenes_train = obtener_imagenes(CARPETA_TRAIN)

    if len(imagenes_train) != CANTIDAD_TRAIN_ESPERADA:
        print(
            f"ERROR: se esperaban {CANTIDAD_TRAIN_ESPERADA} "
            f"recortes en train, pero se encontraron "
            f"{len(imagenes_train)}."
        )
        sys.exit(1)

    limpiar_carpeta(CARPETA_AUMENTADAS)

    ARCHIVO_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transformacion = crear_transformacion()

    fuentes: list[Path] = []

    while len(fuentes) < CANTIDAD_A_GENERAR:
        bloque = imagenes_train.copy()
        random.shuffle(bloque)
        fuentes.extend(bloque)

    fuentes = fuentes[:CANTIDAD_A_GENERAR]

    registros: list[list[str | int]] = []

    print("=" * 68)
    print("AUMENTACIÓN DE RECORTES POSITIVOS")
    print("=" * 68)
    print(f"Recortes originales de train: {len(imagenes_train)}")
    print(f"Recortes a generar:           {CANTIDAD_A_GENERAR}")
    print()

    for indice, ruta_origen in enumerate(
        fuentes,
        start=1,
    ):
        imagen_bgr = cv2.imread(str(ruta_origen))

        if imagen_bgr is None or imagen_bgr.size == 0:
            print(
                f"ERROR: no se pudo leer {ruta_origen.name}"
            )
            sys.exit(1)

        imagen_rgb = cv2.cvtColor(
            imagen_bgr,
            cv2.COLOR_BGR2RGB,
        )

        resultado = transformacion(image=imagen_rgb)

        aumentada_rgb = resultado["image"]

        aumentada_bgr = cv2.cvtColor(
            aumentada_rgb,
            cv2.COLOR_RGB2BGR,
        )

        nombre_salida = (
            f"taxi_recorte_aug_{indice:05d}.jpg"
        )

        ruta_salida = (
            CARPETA_AUMENTADAS
            / nombre_salida
        )

        guardar_imagen(
            aumentada_bgr,
            ruta_salida,
        )

        registros.append(
            [
                indice,
                ruta_origen.name,
                nombre_salida,
                str(ruta_origen),
                str(ruta_salida),
            ]
        )

        if indice % 250 == 0:
            print(
                f"Generadas: {indice}/"
                f"{CANTIDAD_A_GENERAR}"
            )

    with ARCHIVO_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as archivo:
        escritor = csv.writer(archivo)

        escritor.writerow(
            [
                "indice",
                "recorte_origen",
                "recorte_aumentado",
                "ruta_origen",
                "ruta_aumentada",
            ]
        )

        escritor.writerows(registros)

    generadas = len(
        obtener_imagenes(CARPETA_AUMENTADAS)
    )

    print()
    print("=" * 68)
    print("RESULTADO")
    print("=" * 68)
    print(f"Recortes originales train: {len(imagenes_train)}")
    print(f"Recortes aumentados:       {generadas}")
    print(
        f"Total positivo train:      "
        f"{len(imagenes_train) + generadas}"
    )
    print(f"Registro CSV:              {ARCHIVO_CSV}")

    if generadas != CANTIDAD_A_GENERAR:
        print(
            "\nERROR: la cantidad generada no coincide "
            "con la esperada."
        )
        sys.exit(1)

    print("\nAumentación completada correctamente.")


if __name__ == "__main__":
    main()
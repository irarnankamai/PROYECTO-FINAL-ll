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
    / "negativos"
    / "train"
)

CARPETA_AUGMENTED = (
    RAIZ_PROYECTO
    / "dataset"
    / "negativos"
    / "augmented"
)

ARCHIVO_CSV = (
    RAIZ_PROYECTO
    / "resultados"
    / "aumentacion_negativos.csv"
)

EXTENSIONES_VALIDAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

CANTIDAD_ORIGINALES_TOTAL = 900
CANTIDAD_TRAIN_ESPERADA = 630
META_NEGATIVAS = 4000

CANTIDAD_A_GENERAR = (
    META_NEGATIVAS - CANTIDAD_ORIGINALES_TOTAL
)

SEMILLA = 42


def limpiar_carpeta(carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)

    for elemento in carpeta.iterdir():
        if elemento.is_file():
            elemento.unlink()
        elif elemento.is_dir():
            shutil.rmtree(elemento)


def obtener_imagenes(carpeta: Path) -> list[Path]:
    return sorted(
        archivo
        for archivo in carpeta.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in EXTENSIONES_VALIDAS
    )


def crear_transformacion() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.40),

            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-8, 8),
                shear=(-3, 3),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.80,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.65,
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
                p=0.20,
            ),

            A.OneOf(
                [
                    A.GaussNoise(
                        std_range=(0.01, 0.04),
                        p=1.0,
                    ),
                    A.ISONoise(
                        color_shift=(0.01, 0.04),
                        intensity=(0.05, 0.20),
                        p=1.0,
                    ),
                ],
                p=0.15,
            ),

            A.RandomShadow(
                shadow_roi=(0.0, 0.2, 1.0, 1.0),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.15,
            ),
        ]
    )


def guardar_imagen(
    imagen,
    ruta_salida: Path,
) -> None:
    parametros = [
        cv2.IMWRITE_JPEG_QUALITY,
        95,
    ]

    resultado = cv2.imwrite(
        str(ruta_salida),
        imagen,
        parametros,
    )

    if not resultado:
        raise RuntimeError(
            f"No se pudo guardar la imagen: {ruta_salida}"
        )


def main() -> None:
    random.seed(SEMILLA)

    if not CARPETA_TRAIN.exists():
        print("ERROR: no existe la carpeta de entrenamiento:")
        print(CARPETA_TRAIN)
        sys.exit(1)

    imagenes_train = obtener_imagenes(CARPETA_TRAIN)

    if len(imagenes_train) != CANTIDAD_TRAIN_ESPERADA:
        print(
            f"ERROR: se esperaban {CANTIDAD_TRAIN_ESPERADA} "
            f"imágenes negativas en train, pero se encontraron "
            f"{len(imagenes_train)}."
        )
        sys.exit(1)

    limpiar_carpeta(CARPETA_AUGMENTED)

    ARCHIVO_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transformacion = crear_transformacion()

    # Distribuye de manera equilibrada las imágenes fuente.
    fuentes: list[Path] = []

    while len(fuentes) < CANTIDAD_A_GENERAR:
        bloque = imagenes_train.copy()
        random.shuffle(bloque)
        fuentes.extend(bloque)

    fuentes = fuentes[:CANTIDAD_A_GENERAR]

    registros: list[list[str | int]] = []

    print("=" * 65)
    print("DATA AUGMENTATION DE IMÁGENES NEGATIVAS")
    print("=" * 65)
    print(f"Imágenes negativas en train: {len(imagenes_train)}")
    print(f"Imágenes a generar:          {CANTIDAD_A_GENERAR}")
    print()

    for indice, ruta_original in enumerate(
        fuentes,
        start=1,
    ):
        imagen_bgr = cv2.imread(str(ruta_original))

        if imagen_bgr is None or imagen_bgr.size == 0:
            print(
                f"ERROR: no se pudo leer {ruta_original.name}"
            )
            sys.exit(1)

        imagen_rgb = cv2.cvtColor(
            imagen_bgr,
            cv2.COLOR_BGR2RGB,
        )

        resultado = transformacion(
            image=imagen_rgb
        )

        imagen_aumentada_rgb = resultado["image"]

        imagen_aumentada_bgr = cv2.cvtColor(
            imagen_aumentada_rgb,
            cv2.COLOR_RGB2BGR,
        )

        nombre_salida = (
            f"negativo_aug_{indice:05d}.jpg"
        )

        ruta_salida = (
            CARPETA_AUGMENTED
            / nombre_salida
        )

        guardar_imagen(
            imagen_aumentada_bgr,
            ruta_salida,
        )

        registros.append(
            [
                indice,
                ruta_original.name,
                nombre_salida,
                str(ruta_original),
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
                "imagen_origen",
                "imagen_aumentada",
                "ruta_origen",
                "ruta_aumentada",
            ]
        )

        escritor.writerows(registros)

    cantidad_generada = len(
        obtener_imagenes(CARPETA_AUGMENTED)
    )

    total_final = (
        CANTIDAD_ORIGINALES_TOTAL
        + cantidad_generada
    )

    print()
    print("=" * 65)
    print("RESULTADO DE LA AUMENTACIÓN NEGATIVA")
    print("=" * 65)
    print(
        f"Originales totales:     "
        f"{CANTIDAD_ORIGINALES_TOTAL}"
    )
    print(
        f"Aumentadas generadas:   "
        f"{cantidad_generada}"
    )
    print(
        f"Total negativo final:   "
        f"{total_final}"
    )
    print(
        f"Registro CSV:           "
        f"{ARCHIVO_CSV}"
    )

    if total_final == META_NEGATIVAS:
        print()
        print(
            "Aumentación completada correctamente: "
            "4000 imágenes negativas."
        )
    else:
        print()
        print(
            f"ERROR: se esperaban {META_NEGATIVAS} "
            f"imágenes negativas, pero se obtuvieron "
            f"{total_final}."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
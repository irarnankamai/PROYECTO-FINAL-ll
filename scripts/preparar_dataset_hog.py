from pathlib import Path
import csv
import hashlib
import random
import shutil
import sys

import cv2
import numpy as np


RAIZ = Path(__file__).resolve().parent.parent

ANCHO_HOG = 128
ALTO_HOG = 64
RELACION_HOG = ANCHO_HOG / ALTO_HOG

SEMILLA = 42

EXTENSIONES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


RUTAS_POSITIVAS = {
    "train_original": (
        RAIZ
        / "dataset"
        / "positivos"
        / "recortes"
        / "train"
    ),
    "train_augmented": (
        RAIZ
        / "dataset"
        / "positivos"
        / "recortes"
        / "augmented"
    ),
    "validation": (
        RAIZ
        / "dataset"
        / "positivos"
        / "recortes"
        / "validation"
    ),
    "test": (
        RAIZ
        / "dataset"
        / "positivos"
        / "recortes"
        / "test"
    ),
}


RUTAS_NEGATIVAS = {
    "train_original": (
        RAIZ
        / "dataset"
        / "negativos"
        / "train"
    ),
    "train_augmented": (
        RAIZ
        / "dataset"
        / "negativos"
        / "augmented"
    ),
    "validation": (
        RAIZ
        / "dataset"
        / "negativos"
        / "validation"
    ),
    "test": (
        RAIZ
        / "dataset"
        / "negativos"
        / "test"
    ),
}


SALIDAS = {
    "train_positivos": (
        RAIZ
        / "dataset"
        / "hog"
        / "train"
        / "positivos"
    ),
    "train_negativos": (
        RAIZ
        / "dataset"
        / "hog"
        / "train"
        / "negativos"
    ),
    "validation_positivos": (
        RAIZ
        / "dataset"
        / "hog"
        / "validation"
        / "positivos"
    ),
    "validation_negativos": (
        RAIZ
        / "dataset"
        / "hog"
        / "validation"
        / "negativos"
    ),
    "test_positivos": (
        RAIZ
        / "dataset"
        / "hog"
        / "test"
        / "positivos"
    ),
    "test_negativos": (
        RAIZ
        / "dataset"
        / "hog"
        / "test"
        / "negativos"
    ),
}


ARCHIVO_CSV = (
    RAIZ
    / "resultados"
    / "preparacion_dataset_hog.csv"
)


def obtener_imagenes(carpeta: Path) -> list[Path]:
    if not carpeta.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {carpeta}"
        )

    return sorted(
        archivo
        for archivo in carpeta.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in EXTENSIONES
    )


def limpiar_carpeta(carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)

    for elemento in carpeta.iterdir():
        if elemento.is_file():
            elemento.unlink()
        elif elemento.is_dir():
            shutil.rmtree(elemento)


def semilla_para_archivo(
    ruta: Path,
    semilla_base: int,
) -> int:
    texto = f"{ruta.name}-{semilla_base}"

    resumen = hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()

    return int(resumen[:8], 16)


def normalizar_positivo(imagen: np.ndarray) -> np.ndarray:
    alto, ancho = imagen.shape[:2]

    if ancho <= 0 or alto <= 0:
        raise ValueError("Dimensiones de imagen inválidas.")

    escala = min(
        ANCHO_HOG / ancho,
        ALTO_HOG / alto,
    )

    nuevo_ancho = max(
        1,
        int(round(ancho * escala)),
    )

    nuevo_alto = max(
        1,
        int(round(alto * escala)),
    )

    interpolacion = (
        cv2.INTER_AREA
        if escala < 1.0
        else cv2.INTER_CUBIC
    )

    redimensionada = cv2.resize(
        imagen,
        (nuevo_ancho, nuevo_alto),
        interpolation=interpolacion,
    )

    salida = np.zeros(
        (ALTO_HOG, ANCHO_HOG, 3),
        dtype=np.uint8,
    )

    x = (ANCHO_HOG - nuevo_ancho) // 2
    y = (ALTO_HOG - nuevo_alto) // 2

    salida[
        y:y + nuevo_alto,
        x:x + nuevo_ancho,
    ] = redimensionada

    return salida


def extraer_parche_negativo(
    imagen: np.ndarray,
    semilla: int,
) -> np.ndarray:
    alto, ancho = imagen.shape[:2]

    if ancho < 2 or alto < 2:
        raise ValueError("Imagen negativa demasiado pequeña.")

    generador = random.Random(semilla)

    relacion_actual = ancho / alto

    if relacion_actual >= RELACION_HOG:
        alto_recorte = alto
        ancho_recorte = int(
            round(alto_recorte * RELACION_HOG)
        )

        ancho_recorte = min(
            ancho_recorte,
            ancho,
        )

        x_maximo = ancho - ancho_recorte
        x = generador.randint(
            0,
            max(0, x_maximo),
        )
        y = 0
    else:
        ancho_recorte = ancho
        alto_recorte = int(
            round(ancho_recorte / RELACION_HOG)
        )

        alto_recorte = min(
            alto_recorte,
            alto,
        )

        y_maximo = alto - alto_recorte
        y = generador.randint(
            0,
            max(0, y_maximo),
        )
        x = 0

    parche = imagen[
        y:y + alto_recorte,
        x:x + ancho_recorte,
    ]

    if parche.size == 0:
        raise ValueError(
            "No se pudo extraer el parche negativo."
        )

    return cv2.resize(
        parche,
        (ANCHO_HOG, ALTO_HOG),
        interpolation=cv2.INTER_AREA,
    )


def guardar_imagen(
    imagen: np.ndarray,
    ruta_salida: Path,
) -> None:
    guardada = cv2.imwrite(
        str(ruta_salida),
        imagen,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not guardada:
        raise RuntimeError(
            f"No se pudo guardar: {ruta_salida}"
        )


def procesar_positivas(
    rutas: list[Path],
    carpeta_salida: Path,
    prefijo: str,
    registros: list[list[str]],
) -> int:
    contador = 0

    for ruta in rutas:
        imagen = cv2.imread(str(ruta))

        if imagen is None or imagen.size == 0:
            raise RuntimeError(
                f"No se pudo leer: {ruta}"
            )

        preparada = normalizar_positivo(imagen)

        contador += 1

        nombre = (
            f"{prefijo}_{contador:05d}.jpg"
        )

        ruta_salida = carpeta_salida / nombre

        guardar_imagen(
            preparada,
            ruta_salida,
        )

        registros.append(
            [
                "positivo",
                prefijo,
                ruta.name,
                nombre,
                str(ruta),
                str(ruta_salida),
                str(ANCHO_HOG),
                str(ALTO_HOG),
            ]
        )

    return contador


def procesar_negativas(
    rutas: list[Path],
    carpeta_salida: Path,
    prefijo: str,
    registros: list[list[str]],
) -> int:
    contador = 0

    for ruta in rutas:
        imagen = cv2.imread(str(ruta))

        if imagen is None or imagen.size == 0:
            raise RuntimeError(
                f"No se pudo leer: {ruta}"
            )

        semilla = semilla_para_archivo(
            ruta,
            SEMILLA,
        )

        preparada = extraer_parche_negativo(
            imagen,
            semilla,
        )

        contador += 1

        nombre = (
            f"{prefijo}_{contador:05d}.jpg"
        )

        ruta_salida = carpeta_salida / nombre

        guardar_imagen(
            preparada,
            ruta_salida,
        )

        registros.append(
            [
                "negativo",
                prefijo,
                ruta.name,
                nombre,
                str(ruta),
                str(ruta_salida),
                str(ANCHO_HOG),
                str(ALTO_HOG),
            ]
        )

    return contador


def main() -> None:
    random.seed(SEMILLA)

    for carpeta in SALIDAS.values():
        limpiar_carpeta(carpeta)

    registros: list[list[str]] = []

    positivas_train = (
        obtener_imagenes(
            RUTAS_POSITIVAS["train_original"]
        )
        + obtener_imagenes(
            RUTAS_POSITIVAS["train_augmented"]
        )
    )

    negativas_train = (
        obtener_imagenes(
            RUTAS_NEGATIVAS["train_original"]
        )
        + obtener_imagenes(
            RUTAS_NEGATIVAS["train_augmented"]
        )
    )

    positivas_validation = obtener_imagenes(
        RUTAS_POSITIVAS["validation"]
    )

    negativas_validation = obtener_imagenes(
        RUTAS_NEGATIVAS["validation"]
    )

    positivas_test = obtener_imagenes(
        RUTAS_POSITIVAS["test"]
    )

    negativas_test = obtener_imagenes(
        RUTAS_NEGATIVAS["test"]
    )

    print("=" * 68)
    print("PREPARACIÓN DEL DATASET PARA HOG + SVM")
    print("=" * 68)
    print(
        f"Tamaño de ventana: "
        f"{ANCHO_HOG}x{ALTO_HOG}"
    )
    print()

    cantidad_pos_train = procesar_positivas(
        positivas_train,
        SALIDAS["train_positivos"],
        "positivo_train",
        registros,
    )

    cantidad_neg_train = procesar_negativas(
        negativas_train,
        SALIDAS["train_negativos"],
        "negativo_train",
        registros,
    )

    cantidad_pos_validation = procesar_positivas(
        positivas_validation,
        SALIDAS["validation_positivos"],
        "positivo_validation",
        registros,
    )

    cantidad_neg_validation = procesar_negativas(
        negativas_validation,
        SALIDAS["validation_negativos"],
        "negativo_validation",
        registros,
    )

    cantidad_pos_test = procesar_positivas(
        positivas_test,
        SALIDAS["test_positivos"],
        "positivo_test",
        registros,
    )

    cantidad_neg_test = procesar_negativas(
        negativas_test,
        SALIDAS["test_negativos"],
        "negativo_test",
        registros,
    )

    ARCHIVO_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ARCHIVO_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as archivo:
        escritor = csv.writer(archivo)

        escritor.writerow(
            [
                "clase",
                "conjunto",
                "nombre_origen",
                "nombre_salida",
                "ruta_origen",
                "ruta_salida",
                "ancho",
                "alto",
            ]
        )

        escritor.writerows(registros)

    print()
    print("=" * 68)
    print("RESULTADO")
    print("=" * 68)
    print(
        f"Train positivos:       "
        f"{cantidad_pos_train}"
    )
    print(
        f"Train negativos:       "
        f"{cantidad_neg_train}"
    )
    print(
        f"Validation positivos:  "
        f"{cantidad_pos_validation}"
    )
    print(
        f"Validation negativos:  "
        f"{cantidad_neg_validation}"
    )
    print(
        f"Test positivos:        "
        f"{cantidad_pos_test}"
    )
    print(
        f"Test negativos:        "
        f"{cantidad_neg_test}"
    )
    print(f"Registro CSV:          {ARCHIVO_CSV}")

    cantidades_esperadas = {
        "positivos train": (
            cantidad_pos_train,
            3766,
        ),
        "negativos train": (
            cantidad_neg_train,
            3730,
        ),
        "positivos validation": (
            cantidad_pos_validation,
            117,
        ),
        "negativos validation": (
            cantidad_neg_validation,
            135,
        ),
        "positivos test": (
            cantidad_pos_test,
            117,
        ),
        "negativos test": (
            cantidad_neg_test,
            135,
        ),
    }

    errores: list[str] = []

    for nombre, (
        cantidad_real,
        cantidad_esperada,
    ) in cantidades_esperadas.items():
        if cantidad_real != cantidad_esperada:
            errores.append(
                f"{nombre}: se esperaban "
                f"{cantidad_esperada}, pero hay "
                f"{cantidad_real}"
            )

    if errores:
        print("\nERROR EN LAS CANTIDADES:")

        for error in errores:
            print(f" - {error}")

        sys.exit(1)

    print(
        "\nDataset HOG preparado correctamente."
    )


if __name__ == "__main__":
    main()
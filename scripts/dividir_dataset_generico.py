from pathlib import Path
import argparse
import csv
import random
import shutil
import sys


EXTENSIONES_VALIDAS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

SEMILLA = 42


def obtener_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Divide un dataset en train, validation y test."
    )

    parser.add_argument(
        "tipo",
        choices=["positivos", "negativos"],
        help="Dataset que se desea dividir.",
    )

    parser.add_argument(
        "--validation",
        type=int,
        required=True,
        help="Cantidad para validación.",
    )

    parser.add_argument(
        "--test",
        type=int,
        required=True,
        help="Cantidad para prueba.",
    )

    return parser.parse_args()


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


def copiar_imagen(
    origen: Path,
    destino: Path,
    indice: int,
    tipo: str,
    conjunto: str,
) -> str:
    extension = origen.suffix.lower()

    nombre_nuevo = (
        f"{tipo}_{conjunto}_{indice:04d}{extension}"
    )

    ruta_destino = destino / nombre_nuevo
    shutil.copy2(origen, ruta_destino)

    return nombre_nuevo


def main() -> None:
    argumentos = obtener_argumentos()

    raiz = Path(__file__).resolve().parent.parent
    base = raiz / "dataset" / argumentos.tipo

    originales = base / "originales"
    train = base / "train"
    validation = base / "validation"
    test = base / "test"

    archivo_csv = (
        raiz
        / "resultados"
        / f"division_dataset_{argumentos.tipo}.csv"
    )

    if not originales.exists():
        print(f"ERROR: no existe:\n{originales}")
        sys.exit(1)

    imagenes = obtener_imagenes(originales)
    total = len(imagenes)

    cantidad_train = (
        total
        - argumentos.validation
        - argumentos.test
    )

    if cantidad_train <= 0:
        print(
            "ERROR: las cantidades de validación y prueba "
            "superan el total disponible."
        )
        sys.exit(1)

    random.seed(SEMILLA)
    random.shuffle(imagenes)

    imagenes_train = imagenes[:cantidad_train]

    inicio_validation = cantidad_train
    fin_validation = (
        cantidad_train + argumentos.validation
    )

    imagenes_validation = imagenes[
        inicio_validation:fin_validation
    ]

    imagenes_test = imagenes[fin_validation:]

    limpiar_carpeta(train)
    limpiar_carpeta(validation)
    limpiar_carpeta(test)

    archivo_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registros: list[list[str]] = []

    conjuntos = [
        ("train", imagenes_train, train),
        (
            "validation",
            imagenes_validation,
            validation,
        ),
        ("test", imagenes_test, test),
    ]

    for conjunto, lista, carpeta_destino in conjuntos:
        for indice, imagen in enumerate(
            lista,
            start=1,
        ):
            nombre_nuevo = copiar_imagen(
                imagen,
                carpeta_destino,
                indice,
                argumentos.tipo,
                conjunto,
            )

            registros.append(
                [
                    imagen.name,
                    nombre_nuevo,
                    conjunto,
                    str(imagen),
                ]
            )

    with archivo_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as archivo:
        escritor = csv.writer(archivo)

        escritor.writerow(
            [
                "nombre_original",
                "nombre_nuevo",
                "conjunto",
                "ruta_original",
            ]
        )

        escritor.writerows(registros)

    print("=" * 65)
    print(f"DIVISIÓN DEL DATASET: {argumentos.tipo.upper()}")
    print("=" * 65)
    print(f"Total original:    {total}")
    print(f"Entrenamiento:     {len(imagenes_train)}")
    print(f"Validación:        {len(imagenes_validation)}")
    print(f"Prueba:            {len(imagenes_test)}")
    print(f"Semilla:           {SEMILLA}")
    print(f"Registro CSV:      {archivo_csv}")
    print("\nDivisión completada correctamente.")


if __name__ == "__main__":
    main()
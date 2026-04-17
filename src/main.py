from pathlib import Path
from pyexpat import features
import pandas as pd
from segmentacion import segmentar_una_imagen
from feature import extraer_histograma_hsv
from similitud import top_k_similares

DATASET_DIR = Path("dataset")
SALIDA_DIR = Path("salida")
SEGMENTADOS_DIR = SALIDA_DIR / "segmentados"

SALIDA_DIR.mkdir(exist_ok=True)
SEGMENTADOS_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def detectar_vista(nombre):
    n = nombre.lower()
    if "polar" in n:
        return "polar"
    elif "ecuatorial" in n:
        return "ecuatorial"
    return "desconocida"


def crear_dataframe(dataset_dir):
    registros = []

    for carpeta in sorted(dataset_dir.iterdir()):
        if not carpeta.is_dir():
            continue

        especie = carpeta.name

        for img in sorted(carpeta.iterdir()):
            if img.is_file() and img.suffix.lower() in EXTENSIONES:
                registros.append({
                    "especie": especie,
                    "ruta": str(img),
                    "archivo": img.name,
                    "vista": detectar_vista(img.name)
                })

    return pd.DataFrame(registros)


def filtrar_especies_validas(df):
    validas = []

    for especie, grupo in df.groupby("especie"):
        vistas = set(grupo["vista"])
        if "polar" in vistas and "ecuatorial" in vistas:
            validas.append(especie)

    return df[df["especie"].isin(validas)].copy()


def main():
    df = crear_dataframe(DATASET_DIR)

    if df.empty:
        print("No se encontraron imágenes en dataset.")
        return

    df_filtrado = filtrar_especies_validas(df)

    if df_filtrado.empty:
        print("No hay especies con vista polar y ecuatorial.")
        return

    ruta_csv = SALIDA_DIR / "imagenes_validas.csv"
    df_filtrado.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    print("\nEspecies válidas:")
    print(df_filtrado["especie"].unique())

    print("\nTotal imágenes filtradas:", len(df_filtrado))

    for _, fila in df_filtrado.iterrows():
        especie = fila["especie"]
        ruta = fila["ruta"]

        carpeta_salida = SEGMENTADOS_DIR / especie
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        segmentar_una_imagen(ruta, carpeta_salida)

    print("\nProceso terminado.")

    print("\nExtrayendo features...")

    features = []
    rutas = []

    for especie_dir in SEGMENTADOS_DIR.iterdir():
        if not especie_dir.is_dir():
            continue

        for img in especie_dir.iterdir():
            if "_recorte" in img.name:
                vector = extraer_histograma_hsv(img)

                if vector is not None:
                    features.append(vector)
                    rutas.append(str(img))

    print("Total vectores:", len(features))


    print("\nPrueba de similitud:")

    indice_consulta = 0
    vector_consulta = features[indice_consulta]
    ruta_consulta = rutas[indice_consulta]

    resultados = top_k_similares(vector_consulta, features, rutas, k=5)

    print(f"\nImagen consulta: {ruta_consulta}")
    print("\nTop similares:")

    for ruta, score in resultados:
        print(f"{score:.4f} -> {ruta}")

if __name__ == "__main__":
    main()
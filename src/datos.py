"""Carga, conversion a unidades SI y descripcion del conjunto de datos.

La primera vez que se ejecuta:
  1. Se descarga el dataset 'cars' con pydataset (unidades originales:
     velocidad en millas por hora y distancia en pies) y se guarda en
     data/cars.csv.
  2. Se convierte a unidades SI (velocidad en km/h y distancia en metros)
     y se guarda en data/cars_si.csv.

Todos los calculos del proyecto usan exclusivamente data/cars_si.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_DATOS = Path(__file__).resolve().parent.parent / "data"
RUTA_ORIGINAL = CARPETA_DATOS / "cars.csv"      # millas/hora y pies
RUTA_SI = CARPETA_DATOS / "cars_si.csv"         # km/h y metros

# Factores de conversion exactos (definiciones internacionales)
MPH_A_KMH = 1.609344   # 1 milla = 1.609,344 m
PIE_A_M = 0.3048       # 1 pie = 0,3048 m


def descargar_datos(ruta: Path = RUTA_ORIGINAL) -> Path:
    """Descarga el dataset 'cars' con pydataset y lo guarda como CSV (unidades originales)."""
    from pydataset import data

    print("Dataset no encontrado. Descargando 'cars' con pydataset...")
    cars = data("cars")
    # 1. Crear la carpeta 'data' si todavia no existe
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # 2. Guardar el DataFrame en CSV (sin el indice numerico de la izquierda)
    cars.to_csv(ruta, index=False)
    print(f"Archivo guardado con exito en {ruta}")
    return ruta


def convertir_a_si(ruta_origen: Path = RUTA_ORIGINAL, ruta_destino: Path = RUTA_SI) -> Path:
    """Convierte velocidad (mph -> km/h) y distancia (pies -> m) y guarda el CSV en SI."""
    df = pd.read_csv(ruta_origen)
    df_si = pd.DataFrame({
        "velocidad_kmh": df["speed"] * MPH_A_KMH,
        "distancia_m": df["dist"] * PIE_A_M,
    })
    df_si.to_csv(ruta_destino, index=False)
    print(f"Datos convertidos a unidades SI (km/h y m) en {ruta_destino}")
    return ruta_destino


def preparar_datos() -> Path:
    """Garantiza que exista el CSV en unidades SI, descargando y convirtiendo si hace falta."""
    if not RUTA_ORIGINAL.exists():
        descargar_datos(RUTA_ORIGINAL)
    if not RUTA_SI.exists():
        convertir_a_si(RUTA_ORIGINAL, RUTA_SI)
    return RUTA_SI


def cargar_datos() -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (x, y) en unidades SI: velocidad (km/h) y distancia de frenado (m)."""
    df = pd.read_csv(preparar_datos())
    return df["velocidad_kmh"].to_numpy(float), df["distancia_m"].to_numpy(float)


def estadisticos_descriptivos(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Tabla de medidas descriptivas de ambas variables."""
    def resumen(v):
        return {
            "n": len(v),
            "minimo": v.min(),
            "mediana": np.median(v),
            "media": v.mean(),
            "maximo": v.max(),
            "desvio_estandar": v.std(ddof=1),
        }
    return pd.DataFrame({"velocidad (km/h)": resumen(x), "distancia (m)": resumen(y)})

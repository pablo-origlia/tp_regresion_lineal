"""Diagnostico de los supuestos del modelo (contenido de la asignatura).

El diagnostico se basa en el analisis grafico de residuos (ver src/graficos.py)
y en medidas descriptivas que lo respaldan:
  - propiedades algebraicas de los residuos de minimos cuadrados,
  - residuos estandarizados y deteccion de atipicos,
  - comparacion con la regla empirica 68-95-99,7 de la normal (Unidad II),
  - dispersion de los residuos por tramos de x (homocedasticidad).
"""
import numpy as np
import pandas as pd
from scipy import stats

from .regresion import RegresionLinealSimple


def residuos_estandarizados(m: RegresionLinealSimple) -> np.ndarray:
    h = 1 / m.n + (m.x - m.x_media) ** 2 / m.Sxx   # leverage
    return m.residuos / (m.s * np.sqrt(1 - h))


def propiedades_residuos(m: RegresionLinealSimple) -> dict:
    """Propiedades algebraicas que cumplen los residuos de MCO."""
    return {
        "suma_residuos": float(np.sum(m.residuos)),
        "suma_x_por_residuos": float(np.sum(m.x * m.residuos)),
    }


def atipicos(m: RegresionLinealSimple, umbral: float = 2.0) -> list[dict]:
    re = residuos_estandarizados(m)
    idx = np.where(np.abs(re) > umbral)[0]
    return [
        {"obs": int(i + 1), "x": m.x[i], "y": m.y[i], "y_ajustado": m.y_ajustado[i],
         "residuo": m.residuos[i], "residuo_estandarizado": re[i]}
        for i in idx
    ]


def regla_empirica(m: RegresionLinealSimple) -> list[dict]:
    """Proporcion de residuos estandarizados dentro de +-1, +-2 y +-3,
    comparada con lo esperado bajo normalidad (68 %, 95 %, 99,7 %)."""
    re = residuos_estandarizados(m)
    filas = []
    for k in (1, 2, 3):
        filas.append({
            "k": k,
            "observado": float(np.mean(np.abs(re) <= k)),
            "esperado_normal": float(stats.norm.cdf(k) - stats.norm.cdf(-k)),
        })
    return filas


def asimetria_residuos(m: RegresionLinealSimple) -> dict:
    """Comparacion media-mediana y residuos positivos/negativos extremos."""
    return {
        "media": float(np.mean(m.residuos)),
        "mediana": float(np.median(m.residuos)),
        "minimo": float(np.min(m.residuos)),
        "maximo": float(np.max(m.residuos)),
    }


def dispersion_por_tramos(m: RegresionLinealSimple, n_tramos: int = 3) -> list[dict]:
    """Desvio estandar de los residuos en tramos de x (terciles por defecto).

    Bajo homocedasticidad los desvios deberian ser similares entre tramos.
    """
    tramos = pd.qcut(m.x, n_tramos, labels=False, duplicates="drop")
    filas = []
    for t in np.unique(tramos):
        sel = tramos == t
        filas.append({
            "tramo": int(t + 1),
            "x_min": float(m.x[sel].min()),
            "x_max": float(m.x[sel].max()),
            "n": int(sel.sum()),
            "desvio_residuos": float(np.std(m.residuos[sel], ddof=1)),
        })
    base = filas[0]["desvio_residuos"]
    for f in filas:
        f["razon_vs_primer_tramo"] = f["desvio_residuos"] / base
    return filas


def diagnostico_completo(m: RegresionLinealSimple, umbral_atipico: float = 2.0) -> dict:
    return {
        "propiedades": propiedades_residuos(m),
        "atipicos": atipicos(m, umbral_atipico),
        "regla_empirica": regla_empirica(m),
        "asimetria": asimetria_residuos(m),
        "dispersion_por_tramos": dispersion_por_tramos(m),
    }

"""AMPLIACION (fuera del programa de la asignatura).

Pruebas formales de los supuestos, de uso estandar en software estadistico.
Se presentan solo como confirmacion numerica del analisis grafico.
  - Shapiro-Wilk: normalidad de los residuos.
  - Breusch-Pagan (clasico y Koenker): homocedasticidad.
  - Durbin-Watson: autocorrelacion de los residuos.
"""
import numpy as np
from scipy import stats

from .regresion import RegresionLinealSimple


def _r2_auxiliar(x: np.ndarray, v: np.ndarray) -> tuple[float, np.ndarray]:
    coef = np.polyfit(x, v, 1)
    ajuste = np.polyval(coef, x)
    r2 = 1 - np.sum((v - ajuste) ** 2) / np.sum((v - v.mean()) ** 2)
    return r2, ajuste


def shapiro_wilk(m: RegresionLinealSimple) -> dict:
    sw = stats.shapiro(m.residuos)
    return {"W": float(sw.statistic), "p_valor": float(sw.pvalue)}


def breusch_pagan(m: RegresionLinealSimple) -> dict:
    e2 = m.residuos ** 2
    u = e2 / (m.SS_R / m.n)
    _, ajuste = _r2_auxiliar(m.x, u)
    lm_clasico = np.sum((ajuste - u.mean()) ** 2) / 2
    r2_aux, _ = _r2_auxiliar(m.x, e2)
    lm_koenker = m.n * r2_aux
    return {
        "clasico": {"estadistico": lm_clasico, "p_valor": stats.chi2.sf(lm_clasico, 1)},
        "koenker": {"estadistico": lm_koenker, "p_valor": stats.chi2.sf(lm_koenker, 1)},
    }


def durbin_watson(m: RegresionLinealSimple) -> float:
    return float(np.sum(np.diff(m.residuos) ** 2) / m.SS_R)


def pruebas_formales(m: RegresionLinealSimple) -> dict:
    return {
        "shapiro_wilk": shapiro_wilk(m),
        "breusch_pagan": breusch_pagan(m),
        "durbin_watson": durbin_watson(m),
    }

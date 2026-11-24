"""Verifica los resultados contra los valores de referencia de R:
lm(dist ~ speed, data = cars), expresados originalmente en mph y pies.

Los coeficientes de R se convierten a SI con los mismos factores que usa el
proyecto. Las medidas adimensionales (r, r^2, t, F, p-valores) no dependen
de las unidades, por lo que deben coincidir exactamente con R.
"""
import numpy as np
import pytest

from src.datos import MPH_A_KMH, PIE_A_M, cargar_datos
from src.regresion import RegresionLinealSimple

# Valores de referencia de R (unidades originales)
R_ALPHA_PIES = -17.5791
R_BETA_PIES_POR_MPH = 3.9324
R_S_PIES = 15.38


@pytest.fixture(scope="module")
def modelo():
    x, y = cargar_datos()
    return RegresionLinealSimple().ajustar(x, y)


def test_datos_en_si():
    x, y = cargar_datos()
    assert len(x) == 50
    assert x.min() == pytest.approx(4 * MPH_A_KMH)
    assert y.max() == pytest.approx(120 * PIE_A_M)


def test_coeficientes_convertidos(modelo):
    assert modelo.alpha == pytest.approx(R_ALPHA_PIES * PIE_A_M, abs=1e-4)
    assert modelo.beta == pytest.approx(R_BETA_PIES_POR_MPH * PIE_A_M / MPH_A_KMH, abs=1e-4)
    assert modelo.s == pytest.approx(R_S_PIES * PIE_A_M, abs=1e-2)


def test_medidas_invariantes(modelo):
    c = modelo.correlacion()
    assert c["r2"] == pytest.approx(0.6511, abs=1e-4)
    assert c["r2_ajustado"] == pytest.approx(0.6438, abs=1e-4)
    inf = modelo.inferencia_coeficientes()
    assert inf["beta"]["t"] == pytest.approx(9.464, abs=1e-3)
    assert inf["alpha"]["t"] == pytest.approx(-2.601, abs=1e-3)


def test_anova_equivale_a_t(modelo):
    t = modelo.inferencia_coeficientes()["beta"]["t"]
    assert modelo.anova()["F"] == pytest.approx(t ** 2)


def test_propiedades_minimos_cuadrados(modelo):
    assert np.sum(modelo.residuos) == pytest.approx(0, abs=1e-9)
    assert np.sum(modelo.x * modelo.residuos) == pytest.approx(0, abs=1e-8)


def test_prediccion(modelo):
    # 21 mph en km/h debe dar 65,0015 pies expresados en metros
    p = modelo.predecir(21 * MPH_A_KMH)
    assert p["y0"] == pytest.approx(65.0015 * PIE_A_M, abs=1e-3)
    assert p["ip_inf"] < p["ic_media_inf"] < p["y0"] < p["ic_media_sup"] < p["ip_sup"]


def test_regla_empirica_y_tramos(modelo):
    from src.diagnostico import dispersion_por_tramos, regla_empirica
    re = regla_empirica(modelo)
    assert [f["k"] for f in re] == [1, 2, 3]
    assert all(0 <= f["observado"] <= 1 for f in re)
    tramos = dispersion_por_tramos(modelo)
    assert sum(t["n"] for t in tramos) == modelo.n


def test_ampliacion(modelo):
    from src.ampliacion import pruebas_formales
    pf = pruebas_formales(modelo)
    assert pf["shapiro_wilk"]["p_valor"] == pytest.approx(0.0215, abs=1e-3)
    assert pf["durbin_watson"] == pytest.approx(1.676, abs=1e-3)

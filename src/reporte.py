"""Armado del reporte de texto y exportacion de resultados."""
import json
from pathlib import Path

import numpy as np


def _f(v, d=4):
    if abs(v) < 0.5 * 10 ** -d:   # evita mostrar "-0,00"
        v = 0.0
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _p(v):
    return f"{v:.2e}" if v < 1e-4 else _f(v, 4)


def _num(v):
    return f"{v:g}"


def _bloque_diagnostico(d: dict, sangria: str = "  ") -> list[str]:
    s = sangria
    L = []
    pr = d["propiedades"]
    L.append(f"{s}Propiedades MCO: suma residuos = {pr['suma_residuos']:.2e}, "
             f"suma x*residuos = {pr['suma_x_por_residuos']:.2e}")

    a = d["asimetria"]
    L.append(f"{s}Residuos: media = {_f(a['media'],2)}, mediana = {_f(a['mediana'],2)}, "
             f"min = {_f(a['minimo'],2)}, max = {_f(a['maximo'],2)}")

    L.append(f"{s}Regla empirica (residuos estandarizados dentro de +-k):")
    for r in d["regla_empirica"]:
        L.append(f"{s}  k={r['k']}: observado {_f(100*r['observado'],1)} %   "
                 f"esperado normal {_f(100*r['esperado_normal'],1)} %")

    L.append(f"{s}Dispersion de residuos por tramos de x:")
    for t in d["dispersion_por_tramos"]:
        L.append(f"{s}  Tramo {t['tramo']}: x en [{_f(t['x_min'],2)}; {_f(t['x_max'],2)}] km/h, "
                 f"n = {t['n']:>2}, desvio = {_f(t['desvio_residuos'],3)}, "
                 f"razon vs tramo 1 = {_f(t['razon_vs_primer_tramo'],2)}")

    L.append(f"{s}Observaciones con |residuo estandarizado| > 2: {len(d['atipicos'])}")
    for o in d["atipicos"]:
        L.append(f"{s}  Obs {o['obs']:>2}: x={_f(o['x'],2)}, y={_f(o['y'],2)}, "
                 f"y_aj={_f(o['y_ajustado'],2)}, res={_f(o['residuo'],2)}, "
                 f"res_est={_f(o['residuo_estandarizado'],2)}")
    return L


def construir_reporte(r: dict) -> str:
    L = []
    sep = "=" * 76
    L += [sep, "TP FINAL - ESTADISTICA DE DATOS - REGRESION LINEAL SIMPLE", sep,
          "Unidades SI: velocidad x en km/h, distancia de frenado y en metros"]

    L += ["\n[1] ESTADISTICOS DESCRIPTIVOS", r["descriptivos"].round(3).to_string()]

    m = r["modelo"]
    L += ["\n[2] SUMAS Y ESTIMADORES DE MINIMOS CUADRADOS",
          f"  Sxx = {_f(m['Sxx'])}   Syy = {_f(m['Syy'])}   Sxy = {_f(m['Sxy'])}",
          f"  beta  (pendiente) = {_f(m['beta'])}",
          f"  alpha (ordenada)  = {_f(m['alpha'])}",
          f"  Recta estimada: y = {_f(m['alpha'])} + {_f(m['beta'])} x"]

    c = r["correlacion"]
    L += ["\n[3] CORRELACION Y DETERMINACION",
          f"  r = {_f(c['r'])}   r^2 = {_f(c['r2'])}   r^2 ajustado = {_f(c['r2_ajustado'])}"]

    v = r["varianza_error"]
    L += ["\n[4] VARIANZA DEL ERROR",
          f"  s^2 = {_f(v['s2'])} m^2   s = {_f(v['s'])} m",
          f"  IC para sigma^2: [{_f(v['sigma2_inf'])}; {_f(v['sigma2_sup'])}]",
          f"  IC para sigma:   [{_f(v['sigma_inf'])}; {_f(v['sigma_sup'])}]"]

    L += ["\n[5] INFERENCIA SOBRE LOS COEFICIENTES (H0: parametro = 0)",
          f"  {'Param':<7}{'Estim.':>11}{'Err.est.':>11}{'t':>9}{'p-valor':>12}   IC"]
    for nombre, d in r["coeficientes"].items():
        L.append(f"  {nombre:<7}{_f(d['estimacion']):>11}{_f(d['error_estandar']):>11}"
                 f"{_f(d['t'], 3):>9}{_p(d['p_valor']):>12}   "
                 f"[{_f(d['ic_inf'])}; {_f(d['ic_sup'])}]")

    a = r["anova"]
    L += ["\n[6] TABLA ANOVA",
          f"  {'Fuente':<11}{'SC':>13}{'gl':>5}{'CM':>13}{'F':>9}{'p-valor':>12}",
          f"  {'Regresion':<11}{_f(a['SC_regresion'],2):>13}{a['gl_regresion']:>5}"
          f"{_f(a['CM_regresion'],2):>13}{_f(a['F'],2):>9}{_p(a['p_valor']):>12}",
          f"  {'Residual':<11}{_f(a['SC_residual'],2):>13}{a['gl_residual']:>5}{_f(a['CM_residual'],2):>13}",
          f"  {'Total':<11}{_f(a['SC_total'],2):>13}{a['gl_total']:>5}",
          f"  F critico = {_f(a['F_critico'])}"]

    L += ["\n[7] PREDICCION (x0 en km/h; y0 e intervalos en m)",
          f"  {'x0':>5}{'y0':>10}   {'IC respuesta media':<24}{'Intervalo de prediccion'}"]
    for p in r["predicciones"]:
        L.append(f"  {_num(p['x0']):>5}{_f(p['y0'],2):>10}   "
                 f"[{_f(p['ic_media_inf'],2)}; {_f(p['ic_media_sup'],2)}]".ljust(42) +
                 f"[{_f(p['ip_inf'],2)}; {_f(p['ip_sup'],2)}]")

    L += ["\n[8] DIAGNOSTICO DE SUPUESTOS (analisis grafico: fig3_diagnostico.png)"]
    L += _bloque_diagnostico(r["diagnostico"])

    t = r["modelo_raiz"]
    L += ["\n[9] EXTENSION: MODELO sqrt(Y) = alpha' + beta' x (fig4_comparacion_residuos.png)",
          f"  Recta: sqrt(y) = {_f(t['alpha'])} + {_f(t['beta'])} x",
          f"  r^2 = {_f(t['r2'])}",
          f"  Prediccion a x0 = {_num(t['x0'])} km/h: {_f(t['prediccion'],2)} m",
          "  Diagnostico (residuos en escala sqrt):"]
    L += _bloque_diagnostico(t["diagnostico"], sangria="    ")

    if "ampliacion" in r:
        L += ["\n[10] AMPLIACION - PRUEBAS FORMALES (fuera del programa de la asignatura)",
              f"  {'Prueba':<28}{'Modelo Y':>22}{'Modelo sqrt(Y)':>22}"]
        o, q = r["ampliacion"]["modelo_original"], r["ampliacion"]["modelo_raiz"]
        def fila(nombre, a, b):
            L.append(f"  {nombre:<28}{a:>22}{b:>22}")
        fila("Shapiro-Wilk W", _f(o["shapiro_wilk"]["W"]), _f(q["shapiro_wilk"]["W"]))
        fila("Shapiro-Wilk p-valor", _p(o["shapiro_wilk"]["p_valor"]), _p(q["shapiro_wilk"]["p_valor"]))
        for k, et in [("clasico", "Breusch-Pagan clasico"), ("koenker", "Breusch-Pagan Koenker")]:
            fila(f"{et} LM", _f(o["breusch_pagan"][k]["estadistico"]), _f(q["breusch_pagan"][k]["estadistico"]))
            fila(f"{et} p", _p(o["breusch_pagan"][k]["p_valor"]), _p(q["breusch_pagan"][k]["p_valor"]))
        fila("Durbin-Watson", _f(o["durbin_watson"]), _f(q["durbin_watson"]))
    L.append(sep)
    return "\n".join(L)


def _a_serializable(obj):
    if isinstance(obj, dict):
        return {k: _a_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_a_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "to_dict"):
        return _a_serializable(obj.to_dict())
    return obj


def exportar_json(r: dict, ruta: Path):
    ruta.write_text(json.dumps(_a_serializable(r), indent=2, ensure_ascii=False), encoding="utf-8")

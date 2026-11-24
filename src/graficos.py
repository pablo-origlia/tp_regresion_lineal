"""Generacion de los graficos del informe."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .regresion import RegresionLinealSimple

ETIQ_X = "Velocidad (km/h)"
ETIQ_Y = "Distancia de frenado (m)"


def grafico_dispersion(x, y, ruta: Path):
    plt.figure(figsize=(7, 4.5))
    plt.scatter(x, y, color="#2b6cb0", zorder=3, label="Observaciones")
    plt.title("Diagrama de dispersión: velocidad vs. distancia de frenado")
    plt.xlabel(ETIQ_X); plt.ylabel(ETIQ_Y)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(ruta, dpi=130); plt.close()


def grafico_recta_intervalos(m: RegresionLinealSimple, ruta: Path):
    xs = np.linspace(m.x.min() - 1, m.x.max() + 1, 200)
    b = m.bandas(xs)
    pct = int(m.nivel_confianza * 100)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(m.x, m.y, color="#2b6cb0", zorder=3, label="Observaciones")
    plt.plot(xs, b["y"], "r", label=f"ŷ = {m.alpha:.3f} + {m.beta:.3f}x")
    plt.fill_between(xs, b["media_inf"], b["media_sup"], color="red", alpha=0.2,
                     label=f"IC {pct}% respuesta media")
    plt.plot(xs, b["pred_inf"], "k--", lw=1, label=f"Intervalo de predicción {pct}%")
    plt.plot(xs, b["pred_sup"], "k--", lw=1)
    plt.title("Recta de regresión estimada e intervalos")
    plt.xlabel(ETIQ_X); plt.ylabel(ETIQ_Y)
    plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(ruta, dpi=130); plt.close()


def grafico_diagnostico(m: RegresionLinealSimple, ruta: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(m.y_ajustado, m.residuos, color="#2b6cb0")
    ax[0].axhline(0, color="r")
    ax[0].set_xlabel("Valores ajustados ŷ"); ax[0].set_ylabel("Residuos")
    ax[0].set_title("Residuos vs. ajustados"); ax[0].grid(alpha=0.3)
    stats.probplot(m.residuos, dist="norm", plot=ax[1])
    ax[1].set_title("Gráfico Q-Q normal de residuos")
    ax[1].set_xlabel("Cuantiles teóricos"); ax[1].set_ylabel("Cuantiles muestrales")
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(ruta, dpi=130); plt.close(fig)


def grafico_comparacion_residuos(m: RegresionLinealSimple, m_raiz: RegresionLinealSimple, ruta: Path):
    """Residuos estandarizados del modelo original vs. el modelo sqrt(Y)."""
    from .diagnostico import residuos_estandarizados
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, mod, titulo in [(ax[0], m, "Modelo original: Y"),
                           (ax[1], m_raiz, "Modelo transformado: √Y")]:
        a.scatter(mod.x, residuos_estandarizados(mod), color="#2b6cb0")
        a.axhline(0, color="r")
        for k in (-2, 2):
            a.axhline(k, color="gray", ls="--", lw=1)
        a.set_title(titulo); a.set_xlabel(ETIQ_X); a.grid(alpha=0.3)
    ax[0].set_ylabel("Residuos estandarizados")
    plt.tight_layout(); plt.savefig(ruta, dpi=130); plt.close(fig)


def generar_todos(m: RegresionLinealSimple, m_raiz: RegresionLinealSimple, carpeta: Path) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = [carpeta / "fig1_dispersion.png",
             carpeta / "fig2_recta_intervalos.png",
             carpeta / "fig3_diagnostico.png",
             carpeta / "fig4_comparacion_residuos.png"]
    grafico_dispersion(m.x, m.y, rutas[0])
    grafico_recta_intervalos(m, rutas[1])
    grafico_diagnostico(m, rutas[2])
    grafico_comparacion_residuos(m, m_raiz, rutas[3])
    return rutas

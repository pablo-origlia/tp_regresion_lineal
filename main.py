"""Punto de entrada: ejecuta el analisis completo de regresion lineal simple.

Todas las magnitudes estan en unidades SI: velocidad en km/h y distancia en metros.

Uso:
    python main.py                       # analisis completo
    python main.py --x0 15 25 35 --nivel 0.95 --salida outputs
    python main.py --con-ampliacion      # agrega las pruebas formales (fuera del programa)
"""
import argparse
from pathlib import Path

import numpy as np

from src.ampliacion import pruebas_formales
from src.datos import cargar_datos, estadisticos_descriptivos
from src.diagnostico import diagnostico_completo
from src.graficos import generar_todos
from src.regresion import RegresionLinealSimple
from src.reporte import construir_reporte, exportar_json


def analizar(x0s, nivel, carpeta_salida: Path, con_ampliacion: bool = False) -> dict:
    x, y = cargar_datos()

    # Modelo principal
    modelo = RegresionLinealSimple(nivel_confianza=nivel).ajustar(x, y)

    # Extension: modelo transformado sqrt(Y) (discusion critica)
    modelo_raiz = RegresionLinealSimple(nivel_confianza=nivel).ajustar(x, np.sqrt(y))
    x_ref = x0s[-1]
    diag_raiz = diagnostico_completo(modelo_raiz)
    raiz = {
        "alpha": modelo_raiz.alpha,
        "beta": modelo_raiz.beta,
        "r2": modelo_raiz.correlacion()["r2"],
        "x0": x_ref,
        "prediccion": (modelo_raiz.alpha + modelo_raiz.beta * x_ref) ** 2,
        "diagnostico": diag_raiz,
    }

    resultados = {
        "descriptivos": estadisticos_descriptivos(x, y),
        "modelo": {"n": modelo.n, "Sxx": modelo.Sxx, "Syy": modelo.Syy, "Sxy": modelo.Sxy,
                   "alpha": modelo.alpha, "beta": modelo.beta},
        "correlacion": modelo.correlacion(),
        "varianza_error": modelo.ic_varianza_error(),
        "coeficientes": modelo.inferencia_coeficientes(),
        "anova": modelo.anova(),
        "predicciones": [modelo.predecir(v) for v in x0s],
        "diagnostico": diagnostico_completo(modelo),
        "modelo_raiz": raiz,
    }
    if con_ampliacion:
        resultados["ampliacion"] = {
            "modelo_original": pruebas_formales(modelo),
            "modelo_raiz": pruebas_formales(modelo_raiz),
        }

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    figuras = generar_todos(modelo, modelo_raiz, carpeta_salida / "img")
    reporte = construir_reporte(resultados)
    (carpeta_salida / "resultados.txt").write_text(reporte, encoding="utf-8")
    exportar_json(resultados, carpeta_salida / "resultados.json")

    print(reporte)
    print("\nArchivos generados:")
    for f in [*figuras, carpeta_salida / "resultados.txt", carpeta_salida / "resultados.json"]:
        print(f"  - {f}")
    return resultados


def main():
    parser = argparse.ArgumentParser(description="Analisis de regresion lineal simple (dataset cars).")
    parser.add_argument("--x0", type=float, nargs="+", default=[15, 25, 35],
                        help="Velocidades en km/h para predecir (por defecto: 15 25 35)")
    parser.add_argument("--nivel", type=float, default=0.95, help="Nivel de confianza (por defecto 0.95)")
    parser.add_argument("--salida", type=Path, default=Path("outputs"), help="Carpeta de salida")
    parser.add_argument("--con-ampliacion", action="store_true",
                        help="Agrega las pruebas formales que exceden el programa "
                             "(Shapiro-Wilk, Breusch-Pagan, Durbin-Watson)")
    args = parser.parse_args()
    analizar(args.x0, args.nivel, args.salida, con_ampliacion=args.con_ampliacion)


if __name__ == "__main__":
    main()

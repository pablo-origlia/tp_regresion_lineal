# TP Final — Estadística de Datos: Regresión Lineal Simple

**UCASAL — Maestría en Ciencia de Datos**

Proyecto en Python que reproduce de forma completa el análisis de regresión lineal simple del trabajo final, aplicado al dataset `cars` (velocidad vs. distancia de frenado de 50 automóviles).

Todo el análisis se realiza en **unidades SI**: velocidad en **km/h** y distancia en **metros**.

## Contenido del análisis

| Etapa | Unidad del programa | Módulo |
|---|---|---|
| Descarga, conversión a unidades SI y estadísticos descriptivos | — | `src/datos.py` |
| Estimación por mínimos cuadrados ($\hat{\alpha}$, $\hat{\beta}$) | IV | `src/regresion.py` |
| Coeficientes de correlación y determinación | IV | `src/regresion.py` |
| IC para $\sigma^2$ (distribución $\chi^2$) | III | `src/regresion.py` |
| Test $t$ e IC para los coeficientes | III y IV | `src/regresion.py` |
| Tabla ANOVA | IV | `src/regresion.py` |
| IC de la respuesta media e intervalo de predicción | IV | `src/regresion.py` |
| Diagnóstico gráfico y descriptivo de supuestos: residuos estandarizados, atípicos, regla empírica 68-95-99,7, dispersión por tramos | II y IV | `src/diagnostico.py` |
| Extensión: modelo $\sqrt{Y}$ y comparación de residuos | IV | `main.py` |
| **Ampliación (fuera del programa):** Shapiro-Wilk, Breusch-Pagan, Durbin-Watson | — | `src/ampliacion.py` |
| Gráficos del informe | — | `src/graficos.py` |
| Reporte de texto y JSON | — | `src/reporte.py` |

Los estimadores se calculan **"a mano"** con las fórmulas de la teoría ($S_{xx}$, $S_{xy}$, $SS_R$, etc.), sin usar librerías de modelado que oculten los cálculos.

## Estructura

```text
tp_regresion_lineal/
├── data/
│   ├── cars.csv              # Dataset original (mph y pies), se descarga automáticamente
│   └── cars_si.csv           # Dataset convertido a SI (km/h y m), usado en todos los cálculos
├── src/
│   ├── datos.py              # Descarga, conversión a SI, carga y descriptivos
│   ├── regresion.py          # Clase RegresionLinealSimple
│   ├── diagnostico.py        # Diagnóstico de supuestos (contenido de la materia)
│   ├── ampliacion.py         # Pruebas formales (fuera del programa)
│   ├── graficos.py           # Figuras del informe
│   └── reporte.py            # Reporte .txt y exportación .json
├── tests/
│   └── test_regresion.py     # Validación contra la salida de R
├── main.py                   # Punto de entrada
├── requirements.txt
└── README.md
```

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Opciones disponibles:

| Opción | Descripción | Valor por defecto |
|---|---|---|
| `--x0` | Velocidades (km/h) para las que se calcula la predicción | `15 25 35` |
| `--nivel` | Nivel de confianza de los intervalos | `0.95` |
| `--salida` | Carpeta donde se guardan los resultados | `outputs` |
| `--con-ampliacion` | Agrega las pruebas formales que exceden el programa | desactivado |

Ejemplo:

```bash
python main.py --x0 20 30 40 --nivel 0.99 --salida resultados_99
```

## Datos y unidades

El dataset **no viene incluido** en el proyecto. La primera vez que se ejecuta `main.py` (o los tests):

1. Se descarga con la librería `pydataset` y se guarda en `data/cars.csv`, con sus unidades originales (velocidad en millas por hora y distancia en pies).
2. Se convierte a unidades SI y se guarda en `data/cars_si.csv`.

| Variable | Unidad original | Unidad SI | Factor de conversión |
|---|---|---|---|
| Velocidad ($x$) | mph | km/h | 1 mph = 1,609344 km/h |
| Distancia de frenado ($y$) | pies | m | 1 pie = 0,3048 m |

**Todos los cálculos, gráficos y conclusiones usan `data/cars_si.csv`.** En las ejecuciones siguientes se reutilizan los archivos locales. Para regenerarlos, basta con borrar la carpeta `data/` (o los dos CSV).

Las medidas adimensionales ($r$, $r^2$, estadísticos $t$ y $F$, valores p, residuos estandarizados) no cambian con las unidades. Los coeficientes, errores estándar e intervalos sí se expresan en SI.

> **Nota:** `pydataset` guarda una copia de sus datasets en la carpeta `~/.pydataset/` del usuario.

## Salidas

Al ejecutarse, el programa imprime el reporte en consola y genera:

- `outputs/resultados.txt`: reporte completo en texto.
- `outputs/resultados.json`: todos los resultados numéricos en formato estructurado.
- `outputs/img/fig1_dispersion.png`: diagrama de dispersión.
- `outputs/img/fig2_recta_intervalos.png`: recta ajustada con bandas de confianza y predicción.
- `outputs/img/fig3_diagnostico.png`: residuos vs. ajustados y gráfico Q-Q.
- `outputs/img/fig4_comparacion_residuos.png`: residuos estandarizados del modelo original vs. el modelo $\sqrt{Y}$.

Las figuras son las mismas que se usan en el informe en LaTeX.

## Criterio del diagnóstico

El diagnóstico de supuestos se apoya en el **análisis gráfico de residuos**, respaldado por medidas que usan solo contenidos de la asignatura:

- **Normalidad:** gráfico Q-Q y comparación de los residuos estandarizados con la regla empírica 68-95-99,7 de la distribución normal (Unidad II).
- **Homocedasticidad:** gráfico de residuos vs. ajustados y comparación del desvío estándar de los residuos por tramos de velocidad.
- **Atípicos:** residuos estandarizados con valor absoluto mayor que 2.

Las pruebas formales (Shapiro-Wilk, Breusch-Pagan y Durbin-Watson) están en un módulo separado, `src/ampliacion.py`. No se ejecutan por defecto: para incluirlas en una sección aparte del reporte, como confirmación numérica de lo observado en los gráficos, hay que usar `python main.py --con-ampliacion`.

## Validación

Los resultados se verifican contra la salida de referencia de R (`lm(dist ~ speed, data = cars)`). Los coeficientes de R, que están en mph y pies, se convierten a SI con los mismos factores del proyecto; las medidas adimensionales se comparan directamente.

```bash
python -m pytest -q
```

## Uso de la clase como librería

```python
from src.datos import cargar_datos
from src.regresion import RegresionLinealSimple

x, y = cargar_datos()                   # x en km/h, y en metros
modelo = RegresionLinealSimple(nivel_confianza=0.95).ajustar(x, y)

print(modelo.alpha, modelo.beta)        # coeficientes
print(modelo.correlacion())             # r, r², r² ajustado
print(modelo.inferencia_coeficientes()) # SE, t, p-valor, IC
print(modelo.anova())                   # tabla ANOVA
print(modelo.predecir(35))              # predicción a 35 km/h (en metros)
```

## Fuente de los datos

Ezekiel, M. (1930). *Methods of Correlation Analysis*. Wiley. Dataset `cars` del paquete `datasets` de R.

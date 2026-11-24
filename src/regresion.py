"""Regresion lineal simple por minimos cuadrados (notacion de la Unidad IV).

Modelo: Y = alpha + beta * x + e,  con e ~ N(0, sigma^2).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class RegresionLinealSimple:
    nivel_confianza: float = 0.95
    ajustado: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Ajuste por minimos cuadrados
    # ------------------------------------------------------------------
    def ajustar(self, x: np.ndarray, y: np.ndarray) -> "RegresionLinealSimple":
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("Se requieren al menos 3 pares (x, y) de igual longitud.")
        self.x, self.y, self.n = x, y, len(x)
        self.x_media, self.y_media = x.mean(), y.mean()

        self.Sxx = np.sum((x - self.x_media) ** 2)
        self.Syy = np.sum((y - self.y_media) ** 2)
        self.Sxy = np.sum((x - self.x_media) * (y - self.y_media))

        self.beta = self.Sxy / self.Sxx
        self.alpha = self.y_media - self.beta * self.x_media

        self.y_ajustado = self.alpha + self.beta * x
        self.residuos = y - self.y_ajustado

        # Sumas de cuadrados (SS_R = residual, SC_reg = explicada, STC = total)
        self.STC = self.Syy
        self.SS_R = np.sum(self.residuos ** 2)
        self.SC_reg = self.STC - self.SS_R

        self.gl = self.n - 2
        self.s2 = self.SS_R / self.gl
        self.s = np.sqrt(self.s2)
        self.t_critico = stats.t.ppf(1 - (1 - self.nivel_confianza) / 2, self.gl)
        self.ajustado = True
        return self

    def _verificar(self):
        if not self.ajustado:
            raise RuntimeError("Primero debe llamarse a ajustar(x, y).")

    # ------------------------------------------------------------------
    # Correlacion y determinacion
    # ------------------------------------------------------------------
    def correlacion(self) -> dict:
        self._verificar()
        r2 = self.SC_reg / self.STC
        r = np.sign(self.beta) * np.sqrt(r2)
        r2_aj = 1 - (1 - r2) * (self.n - 1) / (self.n - 2)
        return {"r": r, "r2": r2, "r2_ajustado": r2_aj}

    # ------------------------------------------------------------------
    # Inferencia sobre los coeficientes
    # ------------------------------------------------------------------
    def inferencia_coeficientes(self) -> dict:
        self._verificar()
        se_beta = self.s / np.sqrt(self.Sxx)
        se_alpha = self.s * np.sqrt(1 / self.n + self.x_media ** 2 / self.Sxx)
        resultado = {}
        for nombre, est, se in [("alpha", self.alpha, se_alpha), ("beta", self.beta, se_beta)]:
            t = est / se  # H0: parametro = 0
            p = 2 * stats.t.sf(abs(t), self.gl)
            resultado[nombre] = {
                "estimacion": est,
                "error_estandar": se,
                "t": t,
                "p_valor": p,
                "ic_inf": est - self.t_critico * se,
                "ic_sup": est + self.t_critico * se,
            }
        return resultado

    def ic_varianza_error(self) -> dict:
        """IC para sigma^2 usando SS_R / sigma^2 ~ Chi^2(n-2)."""
        self._verificar()
        a = 1 - self.nivel_confianza
        inf = self.SS_R / stats.chi2.ppf(1 - a / 2, self.gl)
        sup = self.SS_R / stats.chi2.ppf(a / 2, self.gl)
        return {"s2": self.s2, "s": self.s, "sigma2_inf": inf, "sigma2_sup": sup,
                "sigma_inf": np.sqrt(inf), "sigma_sup": np.sqrt(sup)}

    # ------------------------------------------------------------------
    # ANOVA
    # ------------------------------------------------------------------
    def anova(self) -> dict:
        self._verificar()
        cm_reg = self.SC_reg / 1
        cm_res = self.SS_R / self.gl
        F = cm_reg / cm_res
        return {
            "SC_regresion": self.SC_reg, "gl_regresion": 1, "CM_regresion": cm_reg,
            "SC_residual": self.SS_R, "gl_residual": self.gl, "CM_residual": cm_res,
            "SC_total": self.STC, "gl_total": self.n - 1,
            "F": F, "p_valor": stats.f.sf(F, 1, self.gl),
            "F_critico": stats.f.ppf(self.nivel_confianza, 1, self.gl),
        }

    # ------------------------------------------------------------------
    # Prediccion
    # ------------------------------------------------------------------
    def predecir(self, x0: float) -> dict:
        """Estimacion puntual, IC de la respuesta media e intervalo de prediccion."""
        self._verificar()
        y0 = self.alpha + self.beta * x0
        termino = (x0 - self.x_media) ** 2 / self.Sxx
        W = np.sqrt((1 + 1 / self.n + termino) * self.s2)   # formula de la Unidad IV
        Wm = np.sqrt((1 / self.n + termino) * self.s2)
        return {
            "x0": x0, "y0": y0,
            "ic_media_inf": y0 - self.t_critico * Wm, "ic_media_sup": y0 + self.t_critico * Wm,
            "ip_inf": y0 - self.t_critico * W, "ip_sup": y0 + self.t_critico * W,
            "W": W, "W_media": Wm,
        }

    def bandas(self, xs: np.ndarray) -> dict:
        """Bandas de confianza y prediccion sobre una grilla (para graficar)."""
        self._verificar()
        ys = self.alpha + self.beta * xs
        termino = (xs - self.x_media) ** 2 / self.Sxx
        W = np.sqrt((1 + 1 / self.n + termino) * self.s2)
        Wm = np.sqrt((1 / self.n + termino) * self.s2)
        return {"y": ys, "media_inf": ys - self.t_critico * Wm, "media_sup": ys + self.t_critico * Wm,
                "pred_inf": ys - self.t_critico * W, "pred_sup": ys + self.t_critico * W}

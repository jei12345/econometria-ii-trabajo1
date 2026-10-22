# -*- coding: utf-8 -*-
"""
Trabajo 1 - Econometría II (2026-2S)
Metodología Box-Jenkins aplicada a las ventas del comercio minorista de Colombia
(índice 2015=100, desestacionalizado, OCDE vía FRED, código COLSLRTTO01IXOBSAM).

Estructura de carpetas esperada (rutas relativas, funciona en cualquier computador):
    Trabajo_Econometria/
        data/     -> COLSLRTTO01IXOBSAM.csv
        code/     -> este script (analisis_arima.py)
        outputs/  -> se crea sola; aquí quedan gráficas y resultados
"""

# ---------------------------------------------------------------
# 0. LIBRERÍAS Y RUTAS
# ---------------------------------------------------------------
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore")

# La raíz del proyecto es la carpeta que contiene a "code". Así no hay que
# escribir rutas como C:\Users\... y el código corre en cualquier computador.
try:
    RAIZ = Path(__file__).resolve().parent.parent
except NameError:  # por si se corre desde Jupyter u otro entorno sin __file__
    RAIZ = Path.cwd().resolve()
    if RAIZ.name == "code":
        RAIZ = RAIZ.parent
ARCHIVO_DATOS = RAIZ / "data" / "COLSLRTTO01IXOBSAM.csv"
CARPETA_SALIDA = RAIZ / "outputs"
CARPETA_SALIDA.mkdir(exist_ok=True)

registro = []  # aquí se guardan los resultados escritos para el archivo de texto


def log(texto=""):
    """Muestra un texto en pantalla y lo guarda para resultados.txt."""
    print(texto)
    registro.append(str(texto))


# ---------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------
df = pd.read_csv(ARCHIVO_DATOS, parse_dates=["observation_date"])
serie = df.set_index("observation_date")["COLSLRTTO01IXOBSAM"].asfreq("MS")
log_serie = np.log(serie)          # logaritmo: trabajamos en porcentajes
dlog = log_serie.diff().dropna()   # primera diferencia del log (≈ crecimiento mensual)

log("=== 1. DATOS ===")
log(f"Observaciones: {len(serie)} | Desde {serie.index[0].date()} hasta {serie.index[-1].date()}")
log(f"Datos faltantes: {int(serie.isna().sum())}")
log(f"Mínimo: {serie.min():.2f} ({serie.idxmin().date()}) | Máximo: {serie.max():.2f} ({serie.idxmax().date()})")

# ---------------------------------------------------------------
# 2. IDENTIFICACIÓN
# ---------------------------------------------------------------
log("\n=== 2. IDENTIFICACIÓN ===")


def prueba_adf(x, nombre, regresion="c"):
    """ADF: H0 = hay raíz unitaria. Se rechaza si p < 0.05."""
    est, pv, rez, _, crit, _ = adfuller(x, regression=regresion, autolag="AIC")
    log(f"ADF {nombre}: estadístico={est:.3f} | p-valor={pv:.3f} | rezagos={rez} | crítico 5%={crit['5%']:.3f}")


def prueba_kpss(x, nombre, regresion="c"):
    """KPSS: H0 = la serie es estacionaria. Se rechaza si el estadístico > crítico."""
    est, pv, rez, crit = kpss(x, regression=regresion, nlags="auto")
    log(f"KPSS {nombre}: estadístico={est:.3f} | crítico 5%={crit['5%']:.3f}")


prueba_adf(log_serie, "log del índice (constante y tendencia)", "ct")
prueba_adf(log_serie, "log del índice (constante)", "c")
prueba_adf(dlog, "primera diferencia del log (constante)", "c")
prueba_kpss(log_serie, "log del índice (tendencia)", "ct")
prueba_kpss(dlog, "primera diferencia del log (constante)", "c")

# ACF y PACF de la serie diferenciada
rezagos = 24
valores_acf = acf(dlog, nlags=rezagos)
valores_pacf = pacf(dlog, nlags=rezagos, method="ywm")
banda = 1.96 / np.sqrt(len(dlog))
log(f"Banda de significancia ±{banda:.3f}")
log("ACF significativa en rezagos: " + str([k for k in range(1, rezagos + 1) if abs(valores_acf[k]) > banda]))
log("PACF significativa en rezagos: " + str([k for k in range(1, rezagos + 1) if abs(valores_pacf[k]) > banda]))

# Figura 1: la serie
fig, ejes = plt.subplots(2, 2, figsize=(12, 7))
serie.plot(ax=ejes[0, 0], color="#1f4e79")
ejes[0, 0].set_title("1. Serie original (índice 2015=100)")
log_serie.plot(ax=ejes[0, 1], color="#1f4e79")
ejes[0, 1].set_title("2. Logaritmo de la serie")
dlog.plot(ax=ejes[1, 0], color="#c0504d")
ejes[1, 0].axhline(0, color="k", lw=0.5)
ejes[1, 0].set_title("3. Primera diferencia del log (≈ crecimiento mensual)")
ejes[1, 1].hist(dlog, bins=25, color="#c0504d")
ejes[1, 1].set_title("4. Histograma de la serie diferenciada")
for eje in ejes.ravel():
    eje.set_xlabel("")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "fig1_serie.png", dpi=130)
plt.show()

# Figura 2: ACF y PACF
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
for eje, vals, titulo in [(ejes[0], valores_acf, "ACF de la serie diferenciada"),
                          (ejes[1], valores_pacf, "PACF de la serie diferenciada")]:
    eje.bar(range(1, rezagos + 1), vals[1:], color="#1f4e79")
    eje.axhline(banda, ls="--", c="r")
    eje.axhline(-banda, ls="--", c="r")
    eje.axhline(0, c="k", lw=0.5)
    eje.set_title(titulo)
    eje.set_xlabel("Rezago (meses)")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "fig2_acf_pacf.png", dpi=130)
plt.show()

# ---------------------------------------------------------------
# 3. ESTIMACIÓN
# ---------------------------------------------------------------
log("\n=== 3. ESTIMACIÓN ===")

# Variables auxiliares (dummies) para el COVID: una por mes, marzo a julio de 2020.
meses_covid = ["2020-03", "2020-04", "2020-05", "2020-06", "2020-07"]
dummies = pd.DataFrame(
    {f"covid_{m}": (dlog.index.strftime("%Y-%m") == m).astype(float) for m in meses_covid},
    index=dlog.index,
)


def ajustar(p, q):
    """ARIMA(p,1,q) = ARMA(p,q) con constante sobre la primera diferencia del log."""
    modelo = SARIMAX(dlog, exog=dummies, order=(p, 0, q), trend="c")
    return modelo.fit(disp=False, maxiter=200)


candidatos = [(2, 0), (0, 2), (2, 2), (1, 0), (0, 1), (1, 1)]
resultados = {}
filas = []
for p, q in candidatos:
    r = ajustar(p, q)
    resultados[(p, q)] = r
    estacionario = (len(r.arroots) == 0) or bool(np.all(np.abs(r.arroots) > 1))
    invertible = (len(r.maroots) == 0) or bool(np.all(np.abs(r.maroots) > 1))
    filas.append({"Modelo": f"ARIMA({p},1,{q})", "AIC": round(r.aic, 1), "BIC": round(r.bic, 1),
                  "Estacionario": estacionario, "Invertible": invertible})
tabla = pd.DataFrame(filas).sort_values("BIC").reset_index(drop=True)
log("Comparación de modelos (menor AIC/BIC es mejor):")
log(tabla.to_string(index=False))

# Modelo elegido: el más simple entre los que empatan en BIC
P, Q = 2, 0
modelo_final = resultados[(P, Q)]
log(f"\nModelo elegido: ARIMA({P},1,{Q}) con dummies de COVID")
log(modelo_final.summary().as_text())

# ---------------------------------------------------------------
# 4. VALIDACIÓN
# ---------------------------------------------------------------
log("\n=== 4. VALIDACIÓN ===")
residuos = modelo_final.resid.iloc[max(P, Q):]

lb = acorr_ljungbox(residuos, lags=[12, 24], model_df=P + Q)
log("Ljung-Box (H0: sin autocorrelación; queremos p > 0.05):")
log(lb.round(3).to_string())

jb, jb_p, asimetria, curtosis = jarque_bera(residuos)
log(f"Jarque-Bera (H0: normalidad): estadístico={jb:.2f} | p-valor={jb_p:.4f}")

lm, lm_p, _, _ = het_arch(residuos, nlags=4)
log(f"ARCH (H0: varianza constante): LM={lm:.2f} | p-valor={lm_p:.4f}")

log("Desviación estándar de los residuos por periodo:")
for inicio, fin in [("2013", "2020-02"), ("2020-08", "2021-12"), ("2022-01", "2024-03")]:
    log(f"  {inicio} a {fin}: {residuos[inicio:fin].std():.4f}")

# Comprobación de robustez: residuos del ARIMA(2,1,2), que tiene el mejor AIC
res_212 = resultados[(2, 2)].resid.iloc[2:]
log("\nComprobación de robustez - residuos del ARIMA(2,1,2):")
lb_212 = acorr_ljungbox(res_212, lags=[12, 24], model_df=4)
log(lb_212.round(3).to_string())
log(f"Jarque-Bera p-valor={jarque_bera(res_212)[1]:.4f} | ARCH p-valor={het_arch(res_212, nlags=4)[1]:.4f}")

# Figura 3: residuos
fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
ejes[0].plot(residuos.index, residuos.values, color="#c0504d")
ejes[0].axhline(0, c="k", lw=0.5)
ejes[0].set_title(f"Residuos del ARIMA({P},1,{Q}) con dummies COVID")
acf_res = acf(residuos, nlags=24)
banda_res = 1.96 / np.sqrt(len(residuos))
ejes[1].bar(range(1, 25), acf_res[1:], color="#1f4e79")
ejes[1].axhline(banda_res, ls="--", c="r")
ejes[1].axhline(-banda_res, ls="--", c="r")
ejes[1].set_title("ACF de los residuos")
ejes[1].set_xlabel("Rezago")
stats.probplot(residuos.values, dist="norm", plot=ejes[2])
ejes[2].set_title("QQ-plot (normalidad)")
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "fig3_residuos.png", dpi=130)
plt.show()

# ---------------------------------------------------------------
# 5. PRONÓSTICO A 10 MESES
# ---------------------------------------------------------------
log("\n=== 5. PRONÓSTICO ===")
H = 10
fechas_futuras = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq="MS")
dummies_futuras = pd.DataFrame(0.0, index=fechas_futuras, columns=dummies.columns)  # sin COVID a futuro

pred = modelo_final.get_forecast(steps=H, exog=dummies_futuras)
pred_dlog = np.asarray(pred.predicted_mean)
pred_log = log_serie.iloc[-1] + np.cumsum(pred_dlog)  # se suman los cambios al último dato

# Varianza del pronóstico acumulado (se usan los pesos psi del modelo AR)
psi = arma2ma(np.r_[1.0, -modelo_final.arparams], [1.0], H)
sigma2 = modelo_final.params["sigma2"]
varianza = sigma2 * np.cumsum(np.cumsum(psi) ** 2)
error = np.sqrt(varianza)

pronostico = pd.DataFrame({
    "pronostico": np.exp(pred_log),
    "inf95": np.exp(pred_log - 1.96 * error),
    "sup95": np.exp(pred_log + 1.96 * error),
}, index=fechas_futuras).round(2)
log(f"Último dato real: {serie.index[-1].date()} = {serie.iloc[-1]:.2f}")
log(pronostico.to_string())
pronostico.to_csv(CARPETA_SALIDA / "pronostico_10_meses.csv")

# Figura 4: pronóstico
fig, eje = plt.subplots(figsize=(11, 4.5))
historia = serie["2019":]
eje.plot(historia.index, historia.values, color="#1f4e79", label="Observado")
eje.plot(pronostico.index, pronostico["pronostico"], color="#c0504d", marker="o", ms=4, label="Pronóstico")
eje.fill_between(pronostico.index, pronostico["inf95"], pronostico["sup95"],
                 color="#c0504d", alpha=0.2, label="Intervalo 95 %")
eje.set_title("Pronóstico a 10 meses - ventas del comercio minorista de Colombia (índice 2015=100)")
eje.legend()
plt.tight_layout()
plt.savefig(CARPETA_SALIDA / "fig4_pronostico.png", dpi=130)
plt.show()

# ---------------------------------------------------------------
# 6. GUARDAR RESULTADOS
# ---------------------------------------------------------------
with open(CARPETA_SALIDA / "resultados.txt", "w", encoding="utf-8") as archivo:
    archivo.write("\n".join(registro))
log(f"\nListo. Gráficas y resultados guardados en: {CARPETA_SALIDA}")

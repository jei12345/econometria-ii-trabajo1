TRABAJO 1 - ECONOMETRIA II (2026-2S)
Metodologia Box-Jenkins aplicada al volumen del comercio minorista de Colombia, excepto
vehiculos automotores y motocicletas (indice 2015=100, desestacionalizado, OCDE via FRED,
codigo COLSLRTTO01IXOBSAM).
Autor: Jeison Pabon

ESTRUCTURA DE LA CARPETA
  data/     COLSLRTTO01IXOBSAM.csv   (base de datos descargada de FRED)
  code/     analisis_arima.py        (script completo del trabajo)
  outputs/  se crea sola al correr el script: graficas (fig1 a fig4),
            resultados.txt y pronostico_10_meses.csv

COMO EJECUTARLO
  1. Abrir Spyder (viene con Anaconda).
  2. Archivo > Abrir... y elegir code/analisis_arima.py
  3. Pulsar el boton verde de ejecutar (F5).
  No hace falta cambiar ninguna ruta: el script usa rutas relativas (pathlib)
  a partir de la carpeta donde esta guardado.

LIBRERIAS NECESARIAS (todas incluidas en Anaconda)
  pandas, numpy, matplotlib, scipy, statsmodels

FUENTE DE LOS DATOS
  OCDE, "Produccion, Ventas, Trabajos Iniciados y Pedidos: Volumen de Comercio Minorista:
  Actividad Economica: Comercio Minorista, Excepto Vehiculos Automotores y Motocicletas
  para Colombia" [COLSLRTTO01IXOBSAM], obtenido de FRED, Federal Reserve Bank of St. Louis.
  https://fred.stlouisfed.org/series/COLSLRTTO01IXOBSAM

# Validación realizada

## Reestructuración modular

La versión modular se verificó en Windows con Python 3.13.7, pandas 2.3.3,
NumPy 2.5.4, scikit-learn 1.8.0, matplotlib 3.11.2 y joblib 1.6.0.
Las tres pruebas de integración pasaron: entrenamiento multiclase y predicción
mediante `ids.py`, entrenamiento binario mediante `python -m ids_core`, y
muestreo con eliminación de conflictos mediante la interfaz compatible.
También se comprobó la compilación sintáctica y la ayuda de ambas entradas.
No se descargaron datos de Kaggle ni se validaron métricas con tráfico real.

## Validación anterior a la reestructuración

Fecha: 2 de noviembre de 2026.

Entorno de comprobación: Python 3.12.14, pandas 2.2.3, NumPy 2.3.5,
scikit-learn 1.8.0, matplotlib 3.10.8 y joblib 1.5.3, sobre Linux.

Se ejecutaron y aprobaron las tres pruebas de integración de `tests/test_pipeline.py`:

1. Entrenamiento multiclase, generación de modelo y matriz gráfica, predicción
   por bloques sin etiquetas y rechazo de variables ausentes.
2. Clasificación binaria con un archivo completo reservado para prueba;
   comprobación de procedencia y conteos de evaluación.
3. Muestreo a través de todos los archivos, límites por clase y eliminación
   de vectores de características con etiquetas contradictorias.

Se verificaron la compilación sintáctica y la ayuda de la interfaz CLI.
Las pruebas incluyen espacios en encabezados, una columna adicional presente
sólo en un archivo, encabezados repetidos, faltantes, infinitos y texto no numérico.

No se descargó el conjunto completo ni se entrenó sobre sus datos reales en esta
entrega. La autenticación/descarga de Kaggle y la ejecución nativa en Windows no
se probaron. La compatibilidad efectiva del esquema se comprobará al ejecutar
`inspeccionar` y `entrenar` con los CSV descargados. No se presentan porcentajes
de desempeño sobre CSE-CIC-IDS2018 ni se incluye un modelo preentrenado.

Los rangos de requirements permiten instalar versiones compatibles; sólo las
versiones indicadas arriba se ejecutaron en esta validación. El modelo guarda
las versiones del entorno de entrenamiento para facilitar su reproducción.

# Validación realizada

## Ejemplos ejecutables del cuerpo de predicción

Pasaron los 14 casos de `pytest tests/test_api.py -q` (2,84 segundos).
La nueva prueba obtiene los ejemplos de OpenAPI y los envía a `/predict`,
verificando también la demostración de imputación y los puertos enteros.
Además se cargó el modelo local `resultados_binario/modelo_ids.joblib`:
los dos ejemplos generados con sus 70 variables devolvieron HTTP 200 mediante
TestClient. Los ejemplos usan las medianas serializadas y no reentrenan.
Persistió la advertencia de deprecación del TestClient de Starlette/HTTPX.

## API FastAPI y Uvicorn

Se ejecutó `.venv\Scripts\python.exe -m pytest -q`: **58 pruebas aprobadas
en 17,49 segundos** en Windows con FastAPI 0.162.0, Uvicorn 0.54.0 y HTTPX
0.28.1. Los 13 casos nuevos verifican carga única del modelo, inferencia sin
`fit`, equivalencia con el pipeline, metadatos, salud, documentación OpenAPI,
validación de peticiones, errores LAZY múltiples, configuración por variable
de entorno y rechazo del arranque cuando no existe modelo.
Las rutas se probaron con `TestClient` y el ciclo de vida completo; se verificó
también el ejecutable con `python -m uvicorn --version`. No se dejó un servidor
en ejecución ni se realizaron pruebas de carga o de red con Uvicorn.
Se registraron dos advertencias de dependencias: concatenación Pandera/pandas
y deprecación del uso de HTTPX en el TestClient de Starlette.

## Preprocesamiento dentro del modelo serializado

Se ejecutó `.venv\Scripts\python.exe -m pytest -q`: **45 pruebas aprobadas
en 16,73 segundos**, con una advertencia interna de Pandera/pandas.
Se comprobó que el pipeline recargado acepta un DataFrame sin preparar,
normaliza encabezados y orden de variables, convierte tipos, limpia texto,
infinitos y desbordamientos y conserva las medianas del imputador.
Las probabilidades coinciden con las obtenidas al aplicar los pasos por
separado. Se bloquearon los métodos `fit` para confirmar que la inferencia
no ajusta ningún paso, y se verificó que no modifica el DataFrame original.
Se comprobó también la predicción con el formato anterior de artefacto.

## Serialización y reutilización del modelo

Se ejecutó `.venv\Scripts\python.exe -m pytest -q`: **43 pruebas aprobadas
en 15,86 segundos**, con el mismo FutureWarning interno de Pandera/pandas.
Los siete casos nuevos comprueban reutilización sin CSV de entrenamiento ni
llamadas a `fit`, predicciones equivalentes a las del pipeline cargado,
reentrenamiento explícito, rechazo de un modo incompatible, modelos ausentes
o corruptos, artefactos sin pipeline y conservación del modelo anterior ante
un fallo de serialización. El modelo y los reportes no cambian al reutilizarlos.

## Suite pytest — 4 de noviembre de 2026

Se ejecutó `.venv\Scripts\python.exe -m pytest -v` en Windows, con
Python 3.13.7 y pytest 9.2.0: **36 pruebas aprobadas en 15,60 segundos**.
La suite incluye los siete casos previos y 29 casos nuevos parametrizados
o unitarios para la CLI, limpieza numérica, lectura por bloques, encabezados,
límites de puertos/protocolo y etiquetas inválidas.
Se verifican también entrenamiento binario/multiclase, persistencia del modelo,
predicción, rechazo de esquemas inválidos, errores LAZY y conservación de
la salida anterior ante un bloque inválido. Se emplean CSV sintéticos locales.
Se registró un FutureWarning interno de Pandera/pandas al reunir errores;
no hubo fallos. Las dependencias de pruebas están en `requirements-dev.txt`
y el descubrimiento de pruebas está configurado en `pytest.ini`.

## Contrato Pandera y validación LAZY — 4 de noviembre de 2026

Con Pandera 0.34.1 en el entorno Windows de la versión modular, pasaron las
siete pruebas de `python -m unittest discover -s tests -v`.
Las cuatro pruebas del contrato verifican tipos y reglas simultáneos,
columnas ausentes, valores infinitos, puertos fraccionarios/fuera de rango,
protocolo, etiquetas binarias, procedencia y filas sin variables disponibles.
También comprueban que los faltantes parciales y columnas adicionales se aceptan.
La integración verifica que un segundo bloque inválido de predicción informa
varios fallos y conserva la salida anterior, eliminando el temporal.
Se mantuvieron las pruebas de entrenamiento, predicción y muestreo y se
verificó la compilación sintáctica. Pandera emitió un FutureWarning interno
de concatenación de pandas al reunir errores; las comprobaciones pasaron.

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

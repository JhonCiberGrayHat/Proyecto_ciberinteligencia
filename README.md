# Clasificación de tráfico de red con Python y VS Code

Proyecto experimental de detección de intrusiones sobre los CSV de
CSE-CIC-IDS2018 publicados en Kaggle. No requiere GPU.

Fuente solicitada: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
Documentación original: https://www.unb.ca/cic/datasets/ids-2018.html
Descarga Python: https://github.com/Kaggle/kagglehub

## 1. Instalar en Windows

Instale Python 3.11 o 3.12 y las extensiones Python y Python Debugger de Microsoft
para VS Code. Descomprima este proyecto y abra **la carpeta IDS_Trafico_ML**
mediante Archivo > Abrir carpeta. Ejecute en su terminal PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si instaló Python 3.12, sustituya `-3.11` por `-3.12`.
Seleccione el intérprete `.venv` con Ctrl+Shift+P > Python: Select Interpreter.
Los comandos siguientes usan la ruta del intérprete para evitar depender de la
activación de PowerShell. En Linux/macOS: `python3 -m venv .venv` y use
`.venv/bin/python` en lugar de `.\.venv\Scripts\python.exe`.

## 2. Obtener los datos

```powershell
.\.venv\Scripts\python.exe ids.py descargar
.\.venv\Scripts\python.exe ids.py inspeccionar
```

`descargar` utiliza `kagglehub.dataset_download` y registra en `data_path.txt`
la ruta devuelta. No copia los CSV dentro del proyecto. La descarga completa
puede ocupar varios GB y el tiempo depende de su conexión.

Si Kaggle pide autenticación o aceptar términos, hágalo en su cuenta; para
iniciar sesión con la biblioteca:

```powershell
.\.venv\Scripts\python.exe -c "import kagglehub; kagglehub.login()"
```

Alternativa manual: descargue el ZIP desde Kaggle y **descomprímalo** en
`data`. Use `--data .\data` en los comandos de inspección o entrenamiento.
No coloque CSV de resultados dentro de la carpeta de entrada: se leen todos
sus CSV, incluidas las subcarpetas. No se descarga información al ejecutar
`entrenar`.

`inspeccionar` informa las columnas reales de cada CSV; sus conteos de etiquetas
corresponden únicamente a las primeras 2000 filas, no a todo el conjunto.

## 3. Entrenar

El modo predeterminado conserva las etiquetas originales de ataque:

```powershell
.\.venv\Scripts\python.exe ids.py entrenar --mode multiclase
```

Para datos descargados manualmente:

```powershell
.\.venv\Scripts\python.exe ids.py entrenar --data .\data --mode multiclase
```

Para distinguir únicamente Benign frente a Attack y guardar un modelo separado:

```powershell
.\.venv\Scripts\python.exe ids.py entrenar --data .\data --mode binario --output resultados_binario
```

También puede pulsar F5 y elegir una configuración incluida en `.vscode/launch.json`.
No se incluye un modelo preentrenado: se genera en su equipo.

### Memoria, muestreo y limpieza

- Se recorren **todos** los CSV en bloques de 25 000 filas; no se cargan completos.
- Se conserva una muestra aleatoria uniforme de hasta 5000 filas por clase de
  ataque y 20 000 benignas. El algoritmo mantiene prioridades aleatorias por
  clase a través de todos los bloques; no selecciona sólo las primeras filas.
- El muestreo reduce memoria y representación de clases frecuentes; las
  métricas describen esta muestra, no la prevalencia real del tráfico.
- Se exige `Label` en cada CSV; se eliminan espacios/BOM de encabezados y filas
  con etiqueta vacía o encabezados `Label` repetidos.
- Sólo se usan columnas comunes a todos los archivos, excluyendo IP, identificador
  de flujo, marca de tiempo, puerto de origen e índices. El puerto de destino
  y protocolo pueden seguir siendo predictores; revise posibles atajos del
  entorno de captura antes de atribuir capacidad de generalización.
- Los valores no numéricos, infinitos o fuera del rango float32 pasan a faltantes.
  Se retiran filas sin ninguna característica numérica válida.
- Se eliminan vectores duplicados y vectores con etiquetas contradictorias
  **en la muestra** antes de dividirla. No se afirma depurar todo el dataset.
- Variables constantes o sin datos se seleccionan usando sólo entrenamiento.
  Los faltantes se imputan con la mediana aprendida sólo en entrenamiento.
- Se entrena un Random Forest de 150 árboles, profundidad máxima 24, ponderación
  de clases y dos hilos CPU. No necesita normalización de escala para el bosque.

Si tiene poca memoria, reduzca límites y tamaño de bloque:

```powershell
.\.venv\Scripts\python.exe ids.py entrenar --data .\data --class-cap 1000 --benign-cap 5000 --chunk-size 5000 --trees 100 --jobs 1
```

Leer por bloques limita la carga de entrada; el bosque se ajusta sobre la muestra
acumulada en RAM, no mediante aprendizaje incremental. La descarga, lectura,
muestra y entrenamiento pueden seguir siendo costosos. Los límites se aplican
antes de la depuración, por lo que los conteos finales pueden ser menores.
Las clases originales se conservan, incluida cualquier grafía particular de
Kaggle; etiquetas diferentes no se fusionan automáticamente excepto Benign.

### Separación de entrenamiento y prueba

Por defecto se realiza una división aleatoria estratificada 80/20 con semilla 42.
Es una evaluación **exploratoria**: flujos correlacionados de una misma sesión,
equipo o día pueden aparecer en ambos grupos incluso después de quitar duplicados.
No demuestra detección en redes reales, ataques desconocidos ni desempeño futuro.
Clases con menos de dos muestras o particiones demasiado pequeñas producen un
error explícito; aumente límites o use clasificación binaria según su objetivo.

Para una prueba por archivo/día, inspeccione los nombres y reserve uno completo:

```powershell
.\.venv\Scripts\python.exe ids.py entrenar --data .\data --test-file "03-02-2018.csv" --output resultados_por_dia
```

El nombre debe coincidir exactamente con un archivo existente. Este comando
aparta sus filas retenidas en la muestra; no evalúa necesariamente todo el día.
El día de prueba puede contener clases ausentes en entrenamiento: quedan
registradas y cuentan como errores si no se reconocen. La deduplicación se
realiza antes de separar, por lo que repeticiones entre días se conservan una
sola vez. Para comparar investigación formal, diseñe particiones por sesión o
varios días, documente las clases de cada partición y mantenga una prueba externa
sin utilizar para elegir parámetros. Este proyecto no implementa ajuste automático
ni validación cruzada y no sustituye esa validación.

## 4. Resultados

Cada ejecución escribe en `resultados` (o en `--output`):

| Archivo | Contenido |
| --- | --- |
| `modelo_ids.joblib` | Pipeline entrenado, variables, modo y versiones |
| `metricas.json` | Accuracy, balanced accuracy, F1 macro/ponderado, conteos y alarmas |
| `reporte_clases.txt` / `.json` | Precision, recall y F1 por etiqueta |
| `matriz_confusion.csv` / `.png` | Filas reales y columnas predichas |
| `importancia_variables.csv` | Importancias del bosque; no representan causalidad |
| `predicciones_prueba.csv` | Etiquetas reales/predichas y archivo de procedencia |
| `auditoria.json` | Archivos, tamaños, muestreo, exclusiones, parámetros y versiones |

La tasa de falsas alarmas se calcula sobre las filas Benign de prueba;
`attack_recall` mide ataques detectados como cualquier ataque. Si falta una clase
binaria, la tasa correspondiente se informa como `null`. No confunda estos
indicadores con acierto en el tipo exacto de ataque.
Las importancias por impureza pueden favorecer variables con muchos valores.
Use carpetas `--output` distintas para conservar experimentos previos.

## 5. Clasificar nuevos flujos

```powershell
.\.venv\Scripts\python.exe ids.py predecir --model .\resultados\modelo_ids.joblib --csv .\nuevos_flujos.csv --output .\predicciones.csv
```

El CSV debe contener **las mismas características de flujo, unidades y nombres**
que se usaron al entrenar, con o sin `Label`. Se valida la presencia de variables;
no se inventan las que falten. No se exige el orden original de columnas.
La salida contiene el registro lógico de origen, predicción y `score_modelo`,
que es el máximo de las probabilidades del bosque y **no una certeza calibrada**.
Si se incluye Label, se conserva como referencia, sin usarla para predecir.
La salida no reproduce todas las columnas originales; `row_csv` permite asociarla.

Este programa clasifica **CSV de flujos**; no captura paquetes ni bloquea tráfico.
Para uso en red necesitaría una etapa de captura y extracción compatible con
CICFlowMeter y su configuración, seguida de inferencia, registro y alertas.
Cambiar el extractor o las unidades requiere verificar compatibilidad y validar
de nuevo el modelo. Cargue únicamente modelos joblib de fuentes confiables.

## 6. Verificación local

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas usan datos sintéticos pequeños para verificar el funcionamiento;
no constituyen métricas sobre Kaggle. Consulte VALIDACION.md.

## Fuentes y atribución

- Dataset original: CSE-CIC-IDS2018, Canadian Institute for Cybersecurity y
  Communications Security Establishment: https://www.unb.ca/cic/datasets/ids-2018.html
- Copia solicitada: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
- I. Sharafaldin, A. H. Lashkari y A. A. Ghorbani, “Toward Generating a New
  Intrusion Detection Dataset and Intrusion Traffic Characterization”, ICISSP, 2018
  (referencia indicada por la documentación del dataset).
- Descarga: https://github.com/Kaggle/kagglehub
- Clasificador: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

No se redistribuyen los datos ni credenciales. Revise las condiciones de uso y
cite las fuentes al publicar resultados.

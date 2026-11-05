# Clasificación de tráfico de red con Python y VS Code

Proyecto experimental de detección de intrusiones sobre los CSV de
CSE-CIC-IDS2018 publicados en Kaggle. No requiere GPU.

## Organización del código

`ids.py` es el punto de entrada y conserva las funciones públicas anteriores.
La implementación está dividida en el paquete `ids_core`:

| Módulo | Responsabilidad |
| --- | --- |
| `config.py` | Dataset, rutas, registro y columnas reservadas/excluidas |
| `contracts.py` | Contratos de tipos y reglas con Pandera y validación LAZY |
| `artifacts.py` | Serialización atómica y carga del pipeline entrenado |
| `preprocessing.py` | Transformador serializable de limpieza y selección de variables |
| `api.py` | API FastAPI de inferencia con el modelo serializado |
| `data.py` | Lectura de CSV por bloques y normalización numérica |
| `datasets.py` | Descarga de Kaggle e inspección de archivos |
| `sampling.py` | Muestreo por clase, limpieza y auditoría de la muestra |
| `training.py` | Partición, ajuste, evaluación y guardado del experimento |
| `prediction.py` | Carga del modelo e inferencia por bloques |
| `reporting.py` | Escritura JSON y gráfico de la matriz de confusión |
| `cli.py` | Argumentos, validación y ejecución de los comandos |

Los módulos de entrenamiento y predicción comparten la lectura y conversión
de datos. La CLI coordina las operaciones; los módulos de datos no dependen
de ella. Importar el paquete no ejecuta comandos ni descarga datos.

Se mantienen los comandos `python ids.py ...`; también se puede ejecutar
`python -m ids_core ...`. Para integrar el muestreo desde Python, use
`from ids_core.sampling import load_sample`. La función `ids_core.cli.main`
acepta una lista opcional de argumentos para su ejecución desde otro programa.
`data_path.txt` sigue ubicado en la raíz del proyecto.

El notebook de `EDA_analisis_modelos` contiene un análisis exploratorio
independiente (PCA, agrupamiento y comparación de clasificadores); conserva
su flujo interactivo y no es una dependencia de la aplicación CLI.

### Contrato de datos con Pandera

El contrato dinámico de `ids_core/contracts.py` exige las características
comunes de entrenamiento o las variables guardadas en el modelo de predicción.
Se aplica **después de la limpieza numérica**, antes de ajustar el modelo y
antes de predecir cada bloque. No modifica ni convierte los tipos: exige
`float32` y valores finitos; permite `NaN` individuales para la imputación,
pero rechaza filas sin ninguna variable disponible. El texto no numérico,
infinitos y desbordamientos del CSV siguen convirtiéndose en faltantes durante
la limpieza previa; el contrato no certifica que esos valores originales sean
válidos. Las filas completamente vacías se eliminan del entrenamiento y se
rechazan en predicción.

Los puertos reconocidos (`Dst Port`, `Destination Port`, `Src Port` y
`Source Port`, ignorando espacios y mayúsculas) deben ser enteros entre 0 y
65535; `Protocol`, entre 0 y 255. Se permiten valores faltantes en estas
columnas. Las demás variables no tienen límites físicos específicos definidos.
En entrenamiento, `Label` debe ser texto no vacío, sin espacios extremos ni
marcadores reservados; el modo binario permite únicamente `Benign` y `Attack`.
`__source` exige una procedencia textual no vacía. En predicción, `Label` es
solo una referencia opcional y no se valida ni se utiliza como predictor.

Todas las validaciones usan `schema.validate(frame, lazy=True)`: Pandera
acumula los fallos del DataFrame validado en lugar de detenerse en el primero.
`DataContractError` conserva el detalle en `failure_cases` (columna, índice,
valor y chequeo) y la CLI lo muestra con código de salida 1. En inferencia,
los errores se acumulan por bloque; el proceso se detiene en el primer bloque
inválido y no publica una salida parcial. LAZY no implica lectura diferida
ni acumulación de errores de todos los archivos.

```python
from ids_core.contracts import DataContractError, feature_schema, validate_data
from ids_core.data import numeric

features = ["Dst Port", "Flow Duration"]
try:
    validated = validate_data(numeric(frame[features]), feature_schema(features))
except DataContractError as exc:
    print(exc.failure_cases)
```

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

El entrenamiento se realiza una vez por carpeta `--output`. Guarda en
`modelo_ids.joblib` el pipeline completo (preprocesador, imputador ajustado y Random Forest),
las variables en su orden, el modo, la semilla y las versiones. Si el archivo
ya existe, `entrenar` comprueba que se puede cargar y omite la lectura del
dataset y el entrenamiento, conservando los reportes existentes.

Para clasificar nuevos CSV, ejecute `predecir` con ese modelo serializado;
no necesita los CSV de entrenamiento ni volver a calcular la imputación.
Los parámetros nuevos de entrenamiento no se aplican al modelo existente.
Para cambiar el modo, los datos o los parámetros, use otra carpeta de salida
o solicite el reemplazo explícito con `--reentrenar`:

```powershell
.\.venv\Scripts\python.exe ids.py entrenar --data .\data --output resultados
.\.venv\Scripts\python.exe ids.py predecir --model .\resultados\modelo_ids.joblib --csv .\nuevos_flujos.csv --output .\predicciones.csv
# Solo cuando se quiera ajustar de nuevo y reemplazar el experimento:
.\.venv\Scripts\python.exe ids.py entrenar --data .\data --output resultados --reentrenar
```

Un modelo ilegible o de un modo distinto produce un error; no se reentrena
automáticamente. La escritura del modelo utiliza un archivo temporal y un
reemplazo al completar la serialización. Los reportes se escriben por separado.

El pipeline serializado de los modelos nuevos recibe directamente un DataFrame
con datos de CSV: normaliza BOM/espacios en encabezados, selecciona y ordena
las variables guardadas, convierte a `float32`, sustituye texto no numérico,
infinitos y desbordamientos por `NaN`, y valida el contrato Pandera con LAZY.
Después aplica las medianas aprendidas en entrenamiento y ejecuta el bosque.
La inferencia no vuelve a ajustar ninguno de estos pasos. También puede usar
`load_model(ruta)["model"].predict_proba(frame)` desde Python.

Los artefactos anteriores siguen funcionando mediante la limpieza compatible
de `predecir`. Para incorporar el preprocesador dentro de esos archivos ya
existentes, genere una nueva versión con `entrenar --reentrenar`.
El muestreo, la depuración de etiquetas y la deduplicación son operaciones del
dataset de entrenamiento; no se aplican a los registros que se van a predecir.

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

## 6. API con FastAPI y Uvicorn

Instale las dependencias y ejecute desde la raíz del proyecto. Si ya dispone
del modelo binario, seleccione su ruta con `IDS_MODEL_PATH`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:IDS_MODEL_PATH = ".\resultados_binario\modelo_ids.joblib"
.\.venv\Scripts\python.exe -m uvicorn ids_core.api:app --host 127.0.0.1 --port 8000
```

Sin esa variable se utiliza `resultados/modelo_ids.joblib`. La ruta relativa
se resuelve desde el directorio de ejecución. El servidor carga el modelo
una vez por proceso al iniciar; si falta o es inválido, no inicia ni entrena
automáticamente. Después de reemplazar el modelo, reinicie el servidor.
La carga utiliza el [ciclo de vida de FastAPI](https://fastapi.tiangolo.com/advanced/events/).

| Endpoint | Función |
| --- | --- |
| `GET /health` | Estado del servicio y confirmación del modelo cargado |
| `GET /model` | Variables requeridas, clases, modo y versión del preprocesamiento |
| `POST /predict` | Predicciones para un lote JSON de 1 a 1000 registros |
| `GET /docs` | Documentación interactiva para probar peticiones |

Abra http://127.0.0.1:8000/docs y consulte `GET /model` para conocer los nombres
exactos de las variables de su modelo. Cada registro debe incluirlas todas;
use `null` para un valor faltante que deba imputarse. Las columnas adicionales
se ignoran al predecir. Los tipos, la limpieza y la validación LAZY se aplican
con el mismo flujo de inferencia utilizado para CSV.

Para una demostración inmediata, despliegue `POST /predict` y pulse **Try it
out**. El cuerpo del request ya incluye `flujo_demo`, generado con todas las
variables y las medianas del modelo cargado; pulse **Execute** para predecir.
El selector de ejemplos también ofrece `imputacion_demo`, un lote de dos
registros con un `null` para demostrar la imputación (si hay al menos dos
variables). Los puertos y el protocolo se ajustan a enteros válidos.
Los ejemplos son sintéticos y no garantizan una predicción Benign o Attack.
Reinicie Uvicorn para que aparezcan los nuevos ejemplos.

Ejemplo de petición (adapte las variables a las devueltas por `/model`):

```json
{
  "records": [
    {"Dst Port": 443, "Flow Duration": 1200, "Flow Byts/s": null}
  ]
}
```

La respuesta contiene `count` y `predictions`, con `row` (índice desde cero),
`prediccion` y `score_modelo` por registro. El score conserva el significado
de probabilidad máxima del bosque sin calibración. Las peticiones inválidas
devuelven HTTP 422; los fallos Pandera incluyen `detail.failure_cases` con los
errores acumulados del lote. Una petición no produce resultados parciales.
La API solo realiza inferencia: la creación y actualización del modelo se
ejecutan con los comandos de entrenamiento.

## 7. Verificación local

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -v
```

Pytest ejecuta las pruebas existentes de integración y contratos, además de
casos parametrizados de límites de puertos/protocolo, etiquetas inválidas,
limpieza numérica, lectura por bloques y errores de argumentos de la CLI.
Los archivos temporales se crean durante las pruebas y no requieren Kaggle,
credenciales ni conexión a internet. Para ejecutar solo los contratos:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_contracts.py tests/test_contract_boundaries.py -v
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

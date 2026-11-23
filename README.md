# Estandarización de Pipelines y Control de Calidad de Datos

**MLOps Santoto | Semana 2 | Universidad Santo Tomás**  
**Autora:** María Fernanda Medina Serrato  
**Proyecto:** clasificación del conjunto de datos Iris con Random Forest y MLflow.

## 1. Objetivo y continuidad del proyecto

Reproducir y automatizar la preparación, validación, entrenamiento y evaluación del mismo modelo de la Semana 1 usando **un solo comando**. El modelo conserva la configuración seleccionada en aquella actividad: `RandomForestClassifier(n_estimators=100, max_depth=1, random_state=42)`. Los datos proceden de `sklearn.datasets.load_iris()` y la división de entrenamiento/prueba es 80/20 con `random_state=42`.

La validación de datos **no importa librerías de terceros**: `validar_datos.py` usa únicamente `argparse`, `csv`, `json`, `math` y `pathlib` (biblioteca estándar). `scikit-learn`, `joblib` y `mlflow` se utilizan exclusivamente para preparación, modelado, almacenamiento y trazabilidad.

## 2. Estructura del repositorio

```text
MLOps_Santoto_Semana2/
├── pipeline.py                   # Orquestación multiplataforma: un comando
├── validar_datos.py              # Comprobaciones sin librerías externas
├── contrato_datos.json           # Contrato de entrada para futura inferencia
├── Makefile                      # Alternativa opcional para entornos con make
├── requirements.txt              # Versiones fijadas de dependencias
├── .gitignore                    # Ignora artefactos regenerables y credenciales
├── README.md                     # Documentación técnica
├── src/
│   ├── __init__.py
│   ├── prepare.py                # Descarga/carga Iris y crea el split 80/20
│   └── train.py                  # Random Forest, métricas y registro en MLflow
├── tests/
│   ├── test_validar_datos.py    # Pruebas de tipos, nulos, rangos y esquema
│   └── test_modelo.py           # Pruebas del modelo y resultados
├── data/processed/              # Archivos CSV regenerados, ignorados por Git
├── models/                      # model.pkl generado, ignorado por Git
└── results/                     # metrics.json generado, ignorado por Git
```

Los directorios vacíos contienen `.gitkeep` para que Git conserve la estructura. Los CSV, modelos, bases SQLite de MLflow y resultados derivados no se suben al repositorio: se generan al ejecutar el pipeline.

## 3. Requisitos e instalación desde cero

- Python 3.11 o superior (se recomienda 3.11–3.13); Git y acceso a Internet para descargar dependencias la primera vez. La version fijada de scikit-learn requiere Python 3.11+.
- Windows 10/11 con PowerShell, macOS o Linux.
- **No se requiere Make**: el comando principal es compatible con los tres sistemas.

Clonar el repositorio personal donde se hayan publicado estos archivos (sustituya la URL por la real):

```bash
git clone <URL_REAL_DEL_REPOSITORIO>
cd MLOps_Santoto_Semana2
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en **Windows PowerShell**:

```powershell
.\.venv\Scripts\Activate.ps1
```

O en **macOS/Linux**:

```bash
source .venv/bin/activate
```

Instalar las dependencias con versiones fijadas:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si Windows no reconoce `python`, probar `py -3` para crear el entorno, activar y luego usar `python`. Si PowerShell bloquea la activación, usar el ejecutable directamente: `.\.venv\Scripts\python.exe pipeline.py` (sin cambiar políticas globales).

## 4. Ejecutar el pipeline con un solo comando

Ubicarse en la raíz del proyecto con el entorno activado y ejecutar:

```bash
python pipeline.py
```

El orquestador ejecuta en orden:

1. **Preparar:** carga Iris desde `scikit-learn` y crea `data/processed/train.csv` (120 filas) y `data/processed/test.csv` (30 filas), con semilla 42.
2. **Validar:** contrasta los CSV con `contrato_datos.json` y revisa tipo numérico, ausencia de nulos, finitud, intervalos y clase objetivo. **Si hay errores, detiene el pipeline antes de entrenar.**
3. **Entrenar y evaluar:** ajusta Random Forest con 100 árboles y profundidad 1; calcula accuracy de entrenamiento y prueba, precisión, recall y F1 macro, y brecha de generalización; registra parámetros, métricas, contrato y modelo en MLflow bajo el experimento `Proyecto_Final_Grado`.
4. **Probar:** ejecuta la suite `unittest` (biblioteca estándar). **Cualquier falla devuelve código de salida distinto de cero.**

El modelo queda en `models/model.pkl` y las métricas en `results/metrics.json`. Este último archivo registra un `mlflow_run_id` **real** cuando MLflow finaliza exitosamente; conservar ese ID para la actividad de serving con FastAPI.

### Comandos complementarios

```bash
python validar_datos.py                      # Validar CSV antes de entrenar
python -m unittest discover -s tests -v     # Ejecutar solo las pruebas
python -m src.prepare                       # Regenerar el dataset
python -m src.train                         # Solo entrenamiento/MLflow
```

En macOS/Linux, `make all` ejecuta el mismo pipeline (si Make está instalado); `make validar` y `make test` son tareas adicionales.

**Modo técnico de prueba sin MLflow:** `python pipeline.py --sin-mlflow`. Úselo únicamente cuando no sea posible instalar la dependencia; comprueba preparación, validación, entrenamiento local y pruebas, **pero no genera RUN_ID ni registra un experimento**. Para la entrega académica y la actividad posterior debe ejecutarse sin esta bandera.

## 5. Política de calidad de datos (sin dependencias externas)

La función reutilizable `validar_registro(registro, contrato)` devuelve una lista de errores; devuelve `[]` para registros válidos. Se verifican:

| Campo JSON | Tipo | Mínimo (cm) | Máximo (cm) | Nulos |
|---|---|---:|---:|---|
| `sepal_length` | número | 4.0 | 8.0 | No |
| `sepal_width` | número | 2.0 | 4.5 | No |
| `petal_length` | número | 1.0 | 7.0 | No |
| `petal_width` | número | 0.1 | 2.5 | No |

Se rechazan: campos faltantes/extra, `null`, strings, booleanos, valores no finitos (`NaN`/infinito) y números fuera de rango. La validación de CSV también exige columnas exactas y objetivo `species` entero en `{0, 1, 2}`. Los rangos son límites de control para este dataset educativo, no una regla botánica universal. El preprocesamiento usa los nombres del contrato y mantiene un orden fijo para evitar inconsistencias entrenamiento/inferencia.

**Ejemplo válido** (aceptado):

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

**Ejemplo inválido** (rechazado):

```json
{"sepal_length": 99.0, "sepal_width": null, "petal_length": 1.4, "petal_width": "0.2"}
```

El documento normativo oficial del contrato es `contrato_datos.json` (`contract_version` = `1.0.0`). Para una futura API, reutilizar las claves y la semántica del contrato en la petición JSON; el campo `species` es el resultado esperado, **no** una variable de entrada para inferencia.

## 6. Trazabilidad y evidencia reproducible

Iniciar la interfaz web de MLflow en otra terminal desde la raíz del proyecto:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir `http://127.0.0.1:5000` y seleccionar `Proyecto_Final_Grado`. La ejecución de esta semana se identifica como `Semana2_Profundidad_1_Pipeline`. Es posible revisar `n_estimators`, `max_depth`, `random_state`, accuracy, F1, precision, recall, contrato y el modelo `iris_model`. El resultado numérico depende de la versión instalada de las librerías y de la partición, por lo que debe registrarse la salida **real**; con las versiones y la semilla del proyecto, la prueba local realizada obtuvo 1.0000 de accuracy sobre 30 muestras (no constituye garantía de generalización sobre datos externos).

El proyecto utiliza `mlflow==3.6.0`, `scikit-learn==1.8.0` y `joblib==1.5.3`. Para depurar diferencias entre equipos comprobar:

```bash
python --version
python -m pip freeze
```

La base de datos `mlflow.db` y el modelo serializado quedan fuera de Git. Nunca subir credenciales ni datos personales al repositorio público.

## 7. Tests y comprobación de errores

La carpeta `tests/` incluye pruebas para: ejemplo válido e inválido del contrato, campos ausentes y extra, tipos incorrectos (incluidos `bool`), nulos, `NaN`, infinito, rangos, CSV vacío, clase fuera de catálogo, y consistencia del modelo y partición. Las pruebas del modelo se omiten explícitamente cuando todavía no se ha ejecutado el pipeline; durante la ejecución completa, deben aprobarse todas.

Para verificar que el control de calidad bloquea datos defectuosos, alterar una **copia temporal** de un CSV con un valor fuera de rango y ejecutar `python validar_datos.py --train <RUTA_TEMPORAL> --test data/processed/test.csv`: el proceso debe devolver código distinto de cero, y no se debe continuar entrenando. No modifique los originales para este ensayo.

### Solución de problemas

- `ModuleNotFoundError: mlflow`: activar el entorno y ejecutar `python -m pip install -r requirements.txt`.
- `FileNotFoundError` de `train.csv` o `test.csv`: ejecutar `python -m src.prepare` o el pipeline completo.
- `VALIDACION FALLIDA`: inspeccionar columna, fila y causa detallada; corregir la fuente, no deshabilitar el validador.
- Error de MLflow o SQLite: comprobar permisos de escritura en el proyecto y que se ejecute desde su carpeta raíz.
- No aparece `mlflow_run_id`: comprobar que se usó **sin** `--sin-mlflow` y que el entrenamiento terminó sin error.

## 8. Publicación y entrega

1. Crear o abrir **el repositorio personal de GitHub** e incorporar todos los archivos fuente de este proyecto (`git add .`, `git commit -m "feat: pipeline y calidad semana 2"`, `git push`). Verificar que `pipeline.py`, `validar_datos.py`, `contrato_datos.json` y `README.md` se ven desde una sesión sin iniciar.
2. Crear en el Drive/SharePoint **académico** una carpeta llamada `MLOps_Santoto_Semana2_Evidencias` y colocar un documento `ENLACE_REPOSITORIO.txt` con la **URL real** de GitHub (opcionalmente añadir una copia del paquete ZIP de código, como respaldo).
3. Configurar acceso de la carpeta en **Cualquier persona con el enlace → Lector** o la opción equivalente que permita ver sin solicitar autorización, respetando la política institucional. Confirmarlo en una ventana privada.
4. Pegar esa URL de carpeta en la segunda página del Word con portada APA y subir **solo el Word** a “Estandarización de Pipelines y Control de Calidad de Datos”, apartado **Actuar** del campus.

La carpeta debe contener un enlace al **repositorio propio**, no al repositorio oficial del docente. No copiar ni inventar URLs de GitHub o Drive.

## 9. Referencias técnicas

- DIALVEBE. (s. f.). *MLOps Open Source: Hands-on Lab* (Semana 2: Pipelines y Calidad). https://github.com/DIALVEBE/mlops-course-santoto/blob/main/index.md
- MLflow. (s. f.). *MLflow tracking*. https://mlflow.org/docs/latest/ml/tracking/
- Scikit-learn. (s. f.). *RandomForestClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Python Software Foundation. (s. f.). *unittest — Unit testing framework*. https://docs.python.org/3/library/unittest.html


## Verificación local de esta entrega (23 noviembre 2026)

Con `python pipeline.py --sin-mlflow` se validaron 150 filas del conjunto Iris (120 entrenamiento y 30 prueba). Después, `python -m unittest discover -s tests -v` completó 20 pruebas. La evaluación local registró `train_accuracy=0.95` y `test_accuracy=1.0` en 30 observaciones de prueba. **No se registró una ejecución de MLflow durante esta verificación**; para esa evidencia se requiere ejecutar `python pipeline.py` con MLflow instalado. Estas cifras no reemplazan la comparación de los cuatro runs de la actividad 1.1.

[README.md](https://github.com/user-attachments/files/35298634/README.md)
# Estandarización de Pipelines y Control de Calidad de Datos

**MLOps Santoto | Universidad Santo Tomás**  
**Autora:** María Fernanda Medina Serrato  
**Proyecto:** Clasificación del conjunto de datos Iris mediante Random Forest y MLflow.

## 1. Descripción del proyecto

Este proyecto integra la preparación de datos, las comprobaciones de calidad, el entrenamiento, la evaluación y las pruebas del modelo de clasificación Iris en un flujo reproducible. El proceso completo se ejecuta mediante un solo comando y detiene el entrenamiento cuando detecta datos que no cumplen las reglas definidas.

Se utiliza `RandomForestClassifier(n_estimators=100, max_depth=1, random_state=42)`, con una partición del conjunto de datos de 80 % para entrenamiento y 20 % para prueba. La fuente es `sklearn.datasets.load_iris()`.

Las validaciones se implementan con módulos de la biblioteca estándar de Python (`argparse`, `csv`, `json`, `math` y `pathlib`). Las dependencias externas se reservan para la preparación de datos, el entrenamiento del modelo, su almacenamiento y el seguimiento de experimentos.

## 2. Estructura del proyecto

```text
mlops-santoto-semana2/
├── .gitignore
├── Makefile
├── README.md
├── contrato_datos.json
├── pipeline.py
├── requirements.txt
├── validar_datos.py
├── src/
│   ├── __init__.py
│   ├── prepare.py
│   └── train.py
└── tests/
    ├── test_modelo.py
    └── test_validar_datos.py
```

Al ejecutar el proyecto se generan archivos de datos procesados en `data/processed/`, el modelo en `models/` y las métricas en `results/`. Estos artefactos no forman parte del código fuente y se excluyen del control de versiones.

## 3. Requisitos e instalación

Se requiere Python 3.11 o superior, Git y conexión a Internet para instalar las dependencias por primera vez. El comando principal funciona en Windows, macOS y Linux y no necesita `make`.

Clonar el repositorio:

```bash
git clone https://github.com/MafeSerratoo/mlops-santoto-semana2.git
cd mlops-santoto-semana2
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows, si el comando `python` no está disponible, se puede utilizar `py -3` para crear el entorno. Si PowerShell impide la activación, es posible ejecutar directamente ` .\.venv\Scripts\python.exe pipeline.py ` sin modificar la política global de ejecución.

## 4. Ejecución del pipeline

Desde la raíz del proyecto, con el entorno virtual activo:

```bash
python pipeline.py
```

El proceso realiza cuatro etapas:

1. **Preparación:** carga Iris y crea los archivos `data/processed/train.csv` y `data/processed/test.csv`, con una partición reproducible de 120 y 30 registros, respectivamente.
2. **Validación:** comprueba la estructura y los valores de los CSV frente a las reglas de `contrato_datos.json`. Cuando encuentra un error, interrumpe el proceso antes del entrenamiento.
3. **Entrenamiento y evaluación:** entrena Random Forest, calcula métricas de clasificación y registra los parámetros, las métricas y los artefactos en MLflow.
4. **Pruebas:** ejecuta las pruebas automatizadas incluidas en `tests/` y reporta los errores encontrados.

El modelo se guarda en `models/model.pkl` y las métricas en `results/metrics.json`. Cuando el registro con MLflow se completa, las métricas incluyen su identificador de ejecución (`mlflow_run_id`).

### Comandos adicionales

```bash
python validar_datos.py
python -m unittest discover -s tests -v
python -m src.prepare
python -m src.train
```

También está disponible `python pipeline.py --sin-mlflow` para comprobar el procesamiento y el entrenamiento local cuando no se dispone de MLflow. Esta opción no registra el experimento ni produce un identificador de ejecución en MLflow.

## 5. Control de calidad de datos

La función `validar_registro(registro, contrato)` revisa las características de cada observación y devuelve los errores que encuentra. Si el registro cumple todas las condiciones, devuelve una lista vacía (`[]`).

| Variable | Tipo esperado | Mínimo (cm) | Máximo (cm) | ¿Admite nulos? |
|---|---|---:|---:|---|
| `sepal_length` | Número | 4.0 | 8.0 | No |
| `sepal_width` | Número | 2.0 | 4.5 | No |
| `petal_length` | Número | 1.0 | 7.0 | No |
| `petal_width` | Número | 0.1 | 2.5 | No |

Se comprueba la presencia de los campos esperados, la ausencia de campos adicionales, los tipos numéricos, la ausencia de valores nulos y no finitos (`NaN` e infinito), y el cumplimiento de los rangos definidos. Para los archivos de entrenamiento y prueba también se revisa que `species` corresponda a una clase válida (`0`, `1` o `2`).

Estos rangos se utilizan como controles de calidad para el conjunto Iris empleado en el proyecto; no representan límites universales para las especies de flores.

### Ejemplo de entrada válida

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Ejemplo de entrada inválida

```json
{
  "sepal_length": 99.0,
  "sepal_width": null,
  "petal_length": 1.4,
  "petal_width": "0.2"
}
```

La segunda entrada incumple el rango de `sepal_length`, contiene un valor nulo en `sepal_width` y utiliza un texto en lugar de un número para `petal_width`.

El archivo `contrato_datos.json` centraliza las reglas de validación y define la versión del contrato (`1.0.0`). La variable `species` es la etiqueta del modelo, por lo que no se incluye entre las características de entrada para una predicción.

## 6. Seguimiento de experimentos con MLflow

Después de ejecutar el pipeline, se puede abrir la interfaz de MLflow desde otra terminal, ubicada en la raíz del proyecto:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

La interfaz queda disponible en `http://127.0.0.1:5000`. El experimento se identifica como `Proyecto_Final_Grado` y la ejecución del pipeline como `Semana2_Profundidad_1_Pipeline`.

En MLflow se pueden consultar los hiperparámetros (`n_estimators`, `max_depth`, `random_state`), las métricas de clasificación y los artefactos registrados. Los resultados deben interpretarse a partir de la ejecución realizada en cada entorno; una métrica alta sobre el conjunto de prueba no garantiza el mismo desempeño con datos nuevos.

## 7. Pruebas automatizadas

Las pruebas de `tests/` contemplan registros válidos e inválidos, valores nulos, campos adicionales o ausentes, tipos incorrectos, límites de rango y consistencia de los datos y del modelo.

Ejecutar las pruebas de forma independiente:

```bash
python -m unittest discover -s tests -v
```

Para comprobar que el validador bloquea datos defectuosos, se puede modificar **una copia** de un archivo CSV e introducir un valor fuera de rango. Después, ejecutar:

```bash
python validar_datos.py --train RUTA_DEL_CSV_DE_PRUEBA --test data/processed/test.csv
```

El comportamiento esperado es que la validación informe el problema y devuelva un código de salida diferente de cero. Los archivos originales no deben alterarse para realizar esta comprobación.

## 8. Solución de problemas

| Situación | Comprobación recomendada |
|---|---|
| `ModuleNotFoundError: mlflow` | Activar el entorno virtual e instalar `requirements.txt`. |
| No aparecen `train.csv` o `test.csv` | Ejecutar `python -m src.prepare` o el pipeline completo. |
| La validación informa un error | Revisar la fila, la columna y la regla incumplida. |
| MLflow no inicia | Comprobar la instalación, los permisos de escritura y el directorio desde el que se ejecuta el comando. |
| No se genera `mlflow_run_id` | Verificar que el pipeline se haya ejecutado sin `--sin-mlflow` y haya finalizado correctamente. |

## 9. Referencias técnicas

- DIALVEBE. (s. f.). *MLOps Open Source: Hands-on Lab*. https://github.com/DIALVEBE/mlops-course-santoto
- MLflow. (s. f.). *MLflow tracking*. https://mlflow.org/docs/latest/ml/tracking/
- Scikit-learn. (s. f.). *RandomForestClassifier*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Python Software Foundation. (s. f.). *unittest — Unit testing framework*. https://docs.python.org/3/library/unittest.html

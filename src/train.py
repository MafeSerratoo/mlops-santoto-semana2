"""Entrena el Random Forest elegido en Semana 1, evalua y registra en MLflow."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from validar_datos import cargar_contrato, validar_csv

RAIZ = Path(__file__).resolve().parents[1]
SEMILLA = 42
ARBOLES = 100
PROFUNDIDAD = 1


def leer_datos(ruta: Path, columnas: list[str]) -> tuple[list[list[float]], list[int]]:
    entradas = []
    etiquetas = []
    with ruta.open("r", encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            entradas.append([float(fila[nombre]) for nombre in columnas])
            etiquetas.append(int(fila["species"]))
    return entradas, etiquetas


def entrenar_modelo(train: Path, test: Path, contrato: dict):
    for ruta in (train, test):
        cantidad, problemas = validar_csv(ruta, contrato)
        if problemas:
            raise ValueError(f"Calidad de datos no aprobada en {ruta}: {problemas}")
    columnas = contrato["required"]
    x_train, y_train = leer_datos(train, columnas)
    x_test, y_test = leer_datos(test, columnas)
    modelo = RandomForestClassifier(
        n_estimators=ARBOLES, max_depth=PROFUNDIDAD, random_state=SEMILLA
    )
    modelo.fit(x_train, y_train)
    preds = modelo.predict(x_test)
    score_train = accuracy_score(y_train, modelo.predict(x_train))
    score_test = accuracy_score(y_test, preds)
    metricas = {
        "train_accuracy": round(float(score_train), 6),
        "test_accuracy": round(float(score_test), 6),
        "test_precision_macro": round(float(precision_score(y_test, preds, average="macro", zero_division=0)), 6),
        "test_recall_macro": round(float(recall_score(y_test, preds, average="macro", zero_division=0)), 6),
        "test_f1_macro": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 6),
        "generalization_gap": round(float(score_train - score_test), 6),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return modelo, metricas


def registrar_mlflow(modelo, metricas: dict, contrato: dict) -> str:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MLflow no esta instalado. Ejecute python -m pip install -r requirements.txt"
        ) from exc
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Proyecto_Final_Grado")
    with mlflow.start_run(run_name="Semana2_Profundidad_1_Pipeline") as run:
        mlflow.log_params({
            "n_estimators": ARBOLES,
            "max_depth": PROFUNDIDAD,
            "random_state": SEMILLA,
            "contract_version": contrato["contract_version"],
            "dataset": "sklearn_iris",
            "test_size": 0.2,
        })
        mlflow.log_metrics({k: float(v) for k, v in metricas.items()})
        mlflow.log_dict(contrato, "contrato_datos.json")
        mlflow.sklearn.log_model(sk_model=modelo, name="iris_model")
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena, evalua, versiona y registra el modelo")
    parser.add_argument("--sin-mlflow", action="store_true", help="Solo para pruebas tecnicas offline")
    args = parser.parse_args()
    contrato = cargar_contrato(RAIZ / "contrato_datos.json")
    ruta_train = RAIZ / "data/processed/train.csv"
    ruta_test = RAIZ / "data/processed/test.csv"
    modelo, metricas = entrenar_modelo(ruta_train, ruta_test, contrato)
    # Registrar la ejecucion en MLflow antes de guardar los resultados locales.
    run_id = None if args.sin_mlflow else registrar_mlflow(modelo, metricas, contrato)
    (RAIZ / "models").mkdir(exist_ok=True)
    (RAIZ / "results").mkdir(exist_ok=True)
    joblib.dump(modelo, RAIZ / "models/model.pkl")
    reporte = {
        **metricas,
        "config": {"model": "RandomForestClassifier", "n_estimators": ARBOLES,
                   "max_depth": PROFUNDIDAD, "random_state": SEMILLA},
        "data_contract": contrato["contract_name"],
        "data_contract_version": contrato["contract_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
        "mlflow_logged": run_id is not None,
    }
    (RAIZ / "results/metrics.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Modelo entrenado: accuracy prueba={metricas['test_accuracy']:.4f}; F1 macro={metricas['test_f1_macro']:.4f}")
    print(f"RUN_ID: {run_id if run_id else 'NO GENERADO (modo tecnico offline)'}")


if __name__ == "__main__":
    main()

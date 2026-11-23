"""Prepara los datos Iris oficiales y la particion original de Semana 1."""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from validar_datos import cargar_contrato

RAIZ = Path(__file__).resolve().parents[1]


def guardar_csv(ruta: Path, entradas, salidas, columnas: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([*columnas, "species"])
        for valores, clase in zip(entradas, salidas):
            escritor.writerow([*[float(v) for v in valores], int(clase)])


def main() -> None:
    contrato = cargar_contrato(RAIZ / "contrato_datos.json")
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    columnas = contrato["required"]
    guardar_csv(RAIZ / "data/processed/train.csv", x_train, y_train, columnas)
    guardar_csv(RAIZ / "data/processed/test.csv", x_test, y_test, columnas)
    print(f"Datos preparados: {len(x_train)} entrenamiento, {len(x_test)} prueba")


if __name__ == "__main__":
    main()

"""Orquestador de un solo comando: python pipeline.py (Windows/macOS/Linux)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def ejecutar(comando: list[str], etiqueta: str) -> None:
    print(f"\n>>> {etiqueta}", flush=True)
    subprocess.run(comando, cwd=RAIZ, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline completo de Semana 2")
    parser.add_argument("--sin-mlflow", action="store_true", help="Solo para validacion offline sin MLflow")
    args = parser.parse_args()
    python = sys.executable
    etapas = [
        ([python, "-m", "src.prepare"], "1/4 Preparar datos Iris 80/20"),
        ([python, "validar_datos.py"], "2/4 Control de calidad previo a entrenamiento"),
        ([python, "-m", "src.train", *(["--sin-mlflow"] if args.sin_mlflow else [])], "3/4 Entrenar/evaluar y registrar el experimento"),
        ([python, "-m", "unittest", "discover", "-s", "tests", "-v"], "4/4 Ejecutar pruebas automaticas"),
    ]
    try:
        for comando, etiqueta in etapas:
            ejecutar(comando, etiqueta)
    except subprocess.CalledProcessError as exc:
        print(f"\nPIPELINE DETENIDO: etapa fallida (codigo {exc.returncode})", file=sys.stderr)
        return exc.returncode or 1
    print("\nPIPELINE FINALIZADO: calidad y pruebas aprobadas", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

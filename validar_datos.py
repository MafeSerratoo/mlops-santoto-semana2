"""Validador del contrato de Iris, SIN librerias externas.

Usa exclusivamente modulos de la biblioteca estandar de Python.
La funcion validar_registro permite validar registros individuales.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def cargar_contrato(ruta: str | Path = RAIZ / "contrato_datos.json") -> dict:
    with Path(ruta).open("r", encoding="utf-8") as archivo:
        contrato = json.load(archivo)
    requeridos = contrato.get("required")
    campos = contrato.get("features")
    if not isinstance(requeridos, list) or not isinstance(campos, dict):
        raise ValueError("El contrato no define required y features correctamente")
    if set(requeridos) != set(campos) or len(requeridos) != len(set(requeridos)):
        raise ValueError("El contrato tiene campos requeridos inconsistentes")
    for nombre, regla in campos.items():
        if regla.get("type") != "number" or regla.get("min") > regla.get("max"):
            raise ValueError(f"Regla de {nombre} invalida")
    return contrato


def validar_registro(registro: object, contrato: dict) -> list[str]:
    """Retorna lista de problemas; lista vacia significa entrada valida.

    No convierte strings en numeros ni acepta booleanos como valores numericos.
    """
    if not isinstance(registro, dict):
        return ["La entrada debe ser un objeto JSON (diccionario)"]
    problemas = []
    requeridos = contrato["required"]
    for nombre in requeridos:
        if nombre not in registro:
            problemas.append(f"Falta el campo obligatorio: {nombre}")
            continue
        valor = registro[nombre]
        if valor is None:
            problemas.append(f"{nombre}: valor nulo")
            continue
        if type(valor) not in (int, float):
            problemas.append(f"{nombre}: se esperaba numero, recibido {type(valor).__name__}")
            continue
        if not math.isfinite(valor):
            problemas.append(f"{nombre}: valor NaN o infinito")
            continue
        regla = contrato["features"][nombre]
        if not regla["min"] <= valor <= regla["max"]:
            problemas.append(f"{nombre}: {valor} fuera del rango [{regla['min']}, {regla['max']}]")
    if contrato.get("additional_properties") is False:
        for nombre in registro:
            if nombre not in requeridos:
                problemas.append(f"Campo inesperado: {nombre}")
    return problemas


def validar_ejemplos(contrato: dict) -> tuple[bool, bool]:
    ejemplos = contrato["examples"]
    entrada_correcta = not validar_registro(ejemplos["valid"], contrato)
    entrada_incorrecta = bool(validar_registro(ejemplos["invalid"], contrato))
    return entrada_correcta, entrada_incorrecta


def validar_csv(ruta: str | Path, contrato: dict) -> tuple[int, list[str]]:
    """Lee CSV con stdlib, comprueba schema, tipos, nulos, rangos y target.

    Los valores CSV son texto por definicion; se convierten a float despues de
    validar su contenido y se rechazan las cadenas no convertibles y no finitas.
    """
    ruta = Path(ruta)
    errores: list[str] = []
    total = 0
    cabecera = [*contrato["required"], contrato["target"]["name"]]
    permitidas = set(contrato["target"]["allowed_values"])
    try:
        with ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
            lector = csv.DictReader(archivo)
            if lector.fieldnames != cabecera:
                return 0, [f"{ruta.name}: columnas incorrectas (esperado {cabecera}, recibido {lector.fieldnames})"]
            for num_fila, fila in enumerate(lector, start=2):
                total += 1
                registro = {}
                for nombre in contrato["required"]:
                    bruto = fila.get(nombre)
                    if bruto is None or not bruto.strip():
                        errores.append(f"{ruta.name} fila {num_fila}: {nombre} nulo o vacio")
                        continue
                    try:
                        registro[nombre] = float(bruto)
                    except (TypeError, ValueError):
                        errores.append(f"{ruta.name} fila {num_fila}: {nombre} no numerico")
                if len(registro) == len(contrato["required"]):
                    errores.extend(f"{ruta.name} fila {num_fila}: {e}" for e in validar_registro(registro, contrato))
                clase = fila.get(contrato["target"]["name"])
                if clase is None or not clase.strip().isdigit() or int(clase) not in permitidas:
                    errores.append(f"{ruta.name} fila {num_fila}: especie invalida {clase!r}")
                if len(errores) >= 25:
                    errores.append("Se detuvo la enumeracion de errores en 25 para limitar la salida")
                    break
    except (OSError, UnicodeError, csv.Error) as exc:
        errores.append(f"No se puede abrir o procesar {ruta}: {exc}")
    if total == 0:
        errores.append(f"{ruta.name}: dataset sin registros")
    return total, errores


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida el contrato y CSV sin dependencias externas")
    parser.add_argument("--train", type=Path, default=RAIZ / "data/processed/train.csv")
    parser.add_argument("--test", type=Path, default=RAIZ / "data/processed/test.csv")
    args = parser.parse_args()
    contrato = cargar_contrato()
    valido, invalido_rechazado = validar_ejemplos(contrato)
    if not valido or not invalido_rechazado:
        print("ERROR: ejemplos del contrato no cumplen su descripcion")
        return 1
    print("Contrato JSON: ejemplo valido ACEPTADO; ejemplo invalido RECHAZADO")
    suma = 0
    errores_total = []
    for ruta in (args.train, args.test):
        registros, errores = validar_csv(ruta, contrato)
        suma += registros
        errores_total.extend(errores)
        print(f"{ruta.name}: {registros} filas; {len(errores)} errores")
    for error in errores_total:
        print(f"  ERROR: {error}")
    if errores_total:
        print("VALIDACION FALLIDA. No se debe entrenar.")
        return 1
    print(f"VALIDACION CORRECTA: {suma} registros aprobados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

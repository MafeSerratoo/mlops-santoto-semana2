"""Casos de calidad de datos e inferencia; usan unittest de la biblioteca estandar."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from validar_datos import validar_registro, validar_csv, validar_ejemplos, cargar_contrato

RAIZ = Path(__file__).resolve().parents[1]
CONTRATO = cargar_contrato(RAIZ / "contrato_datos.json")
VALIDO = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


class TestContrato(unittest.TestCase):
    def test_valida_ejemplo_valido(self):
        self.assertEqual(validar_registro(VALIDO, CONTRATO), [])

    def test_ejemplos_contractuales(self):
        correcto, incorrecto = validar_ejemplos(CONTRATO)
        self.assertTrue(correcto)
        self.assertTrue(incorrecto)

    def test_rechaza_nulo(self):
        self.assertTrue(validar_registro({**VALIDO, "sepal_width": None}, CONTRATO))

    def test_rechaza_variable_faltante(self):
        incompleto = dict(VALIDO)
        del incompleto["petal_width"]
        self.assertTrue(validar_registro(incompleto, CONTRATO))

    def test_rechaza_campo_extra(self):
        self.assertTrue(validar_registro({**VALIDO, "edad": 27}, CONTRATO))

    def test_rechaza_cadena_como_numero(self):
        self.assertTrue(validar_registro({**VALIDO, "petal_length": "1.4"}, CONTRATO))

    def test_rechaza_booleano(self):
        self.assertTrue(validar_registro({**VALIDO, "petal_width": True}, CONTRATO))

    def test_rechaza_nan(self):
        self.assertTrue(validar_registro({**VALIDO, "petal_width": math.nan}, CONTRATO))

    def test_rechaza_infinito(self):
        self.assertTrue(validar_registro({**VALIDO, "petal_width": math.inf}, CONTRATO))

    def test_rechaza_fuera_de_rango(self):
        self.assertTrue(validar_registro({**VALIDO, "sepal_length": 99.0}, CONTRATO))

    def test_acepta_extremos(self):
        extremos = {clave: CONTRATO["features"][clave]["min"] for clave in CONTRATO["required"]}
        self.assertEqual(validar_registro(extremos, CONTRATO), [])

    def test_rechaza_lista_como_registro(self):
        self.assertTrue(validar_registro([1, 2, 3, 4], CONTRATO))

    def _archivo(self, filas, columnas=None):
        temporal = tempfile.TemporaryDirectory()
        ruta = Path(temporal.name) / "dataset.csv"
        nombres = columnas or [*CONTRATO["required"], "species"]
        with ruta.open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=nombres)
            escritor.writeheader()
            escritor.writerows(filas)
        return temporal, ruta

    def test_csv_correcto(self):
        temporal, ruta = self._archivo([{**VALIDO, "species": 0}])
        try:
            self.assertEqual(validar_csv(ruta, CONTRATO), (1, []))
        finally:
            temporal.cleanup()

    def test_csv_rechaza_objetivo_fuera_de_clases(self):
        temporal, ruta = self._archivo([{**VALIDO, "species": 5}])
        try:
            self.assertTrue(validar_csv(ruta, CONTRATO)[1])
        finally:
            temporal.cleanup()

    def test_csv_rechaza_archivo_vacio(self):
        temporal, ruta = self._archivo([])
        try:
            self.assertTrue(validar_csv(ruta, CONTRATO)[1])
        finally:
            temporal.cleanup()

    def test_csv_rechaza_valores_vacios(self):
        temporal, ruta = self._archivo([{**VALIDO, "sepal_width": "", "species": 0}])
        try:
            self.assertTrue(validar_csv(ruta, CONTRATO)[1])
        finally:
            temporal.cleanup()


if __name__ == "__main__":
    unittest.main()

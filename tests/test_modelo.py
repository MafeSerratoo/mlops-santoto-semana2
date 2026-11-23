"""Verificaciones integrales posteriores al entrenamiento del pipeline."""
import json
import unittest
from pathlib import Path

import joblib
from sklearn.datasets import load_iris

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "models" / "model.pkl"
METRICAS = RAIZ / "results" / "metrics.json"


class TestModeloEntrenado(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODELO.exists() or not METRICAS.exists():
            raise unittest.SkipTest("Primero ejecute python pipeline.py para generar artefactos")
        cls.modelo = joblib.load(MODELO)
        cls.metricas = json.loads(METRICAS.read_text(encoding="utf-8"))

    def test_configuracion_es_la_seleccionada_en_semana_uno(self):
        self.assertEqual(self.modelo.n_estimators, 100)
        self.assertEqual(self.modelo.max_depth, 1)
        self.assertEqual(self.modelo.random_state, 42)

    def test_exactitud_minima_en_prueba(self):
        self.assertGreaterEqual(self.metricas["test_accuracy"], 0.80)
        self.assertLessEqual(self.metricas["test_accuracy"], 1.0)

    def test_division_datos_120_30(self):
        self.assertEqual(self.metricas["n_train"], 120)
        self.assertEqual(self.metricas["n_test"], 30)

    def test_predicciones_clases_iris(self):
        datos = load_iris()
        predicciones = self.modelo.predict(datos.data[:5])
        self.assertTrue(all(int(x) in [0, 1, 2] for x in predicciones))


if __name__ == "__main__":
    unittest.main()

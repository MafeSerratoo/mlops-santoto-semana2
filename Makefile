.PHONY: all preparar validar entrenar test
all:
	python pipeline.py
preparar:
	python -m src.prepare
validar:
	python validar_datos.py
entrenar:
	python -m src.train
test:
	python -m unittest discover -s tests -v

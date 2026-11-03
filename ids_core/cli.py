"""Interfaz de comandos del IDS."""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import LOG
from .datasets import download, inspect_data
from .prediction import predict
from .training import train


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo.")
    return n


def main(argv=None):
    parser = argparse.ArgumentParser(description="IDS experimental con CSE-CIC-IDS2018. Ejecutar: python ids.py --help.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("descargar", help="Descargar Kaggle a la caché de kagglehub")
    p.set_defaults(func=download)
    p = sub.add_parser("inspeccionar", help="Listar archivos, columnas y primeras etiquetas")
    p.add_argument("--data")
    p.set_defaults(func=inspect_data)
    p = sub.add_parser("entrenar", help="Entrenar y evaluar Random Forest")
    p.add_argument("--data", help="Carpeta descomprimida o CSV individual")
    p.add_argument("--mode", choices=["binario", "multiclase"], default="multiclase")
    p.add_argument("--class-cap", type=positive, default=5000, help="Muestra máxima por clase de ataque")
    p.add_argument("--benign-cap", type=positive, default=20000)
    p.add_argument("--chunk-size", type=positive, default=25000)
    p.add_argument("--test-size", type=float, default=.2)
    p.add_argument("--test-file", help="Nombre exacto de un CSV reservado para prueba por día")
    p.add_argument("--trees", type=positive, default=150)
    p.add_argument("--jobs", type=int, default=2, help="Hilos CPU; -1 utiliza todos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="resultados")
    p.set_defaults(func=train)
    p = sub.add_parser("predecir", help="Clasificar CSV nuevos con el modelo guardado")
    p.add_argument("--model", default="resultados/modelo_ids.joblib")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="predicciones.csv")
    p.add_argument("--chunk-size", type=positive, default=25000)
    p.set_defaults(func=predict)
    args = parser.parse_args(argv)
    if args.command == "entrenar" and (not 0 < args.test_size < 1 or args.jobs == 0 or args.jobs < -1):
        parser.error("--test-size debe estar entre 0 y 1; --jobs debe ser -1 o positivo.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    try:
        args.func(args)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0

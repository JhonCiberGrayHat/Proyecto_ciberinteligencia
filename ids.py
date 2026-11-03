"""Entrada compatible del IDS. Ejecutar: python ids.py --help.

La implementación se organiza por responsabilidades dentro de ids_core.
Se conservan las funciones públicas para los consumidores existentes.
"""
from ids_core.cli import main, positive
from ids_core.config import DATASET, EXCLUDE, LOG, META, ROOT
from ids_core.data import (canonical, chunks, csv_files, header, normalize_columns,
                           numeric, resolve_data)
from ids_core.datasets import download, inspect_data
from ids_core.prediction import predict
from ids_core.reporting import plot_confusion, write_json
from ids_core.sampling import load_sample
from ids_core.training import train


if __name__ == "__main__":
    raise SystemExit(main())

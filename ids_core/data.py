"""Lectura y normalización de CSV y resolución de rutas."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import META, ROOT


def canonical(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_columns(frame):
    names = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
    if len(names) != len(set(names)):
        raise ValueError("Encabezados duplicados después de quitar espacios.")
    if META.intersection(names):
        raise ValueError("El CSV utiliza nombres de columnas reservados.")
    frame.columns = names
    return frame


def csv_files(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise ValueError(f"No se encontraron CSV en {path}. Descomprima primero el ZIP.")
    return files


def header(path):
    return normalize_columns(pd.read_csv(path, nrows=0, encoding="utf-8-sig"))


def chunks(path, size):
    # dtype=str evita inferencias diferentes entre bloques y permite retirar
    # encabezados repetidos antes de convertir a valores numéricos.
    for frame in pd.read_csv(path, dtype=str, chunksize=size, encoding="utf-8-sig"):
        yield normalize_columns(frame)


def numeric(frame):
    out = frame.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    # El bosque convierte internamente a float32: los desbordamientos se tratan
    # como faltantes, nunca como un número arbitrariamente grande.
    return out.mask(out.abs() > np.finfo(np.float32).max).astype("float32")


def resolve_data(value):
    if value:
        return Path(value).expanduser().resolve()
    pointer = ROOT / "data_path.txt"
    if pointer.exists():
        return Path(pointer.read_text(encoding="utf-8").strip())
    raise ValueError("Use --data RUTA o ejecute primero: python ids.py descargar")

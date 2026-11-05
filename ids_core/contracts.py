"""Contratos de los datos normalizados que consume el modelo."""
from __future__ import annotations

import re

import numpy as np
import pandera.pandas as pa
from pandera.errors import SchemaErrors


class DataContractError(ValueError):
    """Errores acumulados del contrato, accesibles como failure_cases."""

    def __init__(self, context, errors):
        self.failure_cases = errors.failure_cases.copy()
        super().__init__(
            f"Contrato de datos incumplido ({context}):\n"
            + self.failure_cases.to_string(index=False)
        )


def feature_schema(features, *, labeled=False, mode="multiclase"):
    """Exige las variables del modelo sin inventar ni convertir columnas.

    Los NaN individuales se permiten para la imputación posterior. No se
    permiten infinitos ni filas sin ninguna variable numérica disponible.
    Las columnas adicionales se conservan (Label es opcional en inferencia).
    """
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("El contrato requiere variables únicas y no vacías.")
    if mode not in {"binario", "multiclase"}:
        raise ValueError("Modo de contrato desconocido.")
    columns = {}
    for name in features:
        checks = [pa.Check(lambda s: np.isfinite(s), name="valor_finito")]
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in {"dstport", "destinationport", "srcport", "sourceport", "protocol"}:
            maximum = 255 if key == "protocol" else 65535
            checks.extend([
                pa.Check.in_range(0, maximum),
                pa.Check(lambda s: s.mod(1).eq(0), name="entero"),
            ])
        columns[name] = pa.Column("float32", checks=checks, nullable=True)
    if labeled:
        label_checks = [
            pa.Check(lambda s: s.str.strip().eq(s), name="sin_espacios_extremos"),
            pa.Check(lambda s: ~s.str.lower().isin(
                ["", "label", "nan", "null", "none"]), name="etiqueta_valida"),
        ]
        if mode == "binario":
            label_checks.append(pa.Check.isin(["Benign", "Attack"]))
        columns["Label"] = pa.Column(str, checks=label_checks, nullable=False)
        columns["__source"] = pa.Column(
            str, checks=pa.Check.str_length(min_value=1), nullable=False)

    def has_numeric_value(frame):
        # Las columnas ausentes se informan como errores de esquema propios.
        if not set(features).issubset(frame.columns):
            return True
        return frame[features].notna().any(axis=1)

    return pa.DataFrameSchema(
        columns, checks=pa.Check(has_numeric_value, name="alguna_variable_disponible",
                                ignore_na=False),
        coerce=False, strict=False, unique_column_names=True,
        name="muestra_entrenamiento" if labeled else "entrada_inferencia",
    )


def validate_data(frame, schema, *, context="datos"):
    """Valida todos los chequeos del DataFrame y reúne sus fallos con LAZY."""
    try:
        return schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        raise DataContractError(context, exc) from exc

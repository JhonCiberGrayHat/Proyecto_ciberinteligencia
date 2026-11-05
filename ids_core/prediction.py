"""Inferencia por bloques sobre nuevos flujos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import chunks, header, normalize_columns, numeric
from .contracts import feature_schema, validate_data
from .artifacts import load_model


def predict_probabilities(artifact, frame):
    """Inferencia compartida por CSV y API, compatible con artefactos anteriores."""
    model, features = artifact["model"], artifact["features"]
    if "preprocessor" in model.named_steps:
        return model.predict_proba(frame)
    frame = normalize_columns(frame.copy())
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan variables usadas al entrenar: {missing}")
    values = validate_data(numeric(frame[features]), feature_schema(features),
                           context="entrada de inferencia")
    return model.predict_proba(values)


def predict(args):
    # Cargue únicamente archivos joblib propios o de fuentes confiables.
    artifact = load_model(args.model)
    features, model = artifact["features"], artifact["model"]
    source, dest = Path(args.csv).resolve(), Path(args.output).resolve()
    if source == dest:
        raise ValueError("El archivo de salida debe ser distinto al CSV de entrada.")
    if dest == Path(args.model).resolve():
        raise ValueError("La salida de predicciones no puede reemplazar el modelo serializado.")
    missing = sorted(set(features) - set(header(source).columns))
    if missing:
        raise ValueError(f"Faltan variables usadas al entrenar: {missing}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    total = 0
    try:
        for frame in chunks(source, args.chunk_size):
            probabilities = predict_probabilities(artifact, frame)
            best = probabilities.argmax(axis=1)
            # row_csv es el número lógico de registro (encabezado = 1), no una
            # línea física cuando hay campos multilínea.
            result = pd.DataFrame({"row_csv": np.arange(total + 2, total + 2 + len(frame)),
                                   "prediccion": model.classes_[best],
                                   "score_modelo": probabilities.max(axis=1)})
            if "Label" in frame:
                result["etiqueta_original"] = frame["Label"].to_numpy()
            result.to_csv(temp, mode="w" if total == 0 else "a", header=total == 0, index=False)
            total += len(frame)
        if total == 0:
            raise ValueError("CSV sin registros para predecir.")
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{total} flujos clasificados. Salida: {dest}")

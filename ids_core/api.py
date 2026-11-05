"""API de inferencia. Ejecutar: uvicorn ids_core.api:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import MODEL_FILENAME, load_model
from .contracts import DataContractError
from .data import canonical, normalize_columns
from .prediction import predict_probabilities


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict[str, str | float | int | None]] = Field(
        min_length=1, max_length=1000,
        description="Flujos con todas las variables del modelo; null permite imputación.")


class Prediction(BaseModel):
    row: int
    prediccion: str
    score_modelo: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    count: int
    predictions: list[Prediction]


def prediction_examples(artifact):
    """Crea ejemplos sintéticos válidos para las variables del modelo activo."""
    medians = artifact["model"].named_steps["imputer"].statistics_
    record = {}
    for name, median in zip(artifact["features"], medians):
        value = float(median)
        if not math.isfinite(value):
            value = 0.0
        key = canonical(name)
        if key in {"srcport", "sourceport", "dstport", "destinationport", "protocol"}:
            maximum = 255 if key == "protocol" else 65535
            value = max(0, min(maximum, round(value)))
        record[name] = value
    examples = {"flujo_demo": {
        "summary": "Flujo sintético con las variables del modelo",
        "description": "Valores basados en las medianas guardadas. "
                       "Demuestra la inferencia; no garantiza una clase determinada.",
        "value": {"records": [record]},
    }}
    if len(record) > 1:
        incomplete = record.copy()
        incomplete[next(iter(record))] = None
        examples["imputacion_demo"] = {
            "summary": "Lote de dos flujos, uno con un valor faltante",
            "description": "El pipeline imputa null con la mediana aprendida.",
            "value": {"records": [record.copy(), incomplete]},
        }
    return examples


def create_app(model_path=None):
    """Carga un modelo una vez por proceso, sin entrenamiento automático."""
    @asynccontextmanager
    async def lifespan(application):
        path = Path(model_path or os.environ.get(
            "IDS_MODEL_PATH", str(Path("resultados") / MODEL_FILENAME)))
        application.state.artifact = load_model(path)
        # Regenerar el esquema evita ejemplos obsoletos si se consultó OpenAPI
        # antes del inicio o se reutiliza la aplicación con otro modelo.
        application.openapi_schema = None
        schema = application.openapi()
        schema["paths"]["/predict"]["post"]["requestBody"]["content"][
            "application/json"]["examples"] = prediction_examples(application.state.artifact)
        try:
            yield
        finally:
            del application.state.artifact

    application = FastAPI(title="IDS API", version="1.0.0", lifespan=lifespan)

    @application.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    @application.get("/model")
    def model_info(request: Request):
        artifact = request.app.state.artifact
        return {"mode": artifact["mode"], "features": artifact["features"],
                "classes": [str(c) for c in artifact["model"].classes_],
                "preprocessing_version": artifact.get("preprocessing_version", 0)}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        artifact = request.app.state.artifact
        # Comprobar cada registro evita que pandas invente NaN para claves
        # omitidas en una fila. Los faltantes explícitos se envían como null.
        records, failures = [], []
        for index, record in enumerate(payload.records):
            try:
                normalized = normalize_columns(pd.DataFrame([record]))
            except ValueError as exc:
                failures.append({"row": index, "error": str(exc)})
                continue
            missing = sorted(set(artifact["features"]) - set(normalized.columns))
            if missing:
                failures.append({"row": index, "missing_features": missing})
                continue
            records.append(normalized.to_dict(orient="records")[0])
        if failures:
            raise HTTPException(status_code=422, detail=failures)
        try:
            probabilities = predict_probabilities(
                artifact, pd.DataFrame.from_records(records))
        except DataContractError as exc:
            raise HTTPException(status_code=422, detail={
                "error": "Contrato de datos incumplido",
                "failure_cases": json.loads(exc.failure_cases.to_json(orient="records")),
            }) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        best = probabilities.argmax(axis=1)
        classes = artifact["model"].classes_
        return PredictionResponse(count=len(records), predictions=[
            Prediction(row=i, prediccion=str(classes[label]),
                       score_modelo=float(probabilities[i, label]))
            for i, label in enumerate(best)
        ])

    return application


app = create_app()

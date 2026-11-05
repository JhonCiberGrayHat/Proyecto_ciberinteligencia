"""API con un modelo real pequeño, sin entrenamiento durante peticiones."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ids_core import api
from ids_core.artifacts import save_model
from ids_core.preprocessing import FlowPreprocessor


@pytest.fixture
def model_path(tmp_path):
    features = ["Rate", "Dst Port"]
    frame = pd.DataFrame({"Rate": [1, 2, 3, 7, 8, 9], "Dst Port": [80, 443] * 3})
    model = Pipeline([
        ("preprocessor", FlowPreprocessor(features)),
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=3, random_state=42)),
    ]).fit(frame, ["Benign"] * 3 + ["Attack"] * 3)
    path = tmp_path / "model.joblib"
    save_model({"model": model, "features": features, "mode": "binario",
                "preprocessing_version": 1}, path)
    return path


def test_model_loads_once_and_predictions_match_pipeline(model_path, monkeypatch):
    load = api.load_model
    calls = []

    def counted_load(path):
        calls.append(path)
        return load(path)

    def forbid_fit(*args, **kwargs):
        pytest.fail("La API no debe entrenar")

    monkeypatch.setattr(api, "load_model", counted_load)
    monkeypatch.setattr(Pipeline, "fit", forbid_fit)
    records = [{"Rate": "bad", "Dst Port": 80}, {"Rate": 8, "Dst Port": 443}]
    expected = load(model_path)["model"].predict_proba(pd.DataFrame(records))
    application = api.create_app(model_path)
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
        info = client.get("/model").json()
        assert info["features"] == ["Rate", "Dst Port"]
        assert info["mode"] == "binario"
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        for _ in range(2):
            response = client.post("/predict", json={"records": records})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["count"] == 2
            np.testing.assert_allclose(
                [row["score_modelo"] for row in body["predictions"]], expected.max(axis=1))
            assert [row["prediccion"] for row in body["predictions"]] == ["Benign", "Attack"]
        assert len(calls) == 1
    assert not hasattr(application.state, "artifact")


@pytest.mark.parametrize("payload", [
    {}, {"records": []}, {"records": [{}]},
    {"records": [{"Rate": 1}]},
    {"records": [{"Rate": 1, "Dst Port": 80}, {"Dst Port": 443}]},
    {"records": [{"Rate": None, "Dst Port": None}]},
    {"records": [{"Rate": [], "Dst Port": 80}]},
    {"records": [{"Rate": 1, "Dst Port": 80}] * 1001},
    {"records": [{"Rate": 1, " Rate ": 2, "Dst Port": 80}]},
])
def test_invalid_requests_return_422(model_path, payload):
    with TestClient(api.create_app(model_path)) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_lazy_validation_returns_multiple_failures(model_path):
    with TestClient(api.create_app(model_path)) as client:
        response = client.post("/predict", json={"records": [
            {"Rate": 1, "Dst Port": -1}, {"Rate": 2, "Dst Port": 70000}]})
        assert response.status_code == 422
        failures = response.json()["detail"]["failure_cases"]
        assert {0, 1}.issubset({case["index"] for case in failures})
        # La aplicación sigue disponible después de un rechazo.
        assert client.get("/health").status_code == 200


def test_environment_model_path(model_path, monkeypatch):
    monkeypatch.setenv("IDS_MODEL_PATH", str(model_path))
    with TestClient(api.create_app()) as client:
        assert client.get("/health").status_code == 200


def test_swagger_examples_match_active_model_and_can_be_submitted(model_path):
    application = api.create_app(model_path)
    application.openapi()  # También actualiza un esquema generado antes del arranque.
    with TestClient(application) as client:
        schema = client.get("/openapi.json").json()
        examples = schema["paths"]["/predict"]["post"]["requestBody"]["content"][
            "application/json"]["examples"]
        assert set(examples) == {"flujo_demo", "imputacion_demo"}
        # La mediana de puertos 80/443 es fraccionaria: el ejemplo debe ser entero.
        port = examples["flujo_demo"]["value"]["records"][0]["Dst Port"]
        assert port == round(port) and 0 <= port <= 65535
        assert examples["imputacion_demo"]["value"]["records"][1]["Rate"] is None
        for example in examples.values():
            records = example["value"]["records"]
            assert all(set(record) == {"Rate", "Dst Port"} for record in records)
            response = client.post("/predict", json=example["value"])
            assert response.status_code == 200, response.text
            assert response.json()["count"] == len(records)


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(ValueError, match="No existe el modelo"):
        with TestClient(api.create_app(tmp_path / "missing.joblib")):
            pytest.fail("El servidor no debe iniciar sin modelo")

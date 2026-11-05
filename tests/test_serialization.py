"""Entrenar una vez y reutilizar el pipeline persistido en inferencia."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ids_core import artifacts
from ids_core.cli import main


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path / "flows.csv"
    pd.DataFrame({
        "Rate": np.arange(60, dtype=float),
        "Dst Port": np.tile([80, 443], 30),
        "Label": np.tile(["Benign", "DoS"], 30),
    }).to_csv(source, index=False)
    output = tmp_path / "model"
    arguments = ["entrenar", "--data", str(source), "--output", str(output),
                 "--trees", "3", "--jobs", "1"]
    assert main(arguments) == 0
    return source, output, arguments


def test_reuse_and_prediction_never_fit_or_require_training_csv(experiment, monkeypatch, tmp_path, capsys):
    source, output, arguments = experiment
    model_path = output / artifacts.MODEL_FILENAME
    original = model_path.read_bytes()
    report = (output / "metricas.json").read_bytes()
    artifact = artifacts.load_model(model_path)
    frame = pd.read_csv(source).iloc[:6].drop(columns="Label")
    expected = artifact["model"].predict_proba(frame[artifact["features"]].astype("float32"))
    source.unlink()

    def unexpected_fit(*args, **kwargs):
        pytest.fail("La reutilización y predicción no deben volver a entrenar")

    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    assert main(arguments) == 0  # El CSV de entrenamiento ya no existe.
    assert "Se omite el entrenamiento" in capsys.readouterr().out
    new = tmp_path / "new.csv"
    frame.to_csv(new, index=False)
    prediction = tmp_path / "prediction.csv"
    assert main(["predecir", "--model", str(model_path), "--csv", str(new),
                 "--output", str(prediction), "--chunk-size", "2"]) == 0
    result = pd.read_csv(prediction)
    assert result["prediccion"].tolist() == artifact["model"].classes_[expected.argmax(axis=1)].tolist()
    np.testing.assert_allclose(result["score_modelo"], expected.max(axis=1))
    assert model_path.read_bytes() == original
    assert (output / "metricas.json").read_bytes() == report


def test_explicit_retraining_updates_persisted_model(experiment):
    _, output, arguments = experiment
    arguments[arguments.index("--trees") + 1] = "7"
    assert main([*arguments, "--reentrenar"]) == 0
    artifact = artifacts.load_model(output / artifacts.MODEL_FILENAME)
    assert artifact["model"].named_steps["classifier"].n_estimators == 7


def test_serialized_preprocessing_handles_raw_values_and_preserves_medians(experiment, monkeypatch):
    _, output, _ = experiment
    path = output / artifacts.MODEL_FILENAME
    artifact = artifacts.load_model(path)
    model = artifact["model"]
    assert artifact["preprocessing_version"] == 1
    assert list(model.named_steps) == ["preprocessor", "imputer", "classifier"]
    medians = model.named_steps["imputer"].statistics_.copy()
    # Encabezados con BOM/espacios, orden distinto, columna extra, faltantes,
    # texto, infinitos y desbordamiento, sin ninguna preparación externa.
    raw = pd.DataFrame({" Dst Port ": ["80", "443", "80", "443", "80"],
                        "\ufeff Rate ": [None, "bad", "Infinity", "1e40", "2.5"],
                        "Label": ["reference"] * 5})
    original = raw.copy(deep=True)
    clean = model.named_steps["preprocessor"].transform(raw)
    assert list(clean.columns) == artifact["features"]
    assert clean["Rate"].iloc[:4].isna().all()
    assert (clean.dtypes == "float32").all()
    transformed = model.named_steps["imputer"].transform(clean)
    rate_index = artifact["features"].index("Rate")
    np.testing.assert_allclose(transformed[:4, rate_index], medians[rate_index])
    expected = model.named_steps["classifier"].predict_proba(transformed)

    def forbid_fit(*args, **kwargs):
        pytest.fail("El preprocesamiento de inferencia no debe ajustarse")

    for step in model.named_steps.values():
        monkeypatch.setattr(type(step), "fit", forbid_fit)
    loaded = joblib.load(path)["model"]
    np.testing.assert_allclose(loaded.predict_proba(raw), expected)
    np.testing.assert_array_equal(loaded.named_steps["imputer"].statistics_, medians)
    pd.testing.assert_frame_equal(raw, original)
    with pytest.raises(ValueError, match="Faltan variables"):
        loaded.predict_proba(raw.drop(columns=" Dst Port "))


def test_legacy_model_still_predicts(experiment, tmp_path):
    source, output, _ = experiment
    artifact = artifacts.load_model(output / artifacts.MODEL_FILENAME)
    artifact["model"] = Pipeline(artifact["model"].steps[1:])
    artifact.pop("preprocessing_version")
    legacy = tmp_path / "legacy.joblib"
    artifacts.save_model(artifact, legacy)
    assert main(["predecir", "--model", str(legacy), "--csv", str(source),
                 "--output", str(tmp_path / "legacy_predictions.csv")]) == 0


def test_different_mode_requires_explicit_retraining(experiment):
    _, output, arguments = experiment
    path = output / artifacts.MODEL_FILENAME
    original = path.read_bytes()
    assert main([*arguments, "--mode", "binario"]) == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("contents", [None, b"invalid joblib"])
def test_missing_or_corrupt_model_does_not_train(tmp_path, contents):
    path = tmp_path / "model.joblib"
    if contents is not None:
        path.write_bytes(contents)
    with pytest.raises(ValueError, match="modelo serializado"):
        artifacts.load_model(path)


def test_rejects_artifact_without_pipeline(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": None, "features": ["Rate"]}, path)
    with pytest.raises(ValueError, match="pipeline"):
        artifacts.load_model(path)


def test_failed_serialization_preserves_previous_model(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"previous")

    def fail_dump(artifact, temporary, **kwargs):
        temporary.write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(artifacts.joblib, "dump", fail_dump)
    with pytest.raises(OSError, match="disk failure"):
        artifacts.save_model({}, path)
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]

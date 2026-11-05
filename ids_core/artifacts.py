"""Persistencia y carga del pipeline ajustado y sus metadatos."""
from pathlib import Path
import tempfile

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

MODEL_FILENAME = "modelo_ids.joblib"


def load_model(path):
    """Carga un artefacto propio o confiable; nunca entrena al cargar."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"No existe el modelo serializado: {path}. Ejecute entrenar primero.")
    try:
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            raise ValueError("Se esperaba un diccionario de modelo y metadatos.")
        model, features = artifact["model"], artifact["features"]
        if (not isinstance(features, list) or not features
                or not all(isinstance(f, str) for f in features)
                or len(set(features)) != len(features)):
            raise ValueError("Variables del modelo inválidas.")
        if not isinstance(model, Pipeline) or not hasattr(model, "predict_proba"):
            raise ValueError("El artefacto no contiene un pipeline de clasificación.")
        check_is_fitted(model)
        if "imputer" not in model.named_steps:
            raise ValueError("El pipeline no contiene el preprocesamiento de faltantes.")
        check_is_fitted(model.named_steps["imputer"], "statistics_")
        if artifact.get("preprocessing_version") == 1:
            from .preprocessing import FlowPreprocessor
            preprocessor = model.named_steps.get("preprocessor")
            if not isinstance(preprocessor, FlowPreprocessor):
                raise ValueError("Falta el preprocesador serializado del artefacto.")
            check_is_fitted(preprocessor, "feature_names_in_")
            if list(preprocessor.feature_names_in_) != features:
                raise ValueError("Variables del preprocesador incompatibles.")
        if model.n_features_in_ != len(features):
            raise ValueError("Las variables no coinciden con el pipeline ajustado.")
        if list(model.feature_names_in_) != features:
            raise ValueError("El orden de las variables no coincide con el pipeline.")
        if artifact["mode"] not in {"binario", "multiclase"}:
            raise ValueError("Modo de clasificación desconocido.")
    except Exception as exc:
        raise ValueError(f"No se pudo cargar el modelo serializado {path}: {exc}") from exc
    return artifact


def save_model(artifact, path):
    """Publica el modelo completo mediante reemplazo de un temporal local."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(artifact, temporary, compress=3)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

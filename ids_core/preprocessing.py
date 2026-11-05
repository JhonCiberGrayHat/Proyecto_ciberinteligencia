"""Transformación serializable de flujos CSV a entradas del imputador."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .contracts import feature_schema, validate_data
from .data import normalize_columns, numeric


class FlowPreprocessor(TransformerMixin, BaseEstimator):
    """Conserva las variables y aplica la misma limpieza en ajuste e inferencia."""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(self.features, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        frame = normalize_columns(X.copy())
        missing = sorted(set(self.feature_names_in_) - set(frame.columns))
        if missing:
            raise ValueError(f"Faltan variables usadas al entrenar: {missing}")
        values = numeric(frame.loc[:, self.feature_names_in_])
        return validate_data(values, feature_schema(self.feature_names_in_),
                             context="preprocesamiento del modelo serializado")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()

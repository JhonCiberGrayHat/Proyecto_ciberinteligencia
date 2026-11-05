"""Casos límite del contrato parametrizados con pytest."""
import numpy as np
import pandas as pd
import pytest

from ids_core.contracts import DataContractError, feature_schema, validate_data


@pytest.mark.parametrize("column,maximum", [
    ("Dst Port", 65535), ("Destination Port", 65535),
    ("Src Port", 65535), ("Source Port", 65535), ("Protocol", 255),
])
def test_discrete_fields_accept_boundaries_and_reject_invalid_values(column, maximum):
    schema = feature_schema([column])
    valid = pd.DataFrame({column: np.array([0, maximum], dtype="float32")})
    pd.testing.assert_frame_equal(validate_data(valid, schema), valid)
    invalid = pd.DataFrame({column: np.array([-1, maximum + 1, 0.5], dtype="float32")})
    with pytest.raises(DataContractError) as raised:
        validate_data(invalid, schema)
    failures = raised.value.failure_cases
    assert {0, 1, 2}.issubset(set(failures["index"].dropna()))


@pytest.mark.parametrize("label", ["", "Label", "NaN", "null", "none", " Benign", "DoS ", None])
def test_training_rejects_invalid_labels(label):
    frame = pd.DataFrame({"Rate": np.array([1], dtype="float32"),
                          "Label": [label], "__source": ["day.csv"]})
    with pytest.raises(DataContractError) as raised:
        validate_data(frame, feature_schema(["Rate"], labeled=True))
    assert "Label" in raised.value.failure_cases["column"].values

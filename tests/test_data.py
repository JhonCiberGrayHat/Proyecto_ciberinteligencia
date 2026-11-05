"""Pruebas de lectura y limpieza con datos locales temporales."""
import numpy as np
import pandas as pd
import pytest

from ids_core.data import chunks, csv_files, normalize_columns, numeric


def test_numeric_normalizes_invalid_values_without_mutating_input():
    frame = pd.DataFrame({"Rate": ["12.5", "bad", "Infinity", "-inf", "1e40", None]})
    original = frame.copy(deep=True)
    result = numeric(frame)
    assert result["Rate"].dtype == np.dtype("float32")
    assert result.loc[0, "Rate"] == 12.5
    assert result.loc[1:, "Rate"].isna().all()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("columns", [
    ["Label", " Label "], ["__source", "Rate"], ["Rate", "__priority"],
])
def test_rejects_duplicate_or_reserved_headers(columns):
    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame(columns=columns))


def test_csv_discovery_and_chunk_reading(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "flows.CSV"
    pd.DataFrame({" Rate ": [1, 2, 3], "Label": ["Benign"] * 3}).to_csv(
        source, index=False, encoding="utf-8-sig")
    (tmp_path / "ignore.txt").write_text("not CSV", encoding="utf-8")
    assert csv_files(tmp_path) == [source]
    parts = list(chunks(source, 2))
    assert [len(part) for part in parts] == [2, 1]
    assert list(parts[0].columns) == ["Rate", "Label"]
    assert pd.concat(parts)["Rate"].tolist() == ["1", "2", "3"]


def test_empty_directory_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="No se encontraron CSV"):
        csv_files(tmp_path)

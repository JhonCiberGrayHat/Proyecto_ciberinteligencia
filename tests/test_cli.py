"""Pruebas de validación de argumentos y errores de la CLI."""
import logging

import pytest

from ids_core.cli import main


@pytest.mark.parametrize("arguments", [
    [], ["entrenar", "--test-size", "0"],
    ["entrenar", "--test-size", "1"], ["entrenar", "--jobs", "0"],
    ["entrenar", "--jobs", "-2"], ["entrenar", "--trees", "0"],
    ["predecir"], ["predecir", "--csv", "unused.csv", "--chunk-size", "-1"],
])
def test_invalid_arguments_exit_with_usage(arguments, capsys):
    with pytest.raises(SystemExit) as raised:
        main(arguments)
    assert raised.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_inspect_local_csv(tmp_path, capsys):
    source = tmp_path / "flows.csv"
    source.write_text("Rate,Label\n1,Benign\n2,DoS\n", encoding="utf-8")
    assert main(["inspeccionar", "--data", str(source)]) == 0
    output = capsys.readouterr().out
    assert "flows.csv" in output
    assert "Benign" in output and "DoS" in output


def test_missing_data_returns_error_code(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="IDS"):
        code = main(["inspeccionar", "--data", str(tmp_path / "missing")])
    assert code == 1
    assert "No se encontraron CSV" in caplog.text

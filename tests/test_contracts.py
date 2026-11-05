"""Validación de tipos, reglas y acumulación LAZY de errores."""
import unittest

import numpy as np
import pandas as pd

from ids_core.contracts import DataContractError, feature_schema, validate_data


class DataContractTests(unittest.TestCase):
    def test_accepts_partial_missing_and_extra_columns(self):
        frame = pd.DataFrame({
            "Dst Port": np.array([443, np.nan], dtype="float32"),
            "Flow Duration": np.array([1, 2], dtype="float32"),
            "Label": ["unvalidated reference", "Benign"],
        })
        result = validate_data(frame, feature_schema(["Dst Port", "Flow Duration"]))
        pd.testing.assert_frame_equal(result, frame)

    def test_lazy_collects_missing_type_and_rule_failures(self):
        frame = pd.DataFrame({
            "Dst Port": np.array([-1, 70000, 1.5], dtype="float32"),
            "Flow Duration": ["bad", "bad", "bad"],
            "Rate": np.array([np.inf, 1, 2], dtype="float32"),
        })
        with self.assertRaises(DataContractError) as raised:
            validate_data(frame, feature_schema(
                ["Dst Port", "Flow Duration", "Rate", "Missing"]), context="test")
        failures = raised.exception.failure_cases
        self.assertTrue({"Dst Port", "Flow Duration", "Rate"}.issubset(
            set(failures["column"].dropna())))
        self.assertIn("Missing", failures["failure_case"].values)
        self.assertIn("entero", failures["check"].values)
        self.assertIn("test", str(raised.exception))

    def test_rejects_all_missing_row(self):
        frame = pd.DataFrame({"Rate": np.array([np.nan], dtype="float32")})
        with self.assertRaises(DataContractError) as raised:
            validate_data(frame, feature_schema(["Rate"]))
        self.assertIn("alguna_variable_disponible",
                      raised.exception.failure_cases["check"].values)

    def test_labeled_binary_contract(self):
        frame = pd.DataFrame({
            "Protocol": np.array([6, 300], dtype="float32"),
            "Label": ["Benign", "DoS"], "__source": ["day.csv", ""],
        })
        schema = feature_schema(["Protocol"], labeled=True, mode="binario")
        with self.assertRaises(DataContractError) as raised:
            validate_data(frame, schema)
        self.assertTrue({"Protocol", "Label", "__source"}.issubset(
            set(raised.exception.failure_cases["column"].dropna())))
        frame.loc[1, ["Protocol", "Label", "__source"]] = [17, "Attack", "day.csv"]
        validate_data(frame, schema)


if __name__ == "__main__":
    unittest.main()

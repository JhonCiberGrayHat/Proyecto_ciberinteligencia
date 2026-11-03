"""Pruebas de integración sin descarga ni credenciales de Kaggle."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ids", ROOT / "ids.py")
ids = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ids)


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.data = self.base / "data"
        self.data.mkdir()
        rng = np.random.default_rng(42)
        for day in range(2):
            n = 120
            frame = pd.DataFrame({
                " Flow Duration ": rng.uniform(1, 1000, n),
                "Dst Port": rng.integers(1, 60000, n),
                "Flow Byts/s": rng.uniform(1, 5000, n),
                "Timestamp": [f"day{day}"] * n,
                "Constant": [0] * n,
                "Label": np.tile(["Benign", "DoS", "Bot"], n // 3),
            })
            # Desorden, infinitos, faltantes y una fila de encabezado repetida.
            frame["Flow Byts/s"] = frame["Flow Byts/s"].astype(object)
            frame.loc[0, "Flow Byts/s"] = "Infinity"
            frame.loc[1, "Flow Byts/s"] = "not_a_number"
            frame.loc[2, "Flow Byts/s"] = None
            frame.loc[n] = list(frame.columns)
            if day == 1:
                frame["Src IP"] = "192.0.2.1"
            frame.to_csv(self.data / f"day{day}.csv", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args, success=True, module=False):
        entry = ["-m", "ids_core"] if module else [str(ROOT / "ids.py")]
        result = subprocess.run([sys.executable, *entry, *map(str, args)],
                                capture_output=True, text=True, cwd=ROOT)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_multiclass_train_predict_and_schema_rejection(self):
        out = self.base / "multi"
        self.run_cli("entrenar", "--data", self.data, "--output", out,
                     "--chunk-size", 17, "--trees", 5, "--jobs", 1)
        audit = json.loads((out / "auditoria.json").read_text())
        self.assertEqual(audit["invalid_label_rows"], 2)
        self.assertEqual(audit["sample_after_cleaning"], 240)
        artifact = joblib.load(out / "modelo_ids.joblib")
        self.assertNotIn("Constant", artifact["features"])
        self.assertNotIn("Timestamp", artifact["features"])
        self.assertNotIn("Src IP", artifact["features"])
        self.assertNotIn("Label", artifact["features"])
        # Nuevos datos sin Label, columnas desordenadas y valor infinito.
        new = pd.read_csv(self.data / "day0.csv").iloc[:8].drop(columns=["Label"])
        new.to_csv(self.base / "new.csv", index=False)
        dest = self.base / "pred.csv"
        self.run_cli("predecir", "--model", out / "modelo_ids.joblib",
                     "--csv", self.base / "new.csv", "--output", dest, "--chunk-size", 3)
        pred = pd.read_csv(dest)
        self.assertEqual(len(pred), 8)
        self.assertEqual(pred["row_csv"].tolist(), list(range(2, 10)))
        self.assertTrue(pred["score_modelo"].between(0, 1).all())
        bad = new.drop(columns=["Dst Port"])
        bad.to_csv(self.base / "bad.csv", index=False)
        result = self.run_cli("predecir", "--model", out / "modelo_ids.joblib",
                             "--csv", self.base / "bad.csv", "--output", self.base / "bad_out.csv",
                             success=False)
        self.assertIn("Faltan variables", result.stderr)
        self.assertFalse((self.base / "bad_out.csv").exists())
        self.assertTrue((out / "matriz_confusion.png").is_file())

    def test_binary_file_holdout(self):
        out = self.base / "binary"
        self.run_cli("entrenar", "--data", self.data, "--output", out,
                     "--mode", "binario", "--test-file", "day1.csv",
                     "--chunk-size", 19, "--trees", 5, "--jobs", 1, module=True)
        scores = json.loads((out / "metricas.json").read_text())
        self.assertEqual(scores["train_rows"], 120)
        self.assertEqual(scores["test_rows"], 120)
        self.assertEqual(scores["split"], "archivo_excluido")
        preds = pd.read_csv(out / "predicciones_prueba.csv")
        self.assertTrue(preds["archivo"].str.endswith("day1.csv").all())
        self.assertEqual(set(preds["real"]), {"Benign", "Attack"})
        self.assertEqual(sum(scores["binary_alarm"][k] for k in ["TN", "FP", "FN", "TP"]), 120)

    def test_sampler_covers_all_files_and_cleans_conflicts(self):
        args = argparse.Namespace(data=str(self.data), seed=42, chunk_size=11,
                                  mode="multiclase", benign_cap=7, class_cap=7)
        sample, features, audit = ids.load_sample(args)
        self.assertEqual(sum(audit["original_labels_observed_after_mode"].values()), 240)
        self.assertEqual(len(sample), 21)
        self.assertEqual(sample["__source"].nunique(), 2)
        # Una réplica con etiqueta distinta debe eliminar ambas filas, sin fuga.
        first = pd.read_csv(self.data / "day0.csv").iloc[[10]].copy()
        first["Label"] = "AnotherAttack"
        first.to_csv(self.data / "conflict.csv", index=False)
        args.benign_cap = args.class_cap = 500
        sample, features, audit = ids.load_sample(args)
        self.assertEqual(audit["conflicting_sample_rows_removed"], 2)
        self.assertFalse(sample.duplicated(features, keep=False).any())


if __name__ == "__main__":
    unittest.main()

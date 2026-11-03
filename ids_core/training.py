"""Entrenamiento, evaluación y persistencia del modelo."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import DATASET, LOG
from .reporting import plot_confusion, write_json
from .sampling import load_sample


def train(args):
    data, features, audit = load_sample(args)
    if data["Label"].nunique() < 2:
        raise ValueError("Se necesitan al menos dos clases después de limpiar la muestra.")
    if args.test_file:
        # División por archivo/día: el archivo elegido nunca participa en ajuste.
        matches = data["__source"].map(lambda p: Path(p).name == args.test_file)
        train_idx, test_idx = data.index[~matches], data.index[matches]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("--test-file debe coincidir con un CSV y dejar datos de entrenamiento.")
        split_type = "archivo_excluido"
    else:
        counts = data["Label"].value_counts()
        if (counts < 2).any():
            raise ValueError(f"Clases con menos de 2 muestras: {counts[counts < 2].to_dict()}. "
                             "Aumente los límites o use --mode binario.")
        train_idx, test_idx = train_test_split(data.index, test_size=args.test_size,
                                             random_state=args.seed, stratify=data["Label"])
        split_type = "aleatorio_estratificado_exploratorio"
    Xtrain = data.loc[train_idx, features]
    Xtest = data.loc[test_idx, features]
    ytrain, ytest = data.loc[train_idx, "Label"], data.loc[test_idx, "Label"]
    if ytrain.nunique() < 2:
        raise ValueError("El entrenamiento requiere dos clases. Cambie el archivo de prueba.")
    # La selección e imputación dependen exclusivamente del entrenamiento.
    features = Xtrain.columns[Xtrain.nunique(dropna=True) > 1].tolist()
    if not features:
        raise ValueError("No hay características numéricas variables en entrenamiento.")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=args.trees, max_depth=24, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=args.jobs, random_state=args.seed)),
    ])
    LOG.info("Entrenando con %d filas y %d variables...", len(train_idx), len(features))
    model.fit(Xtrain[features], ytrain)
    predicted = model.predict(Xtest[features])
    labels = sorted(set(ytest) | set(model.classes_))
    report = classification_report(ytest, predicted, labels=labels,
                                   output_dict=True, zero_division=0)
    report_text = classification_report(ytest, predicted, labels=labels,
                                        zero_division=0, digits=4)
    scores = {"accuracy": accuracy_score(ytest, predicted),
              "balanced_accuracy": balanced_accuracy_score(ytest, predicted),
              "f1_macro": f1_score(ytest, predicted, average="macro", zero_division=0),
              "f1_weighted": f1_score(ytest, predicted, average="weighted", zero_division=0),
              "train_rows": len(train_idx), "test_rows": len(test_idx),
              "split": split_type,
              "classes_unseen_in_training": sorted(set(ytest) - set(model.classes_)),
              "train_class_counts": ytrain.value_counts().to_dict(),
              "test_class_counts": ytest.value_counts().to_dict()}
    # Métricas de alarma binaria, también cuando el clasificador es multiclase.
    true_attack, pred_attack = ytest.ne("Benign").to_numpy(), predicted != "Benign"
    tn, fp, fn, tp = confusion_matrix(true_attack, pred_attack, labels=[False, True]).ravel()
    scores["binary_alarm"] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "attack_recall": float(tp / (tp + fn)) if tp + fn else None}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    versions = {"python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scikit_learn": sklearn.__version__}
    artifact = {"model": model, "features": features, "mode": args.mode,
                "dataset": DATASET, "versions": versions, "seed": args.seed}
    joblib.dump(artifact, out / "modelo_ids.joblib", compress=3)
    write_json(out / "metricas.json", scores)
    write_json(out / "reporte_clases.json", report)
    write_json(out / "auditoria.json", {**audit, "arguments": vars(args) | {"func": args.func.__name__},
                                      "features": features, "versions": versions})
    (out / "reporte_clases.txt").write_text(report_text, encoding="utf-8")
    cm = confusion_matrix(ytest, predicted, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / "matriz_confusion.csv", index_label="real")
    pd.DataFrame({"variable": features,
                  "importancia": model.named_steps["classifier"].feature_importances_}).sort_values(
                      "importancia", ascending=False).to_csv(out / "importancia_variables.csv", index=False)
    pd.DataFrame({"archivo": data.loc[test_idx, "__source"].to_numpy(),
                  "real": ytest.to_numpy(), "prediccion": predicted}).to_csv(out / "predicciones_prueba.csv", index=False)
    plot_confusion(cm, labels, out)
    print(report_text)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
    print(f"Resultados guardados en {out.resolve()}")

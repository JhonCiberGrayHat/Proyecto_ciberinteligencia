"""IDS experimental con CSE-CIC-IDS2018. Ejecutar: python ids.py --help."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import platform
import re
import sys

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

LOG = logging.getLogger("IDS")
DATASET = "solarmainframe/ids-intrusion-csv"
ROOT = Path(__file__).resolve().parent
# Se excluyen metadatos que pueden identificar el día, el equipo o la captura.
EXCLUDE = {"flowid", "sourceip", "srcip", "destinationip", "dstip", "timestamp",
           "sourceport", "srcport", "sourcefile", "index"}
META = {"__source", "__priority"}


def canonical(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalize_columns(frame):
    names = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
    if len(names) != len(set(names)):
        raise ValueError("Encabezados duplicados después de quitar espacios.")
    if META.intersection(names):
        raise ValueError("El CSV utiliza nombres de columnas reservados.")
    frame.columns = names
    return frame


def csv_files(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise ValueError(f"No se encontraron CSV en {path}. Descomprima primero el ZIP.")
    return files


def header(path):
    return normalize_columns(pd.read_csv(path, nrows=0, encoding="utf-8-sig"))


def chunks(path, size):
    # dtype=str evita inferencias diferentes entre bloques y permite retirar
    # encabezados repetidos antes de convertir a valores numéricos.
    for frame in pd.read_csv(path, dtype=str, chunksize=size, encoding="utf-8-sig"):
        yield normalize_columns(frame)


def numeric(frame):
    out = frame.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    # El bosque convierte internamente a float32: los desbordamientos se tratan
    # como faltantes, nunca como un número arbitrariamente grande.
    return out.mask(out.abs() > np.finfo(np.float32).max).astype("float32")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def resolve_data(value):
    if value:
        return Path(value).expanduser().resolve()
    pointer = ROOT / "data_path.txt"
    if pointer.exists():
        return Path(pointer.read_text(encoding="utf-8").strip())
    raise ValueError("Use --data RUTA o ejecute primero: python ids.py descargar")


def download(args):
    import kagglehub
    try:
        path = Path(kagglehub.dataset_download(DATASET)).resolve()
    except Exception as exc:
        raise ValueError("Falló la descarga de Kaggle. Revise conexión y permisos; "
                         "si requiere autenticación, ejecute python -c "
                         "\"import kagglehub; kagglehub.login()\". También puede "
                         "descargar/descomprimir manualmente y usar --data.") from exc
    (ROOT / "data_path.txt").write_text(str(path), encoding="utf-8")
    print(f"Datos disponibles en: {path}")


def inspect_data(args):
    for path in csv_files(resolve_data(args.data)):
        sample = next(chunks(path, 2000), None)
        print(f"\n{path.name} ({path.stat().st_size / 1024**2:.1f} MiB)")
        print("Columnas:", list(header(path).columns))
        if sample is not None and "Label" in sample:
            print("Etiquetas SOLO de las primeras 2000 filas:",
                  sample["Label"].value_counts().to_dict())


def load_sample(args):
    files = csv_files(resolve_data(args.data))
    schemas = [set(header(p).columns) for p in files]
    for path, cols in zip(files, schemas):
        if "Label" not in cols:
            raise ValueError(f"{path.name}: falta la columna Label.")
    common = set.intersection(*schemas)
    features = sorted(c for c in common if c != "Label"
                      and canonical(c) not in EXCLUDE
                      and not c.lower().startswith("unnamed:"))
    if not features:
        raise ValueError("No hay características comunes entre los CSV.")
    # Mantener las prioridades aleatorias más pequeñas implementa un muestreo
    # uniforme por clase sobre TODOS los bloques; no toma sólo el inicio.
    rng = np.random.default_rng(args.seed)
    pools = {}
    observed = Counter()
    discarded_labels = 0
    for path in files:
        LOG.info("Leyendo %s", path.name)
        for part in chunks(path, args.chunk_size):
            labels = part["Label"].str.strip()
            valid = labels.notna() & ~labels.str.lower().isin(
                ["", "label", "nan", "null", "none"])
            discarded_labels += int((~valid).sum())
            part = part.loc[valid, features].copy()
            labels = labels.loc[valid]
            if args.mode == "binario":
                labels = labels.map(lambda x: "Benign" if x.lower() == "benign" else "Attack")
            else:
                labels = labels.map(lambda x: "Benign" if x.lower() == "benign" else x)
            part["Label"] = labels
            part["__source"] = str(path.resolve())
            part["__priority"] = rng.random(len(part))
            observed.update(labels)
            for label, group in part.groupby("Label", sort=False):
                cap = args.benign_cap if label == "Benign" else args.class_cap
                pool = pd.concat([pools[label], group], ignore_index=True) if label in pools else group
                pools[label] = pool.nsmallest(cap, "__priority")
        LOG.info("Muestra acumulada: %d filas", sum(map(len, pools.values())))
    if not pools:
        raise ValueError("No hay filas etiquetadas válidas.")
    sample = pd.concat(pools.values(), ignore_index=True).drop(columns="__priority")
    X = numeric(sample[features])
    before = len(sample)
    usable = X.notna().any(axis=1)
    X, sample = X.loc[usable], sample.loc[usable]
    # Elimina repeticiones y TODAS las filas de vectores con etiquetas conflictivas.
    # Duplicados se verifican en la muestra; no se afirma depurar todo el dataset.
    combined = X.copy()
    combined["Label"] = sample["Label"]
    combined["__source"] = sample["__source"]
    combined = combined.drop_duplicates(subset=features + ["Label"])
    conflicts = combined.duplicated(subset=features, keep=False)
    conflict_count = int(conflicts.sum())
    combined = combined.loc[~conflicts].reset_index(drop=True)
    audit = {"dataset": DATASET, "files": [str(p.resolve()) for p in files],
             "file_sizes_bytes": {str(p.resolve()): p.stat().st_size for p in files},
             "original_labels_observed_after_mode": dict(observed),
             "invalid_label_rows": discarded_labels,
             "sample_before_cleaning": before,
             "all_missing_feature_rows": int((~usable).sum()),
             "conflicting_sample_rows_removed": conflict_count,
             "sample_after_cleaning": len(combined),
             "noncommon_columns_excluded": sorted(set.union(*schemas) - common),
             "sample_class_counts": combined["Label"].value_counts().to_dict()}
    return combined, features, audit


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


def plot_confusion(cm, labels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(max(7, len(labels)), max(6, len(labels) * .65)))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False,
                                                        xticks_rotation=90, values_format="d")
    ax.set_title("Matriz de confusión · prueba")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion.png", dpi=150)
    plt.close(fig)


def predict(args):
    # Cargue únicamente archivos joblib propios o de fuentes confiables.
    artifact = joblib.load(args.model)
    features, model = artifact["features"], artifact["model"]
    source, dest = Path(args.csv).resolve(), Path(args.output).resolve()
    if source == dest:
        raise ValueError("El archivo de salida debe ser distinto al CSV de entrada.")
    missing = sorted(set(features) - set(header(source).columns))
    if missing:
        raise ValueError(f"Faltan variables usadas al entrenar: {missing}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    total = 0
    try:
        for frame in chunks(source, args.chunk_size):
            probabilities = model.predict_proba(numeric(frame[features]))
            best = probabilities.argmax(axis=1)
            # row_csv es el número lógico de registro (encabezado = 1), no una
            # línea física cuando hay campos multilínea.
            result = pd.DataFrame({"row_csv": np.arange(total + 2, total + 2 + len(frame)),
                                   "prediccion": model.classes_[best],
                                   "score_modelo": probabilities.max(axis=1)})
            if "Label" in frame:
                result["etiqueta_original"] = frame["Label"].to_numpy()
            result.to_csv(temp, mode="w" if total == 0 else "a", header=total == 0, index=False)
            total += len(frame)
        if total == 0:
            raise ValueError("CSV sin registros para predecir.")
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{total} flujos clasificados. Salida: {dest}")


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo.")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("descargar", help="Descargar Kaggle a la caché de kagglehub")
    p.set_defaults(func=download)
    p = sub.add_parser("inspeccionar", help="Listar archivos, columnas y primeras etiquetas")
    p.add_argument("--data")
    p.set_defaults(func=inspect_data)
    p = sub.add_parser("entrenar", help="Entrenar y evaluar Random Forest")
    p.add_argument("--data", help="Carpeta descomprimida o CSV individual")
    p.add_argument("--mode", choices=["binario", "multiclase"], default="multiclase")
    p.add_argument("--class-cap", type=positive, default=5000, help="Muestra máxima por clase de ataque")
    p.add_argument("--benign-cap", type=positive, default=20000)
    p.add_argument("--chunk-size", type=positive, default=25000)
    p.add_argument("--test-size", type=float, default=.2)
    p.add_argument("--test-file", help="Nombre exacto de un CSV reservado para prueba por día")
    p.add_argument("--trees", type=positive, default=150)
    p.add_argument("--jobs", type=int, default=2, help="Hilos CPU; -1 utiliza todos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="resultados")
    p.set_defaults(func=train)
    p = sub.add_parser("predecir", help="Clasificar CSV nuevos con el modelo guardado")
    p.add_argument("--model", default="resultados/modelo_ids.joblib")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="predicciones.csv")
    p.add_argument("--chunk-size", type=positive, default=25000)
    p.set_defaults(func=predict)
    args = parser.parse_args()
    if args.command == "entrenar" and (not 0 < args.test_size < 1 or args.jobs == 0 or args.jobs < -1):
        parser.error("--test-size debe estar entre 0 y 1; --jobs debe ser -1 o positivo.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    try:
        args.func(args)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

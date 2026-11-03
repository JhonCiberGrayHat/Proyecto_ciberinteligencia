"""Muestreo por clase y depuración de datos."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET, EXCLUDE, LOG
from .data import canonical, chunks, csv_files, header, numeric, resolve_data


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

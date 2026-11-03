"""Descarga e inspección del conjunto de datos."""
from __future__ import annotations

from pathlib import Path

from .config import DATASET, ROOT
from .data import chunks, csv_files, header, resolve_data


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

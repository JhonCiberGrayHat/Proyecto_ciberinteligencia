"""Serialización de reportes y visualización de resultados."""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


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

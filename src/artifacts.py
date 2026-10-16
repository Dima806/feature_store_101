"""Save notebook outputs consistently: figures as PNG, numbers/text as JSON.

Every notebook figure goes to ``outputs/figures/<name>.png`` and every textual or
numerical result to ``outputs/json/<name>.json`` (so the Medium article, the
README, and the Streamlit app can cite exact numbers rather than screenshots).

The palette is the validated data-viz default (colorblind-safe in fixed slot
order: blue = honest/correct, orange = naive/divergent, aqua = third series).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.config import PROJECT_ROOT

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
JSON_DIR = OUTPUTS_DIR / "json"

# --- validated categorical palette (fixed slot order) ----------------------
BLUE = "#2a78d6"  # slot 1 — honest / correct / point-in-time / ministore
ORANGE = "#eb6834"  # slot 2 — naive / leaked / divergent / skew
AQUA = "#1baf7a"  # slot 3 — a third series (e.g. Feast)
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

# --- chrome & ink ----------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    """Apply a recessive, ink-on-light Matplotlib style for every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "text.color": TEXT_PRIMARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: Figure, name: str) -> Path:
    """Write ``fig`` to ``outputs/figures/<name>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def save_json(data: dict[str, Any], name: str) -> Path:
    """Write ``data`` to ``outputs/json/<name>.json`` and return the path."""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    path = JSON_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=_json_default))
    return path

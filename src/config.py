"""Typed configuration loaded from ``config/settings.yaml``.

Notebooks and modules read everything (seeds, windows, rates, paths) from here
rather than hard-coding values, so the whole project stays reproducible and a
single edit re-tunes every experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- shared column names (one vocabulary across every module) --------------
ENTITY_COL = "customer_id"
EVENT_TS_COL = "event_timestamp"
FEATURE_COL = "txn_count_30d"
LABEL_COL = "label"
TRUTH_COL = "txn_count_30d_true"
FUTURE_COL = "is_future"
N_TXN_COL = "n_txn"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class Settings(BaseSettings):
    """All knobs for the synthetic scenario and the evaluation."""

    model_config = SettingsConfigDict(env_prefix="FS101_", extra="ignore")

    seed: int = 7
    data_dir: str = "data"

    n_customers: int = 2000
    start_date: str = "2024-01-01"
    n_days: int = 180
    window_days: int = 30

    event_start_day: int = 40
    event_end_day: int = 171
    future_split_day: int = 135

    rate_low: float = 0.05
    rate_high: float = 0.80

    label_alpha: float = 0.12
    label_mu: float = 12.0
    label_bias: float = 0.0

    spike_factor: float = 1.8

    skew_window_days: int = 14

    test_size: float = 0.30

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        """Build ``Settings`` from the YAML file (env vars still override)."""
        path = path or DEFAULT_CONFIG_PATH
        data: dict[str, Any] = {}
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
        return cls(**data)

    @property
    def data_path(self) -> Path:
        """Absolute path to the data directory (created on demand)."""
        root = Path(self.data_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return root

    def file(self, name: str) -> Path:
        """Absolute path to a named data file, e.g. ``events.parquet``."""
        return self.data_path / name


# A ready-to-use instance; call ``Settings.load()`` for an explicit path.
settings = Settings.load()

__all__ = [
    "ENTITY_COL",
    "EVENT_TS_COL",
    "FEATURE_COL",
    "FUTURE_COL",
    "LABEL_COL",
    "N_TXN_COL",
    "TRUTH_COL",
    "Settings",
    "settings",
]

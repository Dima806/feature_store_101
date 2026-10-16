"""Materialize: push the latest offline values into the online store.

This is the link that guarantees train/serve consistency *mechanically*: the
served value is the most recent offline value as of the materialization time, so
it descends from the very same definition (and the very same history) as the
value a model trained on. Serving never recomputes the feature on its own.
"""

from __future__ import annotations

import pandas as pd

from src.config import ENTITY_COL, EVENT_TS_COL


def latest_as_of(
    feature_history: pd.DataFrame,
    as_of: pd.Timestamp,
    features: list[str],
    entity_col: str = ENTITY_COL,
    timestamp_col: str = EVENT_TS_COL,
) -> pd.DataFrame:
    """The most recent value per entity at or before ``as_of`` (one row each)."""
    history = feature_history[feature_history[timestamp_col] <= as_of]
    latest = history.sort_values(timestamp_col).groupby(entity_col, as_index=False).last()
    return latest[[entity_col, *features, timestamp_col]].reset_index(drop=True)

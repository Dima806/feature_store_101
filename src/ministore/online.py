"""The online store: a latest-value key-value table for fast serving.

The online store is a lookup, not a mystery: one row per entity holding the most
recent feature values, backed by SQLite (stdlib). It is *populated from* the
offline definitions (see :mod:`src.ministore.materialize`), never computed on a
separate code path — which is exactly what keeps serving consistent with
training.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import ENTITY_COL, EVENT_TS_COL

_TABLE = "online_features"


class OnlineStore:
    """A SQLite-backed key-value store of the latest value per entity."""

    def __init__(
        self,
        path: str = ":memory:",
        entity_col: str = ENTITY_COL,
        timestamp_col: str = EVENT_TS_COL,
    ) -> None:
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self._conn = sqlite3.connect(path)
        self._features: list[str] = []

    def write(self, latest: pd.DataFrame, features: list[str]) -> None:
        """Replace the online table with one row per entity (the latest values)."""
        self._features = list(features)
        frame = latest[[self.entity_col, *features, self.timestamp_col]].copy()
        # Store the timestamp as ISO text so SQLite round-trips it cleanly.
        frame[self.timestamp_col] = pd.to_datetime(frame[self.timestamp_col]).astype(str)
        frame.to_sql(_TABLE, self._conn, if_exists="replace", index=False)
        self._conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{_TABLE} ON {_TABLE}({self.entity_col})"
        )
        self._conn.commit()

    def get_online_features(
        self, entity_ids: list[int] | pd.Series, features: list[str] | None = None
    ) -> pd.DataFrame:
        """Look up the latest values for the given entity ids (serving path)."""
        features = features or self._features
        ids = list(pd.Series(entity_ids).tolist())
        placeholders = ",".join("?" for _ in ids)
        cols = ", ".join([self.entity_col, *features])
        query = f"SELECT {cols} FROM {_TABLE} WHERE {self.entity_col} IN ({placeholders})"
        result = pd.read_sql_query(query, self._conn, params=ids)
        # Preserve the requested order and surface misses as NaN.
        return (
            pd.DataFrame({self.entity_col: ids})
            .merge(result, on=self.entity_col, how="left")
            .reset_index(drop=True)
        )

    def close(self) -> None:
        self._conn.close()

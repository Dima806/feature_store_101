"""The offline store: timestamped history + the point-in-time (as-of) join.

This is the heart of the project. For each ``(entity, event_time)`` request
row, the point-in-time join fetches the most recent feature value *at or before*
that event time — and never after. That single rule is what makes a training set
honest: a label from six months ago is paired with the feature as it was known
then, not the value computed today.

It is just ``pandas.merge_asof(..., direction="backward")`` over a history
sorted by time, grouped by the entity key.
"""

from __future__ import annotations

import pandas as pd

from src.config import ENTITY_COL, EVENT_TS_COL


class OfflineStore:
    """A timestamped feature history that answers point-in-time requests."""

    def __init__(
        self,
        feature_history: pd.DataFrame,
        entity_col: str = ENTITY_COL,
        timestamp_col: str = EVENT_TS_COL,
    ) -> None:
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self._history = feature_history.sort_values(timestamp_col).reset_index(drop=True)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        keep_feature_timestamp: bool = False,
    ) -> pd.DataFrame:
        """Point-in-time join: attach the value known at/before each event time.

        ``entity_df`` must carry the entity key and ``timestamp_col``. The result
        has one row per request with the requested features joined as-of. Set
        ``keep_feature_timestamp`` to also return when each value was recorded,
        which the tests use to assert no value comes from after the event time.
        """
        ts = self.timestamp_col
        key = self.entity_col
        feature_ts = "_feature_timestamp"

        right_cols = [key, ts, *features]
        right = self._history[right_cols].sort_values(ts)
        if keep_feature_timestamp:
            right = right.assign(**{feature_ts: right[ts]})

        left = entity_df.sort_values(ts)
        merged = pd.merge_asof(
            left,
            right,
            on=ts,
            by=key,
            direction="backward",  # value at or before the event time; never after
            allow_exact_matches=True,
        )
        return merged.reset_index(drop=True)

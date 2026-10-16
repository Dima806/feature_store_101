"""The ~100-line feature store: the two methods every feature store exposes.

``get_historical_features`` builds training sets with a point-in-time join over
the offline store; ``get_online_features`` serves the latest values via a key
lookup over the online store. ``materialize`` links them. Assembled from the
registry, offline store, online store and materialize step, this is — in
miniature — exactly what Feast does for you locally.
"""

from __future__ import annotations

import pandas as pd

from src.config import ENTITY_COL, EVENT_TS_COL
from src.ministore.materialize import latest_as_of
from src.ministore.offline import OfflineStore
from src.ministore.online import OnlineStore
from src.ministore.registry import Registry


class FeatureStore:
    """Registry + offline store + online store, wired together."""

    def __init__(
        self,
        feature_history: pd.DataFrame,
        registry: Registry | None = None,
        online_path: str = ":memory:",
        entity_col: str = ENTITY_COL,
        timestamp_col: str = EVENT_TS_COL,
    ) -> None:
        self.registry = registry or Registry()
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self._history = feature_history
        self.offline = OfflineStore(feature_history, entity_col, timestamp_col)
        self.online = OnlineStore(online_path, entity_col, timestamp_col)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        keep_feature_timestamp: bool = False,
    ) -> pd.DataFrame:
        """Point-in-time training set: feature values known at each event time."""
        return self.offline.get_historical_features(
            entity_df, features, keep_feature_timestamp=keep_feature_timestamp
        )

    def materialize(self, as_of: pd.Timestamp, features: list[str]) -> pd.DataFrame:
        """Push the latest offline values (as of ``as_of``) into the online store."""
        latest = latest_as_of(self._history, as_of, features, self.entity_col, self.timestamp_col)
        self.online.write(latest, features)
        return latest

    def get_online_features(
        self, entity_ids: list[int] | pd.Series, features: list[str] | None = None
    ) -> pd.DataFrame:
        """Serving: look up the latest materialized values per entity."""
        return self.online.get_online_features(entity_ids, features)

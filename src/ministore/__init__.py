"""The feature store, built from scratch in ~100 lines.

Three parts plus a link:
- :mod:`src.ministore.registry`  — the shared catalog of what features mean.
- :mod:`src.ministore.offline`   — timestamped history + point-in-time join.
- :mod:`src.ministore.online`    — latest-value key-value store for serving.
- :mod:`src.ministore.materialize` — pushes offline latest -> online.
- :mod:`src.ministore.store`     — get_historical_features + get_online_features.
"""

from src.ministore.offline import OfflineStore
from src.ministore.online import OnlineStore
from src.ministore.registry import Entity, Feature, FeatureView, Registry
from src.ministore.store import FeatureStore

__all__ = [
    "Entity",
    "Feature",
    "FeatureStore",
    "FeatureView",
    "OfflineStore",
    "OnlineStore",
    "Registry",
]

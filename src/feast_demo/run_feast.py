"""Drive the local Feast store end to end: apply, materialize, retrieve.

Everything is local and keyless (``provider: local``, file offline store, SQLite
online store, no server, no Docker). Notebook 05 calls :func:`demo`; the parity
test uses :mod:`src.evaluation.parity`, which builds an equivalent store in a
temp dir.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.config import ENTITY_COL, EVENT_TS_COL, FEATURE_COL


def demo(feature_history: pd.DataFrame, entity_df: pd.DataFrame, repo_dir: Path) -> dict:
    """Apply, materialize, and run both retrievals on a local Feast store.

    Returns a dict with the point-in-time training set (``historical``) and the
    online serving lookup (``online``).
    """
    from feast import Entity, FeatureStore, FeatureView, Field, FileSource
    from feast.repo_config import RepoConfig
    from feast.types import Int64
    from feast.value_type import ValueType

    repo_dir.mkdir(parents=True, exist_ok=True)
    source_path = repo_dir / "feature_history.parquet"
    history = feature_history[[ENTITY_COL, EVENT_TS_COL, FEATURE_COL]].copy()
    history[EVENT_TS_COL] = pd.to_datetime(history[EVENT_TS_COL], utc=True)
    history.to_parquet(source_path, index=False)

    source = FileSource(path=str(source_path), timestamp_field=EVENT_TS_COL)
    customer = Entity(name="customer", join_keys=[ENTITY_COL], value_type=ValueType.INT64)
    view = FeatureView(
        name="customer_txn",
        entities=[customer],
        ttl=timedelta(days=36500),
        schema=[Field(name=FEATURE_COL, dtype=Int64)],
        source=source,
        online=True,
    )

    config = RepoConfig(
        project="fs101",
        provider="local",
        registry=str(repo_dir / "registry.db"),
        online_store={"type": "sqlite", "path": str(repo_dir / "online.db")},
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
    )
    store = FeatureStore(config=config)
    store.apply([customer, view])

    request = entity_df[[ENTITY_COL, EVENT_TS_COL]].copy()
    request[EVENT_TS_COL] = pd.to_datetime(request[EVENT_TS_COL], utc=True)
    historical = store.get_historical_features(
        entity_df=request, features=["customer_txn:txn_count_30d"]
    ).to_df()

    end = history[EVENT_TS_COL].max().to_pydatetime()
    store.materialize(start_date=history[EVENT_TS_COL].min().to_pydatetime(), end_date=end)
    online = store.get_online_features(
        features=["customer_txn:txn_count_30d"],
        entity_rows=[{ENTITY_COL: int(c)} for c in request[ENTITY_COL].unique()],
    ).to_df()

    return {"historical": historical, "online": online}

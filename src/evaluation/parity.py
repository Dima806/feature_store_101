"""From-scratch store vs local Feast: identical point-in-time training sets.

Feast is an optional dependency and is imported lazily, so the core project and
CI run with or without it. When present, ``feast_historical_features`` stands up
a throwaway *local* Feast store (``provider: local``, file offline + SQLite
online, no server, no account) over the same history and returns its
point-in-time join, which :func:`compare` lines up against the mini-store's.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.config import ENTITY_COL, EVENT_TS_COL, FEATURE_COL
from src.ministore.store import FeatureStore

FEAST_VIEW = "customer_txn"


def ministore_historical_features(
    feature_history: pd.DataFrame, entity_df: pd.DataFrame
) -> pd.DataFrame:
    """Point-in-time join from the hundred-line store, sorted for comparison."""
    store = FeatureStore(feature_history)
    joined = store.get_historical_features(entity_df[[ENTITY_COL, EVENT_TS_COL]], [FEATURE_COL])
    return (
        joined[[ENTITY_COL, EVENT_TS_COL, FEATURE_COL]]
        .sort_values([ENTITY_COL, EVENT_TS_COL])
        .reset_index(drop=True)
    )


def feast_historical_features(
    feature_history: pd.DataFrame, entity_df: pd.DataFrame, repo_dir: Path
) -> pd.DataFrame:
    """Point-in-time join from a local Feast store built in ``repo_dir``."""
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
        name=FEAST_VIEW,
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
    result = store.get_historical_features(
        entity_df=request, features=[f"{FEAST_VIEW}:{FEATURE_COL}"]
    ).to_df()
    return (
        result[[ENTITY_COL, EVENT_TS_COL, FEATURE_COL]]
        .sort_values([ENTITY_COL, EVENT_TS_COL])
        .reset_index(drop=True)
    )


def compare(
    feature_history: pd.DataFrame, entity_df: pd.DataFrame, repo_dir: Path
) -> pd.DataFrame:
    """Return both point-in-time results aligned side by side for assertion.

    Rows are matched by entity and tz-normalised event time; the result has
    ``{FEATURE_COL}_ministore`` and ``{FEATURE_COL}_feast`` columns to compare.
    """
    mine = ministore_historical_features(feature_history, entity_df).assign(
        _ts=lambda d: pd.to_datetime(d[EVENT_TS_COL], utc=True)
    )
    theirs = feast_historical_features(feature_history, entity_df, repo_dir).assign(
        _ts=lambda d: pd.to_datetime(d[EVENT_TS_COL], utc=True)
    )
    return mine.merge(
        theirs,
        on=[ENTITY_COL, "_ts"],
        suffixes=("_ministore", "_feast"),
    )

"""The online store returns the latest materialized value per entity."""

from __future__ import annotations

from src.config import ENTITY_COL, EVENT_TS_COL, FEATURE_COL
from src.datasets.events import Dataset
from src.ministore.store import FeatureStore


def test_online_returns_latest_per_entity(dataset: Dataset) -> None:
    store = FeatureStore(dataset.feature_history)
    as_of = dataset.feature_history[EVENT_TS_COL].max()
    store.materialize(as_of, [FEATURE_COL])

    expected = (
        dataset.feature_history.sort_values(EVENT_TS_COL)
        .groupby(ENTITY_COL, as_index=False)
        .last()[[ENTITY_COL, FEATURE_COL]]
    )
    got = store.get_online_features(expected[ENTITY_COL].tolist(), [FEATURE_COL])
    merged = expected.merge(got, on=ENTITY_COL, suffixes=("_expected", "_online"))
    assert (merged[f"{FEATURE_COL}_expected"] == merged[f"{FEATURE_COL}_online"]).all()


def test_store_serves_both_methods_end_to_end(dataset: Dataset) -> None:
    store = FeatureStore(dataset.feature_history)
    as_of = dataset.feature_history[EVENT_TS_COL].max()
    store.materialize(as_of, [FEATURE_COL])

    requests = dataset.events[[ENTITY_COL, EVENT_TS_COL]].head(5)
    historical = store.get_historical_features(requests, [FEATURE_COL])
    assert FEATURE_COL in historical.columns
    assert len(historical) == 5

    online = store.get_online_features(requests[ENTITY_COL].tolist(), [FEATURE_COL])
    assert FEATURE_COL in online.columns
    assert len(online) == 5
    assert online[FEATURE_COL].notna().all()

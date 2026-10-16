"""The point-in-time join is correct by construction and never peeks ahead."""

from __future__ import annotations

from src.config import ENTITY_COL, EVENT_TS_COL, FEATURE_COL, TRUTH_COL
from src.datasets.events import Dataset
from src.features.definitions import txn_count_window
from src.ministore.store import FeatureStore


def test_asof_reproduces_known_truth(dataset: Dataset) -> None:
    """The as-of join returns exactly the honest value known at each event time."""
    store = FeatureStore(dataset.feature_history)
    joined = store.get_historical_features(
        dataset.events[[ENTITY_COL, EVENT_TS_COL]], [FEATURE_COL]
    )
    merged = dataset.events.merge(joined[[ENTITY_COL, FEATURE_COL]], on=ENTITY_COL, how="left")
    assert (merged[FEATURE_COL] == merged[TRUTH_COL]).all()


def test_never_uses_a_value_after_the_event_time(dataset: Dataset) -> None:
    """Every matched feature value is recorded at or before its event time."""
    store = FeatureStore(dataset.feature_history)
    joined = store.get_historical_features(
        dataset.events[[ENTITY_COL, EVENT_TS_COL]],
        [FEATURE_COL],
        keep_feature_timestamp=True,
    )
    assert (joined["_feature_timestamp"] <= joined[EVENT_TS_COL]).all()


def test_shared_definition_matches_truth(dataset: Dataset, cfg) -> None:
    """The one shared definition, recomputed from raw activity, equals truth."""
    requests = dataset.events[[ENTITY_COL, EVENT_TS_COL]]
    computed = txn_count_window(dataset.activity, requests, cfg.window_days)
    assert (computed == dataset.events[TRUTH_COL].to_numpy()).all()

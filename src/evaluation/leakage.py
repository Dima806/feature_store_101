"""The showpiece: the naive latest-value join leaks the future; as-of does not.

We build the same training set two ways and compare four AUCs:

- ``naive_offline``  — naive model on naive (latest-value) features. Inflated.
- ``honest_offline`` — honest model on point-in-time features. The real number.
- ``honest_future``  — honest model on a true-future holdout. Holds.
- ``naive_future``   — naive model on that same holdout, where the only features
  available are point-in-time (reality). It loses its leaked signal and falls
  back to the honest level.

The gap ``naive_offline - naive_future`` is leakage: it was never skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    ENTITY_COL,
    EVENT_TS_COL,
    FEATURE_COL,
    FUTURE_COL,
    LABEL_COL,
    Settings,
    settings,
)
from src.datasets.events import Dataset
from src.ministore.store import FeatureStore


@dataclass
class LeakageResult:
    """The four headline AUCs of the leakage experiment."""

    naive_offline_auc: float
    honest_offline_auc: float
    naive_future_auc: float
    honest_future_auc: float

    @property
    def inflation(self) -> float:
        """How much the naive offline number overstates the honest one."""
        return self.naive_offline_auc - self.honest_offline_auc

    @property
    def collapse(self) -> float:
        """How far the naive model falls from offline to the true future."""
        return self.naive_offline_auc - self.naive_future_auc


def _fit(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(x.reshape(-1, 1), y)


def _auc(model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, model.predict_proba(x.reshape(-1, 1))[:, 1]))


def run_leakage_experiment(dataset: Dataset, cfg: Settings | None = None) -> LeakageResult:
    """Train naive and honest models and score them offline and on the future."""
    cfg = cfg or settings
    events = dataset.events
    store = FeatureStore(dataset.feature_history)
    as_of = dataset.feature_history[EVENT_TS_COL].max()
    store.materialize(as_of, [FEATURE_COL])

    def honest_features(frame: pd.DataFrame) -> np.ndarray:
        joined = store.get_historical_features(frame[[ENTITY_COL, EVENT_TS_COL]], [FEATURE_COL])
        merged = frame[[ENTITY_COL]].merge(
            joined[[ENTITY_COL, FEATURE_COL]], on=ENTITY_COL, how="left"
        )
        return merged[FEATURE_COL].to_numpy(dtype=float)

    def naive_features(frame: pd.DataFrame) -> np.ndarray:
        online = store.get_online_features(frame[ENTITY_COL].tolist(), [FEATURE_COL])
        merged = frame[[ENTITY_COL]].merge(online, on=ENTITY_COL, how="left")
        return merged[FEATURE_COL].to_numpy(dtype=float)

    offline = events[~events[FUTURE_COL]]
    future = events[events[FUTURE_COL]]
    train, test = train_test_split(
        offline,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=offline[LABEL_COL],
    )

    y_train = train[LABEL_COL].to_numpy()
    y_test = test[LABEL_COL].to_numpy()
    y_future = future[LABEL_COL].to_numpy()

    honest_model = _fit(honest_features(train), y_train)
    naive_model = _fit(naive_features(train), y_train)

    honest_offline = _auc(honest_model, honest_features(test), y_test)
    naive_offline = _auc(naive_model, naive_features(test), y_test)

    # On a genuine future event the only feature available is point-in-time.
    future_honest = honest_features(future)
    honest_future = _auc(honest_model, future_honest, y_future)
    naive_future = _auc(naive_model, future_honest, y_future)

    return LeakageResult(
        naive_offline_auc=naive_offline,
        honest_offline_auc=honest_offline,
        naive_future_auc=naive_future,
        honest_future_auc=honest_future,
    )

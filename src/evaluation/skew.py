"""Train/serve skew: the same feature computed two ways, silently disagreeing.

The feature is defined once (:func:`src.features.definitions.txn_count_window`).
Training uses the correct window; a serving reimplementation uses a different
window. Both run without error, but they disagree on most rows, and the model is
served inputs it never trained on — so the production metric slips. Routing both
paths through the one shared definition (the same window) removes the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ENTITY_COL, EVENT_TS_COL, LABEL_COL, Settings, settings
from src.datasets.events import Dataset
from src.features.definitions import txn_count_window


@dataclass
class SkewResult:
    """How far the two implementations disagree, and what it costs.

    The sharp lesson: AUC (a ranking metric) barely moves, so a quick offline
    check misses the skew entirely, while accuracy at the deployed 0.5 threshold
    collapses because the served feature is on a different scale than training.
    """

    disagreement_rate: float  # fraction of rows where the two values differ
    mean_abs_diff: float  # average absolute difference in the feature
    correct_auc: float  # model served the shared (correct) feature
    skew_auc: float  # same model served the divergent feature
    correct_accuracy: float  # accuracy @0.5 on the shared feature
    skew_accuracy: float  # accuracy @0.5 on the divergent feature

    @property
    def auc_slip(self) -> float:
        """AUC lost to serving the divergent feature (often misleadingly small)."""
        return self.correct_auc - self.skew_auc

    @property
    def accuracy_slip(self) -> float:
        """Accuracy lost at the deployed threshold — the real production damage."""
        return self.correct_accuracy - self.skew_accuracy

    @property
    def slip(self) -> float:
        """Back-compat alias for :attr:`auc_slip`."""
        return self.auc_slip


def run_skew_experiment(dataset: Dataset, cfg: Settings | None = None) -> SkewResult:
    """Measure feature disagreement and the production slip it causes."""
    cfg = cfg or settings
    events = dataset.events
    requests = events[[ENTITY_COL, EVENT_TS_COL]]

    correct = txn_count_window(dataset.activity, requests, cfg.window_days)
    divergent = txn_count_window(dataset.activity, requests, cfg.skew_window_days)

    disagreement_rate = float(np.mean(correct != divergent))
    mean_abs_diff = float(np.mean(np.abs(correct - divergent)))

    frame = events.assign(_correct=correct.astype(float), _divergent=divergent.astype(float))
    train, test = train_test_split(
        frame,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=frame[LABEL_COL],
    )

    model = LogisticRegression(max_iter=1000).fit(
        train[["_correct"]].to_numpy(), train[LABEL_COL].to_numpy()
    )
    y_test = test[LABEL_COL].to_numpy()
    proba_correct = model.predict_proba(test[["_correct"]].to_numpy())[:, 1]
    proba_skew = model.predict_proba(test[["_divergent"]].to_numpy())[:, 1]

    return SkewResult(
        disagreement_rate=disagreement_rate,
        mean_abs_diff=mean_abs_diff,
        correct_auc=float(roc_auc_score(y_test, proba_correct)),
        skew_auc=float(roc_auc_score(y_test, proba_skew)),
        correct_accuracy=float(accuracy_score(y_test, proba_correct >= 0.5)),
        skew_accuracy=float(accuracy_score(y_test, proba_skew >= 0.5)),
    )

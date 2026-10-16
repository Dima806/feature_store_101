"""The naive latest-value join inflates offline AUC and collapses on the future."""

from __future__ import annotations

from src.config import Settings
from src.datasets.events import Dataset
from src.evaluation.leakage import run_leakage_experiment


def test_naive_join_inflates_and_collapses(dataset: Dataset, cfg: Settings) -> None:
    result = run_leakage_experiment(dataset, cfg)

    # Inflation: the naive offline number is materially higher than the honest one.
    assert result.naive_offline_auc > result.honest_offline_auc + 0.05

    # The honest number is honest: it holds from offline to the true future.
    assert abs(result.honest_offline_auc - result.honest_future_auc) < 0.05

    # Collapse: on a genuine future holdout the naive model loses its edge...
    assert result.naive_offline_auc > result.naive_future_auc + 0.05

    # ...falling to about the honest model's level. The gap was leakage, not skill.
    assert abs(result.naive_future_auc - result.honest_future_auc) < 0.03

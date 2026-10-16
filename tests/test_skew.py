"""Train/serve skew: two implementations disagree and the production metric slips."""

from __future__ import annotations

from src.config import Settings
from src.datasets.events import Dataset
from src.evaluation.skew import run_skew_experiment


def test_divergent_implementation_disagrees_and_costs(dataset: Dataset, cfg: Settings) -> None:
    result = run_skew_experiment(dataset, cfg)

    # The two implementations silently disagree on a large share of rows.
    assert result.disagreement_rate > 0.2
    assert result.mean_abs_diff > 0.0

    # AUC barely notices the skew — which is exactly why it goes unnoticed...
    assert result.auc_slip >= 0.0

    # ...but accuracy at the deployed threshold takes a measurable hit.
    assert result.correct_accuracy >= result.skew_accuracy
    assert result.accuracy_slip > 0.02

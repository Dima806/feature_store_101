"""The arena harness: run the experiments and collect the headline numbers."""

from __future__ import annotations

import pandas as pd

from src.config import Settings, settings
from src.datasets.events import Dataset
from src.evaluation.leakage import run_leakage_experiment
from src.evaluation.skew import run_skew_experiment


def run_all(dataset: Dataset, cfg: Settings | None = None) -> pd.DataFrame:
    """Return a tidy table of the leakage and skew results for display."""
    cfg = cfg or settings
    leak = run_leakage_experiment(dataset, cfg)
    skew = run_skew_experiment(dataset, cfg)
    rows = [
        ("leakage", "naive offline AUC", leak.naive_offline_auc),
        ("leakage", "honest offline AUC", leak.honest_offline_auc),
        ("leakage", "naive future AUC", leak.naive_future_auc),
        ("leakage", "honest future AUC", leak.honest_future_auc),
        ("leakage", "inflation (naive-honest offline)", leak.inflation),
        ("leakage", "collapse (naive offline-future)", leak.collapse),
        ("skew", "feature disagreement rate", skew.disagreement_rate),
        ("skew", "mean abs feature diff", skew.mean_abs_diff),
        ("skew", "correct-feature AUC", skew.correct_auc),
        ("skew", "divergent-feature AUC", skew.skew_auc),
        ("skew", "AUC slip (hides the skew)", skew.auc_slip),
        ("skew", "correct-feature accuracy", skew.correct_accuracy),
        ("skew", "divergent-feature accuracy", skew.skew_accuracy),
        ("skew", "accuracy slip (the real damage)", skew.accuracy_slip),
    ]
    return pd.DataFrame(rows, columns=pd.Index(["experiment", "metric", "value"]))

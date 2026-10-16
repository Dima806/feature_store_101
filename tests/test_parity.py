"""From-scratch get_historical_features matches local Feast, value for value."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import FEATURE_COL, Settings
from src.datasets.events import generate


def test_from_scratch_matches_feast(tmp_path: Path) -> None:
    pytest.importorskip("feast")
    from src.evaluation.parity import compare

    cfg = Settings.load().model_copy(update={"n_customers": 60})
    dataset = generate(cfg)
    requests = dataset.events.head(40)

    merged = compare(dataset.feature_history, requests, tmp_path / "feast_repo")

    assert len(merged) == len(requests)
    mine = merged[f"{FEATURE_COL}_ministore"].to_numpy()
    feast = merged[f"{FEATURE_COL}_feast"].to_numpy()
    assert np.array_equal(mine, feast)

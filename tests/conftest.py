"""Shared fixtures: a single seeded dataset built once for the whole suite."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.datasets.events import Dataset, generate


@pytest.fixture(scope="session")
def cfg() -> Settings:
    return Settings.load()


@pytest.fixture(scope="session")
def dataset(cfg: Settings) -> Dataset:
    return generate(cfg)

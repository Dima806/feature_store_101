"""Synthetic entity-timestamp event stream with a planted, time-varying feature.

The whole project's central claim — "the naive join leaks the future, the
point-in-time join does not" — is only *measurable* if the honest feature value
at every event time is known by construction. This generator guarantees that.

The scenario
------------
Each customer transacts over time. The feature is ``txn_count_30d`` (the count
of transactions in the trailing 30 days). For each customer we pick one labeled
event; its label depends ONLY on the feature value *as of* the event time, so a
point-in-time join recovers the true relationship exactly.

The leak: after a positive (``label == 1``) event the customer's transaction
rate is multiplied by ``spike_factor``. That future behaviour inflates the
*latest* 30-day count — the value a naive "join the latest value" training set
would use — so the latest value encodes the outcome. Because the spike only
happens strictly *after* the event, the value at or before the event time is
untouched: ground truth stays clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ENTITY_COL,
    EVENT_TS_COL,
    FEATURE_COL,
    FUTURE_COL,
    LABEL_COL,
    N_TXN_COL,
    TRUTH_COL,
    Settings,
    settings,
)
from src.features.definitions import rolling_window_count


@dataclass
class Dataset:
    """The three tables the rest of the project consumes.

    - ``events``: one labeled row per customer (the training/serving requests).
    - ``feature_history``: the offline store's source — a daily timestamped
      history of ``txn_count_30d`` per customer.
    - ``activity``: the raw daily transaction counts, used to recompute the
      feature a *different* way for the train/serve-skew demo.
    """

    events: pd.DataFrame
    feature_history: pd.DataFrame
    activity: pd.DataFrame


def generate(cfg: Settings | None = None) -> Dataset:
    """Generate the planted-truth dataset deterministically from ``cfg``."""
    cfg = cfg or settings
    rng = np.random.default_rng(cfg.seed)

    n_customers = cfg.n_customers
    n_days = cfg.n_days
    window = cfg.window_days
    start = pd.Timestamp(cfg.start_date)
    customer_ids = np.arange(n_customers)

    # Baseline transaction intensity and the single labeled event per customer.
    rates = rng.uniform(cfg.rate_low, cfg.rate_high, n_customers)
    event_day = rng.integers(cfg.event_start_day, cfg.event_end_day, n_customers)

    # Baseline daily counts, then the honest as-of feature at each event time.
    daily = rng.poisson(rates[:, None], size=(n_customers, n_days)).astype(np.int64)
    feat_baseline = rolling_window_count(daily, window)
    f_asof = feat_baseline[customer_ids, event_day]

    # Label depends ONLY on the as-of feature (clip keeps the sigmoid stable).
    z = np.clip(cfg.label_alpha * (f_asof - cfg.label_mu) + cfg.label_bias, -30, 30)
    prob = 1.0 / (1.0 + np.exp(-z))
    label = (rng.random(n_customers) < prob).astype(np.int64)

    # Inject the leak: strictly AFTER a positive event, multiply the rate.
    day_index = np.arange(n_days)[None, :]
    after_event = day_index > event_day[:, None]
    leak_mask = after_event & (label[:, None] == 1)
    spiked_draw = rng.poisson(
        (rates * cfg.spike_factor)[:, None], size=(n_customers, n_days)
    ).astype(np.int64)
    daily_leaked = np.where(leak_mask, spiked_draw, daily)

    # The feature history the offline store serves (leak now visible post-event).
    feat_full = rolling_window_count(daily_leaked, window)

    dates = start + pd.to_timedelta(np.arange(n_days), unit="D")

    feature_history = pd.DataFrame(
        {
            ENTITY_COL: np.repeat(customer_ids, n_days),
            EVENT_TS_COL: np.tile(dates.to_numpy(), n_customers),
            FEATURE_COL: feat_full.reshape(-1),
        }
    )

    events = pd.DataFrame(
        {
            ENTITY_COL: customer_ids,
            EVENT_TS_COL: start + pd.to_timedelta(event_day, unit="D"),
            LABEL_COL: label,
            TRUTH_COL: f_asof,  # the honest as-of feature, known by construction
            FUTURE_COL: event_day >= cfg.future_split_day,
        }
    )

    # Raw daily activity (nonzero days only) for the skew reimplementation.
    cust_grid = np.repeat(customer_ids, n_days)
    date_grid = np.tile(dates.to_numpy(), n_customers)
    counts = daily_leaked.reshape(-1)
    nonzero = counts > 0
    activity = pd.DataFrame(
        {
            ENTITY_COL: cust_grid[nonzero],
            EVENT_TS_COL: date_grid[nonzero],
            N_TXN_COL: counts[nonzero],
        }
    )

    return Dataset(events=events, feature_history=feature_history, activity=activity)


def write_parquet(dataset: Dataset, cfg: Settings | None = None) -> None:
    """Persist the dataset as local parquet under the configured data dir."""
    cfg = cfg or settings
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    dataset.events.to_parquet(cfg.file("events.parquet"), index=False)
    dataset.feature_history.to_parquet(cfg.file("feature_history.parquet"), index=False)
    dataset.activity.to_parquet(cfg.file("activity.parquet"), index=False)


def load(cfg: Settings | None = None) -> Dataset:
    """Load the dataset from parquet, generating and writing it if missing."""
    cfg = cfg or settings
    events_path = cfg.file("events.parquet")
    if not events_path.exists():
        dataset = generate(cfg)
        write_parquet(dataset, cfg)
        return dataset
    return Dataset(
        events=pd.read_parquet(events_path),
        feature_history=pd.read_parquet(cfg.file("feature_history.parquet")),
        activity=pd.read_parquet(cfg.file("activity.parquet")),
    )


def main() -> None:
    """``python -m src.datasets.events`` — generate and write the streams."""
    cfg = settings
    dataset = generate(cfg)
    write_parquet(dataset, cfg)
    n_pos = int(dataset.events[LABEL_COL].sum())
    n_future = int(dataset.events[FUTURE_COL].sum())
    print(
        f"wrote {len(dataset.events):,} events "
        f"({n_pos:,} positive, {n_future:,} future-holdout), "
        f"{len(dataset.feature_history):,} feature-history rows, "
        f"{len(dataset.activity):,} activity rows -> {cfg.data_path}"
    )


if __name__ == "__main__":
    main()

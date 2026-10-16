"""The single, shared definition of every feature.

Train/serve consistency is structural: if the training pipeline and the serving
path each hand-write the feature, they drift (a different window, a different
null rule) and the model is silently fed inputs it never trained on. The cure is
*one* definition that both paths call. ``txn_count_window`` is that definition
for ``txn_count_30d``.

The skew demo (:mod:`src.evaluation.skew`) deliberately calls this one function
with the *wrong* window to reproduce the bug, then shows that routing both paths
through the same call with the same window makes the gap vanish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ENTITY_COL, EVENT_TS_COL, FEATURE_COL, N_TXN_COL

DEFAULT_WINDOW_DAYS = 30


def rolling_window_count(daily: np.ndarray, window: int) -> np.ndarray:
    """Trailing ``window``-day sum for each (customer, day) in a (C, D) array.

    Entry ``[c, d]`` sums ``daily[c, d-window+1 .. d]`` inclusive — the count
    over the window ending at and including day ``d``.
    """
    cumsum = np.cumsum(daily, axis=1)
    out = cumsum.copy()
    out[:, window:] = cumsum[:, window:] - cumsum[:, :-window]
    return out


def txn_count_window(
    activity: pd.DataFrame,
    entity_df: pd.DataFrame,
    window_days: int = DEFAULT_WINDOW_DAYS,
    entity_col: str = ENTITY_COL,
    timestamp_col: str = EVENT_TS_COL,
    n_col: str = N_TXN_COL,
) -> np.ndarray:
    """Point-in-time windowed transaction count for each request row.

    For request ``(entity, as_of)`` this counts activity in the half-open window
    ``(as_of - window_days, as_of]`` — strictly after the window start, at or
    before the event time. This is the canonical definition both training and
    serving must call.
    """
    window = np.timedelta64(window_days, "D")
    act = activity.sort_values(timestamp_col)
    groups: dict[object, tuple[np.ndarray, np.ndarray]] = {}
    for cid, grp in act.groupby(entity_col):
        ts = grp[timestamp_col].to_numpy(dtype="datetime64[ns]")
        cum = np.cumsum(grp[n_col].to_numpy())
        groups[cid] = (ts, cum)

    ids = entity_df[entity_col].to_numpy()
    times = entity_df[timestamp_col].to_numpy(dtype="datetime64[ns]")
    result = np.zeros(len(entity_df), dtype=np.int64)
    for i in range(len(entity_df)):
        found = groups.get(ids[i])
        if found is None:
            continue
        ts, cum = found
        at_or_before = np.searchsorted(ts, times[i], side="right")
        before_window = np.searchsorted(ts, times[i] - window, side="right")
        total_hi = int(cum[at_or_before - 1]) if at_or_before > 0 else 0
        total_lo = int(cum[before_window - 1]) if before_window > 0 else 0
        result[i] = total_hi - total_lo
    return result


def txn_count_asof(
    activity: pd.DataFrame,
    customer_id: int,
    as_of: pd.Timestamp,
    window_days: int = DEFAULT_WINDOW_DAYS,
    entity_col: str = ENTITY_COL,
    timestamp_col: str = EVENT_TS_COL,
    n_col: str = N_TXN_COL,
) -> int:
    """The shared definition evaluated at a single ``(customer, as_of)`` point."""
    request = pd.DataFrame({entity_col: [customer_id], timestamp_col: [as_of]})
    return int(
        txn_count_window(activity, request, window_days, entity_col, timestamp_col, n_col)[0]
    )


def build_feature_history(
    activity: pd.DataFrame,
    dates: pd.DatetimeIndex,
    window_days: int = DEFAULT_WINDOW_DAYS,
    entity_col: str = ENTITY_COL,
    timestamp_col: str = EVENT_TS_COL,
    n_col: str = N_TXN_COL,
    feature_col: str = FEATURE_COL,
) -> pd.DataFrame:
    """Build the daily feature history from raw activity via the one definition.

    Produces the long-format table the offline store serves: one row per
    ``(customer, date)`` with the trailing-window count. Vectorized over a dense
    customer-by-date grid so it stays fast.
    """
    pivot = (
        activity.pivot_table(
            index=entity_col,
            columns=timestamp_col,
            values=n_col,
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=dates, fill_value=0)
        .sort_index()
    )
    daily = pivot.to_numpy()
    feat = rolling_window_count(daily, window_days)
    customers = pivot.index.to_numpy()
    n_days = len(dates)
    return pd.DataFrame(
        {
            entity_col: np.repeat(customers, n_days),
            timestamp_col: np.tile(dates.to_numpy(), len(customers)),
            feature_col: feat.reshape(-1),
        }
    )

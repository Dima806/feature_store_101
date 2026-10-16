"""Streamlit explorer for feature_store_101.

Four panels, all driven by the ``src`` library so nothing is reimplemented here:
a time-travel explorer, a leakage meter, an online lookup, and a from-scratch
vs Feast parity check. Run with ``make run``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    ENTITY_COL,
    EVENT_TS_COL,
    FEATURE_COL,
    settings,
)
from src.datasets.events import Dataset, generate
from src.evaluation.leakage import run_leakage_experiment
from src.ministore.store import FeatureStore


@st.cache_data
def _load() -> Dataset:
    return generate(settings)


@st.cache_resource
def _store() -> FeatureStore:
    dataset = _load()
    store = FeatureStore(dataset.feature_history)
    store.materialize(dataset.feature_history[EVENT_TS_COL].max(), [FEATURE_COL])
    return store


def time_travel_panel(dataset: Dataset) -> None:
    st.header("Time-travel explorer")
    st.caption("The value as it was known then (point-in-time) vs the latest value.")
    customer = st.selectbox("Customer", sorted(dataset.events[ENTITY_COL].unique()))
    history = dataset.feature_history
    series = history[history[ENTITY_COL] == customer].sort_values(EVENT_TS_COL)
    as_of = st.select_slider("As of", options=list(series[EVENT_TS_COL]))

    point_in_time = int(series[series[EVENT_TS_COL] <= as_of][FEATURE_COL].iloc[-1])
    latest = int(series[FEATURE_COL].iloc[-1])
    left, right = st.columns(2)
    left.metric("Point-in-time value", point_in_time)
    right.metric("Latest value (naive would use this)", latest, delta=latest - point_in_time)
    st.line_chart(series.set_index(EVENT_TS_COL)[FEATURE_COL])


def leakage_panel(dataset: Dataset) -> None:
    st.header("Leakage meter")
    st.caption("Naive 'join the latest value' inflates offline AUC; the future erases it.")
    result = run_leakage_experiment(dataset, settings)
    chart = pd.DataFrame(
        {
            "AUC": [
                result.naive_offline_auc,
                result.honest_offline_auc,
                result.naive_future_auc,
                result.honest_future_auc,
            ]
        },
        index=["naive offline", "honest offline", "naive future", "honest future"],
    )
    st.bar_chart(chart)
    st.metric("Leakage (naive offline - naive future)", f"{result.collapse:.3f}")


def online_panel(dataset: Dataset) -> None:
    st.header("Online lookup")
    st.caption("The served (latest) value vs the point-in-time value at the event time.")
    store = _store()
    customer = st.selectbox(
        "Customer", sorted(dataset.events[ENTITY_COL].unique()), key="online_customer"
    )
    event_row = dataset.events[dataset.events[ENTITY_COL] == customer].iloc[0]
    served = store.get_online_features([int(customer)], [FEATURE_COL])
    historical = store.get_historical_features(
        pd.DataFrame({ENTITY_COL: [customer], EVENT_TS_COL: [event_row[EVENT_TS_COL]]}),
        [FEATURE_COL],
    )
    left, right = st.columns(2)
    left.metric("Served (online, latest)", int(served[FEATURE_COL].iloc[0]))
    right.metric("Trained on (point-in-time)", int(historical[FEATURE_COL].iloc[0]))


def parity_panel(dataset: Dataset) -> None:
    st.header("From-scratch vs Feast")
    st.caption("The same historical-features request through both stores.")
    try:
        from src.evaluation.parity import compare
    except ImportError:
        st.warning("Feast is not installed — `uv sync --all-extras` to enable this panel.")
        return
    if st.button("Run parity check"):
        sample = dataset.events.head(25)
        with tempfile.TemporaryDirectory() as tmp:
            merged = compare(dataset.feature_history, sample, Path(tmp) / "repo")
        match = bool((merged[f"{FEATURE_COL}_ministore"] == merged[f"{FEATURE_COL}_feast"]).all())
        st.success("Identical point-in-time results ✅" if match else "Mismatch ❌")
        st.dataframe(merged[[ENTITY_COL, f"{FEATURE_COL}_ministore", f"{FEATURE_COL}_feast"]])


def main() -> None:
    st.set_page_config(page_title="feature_store_101", layout="wide")
    st.title("feature_store_101")
    st.write("A feature store solves **time** and **consistency**, not storage.")
    dataset = _load()

    panels = {
        "Time-travel explorer": time_travel_panel,
        "Leakage meter": leakage_panel,
        "Online lookup": online_panel,
        "From-scratch vs Feast": parity_panel,
    }
    choice = st.sidebar.radio("Panel", list(panels))
    panels[choice](dataset)


if __name__ == "__main__":
    main()

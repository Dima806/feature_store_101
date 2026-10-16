"""Feast feature definitions for the local repo (`feast apply` scans this file).

One entity (``customer``) and one feature view (``customer_txn``) over the same
``feature_history.parquet`` the from-scratch store uses, so Feast's point-in-time
join can be checked against the mini-store's. Run ``make data`` first so the
parquet exists, then ``make feast-apply``.
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64
from feast.value_type import ValueType

_REPO_DIR = Path(__file__).resolve().parent
_DATA_DIR = _REPO_DIR.parent.parent.parent / "data"

customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    value_type=ValueType.INT64,
    description="A customer the features are keyed on.",
)

customer_activity_source = FileSource(
    name="customer_activity_source",
    path=str(_DATA_DIR / "feature_history.parquet"),
    timestamp_field="event_timestamp",
)

customer_txn = FeatureView(
    name="customer_txn",
    entities=[customer],
    ttl=timedelta(days=36500),
    schema=[Field(name="txn_count_30d", dtype=Int64)],
    source=customer_activity_source,
    online=True,
    description="Transactions in the trailing 30 days (one shared definition).",
)

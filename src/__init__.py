"""feature_store_101 — a feature store from first principles, on CPU, no server.

See CLAUDE.md for the thesis: a feature store solves TIME and CONSISTENCY, not
storage. The two problems are point-in-time correctness and train/serve
consistency; the three parts that solve them are the offline store, the online
store, and the registry.
"""

__version__ = "0.1.0"

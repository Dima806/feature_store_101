# feature_store_101 — Agent Guide

Educational "101" portfolio project for a data-science audience. Read this before
touching code; it encodes the thesis the whole repo exists to prove and the
invariants that must never break.

---

## 1. What this project is

A hands-on introduction to **feature stores**, built from scratch on CPU with no
server, no Docker, and no account. Format: a GitHub repo with 6 Jupyter notebooks,
a `src/` library, a Streamlit app, and tests — all runnable inside a
**2-CPU / 8 GB GitHub Codespace**.

**Status (built & green):** the `src/` library, all 6 notebooks, the Streamlit app, and
the test suite are implemented. `make ci` (sync + ruff + ty + 8 tests) and `make notebooks`
both pass. Python is pinned to **3.12** (`.python-version`; `feast` does not support 3.14).
Running the notebooks writes every figure to `outputs/figures/*.png` and every textual/
numerical result to `outputs/json/*.json` — these are tracked deliverables the README and the
article cite. Keep them in sync: if you change a number, re-run the notebook that produces it.

The one-sentence framing, which every notebook and module should reinforce:

> A feature store is **one feature definition served two ways** — historically for
> training (point-in-time correct), and latest for serving (fast online lookup) —
> backed by a shared registry of what each feature means.

It is really just three plain things:
1. **Offline store** — a place to look up *historical* feature values correctly.
2. **Online store** — a place to look up the *latest* values fast.
3. **Registry** — a shared catalog of what each feature means.

## 2. Thesis (never lose this)

A feature store solves **TIME and CONSISTENCY, not storage.** Two problems:

- **Point-in-time correctness.** Build a training set with each label joined to the
  feature value *as it was known at that label's timestamp* — never a value computed
  later. The naive "join the latest value" approach leaks the future: offline AUC
  looks great (~0.86) and collapses in production (~0.71). The honest point-in-time
  join (`pandas.merge_asof`) holds (~0.74 offline, ~0.74 on a true-future holdout).
  The ~12-point gap is **leakage, not skill.**
- **Train/serve consistency.** The feature a model sees in production must be computed
  the *same way* as the feature it trained on. Two hand-written implementations drift
  (a different window, null rule, rounding) and silently feed the model inputs it never
  trained on — both paths run without error; only the production metric suffers. The fix
  is **one shared definition** used by training and serving, materialized to the online
  store.

The payoff: build the whole store in ~100 lines and you understand exactly what Feast
does for you locally — and exactly when you don't need it. **The provocative claim is
the project's spine; preserve its numbers and its measured-not-asserted character.**

## 3. Environment & constraints

| Constraint | Decision |
|---|---|
| CPU / RAM | 2 cores, 8 GB. Event streams are tens of thousands of rows; everything runs in seconds |
| GPU | None |
| Docker / servers | None. Feast runs `provider: local` (file offline + SQLite online), in-process |
| Registration / API keys | None. All local files, no cloud, no account |
| Python | 3.11+ |
| Package manager | **`uv` only.** Never `pip install` directly. No global installs |
| Notebook runtime | Each notebook must run in **< 5 minutes** |
| Reproducibility | Seeded streams; correctness, inflation, and parity **asserted against known truth** |

**From-scratch store (notebooks 01–04)** needs nothing heavier than `pandas`
(`merge_asof`), `pyarrow` (parquet), and stdlib `sqlite3`. **Feast is an optional
extra** used only in notebook 05 — the project and CI must run and pass even if Feast
is not installed.

## 4. Commands (use the Makefile; don't invent ad-hoc invocations)

```bash
make setup        # First-time: install uv if missing, uv sync --all-extras, register kernel
make sync         # uv sync --all-extras
make data         # Generate synthetic event streams -> data/ (run before tests/notebooks)
make feast-apply  # Apply the local Feast repo (provider: local; no server, no account)
make lint         # format + check + typecheck (ruff format, ruff check --fix, ty check)
make format       # ruff format src/ tests/ app/
make check        # ruff check --fix src/ tests/ app/
make typecheck    # ty check src/
make test         # pytest
make test-cov     # pytest with coverage
make notebooks    # Execute all notebooks headless (nbconvert, 300s timeout each)
make run          # Streamlit app on :8501
make lab          # JupyterLab on :8888
make ci           # sync + lint + test  (what CI runs)
make dev          # lint + test  (fast local loop)
make clean / make reset
```

Everything runs through `uv run ...`. If you need a one-off, prefer `uv run <cmd>`.

## 5. Repository layout

```
src/
  config.py                 # pydantic-settings loader for config/settings.yaml
  ministore/                # the from-scratch feature store (~100 lines total)
    registry.py             # entity + feature definitions (the shared catalog)
    offline.py              # timestamped history + point-in-time (as-of) join  <- heart of the project
    online.py               # latest-value key-value store (SQLite / dict)
    materialize.py          # push latest offline values -> online store
    store.py                # get_historical_features + get_online_features
  features/
    definitions.py          # ONE shared definition per feature (used by training AND serving)
  feast_demo/
    feature_repo/           # local Feast repo: feature_store.yaml (provider: local, sqlite + file)
    run_feast.py            # apply / materialize / get_historical / get_online — all local
  datasets/
    events.py               # synthetic entity-timestamp stream + time-varying feature + known-truth label
  evaluation/
    leakage.py              # naive-join vs point-in-time: offline AUC and true-future AUC
    skew.py                 # training vs serving feature disagreement
    parity.py               # from-scratch store vs Feast: identical point-in-time results
    comparison.py           # arena harness
  visualisation.py          # matplotlib figure helpers for the app/notebooks
  artifacts.py              # save_figure (PNG) + save_json + the validated chart palette/style
app/streamlit_app.py
config/settings.yaml        # windows, entities, timestamps, seeds
conftest.py                 # puts the repo root on sys.path so `import src` works under pytest
.python-version             # 3.12 (feast has no 3.14 wheels)
notebooks/01..06            # each imports from src/, saves figures to outputs/figures, numbers to outputs/json
tests/                      # test_point_in_time, test_naive_leaks, test_online, test_parity, test_skew (8 tests)
data/                       # generated parquet (git-ignored)
outputs/figures/*.png  outputs/json/*.json   # tracked notebook deliverables
```

Notebook build-up (keep this order and dependency):
`01` what a feature store is → `02` point-in-time from scratch → `03` build a store in
100 lines → `04` train/serve skew → `05` Feast locally → `06` decision framework.

## 6. Core concepts to preserve in code & prose

- **Point-in-time join = `merge_asof`**: each request row `(entity, event_time)` gets the
  most recent feature value *at or before* `event_time`, and **never after**. This
  invariant is asserted in tests — do not relax it.
- **Naive join leaks**: joins the latest value onto historical labels → inflated offline
  AUC that collapses on a true-future holdout. Always measure *both* AUCs; never just claim it.
- **Materialize is the consistency link**: the online (served) value descends from the
  *same* definition as the offline (trained) value. Don't let serving recompute features
  independently — that is exactly the skew bug the project warns against.
- **Registry = shared catalog**: one definition, used everywhere. It's a structural idea,
  not bureaucracy.
- **Feast local = the 100-line idea done properly**: `provider: local`, file offline +
  SQLite online, no server/account. Notebook 05 asserts **parity** with the from-scratch
  store, then shows what Feast *adds* (managed registry, materialization/freshness, scale).
- **Decision framework**: a dataframe + correct `merge_asof` for a single batch model
  owned by one person; Feast for feature reuse across models/teams, low-latency online
  serving, or point-in-time over large histories with freshness; managed/streaming only at
  genuine scale or strict SLAs. **The rule that outranks the tool choice: you may not need a
  store, but you ALWAYS need point-in-time correctness and train/serve consistency — if you
  skip the store, do not skip the discipline.**

## 7. Datasets & reproducibility

- Synthetic, **seeded**, small (tens of thousands of rows), stored as local **parquet +
  SQLite**, generated by `src/datasets/events.py` (`make data`).
- The generator produces a customer-style stream (transactions over time, a rolling-window
  feature, labeled events) where the **honest as-of feature value at every event time is
  computable exactly**. The inflated-vs-honest AUC and the future-holdout collapse are
  measured **against ground truth, not asserted by hand.**
- A second small scenario with two deliberately divergent feature implementations drives the
  train/serve-skew chapter.
- All seeds live in `config/settings.yaml`; read config via `src/config.py`
  (pydantic-settings). Don't hardcode seeds, windows, or paths in notebooks.

## 8. Coding conventions

- **Lint/format: ruff.** Line length 99 (`E501` ignored). Rule sets:
  `E, F, W, I, UP, N, B, A, SIM, PTH` — note `PTH` (use `pathlib`, not `os.path`) and `UP`
  (modern syntax). Run `make format` and `make check` before committing.
- **Type checking: `ty`** (`make typecheck`, `ty check src/`). Keep `src/` typed and clean.
- **Tests: pytest** (`testpaths = ["tests"]`, `-v --tb=short`). Prefer asserting measurable
  truths (see §9) over smoke tests.
- Keep the mini-store genuinely **small and readable** — the "~100 lines" is a teaching
  promise, not a slogan. Clarity beats cleverness; a data scientist should read
  `ministore/` and see the whole idea.
- Use `pathlib` for paths, pandas vectorized ops over loops, and prefer pure functions in
  `features/definitions.py` so the same definition is trivially callable from both training
  and serving.

## 9. Success criteria / what tests must assert

| Criterion | Threshold |
|---|---|
| Point-in-time join | Never uses a feature value after the event time (asserted) |
| Naive-join inflation | Offline AUC materially > honest point-in-time AUC (asserted) |
| Leakage collapse | Naive model falls to ≈ honest model on a true-future holdout (asserted) |
| Hundred-line store | `get_historical_features` and `get_online_features` both work end to end |
| Online store | Returns the latest materialized value per entity (asserted) |
| Train/serve skew | Divergent impls measurably disagree; shared definition removes the gap (demonstrated) |
| Feast runs locally | `apply`/`materialize`/`get_historical`/`get_online`, no server/Docker/account (verified) |
| From-scratch vs Feast | Identical point-in-time training sets (asserted) |
| Open-source, CPU, keyless | Verified |
| Notebooks | Each runs < 5 min on a 2-CPU Codespace |
| `make lint` & `make test` | Zero errors |

Named tests (8 tests total): `test_point_in_time.py` (also asserts the one shared definition
reproduces ground truth), `test_naive_leaks.py`, `test_online.py`, `test_parity.py`
(`importorskip("feast")`), `test_skew.py`. Current measured results: naive offline **0.864** →
future **0.712**; honest **0.724 / 0.712**; skew disagreement **96%**, accuracy **0.672 →
0.543** at a 30-vs-14-day window; Feast parity **identical** (max │diff│ 0); store **114
statements**.

## 10. Feast specifics

- Keep Feast **optional** (`[project.optional-dependencies] feast`). Guard imports so the
  core project and CI pass without it installed.
- `feast apply` runs from `src/feast_demo/feature_repo/` (see `make feast-apply`).
- Pin the Feast version; use the documented `provider: local` file + SQLite config. The
  from-scratch store is the **primary teaching artifact**; the parity check pins the behavior
  being demonstrated. `duckdb` is an optional backing for the file offline store.

## 11. Streamlit app (`app/streamlit_app.py`)

Four panels, all driven by the same `src/` library (no logic duplicated in the app):
1. **Time-travel explorer** — pick an entity + timestamp; see the point-in-time value next
   to the latest value (makes leakage visible).
2. **Leakage meter** — toggle naive vs point-in-time; watch offline AUC and true-future AUC
   move apart and back.
3. **Online lookup** — query served features vs the value the model trained on.
4. **From-scratch vs Feast** — same historical-features request through both; show the match.

## 12. Notebooks

- Each is self-contained, runs headless under `make notebooks` (300s timeout), and imports
  logic from `src/` rather than redefining it. The first code cell bootstraps `sys.path` so the
  notebook imports `src` whether run from `notebooks/` or the repo root.
- **Save everything:** every figure via `artifacts.save_figure(fig, "NN_name")` → PNG, and every
  textual/numerical result via `artifacts.save_json(data, "NN_name")` → JSON. Use the validated
  palette (`BLUE` = honest/correct, `ORANGE` = naive/divergent, `AQUA` = third) and `apply_style()`.
- Cross-link to sibling projects where relevant (real repos under
  [github.com/Dima806](https://github.com/Dima806); **there is no `train_test_split_leakage`
  repo** — use `cv_arena`): point-in-time leakage ↔ `cv_arena`; feature freshness/staleness ↔
  `model_drift_detector`; the engineered features a store serves ↔ `feature_engineering_arena`;
  temporal thinking ↔ `timeseries_101`; the AUC/accuracy metrics ↔ `metrics_101` /
  `calibration_arena`. See the README's Related projects table.
- Narrative must stay **measured, not asserted** — show the AUCs, don't just state them.

## 13. Workflow & token rules

- **Caveman mode: ON.** Be code-first and terse; minimize prose in agent replies.
- Run `/compact` at notebook boundaries to keep context lean.
- Prefer editing `src/` and letting notebooks/tests import it over pasting logic in multiple
  places (this is also the train/serve-consistency discipline, applied to the repo itself).
- License: **Apache-2.0.** All dependencies must be open-source and CPU-only.

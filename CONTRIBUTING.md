# Contributing to tsfeat

## Setup

```bash
git clone https://github.com/quantxaashish/tsfeat.git
cd tsfeat
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
ruff check .
mypy src/tsfeat
pytest --cov=tsfeat
```

All three must pass. New primitives should include:

- a docstring stating exactly what is computed (parameters, return shape,
  NaN behaviour, and any convention that has more than one common
  definition in the literature)
- deterministic unit tests with hand-computable expected values
- at least one Hypothesis property test if a meaningful invariant exists
- **no row-wise iteration** (`iterrows`, `itertuples`, `apply(axis=1)`, or
  any disguised per-observation Python loop) in `src/tsfeat`

## Design constraints

- Loops over a small, fixed set of horizons/groups/columns are fine;
  looping over observations is not.
- Every function must state whether it is backward-looking,
  contemporaneous, or forward-looking, and must not look ahead unless
  that is the explicit, documented purpose (forward returns, IC decay
  outcomes).
- Ambiguous conventions (e.g. turnover's factor of 0.5, EWMA's
  `adjust=False`) must be resolved to one documented default rather than
  left implicit.

## Naive reference implementations

`benchmarks/naive.py` contains intentionally slow, loop-based reference
implementations used only for benchmarking and independent correctness
checks. `src/tsfeat` must never import from `benchmarks`.

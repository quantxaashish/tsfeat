# tsfeat

Vectorised primitives for quantitative time-series research.

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Motivation

Quantitative research repeatedly needs the same small set of operations:
rolling normalisation, forward-return construction, cross-sectional
transforms, risk statistics, portfolio turnover, and IC analysis. A
careless implementation of any of these can quietly introduce:

- Python-loop bottlenecks that make research iteration painfully slow
- timestamp bugs (duplicate trades, non-monotonic indices)
- misaligned returns (off-by-one shifts, wrong axis)
- lookahead bias (a signal that accidentally sees its own outcome)
- inconsistent NaN semantics across otherwise-similar functions

`tsfeat` is a compact, fully tested collection of vectorised primitives
that gets these details right once, so they don't need to be re-derived
(or re-debugged) in every research script. It is also, candidly, shaped by
the kind of vectorisation and time-series coding problems that come up in
quantitative research interviews — each primitive here is one an
interviewer could reasonably ask you to implement and defend from first
principles.

`tsfeat` is not affiliated with any trading firm, and makes no claims of
being "institutional-grade" or "production-proven" — it is a well-tested
research utility library, not a live trading system.

## Quickstart

```bash
pip install -e ".[dev]"
```

```python
import numpy as np
import pandas as pd
import tsfeat as tf

prices = pd.DataFrame(
    100 * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, size=(252, 4)), axis=0),
    columns=["AAPL", "MSFT", "NVDA", "AMZN"],
)
returns = prices.pct_change()

z = tf.rolling_zscore(prices, window=20, min_periods=10)
vol = tf.ew_volatility(returns, half_life=20, annualization=252)
fwd = tf.forward_returns(prices, horizons=[1, 5, 10, 20])

signal = tf.cross_sectional_demean(tf.cross_sectional_rank(z))
ic = tf.ic_decay(signal, fwd, method="spearman")
turnover = tf.signal_turnover(signal.dropna(how="all"))
```

## Feature table

| Primitive              | Vectorised | Handles NaN | Multi-asset | Lookahead-safe |
| ----------------------- | ---------: | ----------: | ----------: | -------------: |
| `rolling_zscore`        |          ✓ |           ✓ |           ✓ |              ✓ |
| `ew_volatility`         |          ✓ |           ✓ |           ✓ |              ✓ |
| `rolling_correlation`   |          ✓ |           ✓ |           ✓ |              ✓ |
| `vwap` / `cumulative_vwap` |       ✓ |           ✓ |           n/a |           ✓ |
| `max_drawdown`          |          ✓ |           ✓ |           n/a |           n/a |
| `cross_sectional_rank`  |          ✓ |           ✓ |           ✓ |              ✓ |
| `cross_sectional_demean`|          ✓ |           ✓ |           ✓ |              ✓ |
| `grouped_winsorize`     |          ✓ |           ✓ |           ✓ |              ✓ |
| `forward_returns`       |          ✓ |           ✓ |           ✓ |              ✓ |
| `signal_turnover`       |          ✓ |           ✓ |           ✓ |              ✓ |
| `ic_decay`              |          ✓ |           ✓ |           ✓ |              ✓ |

`max_drawdown` operates on a single wealth series, so "multi-asset" does
not apply; it is inherently backward-looking (drawdown from a
high-water mark) rather than a forward/backward comparison, so
"lookahead-safe" doesn't apply in the same sense either.

## Performance

Naive Python-loop reference implementations live in `benchmarks/naive.py`
(used only for benchmarking and independent correctness checks — the
production package never imports them; a dedicated test,
`tests/test_naive_equivalence.py`, asserts they produce identical output
to the vectorised implementations, so the speedups below reflect
vectorisation only, not two different algorithms). Naive implementations
are skipped at sizes where their asymptotic complexity (`O(T*W)` for
rolling z-score, `O(T*N^2)` for cross-sectional rank) would make them
impractically slow; this is noted explicitly in the tables rather than
silently omitted.

Measured on: Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, Windows 11
(single run per environment; timings are the median of several repeated
calls per size — see `benchmarks/_timing.py`). Absolute numbers will vary
by hardware; relative speedups should be broadly representative.

**Rolling z-score** (`window=20`):

| Rows      | Naive     | tsfeat  | Speedup |
| --------- | --------: | ------: | ------: |
| 10,000    | 86.9 ms   | 377 us  | 230x    |
| 100,000   | 881 ms    | 2.76 ms | 319x    |
| 1,000,000 | skipped (`O(T*W)`) | 30.6 ms | n/a |

**Forward returns** (`horizons=[1, 5, 10, 20]`):

| Rows      | Naive     | tsfeat  | Speedup |
| --------- | --------: | ------: | ------: |
| 10,000    | 6.39 ms   | 273 us  | 23x     |
| 100,000   | 66.1 ms   | 1.55 ms | 43x     |
| 1,000,000 | skipped (`O(T*H)`) | 16.5 ms | n/a |

**Cross-sectional rank** (dates x assets):

| Shape     | Naive     | tsfeat  | Speedup |
| --------- | --------: | ------: | ------: |
| 250x50    | 73.6 ms   | 235 us  | 313x    |
| 500x100   | 564 ms    | 1.36 ms | 414x    |
| 2,500x500 | skipped (`O(T*N^2)`) | 50.4 ms | n/a |

**Cross-sectional demean** (dates x assets):

| Shape     | Naive    | tsfeat  | Speedup |
| --------- | -------: | ------: | ------: |
| 500x100   | 57.4 ms  | 206 us  | 278x    |
| 2,500x500 | 1.34 s   | 4.25 ms | 316x    |

**Signal turnover** (dates x assets):

| Shape       | Naive   | tsfeat  | Speedup |
| ----------- | ------: | ------: | ------: |
| 500x100     | 25.3 ms | 561 us  | 45x     |
| 2,500x500   | 647 ms  | 10.0 ms | 65x     |
| 10,000x1,000| 5.16 s  | 95.7 ms | 54x     |

Reproduce with:

```bash
python -m benchmarks.benchmark_rolling
python -m benchmarks.benchmark_cross_sectional
python -m benchmarks.benchmark_portfolio
```

The speedups come from avoiding Python-level per-observation dispatch, not
from pandas somehow avoiding loops altogether — pandas' own rolling and
groupby operations are themselves implemented as compiled (Cython) loops
internally. The distinction that matters here is *where* the loop runs:
once in compiled code across the whole array, versus once per
observation in the Python interpreter.

## Lookahead safety

Every function in `tsfeat` documents whether it is backward-looking,
contemporaneous, or forward-looking:

- **Backward-looking** (`rolling_zscore`, `ew_volatility`,
  `rolling_correlation`, `max_drawdown`): a value at `t` depends only on
  observations at or before `t`.
- **Contemporaneous** (`cross_sectional_rank`, `cross_sectional_demean`,
  `grouped_winsorize`): a value at `t` depends only on other assets'
  observations at that same `t`, never on another date.
- **Forward-looking, by design** (`forward_returns`, and the forward-return
  side of `ic_decay`): a signal at date `t` represents information known at
  `t`; a forward return at `t` represents the *outcome* realised from `t`
  to `t + h`. It is correct and expected for the outcome to depend on
  `price(t + h)` — that is the entire point of a forward return — but it
  must never leak into anything computed *as of* `t`.

`forward_returns` uses `prices.shift(-h) / prices - 1`: the final `h`
observations of each horizon become genuinely unavailable (`NaN`) rather
than being backfilled, wrapped around, or otherwise invented. A dedicated
regression test (`TestForwardReturnsShiftDirection` in
`tests/test_returns.py`) exists specifically to catch an accidental
shift-direction (sign) bug, which is the single easiest way to
accidentally leak future information into a signal.

## Testing

There is no CI pipeline in this version — all checks below are run and
verified locally before each commit:

```bash
ruff check .
ruff format --check .
mypy src/tsfeat
pytest
pytest --cov=tsfeat --cov-report=term-missing
```

The suite combines deterministic unit tests with hand-computable expected
values (important for alignment-sensitive functions like
`forward_returns` and `vwap`) and Hypothesis property-based tests for
invariants that a fixed example can't fully exercise — e.g. affine
invariance of the rolling z-score, symmetry and boundedness of rolling
correlation, the zero-mean invariant of cross-sectional demean, and
"drawdown is never positive."

As of the last local run: **110 tests passing, 100% statement coverage**
on `src/tsfeat` (see the coverage table produced by
`pytest --cov=tsfeat --cov-report=term-missing`).

## Benchmarks

```bash
python -m benchmarks.benchmark_rolling
python -m benchmarks.benchmark_cross_sectional
python -m benchmarks.benchmark_portfolio
```

Each script prints its own results table (see [Performance](#performance)
above for a snapshot) along with the Python/NumPy/pandas versions and
platform used, since absolute timings depend on hardware.

## Example data

`examples/yahoo_finance_demo.py` and `examples/research_pipeline.py` use
[yfinance](https://github.com/ranaroussi/yfinance) to fetch real prices
for a small US equity universe (AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM,
XOM) purely to demonstrate the library end-to-end. Both scripts require
internet access and are never imported by the test suite; all tests use
deterministic local/synthetic fixtures. The example research pipeline's
"signal" is arbitrary and has not been evaluated for economic merit — it
exists to exercise the API, not to demonstrate a trading strategy.

## Design principles

**Vectorise observations.** No Python row loops (`iterrows`,
`itertuples`, `apply(axis=1)`, or disguised equivalents) in any
production primitive. Loops over a small, fixed set of horizons, groups,
or columns are fine — a loop over dates or assets themselves is not.

**Make temporal alignment explicit.** Every function's docstring states
whether it is backward-looking, contemporaneous, or forward-looking, and
why.

**Treat missing data intentionally.** NaN handling is part of each
function's documented contract, not an accident of whatever pandas
happens to do by default.

**Test invariants, not just examples.** Property-based tests complement
fixed, hand-computable examples so that a refactor changing *how*
something is computed still has to satisfy *what* it must be true of the
output.

**Benchmark claims.** Every performance claim in this README comes from
an actual, reproducible run of the scripts in `benchmarks/`, compared
against a naive reference implementation that is verified to produce
equivalent output.

## Project layout

```text
src/tsfeat/
    rolling.py          rolling_zscore, ew_volatility, rolling_correlation
    returns.py           forward_returns
    cross_sectional.py   cross_sectional_rank, cross_sectional_demean, grouped_winsorize
    portfolio.py         signal_turnover
    statistics.py        vwap, cumulative_vwap, max_drawdown, ic_decay
    validation.py        internal input validation (not part of the public API)
tests/                   deterministic + Hypothesis property tests
benchmarks/              naive reference implementations + benchmark scripts
examples/                Yahoo Finance demo and an end-to-end research pipeline
```

## License

[MIT](LICENSE)

# Changelog

All notable changes to this project are documented in this file.
This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-05

Initial release.

### Added

- `rolling_zscore` — rolling z-score with an explicit `min_periods` guard
  and documented zero-variance handling.
- `ew_volatility` — exponentially weighted volatility parameterised by
  half-life.
- `rolling_correlation` — rolling pairwise correlation across N assets.
- `vwap` / `cumulative_vwap` — volume-weighted average price with explicit
  duplicate-timestamp handling.
- `max_drawdown` — drawdown statistics (`DrawdownStats`), including a
  single unified duration definition for recovered and unrecovered
  episodes.
- `cross_sectional_rank` / `cross_sectional_demean` — same-date,
  across-asset transforms.
- `grouped_winsorize` — group-specific quantile winsorisation.
- `forward_returns` — leakage-safe forward returns at multiple horizons.
- `signal_turnover` — portfolio/signal turnover with an explicit one-way
  convention.
- `ic_decay` — cross-sectional Information Coefficient, summarised per
  forward-return horizon.
- Deterministic unit tests, Hypothesis property-based tests, and naive
  Python-loop benchmark references for the core primitives.
- GitHub Actions CI (lint, type-check, test) and pre-commit configuration.

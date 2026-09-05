"""tsfeat: vectorised primitives for quantitative time-series research."""

from tsfeat.cross_sectional import cross_sectional_demean, cross_sectional_rank, grouped_winsorize
from tsfeat.portfolio import signal_turnover
from tsfeat.returns import forward_returns
from tsfeat.rolling import ew_volatility, rolling_correlation, rolling_zscore
from tsfeat.statistics import DrawdownStats, cumulative_vwap, ic_decay, max_drawdown, vwap

__version__ = "0.1.0"

__all__ = [
    "rolling_zscore",
    "ew_volatility",
    "rolling_correlation",
    "vwap",
    "cumulative_vwap",
    "DrawdownStats",
    "max_drawdown",
    "cross_sectional_rank",
    "cross_sectional_demean",
    "grouped_winsorize",
    "forward_returns",
    "signal_turnover",
    "ic_decay",
    "__version__",
]

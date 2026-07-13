"""
Market regime classification - trending vs choppy.

Shared between backtest.py (regime-split win rates) and main.py (live
regime labeling), so both use exactly the same definition.
"""

import numpy as np

REGIME_LOOKBACK = 20    # candles used to measure trend efficiency
REGIME_THRESHOLD = 0.3  # efficiency ratio >= this counts as "trending"


def efficiency_ratio(closes: np.ndarray, i: int, n: int = REGIME_LOOKBACK) -> float | None:
    """
    Kaufman's Efficiency Ratio: net price change / total path length over
    the window, causal (only uses closes up to and including i). Near 1.0
    means price moved in a straight line (trending); near 0 means it
    churned back and forth with little net progress (choppy/ranging).
    """
    if i < n:
        return None
    window = closes[i - n:i + 1]
    net = abs(window[-1] - window[0])
    path = np.abs(np.diff(window)).sum()
    return net / path if path > 0 else 0.0


def regime_label(closes: np.ndarray, i: int) -> str:
    er = efficiency_ratio(closes, i)
    if er is None:
        return "unknown"
    return "trending" if er >= REGIME_THRESHOLD else "choppy"

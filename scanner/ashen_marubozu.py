"""
Ashen Marubozu Detector ("Trading with Ashen" Part 03)

Philosophy: a Marubozu candle (near-zero wicks on both ends, body spans
almost the full high-low range) means one side dominated the entire
candle with no real pushback - a directional "market dominance" candle.
White/bullish Marubozu -> long signal. Black/bearish Marubozu -> short
signal. No moving average, volume, or oscillator is used - detection is
purely about the shape of the single most recent CLOSED candle.

Ashen's exact rule (confirmed directly by the user against the source
video, correcting an earlier reading of the garbled transcript): once
the Marubozu is confirmed (closed), enter the trade immediately - no
next-candle breakout wait, unlike b2b_ashen. Stop sits a small buffer
beyond the candle's own START point (its open), not its wick extreme
(low for a long, high for a short) - though on a true Marubozu the two
are close since wicks are negligible by definition. Target is 1.2x
that risk projected the same direction ("1 candle = 1 risk unit"). He
deliberately uses a LOWER reward:risk target here (1:1.2, not the
usual 1:1.5) with the explicit math: at 1.2:1 you need fewer wins out
of 100 trades to break even than at 1.5:1, so it's meant to bank
profit faster rather than chase a bigger move.

He also excludes an otherwise-valid Marubozu if it's already too
"extended" beyond the 100/200-period moving averages (e.g. a bullish
Marubozu that has already run far above both MAs) - chasing a
candle that's already stretched far from its own trend baseline is
not the same setup as one still near it.
"""

import pandas as pd

_DEFAULT_CFG = {
    "min_body_ratio": 0.85,   # body must be at least this fraction of the full high-low range
    "max_wick_ratio": 0.10,   # each wick must be no more than this fraction of the range
    "reward_risk_ratio": 1.2,  # Ashen's stated target for this strategy specifically (not the global default)
    "stop_buffer_pct": 0.1,   # stop sits this % beyond the candle's OPEN, not its wick extreme
    "extension_ma_periods": [100, 200],  # candle must not be too far beyond either of these SMAs
    "max_extension_multiplier": 5.0,     # "too far" = distance to an MA exceeds this many candle-ranges
}


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _is_extended(df: pd.DataFrame, candle, direction: str, periods: list[int],
                  max_extension_multiplier: float) -> bool:
    """
    True if the candle's close already sits too far beyond ANY of the
    given SMAs, in units of the candle's own high-low range - i.e. the
    move is "chased" rather than a dominance candle still near its
    trend baseline.
    """
    candle_range = candle["high"] - candle["low"]
    if candle_range <= 0:
        return False
    close = float(candle["close"])
    for period in periods:
        if len(df) < period:
            continue
        ma = _sma(df["close"], period).iloc[-1]
        if pd.isna(ma):
            continue
        distance = (close - ma) if direction == "bullish" else (ma - close)
        if distance > candle_range * max_extension_multiplier:
            return True
    return False


def _is_bullish_marubozu(candle, min_body_ratio: float, max_wick_ratio: float) -> bool:
    rng = candle["high"] - candle["low"]
    if rng <= 0 or candle["close"] <= candle["open"]:
        return False
    body = candle["close"] - candle["open"]
    lower_wick = candle["open"] - candle["low"]
    upper_wick = candle["high"] - candle["close"]
    return (body / rng >= min_body_ratio
            and lower_wick / rng <= max_wick_ratio
            and upper_wick / rng <= max_wick_ratio)


def _is_bearish_marubozu(candle, min_body_ratio: float, max_wick_ratio: float) -> bool:
    rng = candle["high"] - candle["low"]
    if rng <= 0 or candle["close"] >= candle["open"]:
        return False
    body = candle["open"] - candle["close"]
    upper_wick = candle["high"] - candle["open"]
    lower_wick = candle["close"] - candle["low"]
    return (body / rng >= min_body_ratio
            and upper_wick / rng <= max_wick_ratio
            and lower_wick / rng <= max_wick_ratio)


def detect_signals(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """
    Returns [] or a single-signal list. Only looks at the last CLOSED
    candle (df.iloc[-1]) - Ashen's rule is you never act mid-candle.
    """
    mcfg = cfg.get("ashen", {}).get("marubozu", {})
    if not mcfg.get("enabled", True):
        return []
    if len(df) < 2:
        return []

    min_body_ratio = mcfg.get("min_body_ratio", _DEFAULT_CFG["min_body_ratio"])
    max_wick_ratio = mcfg.get("max_wick_ratio", _DEFAULT_CFG["max_wick_ratio"])
    reward_risk = mcfg.get("reward_risk_ratio", _DEFAULT_CFG["reward_risk_ratio"])
    stop_buffer_pct = mcfg.get("stop_buffer_pct", _DEFAULT_CFG["stop_buffer_pct"])
    extension_ma_periods = mcfg.get("extension_ma_periods", _DEFAULT_CFG["extension_ma_periods"])
    max_extension_multiplier = mcfg.get("max_extension_multiplier", _DEFAULT_CFG["max_extension_multiplier"])

    candle = df.iloc[-1]
    close = float(candle["close"])
    open_ = float(candle["open"])

    if _is_bullish_marubozu(candle, min_body_ratio, max_wick_ratio):
        if _is_extended(df, candle, "bullish", extension_ma_periods, max_extension_multiplier):
            return []
        stop = open_ * (1 - stop_buffer_pct / 100)
        target = close + (close - stop) * reward_risk
        body_ratio = (candle["close"] - candle["open"]) / (candle["high"] - candle["low"])
        return [{
            "name": "marubozu_ashen",
            "direction": "bullish",
            "detail": f"Bullish (white) Marubozu - body {body_ratio:.0%} of range, negligible wicks",
            "stop": stop,
            "target": target,
        }]

    if _is_bearish_marubozu(candle, min_body_ratio, max_wick_ratio):
        if _is_extended(df, candle, "bearish", extension_ma_periods, max_extension_multiplier):
            return []
        stop = open_ * (1 + stop_buffer_pct / 100)
        target = close - (stop - close) * reward_risk
        body_ratio = (candle["open"] - candle["close"]) / (candle["high"] - candle["low"])
        return [{
            "name": "marubozu_ashen",
            "direction": "bearish",
            "detail": f"Bearish (black) Marubozu - body {body_ratio:.0%} of range, negligible wicks",
            "stop": stop,
            "target": target,
        }]

    return []

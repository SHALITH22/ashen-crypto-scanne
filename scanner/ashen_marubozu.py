"""
Ashen Marubozu Detector ("Trading with Ashen" Part 03)

Philosophy: a Marubozu candle (near-zero wicks on both ends, body spans
almost the full high-low range) means one side dominated the entire
candle with no real pushback - a directional "market dominance" candle.
White/bullish Marubozu -> long signal. Black/bearish Marubozu -> short
signal. No moving average, volume, or oscillator is used - detection is
purely about the shape of the single most recent CLOSED candle.

Ashen's exact rule (from the transcript): enter at the start of the
candle right after the Marubozu closes, above its high (long) / below
its low (short); stop at the Marubozu's own opposite extreme (the
candle's low for a long, its high for a short) - "1 candle = 1 risk
unit". He deliberately uses a LOWER reward:risk target here (1:1.2, not
the usual 1:1.5) with the explicit math: at 1.2:1 you need fewer wins
out of 100 trades to break even than at 1.5:1, so it's meant to bank
profit faster rather than chase a bigger move.
"""

import pandas as pd

_DEFAULT_CFG = {
    "min_body_ratio": 0.85,   # body must be at least this fraction of the full high-low range
    "max_wick_ratio": 0.10,   # each wick must be no more than this fraction of the range
    "reward_risk_ratio": 1.2,  # Ashen's stated target for this strategy specifically (not the global default)
}


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

    candle = df.iloc[-1]
    close = float(candle["close"])

    if _is_bullish_marubozu(candle, min_body_ratio, max_wick_ratio):
        stop = float(candle["low"])
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
        stop = float(candle["high"])
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

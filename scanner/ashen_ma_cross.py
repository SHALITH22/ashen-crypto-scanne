"""
Ashen MA Cross + Dominant Candle Detector ("Trading with Ashen" Part 02)

Philosophy: trade in the direction of a 20/200 SMA cross (20 above 200 =
bullish regime, 20 below 200 = bearish regime), triggered by a single
"dominant" candle - one visibly larger than every other candle in the
recent lookback window, closing near its extreme (little to no
opposing wick).

Ashen's exact rule (confirmed against the source video, interpretation
A of two candidates - see ashen_marubozu.py's note on entry timing):
price above both SMAs with 20 SMA above 200 SMA, then a big bullish
candle (dwarfing recent candles, closing near its high) - enter
immediately once that dominant candle closes, no next-candle breakout
wait (unlike b2b_ashen). Entry above that candle's high, stop below
its low. Mirrored for bearish (price below both SMAs, 20 below 200,
big bearish candle, entry below its low, stop above its high).
Standard 1:1.5 target here (same as Jayantha's and Ashen's other
videos), unlike the deliberately lower 1:1.2 used in the Marubozu
strategy (see ashen_marubozu.py). The "~80% accuracy" claim in this
video series belongs to the VWAP Breakout strategy (Part 01, see
ashen_vwap_breakout.py) - not this one; that quote was previously
misattributed here.
"""

import pandas as pd

_DEFAULT_CFG = {
    "fast_period": 20,
    "slow_period": 200,
    "dominance_lookback": 10,   # candle must be bigger than every one of these preceding candles
    "dominance_multiplier": 1.5,  # ...by at least this much (vs. the largest of them)
    "max_opposing_wick_ratio": 0.25,  # opposing wick must be no more than this fraction of the candle's range
    "reward_risk_ratio": 1.5,
}


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def detect_signals(df: pd.DataFrame, cfg: dict) -> list[dict]:
    macfg = cfg.get("ashen", {}).get("ma_cross", {})
    if not macfg.get("enabled", True):
        return []

    fast_period = macfg.get("fast_period", _DEFAULT_CFG["fast_period"])
    slow_period = macfg.get("slow_period", _DEFAULT_CFG["slow_period"])
    lookback = macfg.get("dominance_lookback", _DEFAULT_CFG["dominance_lookback"])
    dominance_mult = macfg.get("dominance_multiplier", _DEFAULT_CFG["dominance_multiplier"])
    max_wick_ratio = macfg.get("max_opposing_wick_ratio", _DEFAULT_CFG["max_opposing_wick_ratio"])
    reward_risk = macfg.get("reward_risk_ratio", _DEFAULT_CFG["reward_risk_ratio"])

    min_candles = slow_period + lookback + 5
    if len(df) < min_candles:
        return []

    sma_fast = _sma(df["close"], fast_period)
    sma_slow = _sma(df["close"], slow_period)
    if pd.isna(sma_fast.iloc[-1]) or pd.isna(sma_slow.iloc[-1]):
        return []

    candle = df.iloc[-1]
    prior = df.iloc[-1 - lookback:-1]
    if len(prior) < lookback:
        return []
    prior_range = (prior["high"] - prior["low"]).max()
    candle_range = candle["high"] - candle["low"]
    if prior_range <= 0 or candle_range <= 0:
        return []
    is_dominant = candle_range >= prior_range * dominance_mult

    close = float(candle["close"])
    fast, slow = float(sma_fast.iloc[-1]), float(sma_slow.iloc[-1])
    bullish_regime = close > fast > slow
    bearish_regime = close < fast < slow

    if is_dominant and bullish_regime and candle["close"] > candle["open"]:
        upper_wick = candle["high"] - candle["close"]
        if upper_wick / candle_range <= max_wick_ratio:
            stop = float(candle["low"])
            target = close + (close - stop) * reward_risk
            return [{
                "name": "ma_cross_ashen",
                "direction": "bullish",
                "detail": (f"20SMA>200SMA regime, dominant bullish candle "
                           f"({candle_range / prior_range:.1f}x recent range) closing near high"),
                "stop": stop,
                "target": target,
            }]

    if is_dominant and bearish_regime and candle["close"] < candle["open"]:
        lower_wick = candle["close"] - candle["low"]
        if lower_wick / candle_range <= max_wick_ratio:
            stop = float(candle["high"])
            target = close - (stop - close) * reward_risk
            return [{
                "name": "ma_cross_ashen",
                "direction": "bearish",
                "detail": (f"20SMA<200SMA regime, dominant bearish candle "
                           f"({candle_range / prior_range:.1f}x recent range) closing near low"),
                "stop": stop,
                "target": target,
            }]

    return []

"""
Ashen VWAP Breakout Detector ("Trading with Ashen" Part 01)

The only one of the four Ashen strategies that needs TWO nested
timeframes rather than one: a higher timeframe (HTF) to read trend and
market structure, and a lower timeframe (LTF) to time the actual entry
off VWAP. Ashen is explicit that this pairing must "flow continuously"
(e.g. 1h+15m, or daily+4h) - it does NOT apply to markets with daily
opening gaps (stocks); every symbol scanned here is a Binance USDT
perpetual, which trades continuously, so that caveat is a non-issue in
this codebase and isn't encoded as a check.

Ashen's exact rule:
  1. On the HTF: identify trend direction (price vs. its own recent
     swing structure) and a market-structure level (recent swing
     high/low) that price has broken, faked back through, and is now
     retesting.
  2. Drop to the LTF: once price closes beyond that session's VWAP
     (above for long, below for short) near the retest, enter at the
     START of the next candle.
  3. Stop: the entry candle's opposite extreme, OR (the "more technical"
     method he also demonstrates) the entry candle's low/high minus/plus
     that candle's own ATR value. This detector uses the ATR method
     since it's the one Ashen calls more precise, and matches this
     codebase's existing preference for ATR-based stops elsewhere.
  4. Target: minimum 1:1.5 reward:risk (same floor as Jayantha's and
     Ashen's other setups here).

VWAP itself resets each session; on Binance's continuous 24/7 market
there's no exchange-defined "session close", so VWAP here is computed
over a rolling window of the LTF's own candles (config `vwap_window`),
which is the standard practical adaptation for a market with no natural
session boundary.
"""

import pandas as pd

_DEFAULT_CFG = {
    "htf_swing_lookback": 20,   # candles used to find the HTF swing high/low (structure level)
    "vwap_window": 96,          # rolling candle window for the LTF's own VWAP (~24h on 15m candles)
    "atr_period": 14,
    "atr_multiplier": 1.0,      # stop = entry candle low/high -+ ATR * this
    "reward_risk_ratio": 1.5,
}


def _rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    return pv.rolling(window).sum() / df["volume"].rolling(window).sum()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _htf_trend_and_structure(htf_df: pd.DataFrame, swing_lookback: int) -> dict:
    """
    Trend = close vs its own 20-EMA (consistent with get_market_trend
    elsewhere in this codebase). Structure level = the most recent swing
    high (downtrend) or swing low (uptrend) - the level Ashen says price
    must have broken through and be retesting.
    """
    ema20 = htf_df["close"].ewm(span=20, adjust=False).mean()
    bullish = bool(htf_df["close"].iloc[-1] > ema20.iloc[-1])
    recent = htf_df.iloc[-swing_lookback:]
    return {
        "bullish": bullish,
        "swing_high": float(recent["high"].max()),
        "swing_low": float(recent["low"].min()),
    }


def detect_signals(df: pd.DataFrame, htf_df: pd.DataFrame | None, cfg: dict) -> list[dict]:
    """
    df: LTF candles (entry timeframe). htf_df: HTF candles (trend/structure
    timeframe) - None if no HTF pairing is configured for this timeframe,
    in which case this strategy simply doesn't fire (it structurally
    requires both).
    """
    vcfg = cfg.get("ashen", {}).get("vwap_breakout", {})
    if not vcfg.get("enabled", True) or htf_df is None:
        return []

    swing_lookback = vcfg.get("htf_swing_lookback", _DEFAULT_CFG["htf_swing_lookback"])
    vwap_window = vcfg.get("vwap_window", _DEFAULT_CFG["vwap_window"])
    atr_period = vcfg.get("atr_period", _DEFAULT_CFG["atr_period"])
    atr_mult = vcfg.get("atr_multiplier", _DEFAULT_CFG["atr_multiplier"])
    reward_risk = vcfg.get("reward_risk_ratio", _DEFAULT_CFG["reward_risk_ratio"])

    if len(htf_df) < swing_lookback + 5 or len(df) < vwap_window + 5:
        return []

    structure = _htf_trend_and_structure(htf_df, swing_lookback)
    vwap = _rolling_vwap(df, vwap_window)
    atr = _atr(df, atr_period)
    if pd.isna(vwap.iloc[-1]) or pd.isna(vwap.iloc[-2]) or pd.isna(atr.iloc[-1]):
        return []

    entry_candle = df.iloc[-1]
    prior_candle = df.iloc[-2]
    close = float(entry_candle["close"])
    atr_val = float(atr.iloc[-1])

    # Entry rule: the PRIOR candle closed beyond VWAP (the confirmation
    # candle Ashen describes), and price is now trading through that level
    # on this (the "next candle") close - only meaningful near the HTF's
    # broken structure level, which the retest requirement approximates by
    # gating on trend direction rather than a precise retest-zone check
    # (retest recognition is visual/discretionary in the source video).
    if (structure["bullish"]
            and prior_candle["close"] > vwap.iloc[-2]
            and close > vwap.iloc[-1]):
        stop = float(entry_candle["low"]) - atr_val * atr_mult
        target = close + (close - stop) * reward_risk
        return [{
            "name": "vwap_breakout_ashen",
            "direction": "bullish",
            "detail": (f"HTF uptrend (structure high {structure['swing_high']:.6g}), "
                       f"LTF closed above rolling VWAP with ATR-based stop"),
            "stop": stop,
            "target": target,
        }]

    if (not structure["bullish"]
            and prior_candle["close"] < vwap.iloc[-2]
            and close < vwap.iloc[-1]):
        stop = float(entry_candle["high"]) + atr_val * atr_mult
        target = close - (stop - close) * reward_risk
        return [{
            "name": "vwap_breakout_ashen",
            "direction": "bearish",
            "detail": (f"HTF downtrend (structure low {structure['swing_low']:.6g}), "
                       f"LTF closed below rolling VWAP with ATR-based stop"),
            "stop": stop,
            "target": target,
        }]

    return []

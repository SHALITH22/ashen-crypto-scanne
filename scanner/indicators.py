"""
Indicator layer - pure pandas implementations (no TA-Lib compilation headaches).
Every function takes a klines DataFrame and returns it with new columns.
"""

import pandas as pd


def add_emas(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    for p in periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_stochrsi(df: pd.DataFrame, rsi_length: int = 14,
                 stoch_length: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    if "rsi" not in df.columns:
        df = add_rsi(df, rsi_length)
    rsi = df["rsi"]
    lowest = rsi.rolling(stoch_length).min()
    highest = rsi.rolling(stoch_length).max()
    stoch = 100 * (rsi - lowest) / (highest - lowest).replace(0, 1e-10)
    df["stochrsi_k"] = stoch.rolling(k).mean()
    df["stochrsi_d"] = df["stochrsi_k"].rolling(d).mean()
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df["volume_ma"] = df["volume"].rolling(period).mean()
    return df


def add_qqe(df: pd.DataFrame, rsi_length: int = 14, smoothing: int = 5, factor: float = 4.236) -> pd.DataFrame:
    """
    QQE (Quantitative Qualitative Estimation) - a smoothed-RSI trend
    indicator with ATR-of-RSI trailing bands (Wilder's smoothing), the same
    algorithm as the widely-used TradingView QQE/QQE Mod scripts. Not a
    black-box replica of any paid indicator - this is the openly published,
    standard QQE formula.

    RSI is smoothed with an EMA to get a less noisy "RSI MA" line. A
    trailing band (like a chandelier stop, but on the RSI's own volatility)
    hugs that line; qqe_trend flips to 1 (bullish) when RSI MA crosses
    above the trailing band from below, and -1 (bearish) on the reverse
    cross. qqe_line is whichever band is currently "active" (the trailing
    stop level itself), useful for plotting/distance-based strength.
    """
    if "rsi" not in df.columns:
        df = add_rsi(df, rsi_length)
    rsi_ma = df["rsi"].ewm(span=smoothing, adjust=False).mean()

    wilders_period = rsi_length * 2 - 1
    atr_rsi = rsi_ma.diff().abs()
    # Wilder's smoothing is just an EMA with alpha=1/period, applied twice
    # (once to get the ATR of RSI, once more to smooth that) - this is the
    # standard QQE double-smoothing, not an arbitrary choice.
    ma_atr_rsi = atr_rsi.ewm(alpha=1 / wilders_period, adjust=False).mean()
    dar = ma_atr_rsi.ewm(alpha=1 / wilders_period, adjust=False).mean() * factor

    rsi_ma_vals = rsi_ma.to_numpy()
    dar_vals = dar.to_numpy()
    n = len(rsi_ma_vals)
    longband = [0.0] * n
    shortband = [0.0] * n
    trend = [1] * n

    for i in range(1, n):
        newlong = rsi_ma_vals[i] - dar_vals[i]
        newshort = rsi_ma_vals[i] + dar_vals[i]

        if rsi_ma_vals[i - 1] > longband[i - 1] and rsi_ma_vals[i] > longband[i - 1]:
            longband[i] = max(longband[i - 1], newlong)
        else:
            longband[i] = newlong

        if rsi_ma_vals[i - 1] < shortband[i - 1] and rsi_ma_vals[i] < shortband[i - 1]:
            shortband[i] = min(shortband[i - 1], newshort)
        else:
            shortband[i] = newshort

        if rsi_ma_vals[i] > shortband[i - 1]:
            trend[i] = 1
        elif rsi_ma_vals[i] < longband[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    trend_s = pd.Series(trend, index=df.index)
    longband_s = pd.Series(longband, index=df.index)
    shortband_s = pd.Series(shortband, index=df.index)

    df["qqe_rsi_ma"] = rsi_ma
    df["qqe_line"] = longband_s.where(trend_s == 1, shortband_s)
    df["qqe_trend"] = trend_s
    return df


def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Average True Range - used for stop distance / volatility context."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / length, adjust=False).mean()
    return df


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply all configured indicators in one pass."""
    ind = cfg["indicators"]
    df = add_emas(df, ind["ema_periods"])
    s = ind["stochrsi"]
    df = add_stochrsi(df, s["rsi_length"], s["stoch_length"], s["k"], s["d"])
    df = add_volume_ma(df, ind["volume_ma_period"])
    df = add_atr(df)
    q = ind.get("qqe", {})
    df = add_qqe(df, q.get("rsi_length", 14), q.get("smoothing", 5), q.get("factor", 4.236))
    return df

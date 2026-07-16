"""
Offline smoke test: runs the full live pipeline (indicators + Jayantha +
Ashen detectors + confluence) on synthetic and/or fixture klines - no
network needed. Exercises the exact same run_jayantha_detectors() /
run_ashen_detectors() calls main.py's scan_pair() makes, so a passing run
here means the live code path is syntactically and logically sound
without needing Binance access.
Usage: python smoke_test.py [fixture.json ...]
"""

import json
import sys

import numpy as np
import pandas as pd
import yaml

from scanner.indicators import enrich
from scanner.jayantha_detectors import run_jayantha_detectors
from scanner.ashen_detectors import run_ashen_detectors
from main import confluence_score, load_config

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def synthetic_df(n=300, seed=42, trend=0.001):
    rng = np.random.default_rng(seed)
    ret = rng.normal(trend, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    o = np.roll(close, 1); o[0] = close[0]
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    df = pd.DataFrame({
        "open": o, "close": close,
        "high": np.maximum(o, close) + spread,
        "low": np.minimum(o, close) - spread,
        "volume": rng.uniform(100, 1000, n),
    })
    # inject a volume spike on last candle
    df.loc[n - 1, "volume"] = df["volume"].mean() * 5
    return df


def crafted_b2b_df(direction="bullish", n_trend=400, n_pullback=12,
                   trend_drift=0.004, pullback_mult=1.5, seed=11):
    """
    A clean, established trend followed by a pullback/rally that dips into
    (bullish) or pokes above (bearish) the 50-EMA zone and closes back on
    the trend side of it - i.e. a fixture engineered to actually trigger
    jayantha_b2b's "found" branch, rather than relying on random synthetic
    noise to stumble into this fairly narrow pattern by chance (which the
    plain synthetic_df cases essentially never do). This is what actually
    exercises the confirmation-scoring and signal-building logic, not just
    the "no signal, no crash" empty-list path.
    """
    rng = np.random.default_rng(seed)
    d = trend_drift if direction == "bullish" else -trend_drift
    ret = rng.normal(d, 0.0015, n_trend)
    close = 100 * np.exp(np.cumsum(ret))
    pullback_drift = -d * pullback_mult
    ret2 = rng.normal(pullback_drift, 0.0012, n_pullback)
    close2 = close[-1] * np.exp(np.cumsum(ret2))
    full_close = np.concatenate([close, close2])
    o = np.roll(full_close, 1); o[0] = full_close[0]
    spread = np.abs(rng.normal(0, 0.0008, len(full_close))) * full_close
    df = pd.DataFrame({
        "open": o, "close": full_close,
        "high": np.maximum(o, full_close) + spread,
        "low": np.minimum(o, full_close) - spread,
        "volume": rng.uniform(100, 1000, len(full_close)),
    })
    # EMA50 computed on the series so far, then the final candle is placed
    # explicitly relative to it (touch + reclaim for bullish, touch +
    # rejection for bearish) - the small shift EMA50 undergoes once this
    # last close is folded in is far smaller than the margins used here.
    ema50 = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    if direction == "bullish":
        new_open, new_low, new_close = ema50 * 1.003, ema50 * 0.996, ema50 * 1.008
        new_high = max(new_open, new_close) * 1.001
    else:
        new_open, new_high, new_close = ema50 * 0.997, ema50 * 1.004, ema50 * 0.992
        new_low = min(new_open, new_close) * 0.999
    new_row = pd.DataFrame([{"open": new_open, "high": new_high, "low": new_low,
                             "close": new_close, "volume": df["volume"].mean()}])
    return pd.concat([df, new_row], ignore_index=True)


def crafted_marubozu_df(direction="bullish", n=250, seed=17):
    """
    Marubozu only looks at the single last CLOSED candle, so this is the
    simplest possible crafted fixture: any trending/noisy series underneath,
    then a last candle explicitly shaped to have a near-full body and
    negligible wicks in the requested direction.
    """
    df = synthetic_df(n=n, seed=seed, trend=0.0005 if direction == "bullish" else -0.0005)
    last_close = df["close"].iloc[-2]
    if direction == "bullish":
        o, c = last_close, last_close * 1.02
        h, l = c * 1.0005, o * 0.9995  # wicks small enough to keep body_ratio well above the 0.85 threshold
    else:
        o, c = last_close, last_close * 0.98
        h, l = o * 1.0005, c * 0.9995
    df.loc[df.index[-1], ["open", "high", "low", "close"]] = [o, h, l, c]
    return df


def fixture_df(path):
    raw = json.load(open(path))
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df.iloc[:-1].reset_index(drop=True)  # drop forming candle


def run_case(name, df, cfg, expect_jayantha=None, expect_ashen=None):
    df = enrich(df, cfg)
    assert not df[["ema_20", "ema_50", "ema_200", "stochrsi_k", "atr"]].iloc[-1].isna().any(), \
        f"{name}: NaN indicators on last candle"
    jayantha_signals = run_jayantha_detectors(df, cfg)
    ashen_signals = run_ashen_detectors(df, cfg, htf_df=None)  # no HTF pairing in these single-timeframe fixtures
    signals = jayantha_signals + ashen_signals
    bias, strength = confluence_score(signals) if signals else ("none", 0)
    print(f"{name}: {len(signals)} signals ({len(jayantha_signals)} jayantha, {len(ashen_signals)} ashen) "
          f"| bias={bias} strength={strength}")
    for s in signals:
        extra = f" stop={s['stop']:.6g} target={s['target']:.6g}" if "stop" in s else ""
        print(f"    - {s['name']} [{s['direction']}]: {s['detail']}{extra}")
    if expect_jayantha is not None:
        assert bool(jayantha_signals) == expect_jayantha, \
            f"{name}: expected {'' if expect_jayantha else 'no '}jayantha signal, got {len(jayantha_signals)}"
    if expect_ashen is not None:
        assert bool(ashen_signals) == expect_ashen, \
            f"{name}: expected {'' if expect_ashen else 'no '}ashen signal, got {len(ashen_signals)}"
    if jayantha_signals:
        # A structural stop/target must bracket the close in the trade's
        # own direction, or attach_atr_risk's own bracket check would have
        # silently discarded it downstream anyway - catching that here
        # means a broken geometry calc fails loudly in the smoke test
        # instead of quietly vanishing three layers deeper in main.py.
        close = df["close"].iloc[-1]
        b2b = next(s for s in jayantha_signals if s["name"] == "jayantha_b2b")
        if bias == "bullish":
            assert b2b["stop"] < close < b2b["target"], f"{name}: bullish stop/target don't bracket close"
        else:
            assert b2b["stop"] > close > b2b["target"], f"{name}: bearish stop/target don't bracket close"
    for s in ashen_signals:
        close = df["close"].iloc[-1]
        if s["direction"] == "bullish":
            assert s["stop"] < close < s["target"], f"{name}: {s['name']} bullish stop/target don't bracket close"
        else:
            assert s["stop"] > close > s["target"], f"{name}: {s['name']} bearish stop/target don't bracket close"


if __name__ == "__main__":
    cfg = load_config()
    # Longer synthetic series than the default 300: jayantha_b2b needs
    # trend_ma_period (200) + 10 candles minimum just to evaluate, and a
    # trending series needs real room beyond that for an actual pullback
    # to the 50 MA to form within the trend, not just satisfy the floor.
    # Plain random-walk noise essentially never lands the exact "touched
    # the 50 MA and reclaimed/rejected on the last candle" pattern by
    # chance, so these three are expected to find nothing - they exist to
    # prove the "no signal" path never crashes, not to exercise detection.
    # jayantha_b2b is asserted False on plain noise (essentially never fires
    # by chance); ashen is left unconstrained here since marubozu in
    # particular can legitimately appear on random noise (it only looks at
    # the shape of the single last candle) - constraining it to False would
    # make this test flaky rather than actually catching a real bug.
    run_case("synthetic-uptrend", synthetic_df(n=500, trend=0.002), cfg, expect_jayantha=False)
    run_case("synthetic-downtrend", synthetic_df(n=500, seed=7, trend=-0.002), cfg, expect_jayantha=False)
    run_case("synthetic-flat", synthetic_df(n=500, seed=3, trend=0.0), cfg, expect_jayantha=False)
    # These ARE engineered to trigger a setup - the actual detection,
    # confirmation-scoring, and stop/target logic gets exercised here.
    run_case("crafted-bullish-b2b", crafted_b2b_df(direction="bullish"), cfg, expect_jayantha=True)
    run_case("crafted-bearish-b2b", crafted_b2b_df(direction="bearish"), cfg, expect_jayantha=True)
    run_case("crafted-bullish-marubozu", crafted_marubozu_df(direction="bullish"), cfg, expect_ashen=True)
    run_case("crafted-bearish-marubozu", crafted_marubozu_df(direction="bearish"), cfg, expect_ashen=True)
    for path in sys.argv[1:]:
        run_case(f"fixture:{path}", fixture_df(path), cfg)
    print("\nSmoke test PASSED")

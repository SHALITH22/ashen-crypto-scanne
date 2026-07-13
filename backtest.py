"""
Backtest: replay historical klines through the detectors and measure
what price actually did N candles later.

Usage:
  python backtest.py                          # pairs/timeframes from settings.yaml
  python backtest.py --pairs BTCUSDT,ETHUSDT --timeframes 1h,4h
  python backtest.py --synthetic              # offline self-test (no network)

Output: console table + backtest_results.json

Method (no lookahead):
  - indicators are causal (EWM/rolling use past data only), computed once
  - at each candle i, detectors see only df.iloc[:i+1]
  - forward return = close[i+h] / close[i] - 1 for each horizon h
  - a bullish signal "wins" if forward return > 0, bearish if < 0
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scanner.data import get_klines, get_all_usdt_pairs
from scanner.indicators import enrich
from scanner.patterns import run_all_detectors
from scanner.regime import regime_label

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
WARMUP = 210  # candles before signals count (EMA200 needs history)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def synthetic_df(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.012, n)
    close = 100 * np.exp(np.cumsum(ret))
    o = np.roll(close, 1); o[0] = close[0]
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    return pd.DataFrame({
        "open": o, "close": close,
        "high": np.maximum(o, close) + spread,
        "low": np.minimum(o, close) - spread,
        "volume": rng.uniform(100, 1000, n),
    })


def backtest_df(df: pd.DataFrame, cfg: dict, horizons: list[int],
                label: str) -> list[dict]:
    df = enrich(df.copy(), cfg)
    closes = df["close"].values
    rows = []
    end = len(df) - max(horizons)
    for i in range(WARMUP, end):
        window = df.iloc[:i + 1]
        for s in run_all_detectors(window, cfg):
            if s["direction"] not in ("bullish", "bearish"):
                continue
            row = {"source": label, "candle": i, "detector": s["name"],
                   "direction": s["direction"], "regime": regime_label(closes, i)}
            for h in horizons:
                row[f"ret_{h}"] = closes[i + h] / closes[i] - 1
            rows.append(row)
    return rows


def _summarize_groups(groups: dict, horizons: list[int]) -> dict:
    summary = {}
    for key, rs in sorted(groups.items()):
        entry = {"signals": len(rs)}
        direction = key[1] if isinstance(key, tuple) else key.split("/")[1]
        for h in horizons:
            rets = np.array([r[f"ret_{h}"] for r in rs])
            wins = (rets > 0) if direction == "bullish" else (rets < 0)
            entry[f"h{h}"] = {"win_rate": round(float(wins.mean()), 3),
                              "avg_ret_pct": round(float(rets.mean()) * 100, 3)}
        name = "/".join(key) if isinstance(key, tuple) else key
        summary[name] = entry
    return summary


def summarize(rows: list[dict], horizons: list[int]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["detector"], r["direction"])].append(r)
        groups[("ALL", r["direction"])].append(r)
    return _summarize_groups(groups, horizons)


def summarize_by_regime(rows: list[dict], horizons: list[int]) -> dict:
    """
    Same win-rate breakdown, but split by trending vs choppy market regime -
    a detector that only "works" in a trending bull market isn't the same
    as one that holds up regardless of regime.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["regime"] == "unknown":
            continue
        groups[(r["detector"], r["direction"], r["regime"])].append(r)
        groups[("ALL", r["direction"], r["regime"])].append(r)
    return _summarize_groups(groups, horizons)


def print_summary(summary: dict, horizons: list[int], title: str = ""):
    hdr = f"{'detector/direction':<38}{'n':>6}"
    for h in horizons:
        hdr += f"{f'win@{h}':>9}{f'avg%@{h}':>9}"
    print(f"\n{title}" if title else "")
    print(hdr)
    print("-" * len(hdr))
    for key, e in summary.items():
        line = f"{key:<38}{e['signals']:>6}"
        for h in horizons:
            line += f"{e[f'h{h}']['win_rate']:>9.1%}{e[f'h{h}']['avg_ret_pct']:>9.2f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="comma-separated, default from settings.yaml")
    ap.add_argument("--timeframes", help="comma-separated, default from settings.yaml")
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--limit", type=int, default=1000, help="candles per pair/tf (max 1000)")
    ap.add_argument("--synthetic", action="store_true", help="offline self-test, no network")
    ap.add_argument("--all-pairs", action="store_true",
                    help="backtest every USDT perpetual instead of just settings.yaml's pairs")
    ap.add_argument("--regime-split", action="store_true",
                    help="also print a trending-vs-choppy breakdown per detector")
    args = ap.parse_args()

    cfg = load_config()
    horizons = [int(h) for h in args.horizons.split(",")]
    rows = []

    if args.synthetic:
        for seed in range(3):
            rows += backtest_df(synthetic_df(seed=seed), cfg, horizons, f"synthetic{seed}")
    else:
        pairs = get_all_usdt_pairs() if args.all_pairs else (args.pairs.split(",") if args.pairs else cfg["pairs"])
        tfs = args.timeframes.split(",") if args.timeframes else cfg["timeframes"]
        print(f"Backtesting {len(pairs)} pair(s) x {len(tfs)} timeframe(s)...")
        for sym in pairs:
            for tf in tfs:
                df = get_klines(sym, tf, args.limit)
                if df is None or len(df) < WARMUP + max(horizons) + 10:
                    print(f"[skip] {sym} {tf}: not enough data")
                    continue
                print(f"[bt] {sym} {tf}: {len(df)} candles")
                rows += backtest_df(df, cfg, horizons, f"{sym}/{tf}")
                time.sleep(0.2)

    if not rows:
        print("No signals generated.")
        return
    summary = summarize(rows, horizons)
    print_summary(summary, horizons)

    output = {"horizons": horizons, "n_signals": len(rows), "summary": summary}
    if args.regime_split:
        regime_summary = summarize_by_regime(rows, horizons)
        print_summary(regime_summary, horizons, title="=== Trending vs choppy regime split ===")
        output["summary_by_regime"] = regime_summary

    out = Path(__file__).parent / "backtest_results.json"
    out.write_text(json.dumps(output, indent=2))
    print(f"\nDetail written to {out}")


if __name__ == "__main__":
    main()

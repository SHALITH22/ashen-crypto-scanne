"""
Tests whether the Crypto Fear & Greed Index - a free, daily, market-wide
macro sentiment reading (alternative.me, no API key, history back to 2018)
- adds real edge on top of the already-proven, BTC/ETH-filtered detector
set. Direct sibling of funding_rate_backtest.py; same walk-forward
methodology, same PROVEN_NAMES population, same report shape, so results
are directly comparable.

Concept (standard contrarian sentiment read): Extreme Greed means the
broad market mood is euphoric - a classic "everyone's already long"
setup. Extreme Fear is the mirror image. This checks whether a trade
going WITH the crowd's mood (same direction as the extreme reading)
performs differently than one going AGAINST it, on the same setups
already being alerted on live. Unlike funding rate, this is NOT assumed
to resolve the same way (with-crowd wins) - classify_fear_greed in
scanner/risk.py deliberately doesn't presuppose an answer; that's what
this script is for.

Fear & Greed is daily; funding data was per-8h and needed exact lookup.
Here, date-level lookup is enough - the index doesn't move faster than
that.

Usage:
  python fear_greed_backtest.py --max-trades 2000
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scanner.data import get_klines, get_top_pairs_by_volume, get_fear_greed_history
from scanner.indicators import enrich
from scanner.patterns import run_all_detectors
from scanner.risk import attach_atr_risk, setup_risk_plan, STRUCTURAL_NAMES, classify_fear_greed

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
WARMUP = 210
PROVEN_NAMES = STRUCTURAL_NAMES | {"ema_stack"}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def confluence_score(signals: list[dict]) -> tuple[str, float]:
    bull = sum(1.0 for s in signals if s["direction"] == "bullish")
    bear = sum(1.0 for s in signals if s["direction"] == "bearish")
    if bull > bear:
        return "bullish", round(bull, 2)
    if bear > bull:
        return "bearish", round(bear, 2)
    return "mixed", round(max(bull, bear), 2)


def trend_lookup(df: pd.DataFrame) -> dict:
    enriched = df.copy()
    enriched["ema20"] = enriched["close"].ewm(span=20, adjust=False).mean()
    return dict(zip(enriched["open_time"], enriched["close"] > enriched["ema20"]))


def fear_greed_arrays(fg_df: pd.DataFrame | None) -> tuple:
    """
    Raw (dates, classifications) numpy arrays, not a closure - closures
    aren't picklable for ProcessPoolExecutor (Windows uses spawn, which
    pickles every argument), same reasoning as funding_arrays in
    funding_rate_backtest.py.
    """
    if fg_df is None or fg_df.empty:
        return (np.array([], dtype="datetime64[ns]"), np.array([], dtype=object))
    return (fg_df["date"].to_numpy(), fg_df["classification"].to_numpy())


def lookup_fear_greed(dates: np.ndarray, classes: np.ndarray, ts) -> str | None:
    if len(dates) == 0:
        return None
    idx = np.searchsorted(dates, np.datetime64(ts).astype("datetime64[D]"), side="right") - 1
    if idx < 0:
        return None
    return str(classes[idx])


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int,
                     btc_trend: dict, eth_trend: dict, fg_dates: np.ndarray, fg_classes: np.ndarray) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    trades = []
    blocked_until: dict[str, int] = {}

    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[:i + 1]
        signals = run_all_detectors(window, cfg)
        if not signals:
            continue
        proven_signals = [s for s in signals if s["name"] in PROVEN_NAMES]
        if not proven_signals:
            continue
        close = float(window["close"].iloc[-1])
        atr = float(window["atr"].iloc[-1])
        signals = attach_atr_risk(signals, close, atr,
                                  risk_cfg.get("atr_multiplier", 1.5),
                                  risk_cfg.get("reward_risk_ratio", 2.0),
                                  risk_cfg.get("max_stop_pct"))
        bias, strength = confluence_score(signals)
        if strength < min_confluence:
            continue
        proven_bias_signals = [s for s in proven_signals if s["direction"] == bias]
        if not proven_bias_signals:
            continue
        risk = setup_risk_plan(signals, bias, close, risk_cfg.get("min_risk_reward", 1.0),
                               market_disagrees=True,
                               target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not risk or risk["based_on"] not in PROVEN_NAMES:
            continue
        key = risk["based_on"]
        if blocked_until.get(key, -1) >= i:
            continue

        open_time = window["open_time"].iloc[-1]
        btc_bull = btc_trend.get(open_time)
        eth_bull = eth_trend.get(open_time)
        if btc_bull is None or eth_bull is None:
            continue
        trade_is_bullish = bias == "bullish"
        if not ((btc_bull != trade_is_bullish) and (eth_bull != trade_is_bullish)):
            continue

        fg_class = lookup_fear_greed(fg_dates, fg_classes, open_time)
        fg_sentiment = classify_fear_greed(fg_class, bias)

        outcome, outcome_price, resolved_at = None, None, None
        end = min(i + 1 + horizon_candles, len(df))
        for j in range(i + 1, end):
            candle = df.iloc[j]
            if bias == "bullish":
                if candle["low"] <= risk["stop"]:
                    outcome, outcome_price, resolved_at = "loss", risk["stop"], j
                    break
                if candle["high"] >= risk["target"]:
                    outcome, outcome_price, resolved_at = "win", risk["target"], j
                    break
            else:
                if candle["high"] >= risk["stop"]:
                    outcome, outcome_price, resolved_at = "loss", risk["stop"], j
                    break
                if candle["low"] <= risk["target"]:
                    outcome, outcome_price, resolved_at = "win", risk["target"], j
                    break
        if outcome is None:
            resolved_at = end - 1
            outcome, outcome_price = "expired", float(df["close"].iloc[resolved_at])

        trades.append({
            "symbol": symbol, "timeframe": tf, "based_on": risk["based_on"], "direction": bias,
            "risk_reward": risk["risk_reward"], "outcome": outcome,
            "outcome_pct": round((outcome_price - risk["entry"]) / risk["entry"] * 100, 3),
            "fear_greed_class": fg_class, "fear_greed_sentiment": fg_sentiment,
        })
        blocked_until[key] = resolved_at

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    min_conf = cfg["output"]["min_confluence"]
    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = [p for p in dict.fromkeys(static_pairs + top_pairs) if p not in ("BTCUSDT", "ETHUSDT")]
    timeframes = cfg["timeframes"]

    print("Fetching BTC/ETH trend reference per timeframe...", flush=True)
    btc_trends, eth_trends = {}, {}
    for tf in timeframes:
        btc_df = get_klines("BTCUSDT", tf, 1000)
        eth_df = get_klines("ETHUSDT", tf, 1000)
        btc_trends[tf] = trend_lookup(btc_df) if btc_df is not None else {}
        eth_trends[tf] = trend_lookup(eth_df) if eth_df is not None else {}

    print("Fetching Fear & Greed Index history (single series, market-wide, not per-pair)...", flush=True)
    fg_df = get_fear_greed_history(limit=0)
    fg_dates, fg_classes = fear_greed_arrays(fg_df)
    print(f"  {len(fg_dates)} days of history, {fg_dates.min() if len(fg_dates) else 'n/a'} "
          f"to {fg_dates.max() if len(fg_dates) else 'n/a'}", flush=True)

    jobs = [(s, tf) for tf in timeframes for s in pairs]
    print(f"Simulating up to {len(pairs)} pairs x {len(timeframes)} timeframes "
          f"({args.workers} workers, target: {args.max_trades} trades)...", flush=True)

    all_trades = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon, min_conf,
                                   btc_trends[tf], eth_trends[tf], fg_dates, fg_classes): (s, tf) for s, tf in jobs}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                trades = future.result()
            except Exception as e:
                print(f"  [error] {futures[future]}: {e}", flush=True)
                continue
            all_trades.extend(trades)
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {len(all_trades)} trades ({time.time()-t0:.0f}s)", flush=True)
            if len(all_trades) >= args.max_trades:
                print(f"  reached {len(all_trades)} trades, stopping early", flush=True)
                for f in futures:
                    f.cancel()
                break

    print(f"\nDone: {len(all_trades)} trades from {done}/{len(jobs)} series in {time.time()-t0:.0f}s\n", flush=True)

    decided = [t for t in all_trades if t["outcome"] in ("win", "loss")]

    def report(label, rows):
        n = len(rows)
        if n < 15:
            print(f"{label:<35}n={n:<6}(too few to trust)")
            return
        wins = sum(1 for r in rows if r["outcome"] == "win")
        wr = wins / n
        avg_rr = sum(r["risk_reward"] for r in rows if r["risk_reward"]) / n
        exp = wr * (1 + avg_rr) - 1
        print(f"{label:<35}n={n:<6}win_rate={wr:>6.1%}  avg_rr={avg_rr:>5.2f}  expectancy={exp:+.3f}R")

    print("=== Fear & Greed sentiment classification (all proven+BTC/ETH-filtered trades) ===")
    by_class = defaultdict(list)
    for t in decided:
        if t["fear_greed_sentiment"]:
            by_class[t["fear_greed_sentiment"]].append(t)
    for c in ("against_crowd", "neutral", "with_crowd"):
        report(c, by_class.get(c, []))

    print("\n=== Raw Fear & Greed reading (all 5 buckets, unconditional on direction) ===")
    by_reading = defaultdict(list)
    for t in decided:
        if t["fear_greed_class"]:
            by_reading[t["fear_greed_class"]].append(t)
    for c in ("Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"):
        report(c, by_reading.get(c, []))

    out = Path(__file__).parent / "fear_greed_results.json"
    out.write_text(json.dumps({"n_trades": len(all_trades), "trades": all_trades}, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

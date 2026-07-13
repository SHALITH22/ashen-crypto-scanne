"""
Tests whether MULTIPLE proven detectors agreeing at once (e.g.
ascending_triangle AND ema_stack both firing bullish on the same setup)
beats a single proven detector alone - a different question from generic
confluence count (already tested, found no benefit: confluence_btc_backtest.py
showed expectancy declining as more signals of ANY kind pile on). This
specifically checks whether two things ALREADY SHOWN to have real edge
overlapping is a stronger signal than either alone, not just "more signals."

Runs on the same BTC/ETH-filtered setup as confluence_btc_backtest.py
(that's what's actually live), and only counts overlap among the 4
detectors already confirmed profitable: ascending_triangle, descending_
triangle, rising_wedge, ema_stack.

Usage:
  python proven_overlap_backtest.py --max-trades 2000
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from scanner.data import get_klines, get_top_pairs_by_volume
from scanner.indicators import enrich
from scanner.patterns import run_all_detectors
from scanner.risk import attach_atr_risk, setup_risk_plan, STRUCTURAL_NAMES

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


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int,
                     btc_trend: dict, eth_trend: dict) -> list[dict]:
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
        close = float(window["close"].iloc[-1])
        atr = float(window["atr"].iloc[-1])
        signals = attach_atr_risk(signals, close, atr,
                                  risk_cfg.get("atr_multiplier", 1.5),
                                  risk_cfg.get("reward_risk_ratio", 2.0),
                                  risk_cfg.get("max_stop_pct"))
        bias, strength = confluence_score(signals)
        if strength < min_confluence:
            continue

        # How many DISTINCT proven detectors agree with this bias - the
        # thing actually being tested here, separate from raw signal count.
        proven_agreeing = {s["name"] for s in signals if s["direction"] == bias and s["name"] in PROVEN_NAMES}
        if not proven_agreeing:
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
            "proven_names": sorted(proven_agreeing), "proven_count": len(proven_agreeing),
            "risk_reward": risk["risk_reward"], "outcome": outcome,
            "outcome_pct": round((outcome_price - risk["entry"]) / risk["entry"] * 100, 3),
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

    jobs = [(s, tf) for tf in timeframes for s in pairs]
    print(f"Simulating up to {len(pairs)} pairs x {len(timeframes)} timeframes "
          f"({args.workers} workers, target: {args.max_trades} trades)...", flush=True)

    all_trades = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon, min_conf,
                                   btc_trends[tf], eth_trends[tf]): (s, tf) for s, tf in jobs}
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
            print(f"{label:<45}n={n:<6}(too few to trust)")
            return
        wins = sum(1 for r in rows if r["outcome"] == "win")
        wr = wins / n
        avg_rr = sum(r["risk_reward"] for r in rows if r["risk_reward"]) / n
        exp = wr * (1 + avg_rr) - 1
        print(f"{label:<45}n={n:<6}win_rate={wr:>6.1%}  avg_rr={avg_rr:>5.2f}  expectancy={exp:+.3f}R")

    print("=== Single proven detector vs multiple overlapping ===")
    report("Exactly 1 proven detector", [t for t in decided if t["proven_count"] == 1])
    report("2+ proven detectors overlapping", [t for t in decided if t["proven_count"] >= 2])

    print("\n=== By overlap count ===")
    by_count = defaultdict(list)
    for t in decided:
        by_count[t["proven_count"]].append(t)
    for c in sorted(by_count):
        report(f"proven_count={c}", by_count[c])

    print("\n=== Which specific pairs overlap most, and how they perform ===")
    by_combo = defaultdict(list)
    for t in decided:
        if t["proven_count"] >= 2:
            by_combo[tuple(t["proven_names"])].append(t)
    for combo, rows in sorted(by_combo.items(), key=lambda kv: -len(kv[1])):
        report(" + ".join(combo), rows)

    out = Path(__file__).parent / "proven_overlap_results.json"
    out.write_text(json.dumps({"n_trades": len(all_trades), "trades": all_trades}, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

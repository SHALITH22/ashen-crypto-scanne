"""
Tests three concrete levers for raising win rate, on the same live setup
(proven detectors + BTC/ETH filter, matching what's actually deployed):

  1. Regime filter: trending vs choppy at entry - does blocking choppy-
     regime trades raise win rate AND expectancy (a free improvement), or
     just shrink the sample without helping?
  2. Partial targets: taking profit at a FRACTION of the full geometric
     target instead of the whole thing - mechanically raises win rate,
     but shrinks the payoff per win. Finds where (if anywhere) that
     trade-off is actually worth it.
  3. Extended horizon: giving a trade more candles to resolve before
     calling it "expired" - some trades may just need more time, not a
     different setup.

One forward walk per trade covers all three: track the stop-hit candle,
the first candle reaching each target fraction, and the regime label at
entry, then run tables at both the current 20-candle horizon and a longer
60-candle horizon.

Usage:
  python win_rate_levers_backtest.py --max-trades 3000
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
from scanner.regime import regime_label
from scanner.risk import attach_atr_risk, setup_risk_plan, STRUCTURAL_NAMES

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
WARMUP = 210
PROVEN_NAMES = STRUCTURAL_NAMES | {"ema_stack", "qqe_cross"}
TARGET_FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
LONG_HORIZON = 60


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


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, min_confluence: int,
                     btc_trend: dict, eth_trend: dict) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + LONG_HORIZON + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    closes = df["close"].values
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

        regime = regime_label(closes, i)
        stop_distance = abs(risk["entry"] - risk["stop"])
        target_distance = abs(risk["target"] - risk["entry"])
        if stop_distance <= 0 or target_distance <= 0:
            continue

        stop_hit_at = None
        target_frac_hit_at = {f: None for f in TARGET_FRACTIONS}
        end = min(i + 1 + LONG_HORIZON, len(df))
        for j in range(i + 1, end):
            candle = df.iloc[j]
            if bias == "bullish":
                stop_hit = candle["low"] <= risk["stop"]
                favorable_px = candle["high"]
                reached = lambda frac: favorable_px >= risk["entry"] + target_distance * frac
            else:
                stop_hit = candle["high"] >= risk["stop"]
                favorable_px = candle["low"]
                reached = lambda frac: favorable_px <= risk["entry"] - target_distance * frac

            if stop_hit:
                stop_hit_at = j
                break
            for f in TARGET_FRACTIONS:
                if target_frac_hit_at[f] is None and reached(f):
                    target_frac_hit_at[f] = j

        trades.append({
            "symbol": symbol, "timeframe": tf, "based_on": risk["based_on"], "direction": bias,
            "regime": regime, "risk_reward": risk["risk_reward"],
            # Offsets relative to the entry candle (i), not absolute df index -
            # outcome_at() below compares these against a small horizon like 20.
            "stop_hit_offset": (stop_hit_at - i) if stop_hit_at is not None else None,
            "target_frac_offset": {str(f): (target_frac_hit_at[f] - i) if target_frac_hit_at[f] is not None else None
                                   for f in TARGET_FRACTIONS},
        })
        blocked_until[key] = stop_hit_at if stop_hit_at is not None else (target_frac_hit_at[1.0] or end - 1)

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=3000)
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
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, min_conf,
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

    def outcome_at(t, horizon, target_frac):
        """Resolve a trade under a given horizon and target-taking fraction."""
        stop_at = t["stop_hit_offset"]
        target_at = t["target_frac_offset"][str(target_frac)]
        if stop_at is not None and stop_at <= horizon and (target_at is None or stop_at < target_at):
            return "loss"
        if target_at is not None and target_at <= horizon:
            return "win"
        if stop_at is not None and stop_at <= horizon:
            return "loss"
        return "expired"

    def report(label, rows, horizon, target_frac):
        outcomes = [outcome_at(t, horizon, target_frac) for t in rows]
        decided = [(t, o) for t, o in zip(rows, outcomes) if o in ("win", "loss")]
        n = len(decided)
        if n < 20:
            print(f"{label:<38}n={n:<6}(too few to trust)")
            return
        wins = sum(1 for _, o in decided if o == "win")
        wr = wins / n
        avg_rr = sum(t["risk_reward"] * target_frac for t, _ in decided if t["risk_reward"]) / n
        exp = wr * (1 + avg_rr) - 1
        print(f"{label:<38}n={n:<6}win_rate={wr:>6.1%}  eff_rr={avg_rr:>5.2f}  expectancy={exp:+.3f}R")

    print("=== 1. Regime filter (horizon=20, full target) ===")
    for regime in ("trending", "choppy", "unknown"):
        rows = [t for t in all_trades if t["regime"] == regime]
        report(regime, rows, 20, 1.0)
    report("all regimes combined", all_trades, 20, 1.0)

    print("\n=== 2. Partial target fraction (horizon=20) ===")
    for f in TARGET_FRACTIONS:
        report(f"target={f}", all_trades, 20, f)

    print("\n=== 3. Extended horizon (full target) ===")
    for h in (20, 30, 40, 60):
        report(f"horizon={h}", all_trades, h, 1.0)

    out = Path(__file__).parent / "win_rate_levers_results.json"
    out.write_text(json.dumps({"n_trades": len(all_trades), "trades": all_trades}, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

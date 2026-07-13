"""
Tests the user's concrete concern (TARUSDT qqe_cross trade hit stop before
reaching target) that the ATR-based stop may be systematically too tight,
cutting winners short before they have room to develop.

Only applies to the two non-structural tradeable detectors - ema_stack/bullish
and qqe_cross/bullish - since structural patterns (ascending_triangle,
descending_triangle, rising_wedge) get their stop from real chart geometry in
patterns.py, not from scanner.risk.attach_atr_risk's ATR multiplier.

Two independent variants at each widening multiplier m (relative to config's
atr_multiplier=1.5), since they answer different questions:

  A) "rescaled" - stop AND target both scale by m, same as re-tuning
     atr_multiplier itself (the actual lever in attach_atr_risk). R:R stays
     fixed at the config's reward_risk_ratio=2.0.
  B) "stop_only" - only the stop widens by m; target stays at the m=1.0
     baseline distance. Directly tests "just give the stop more room" -
     this worsens R:R as m grows, mirroring the trade-off win_rate_levers_
     backtest.py found on the target side (taking profit early raises win
     rate but can cost more expectancy than it gains).

One forward walk per candle resolves every variant simultaneously - they
share the same underlying price path, only stop/target distances differ.

Usage:
  python stop_width_backtest.py --max-trades 4000
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
from scanner.risk import MARKET_FILTER_NAMES

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
WARMUP = 210
MULTIPLIERS = [1.0, 1.2, 1.5, 2.0, 3.0]  # 1.0 = current live behavior (baseline)
TARGET_NAMES = {"ema_stack", "qqe_cross"}  # the two ATR-based tradeable detectors


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def trend_lookup(df: pd.DataFrame) -> dict:
    enriched = df.copy()
    enriched["ema20"] = enriched["close"].ewm(span=20, adjust=False).mean()
    return dict(zip(enriched["open_time"], enriched["close"] > enriched["ema20"]))


def build_variants(close: float, bias: str, base_distance: float, reward_risk: float,
                   target_fraction: float, multipliers: list[float]) -> dict:
    """
    variant key -> {"stop": price, "target": price, "outcome": None, "price": None}
    "1.0" is the single shared baseline. "stop_only" variants only exist for m > 1.0
    (at m=1.0 they're identical to baseline).
    """
    sign = 1 if bias == "bullish" else -1
    variants = {}
    for m in multipliers:
        stop_dist = base_distance * m
        stop = close - sign * stop_dist
        rescaled_target_dist = stop_dist * reward_risk * target_fraction
        rescaled_target = close + sign * rescaled_target_dist
        label = "baseline" if m == 1.0 else f"rescaled_{m}"
        variants[label] = {"stop": stop, "target": rescaled_target, "outcome": None, "price": None}
        if m != 1.0:
            stop_only_target_dist = base_distance * reward_risk * target_fraction  # fixed at baseline
            stop_only_target = close + sign * stop_only_target_dist
            variants[f"stop_only_{m}"] = {"stop": stop, "target": stop_only_target, "outcome": None, "price": None}
    return variants


def resolve_variants(df, start_i: int, bias: str, variants: dict, horizon_candles: int) -> None:
    """Mutates variants in place with outcome + price, one shared forward walk."""
    end = min(start_i + 1 + horizon_candles, len(df))
    for j in range(start_i + 1, end):
        candle = df.iloc[j]
        pending = False
        for v in variants.values():
            if v["outcome"] is not None:
                continue
            if bias == "bullish":
                if candle["low"] <= v["stop"]:
                    v["outcome"], v["price"] = "loss", v["stop"]
                    continue
                if candle["high"] >= v["target"]:
                    v["outcome"], v["price"] = "win", v["target"]
                    continue
            else:
                if candle["high"] >= v["stop"]:
                    v["outcome"], v["price"] = "loss", v["stop"]
                    continue
                if candle["low"] <= v["target"]:
                    v["outcome"], v["price"] = "win", v["target"]
                    continue
            pending = True
        if not pending:
            break

    end_close = float(df["close"].iloc[end - 1]) if end > start_i + 1 else None
    for v in variants.values():
        if v["outcome"] is None:
            v["outcome"], v["price"] = "expired", end_close


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int,
                     btc_trend: dict, eth_trend: dict) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    atr_mult = risk_cfg.get("atr_multiplier", 1.5)
    reward_risk = risk_cfg.get("reward_risk_ratio", 2.0)
    max_stop_pct = risk_cfg.get("max_stop_pct")
    target_fraction = risk_cfg.get("target_fraction", 1.0)
    trades = []
    blocked_until: dict[str, int] = {}

    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[:i + 1]
        signals = run_all_detectors(window, cfg)
        if not signals:
            continue

        bull_n = sum(1 for s in signals if s["direction"] == "bullish")
        bear_n = sum(1 for s in signals if s["direction"] == "bearish")
        bias = "bullish" if bull_n > bear_n else ("bearish" if bear_n > bull_n else None)
        if bias != "bullish":  # both tradeable detectors here are bullish-only per the live blacklist
            continue
        strength = bull_n
        if strength < min_confluence:
            continue

        # Only ema_stack/qqe_cross, only if they don't already carry a
        # structural stop/target (they never do - they're generic - but
        # this mirrors rr_sweep_backtest.py's exact filter for consistency).
        candidates = [s for s in signals if s["name"] in TARGET_NAMES
                      and s["direction"] == "bullish" and "stop" not in s]
        if not candidates:
            continue

        close = float(window["close"].iloc[-1])
        atr = float(window["atr"].iloc[-1])
        if atr is None or atr <= 0:
            continue
        base_distance = atr * atr_mult
        if max_stop_pct is not None:
            base_distance = min(base_distance, close * max_stop_pct / 100)
        if base_distance <= 0:
            continue

        open_time = window["open_time"].iloc[-1]
        for s in candidates:
            name = s["name"]
            if blocked_until.get(name, -1) >= i:
                continue
            # ema_stack is subject to the live BTC/ETH independence filter
            # (MARKET_FILTER_NAMES); qqe_cross is not - matches setup_risk_plan.
            if name in MARKET_FILTER_NAMES:
                btc_bull = btc_trend.get(open_time)
                eth_bull = eth_trend.get(open_time)
                if btc_bull is None or eth_bull is None or btc_bull or eth_bull:
                    continue  # BTC/ETH must both disagree (be bearish) for a bullish trade

            variants = build_variants(close, bias, base_distance, reward_risk, target_fraction, MULTIPLIERS)
            resolve_variants(df, i, bias, variants, horizon_candles)

            row = {"symbol": symbol, "timeframe": tf, "detector": name, "open_time": str(open_time)}
            for key, v in variants.items():
                row[f"{key}_outcome"] = v["outcome"]
                row[f"{key}_rr"] = (abs(v["target"] - close) / abs(close - v["stop"])
                                    if v["outcome"] in ("win", "loss") and abs(close - v["stop"]) > 0 else None)
            trades.append(row)

            # Block by the widest variant's horizon so overlapping windows
            # for the same detector name don't double-count the same move.
            blocked_until[name] = min(i + 1 + horizon_candles, len(df)) - 1

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    min_conf = cfg["output"]["min_confluence"]
    horizon = args.horizon or cfg.get("journal", {}).get("horizon_candles", 60)
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
    print(f"Simulating up to {len(pairs)} pairs x {len(timeframes)} timeframes, "
          f"ema_stack/bullish + qqe_cross/bullish only "
          f"({args.workers} workers, horizon={horizon}, target: {args.max_trades} trades)...", flush=True)

    all_trades = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, horizon, min_conf,
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

    def report(label: str, outcome_key: str, rr_key: str, rows: list[dict]):
        decided = [r for r in rows if r[outcome_key] in ("win", "loss")]
        n = len(decided)
        if n < 15:
            print(f"{label:<28}n={n:<6}(too few to trust)")
            return None
        wins = sum(1 for r in decided if r[outcome_key] == "win")
        wr = wins / n
        avg_rr = sum(r[rr_key] for r in decided if r[rr_key]) / n
        exp = wr * (1 + avg_rr) - 1
        print(f"{label:<28}n={n:<6}win_rate={wr:>6.1%}  avg_rr={avg_rr:>5.2f}  expectancy={exp:+.3f}R")
        return {"n": n, "win_rate": round(wr, 3), "avg_rr": round(avg_rr, 2), "expectancy": round(exp, 3)}

    summary = {}
    for name in sorted(TARGET_NAMES):
        rows = [t for t in all_trades if t["detector"] == name]
        print(f"\n=== {name}/bullish (n_opportunities={len(rows)}) ===")
        print(f"{'variant':<28}")
        summary[name] = {}
        summary[name]["baseline"] = report("baseline (current, m=1.0)", "baseline_outcome", "baseline_rr", rows)
        for m in MULTIPLIERS:
            if m == 1.0:
                continue
            summary[name][f"rescaled_{m}"] = report(f"rescaled x{m} (stop+target)",
                                                     f"rescaled_{m}_outcome", f"rescaled_{m}_rr", rows)
            summary[name][f"stop_only_{m}"] = report(f"stop_only x{m} (target fixed)",
                                                      f"stop_only_{m}_outcome", f"stop_only_{m}_rr", rows)

    print(f"\n=== Combined (ema_stack + qqe_cross pooled) ===")
    all_rows = all_trades
    summary["combined"] = {}
    summary["combined"]["baseline"] = report("baseline (current, m=1.0)", "baseline_outcome", "baseline_rr", all_rows)
    for m in MULTIPLIERS:
        if m == 1.0:
            continue
        summary["combined"][f"rescaled_{m}"] = report(f"rescaled x{m} (stop+target)",
                                                       f"rescaled_{m}_outcome", f"rescaled_{m}_rr", all_rows)
        summary["combined"][f"stop_only_{m}"] = report(f"stop_only x{m} (target fixed)",
                                                        f"stop_only_{m}_outcome", f"stop_only_{m}_rr", all_rows)

    out = Path(__file__).parent / "stop_width_results.json"
    out.write_text(json.dumps({"n_trades": len(all_trades), "summary": summary, "trades": all_trades}, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

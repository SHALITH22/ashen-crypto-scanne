"""
Reward:risk sweep + confluence-strength check, built on the same walk-
forward simulation as realistic_backtest.py.

Two questions realistic_backtest.py couldn't answer on its own:
  1. Its detectors were only ever tested at the config's fixed 2:1 reward:
     risk. A detector losing at 2:1 could easily be profitable at 1:1 (an
     easier, closer target against the SAME stop) - this sweeps several R:R
     candidates per detector using ONE shared forward walk (the stop is
     identical across candidates; only the target distance differs, so
     smaller targets naturally resolve first without extra API calls).
  2. Detectors exist mainly to feed the "confluence strength" score (how
     many signals agree), not necessarily to each independently support a
     full trade alone. This also buckets simulated trades by how many
     signals agreed at entry, to check whether higher confluence actually
     predicts a better win rate - the real justification (or not) for
     having this many detectors at all.

Usage:
  python rr_sweep_backtest.py
  python rr_sweep_backtest.py --max-trades 4000
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from scanner.data import get_klines, get_top_pairs_by_volume
from scanner.indicators import enrich
from scanner.patterns import run_all_detectors

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
WARMUP = 210
RR_CANDIDATES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_multi_rr(df, start_i: int, bias: str, close: float, distance: float,
                     rr_candidates: list[float], horizon_candles: int) -> dict[float, tuple[str, float]]:
    """
    One forward walk resolving every RR candidate at once. All candidates
    share the same stop (only the target distance differs), so once the
    stop is hit, every still-unresolved candidate is a loss at that same
    point; smaller RR targets are closer and naturally resolve first.
    """
    stop = close - distance if bias == "bullish" else close + distance
    remaining = {rr: None for rr in rr_candidates}
    end = min(start_i + 1 + horizon_candles, len(df))
    for j in range(start_i + 1, end):
        candle = df.iloc[j]
        stop_hit = (candle["low"] <= stop) if bias == "bullish" else (candle["high"] >= stop)
        if stop_hit:
            for rr in remaining:
                if remaining[rr] is None:
                    remaining[rr] = ("loss", stop)
            break
        for rr in remaining:
            if remaining[rr] is not None:
                continue
            target = close + distance * rr if bias == "bullish" else close - distance * rr
            hit = (candle["high"] >= target) if bias == "bullish" else (candle["low"] <= target)
            if hit:
                remaining[rr] = ("win", target)
    end_close = float(df["close"].iloc[end - 1]) if end > start_i + 1 else close
    for rr in remaining:
        if remaining[rr] is None:
            remaining[rr] = ("expired", end_close)
    return remaining


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    atr_mult = risk_cfg.get("atr_multiplier", 1.5)
    max_stop_pct = risk_cfg.get("max_stop_pct")
    trades = []
    blocked_until: dict[str, int] = {}

    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[:i + 1]
        signals = run_all_detectors(window, cfg)
        if not signals:
            continue
        close = float(window["close"].iloc[-1])
        atr = float(window["atr"].iloc[-1])
        if atr is None or atr <= 0:
            continue
        distance = atr * atr_mult
        if max_stop_pct is not None:
            distance = min(distance, close * max_stop_pct / 100)

        bull_n = sum(1 for s in signals if s["direction"] == "bullish")
        bear_n = sum(1 for s in signals if s["direction"] == "bearish")
        bias = "bullish" if bull_n > bear_n else ("bearish" if bear_n > bull_n else None)
        if bias is None:
            continue
        strength = max(bull_n, bear_n)
        if strength < min_confluence:
            continue

        # Only generic (non-structural) signals share this uniform ATR
        # stop/target formula - structural patterns compute their own fixed
        # geometry and aren't meaningfully "swept" by a global R:R knob.
        generic_names = {s["name"] for s in signals if s["direction"] == bias and "stop" not in s}
        for name in generic_names:
            if blocked_until.get(name, -1) >= i:
                continue
            results = resolve_multi_rr(df, i, bias, close, distance, RR_CANDIDATES, horizon_candles)
            resolved_ats = []
            for rr, (outcome, price) in results.items():
                trades.append({
                    "symbol": symbol, "timeframe": tf, "detector": name, "direction": bias,
                    "strength": strength, "rr": rr, "outcome": outcome,
                    "outcome_pct": round((price - close) / close * 100, 3),
                })
            # Block until the largest-RR candidate's resolution point (most
            # conservative - avoids overlapping windows for the same name).
            stop = close - distance if bias == "bullish" else close + distance
            end = min(i + 1 + horizon_candles, len(df))
            blocked_until[name] = end - 1  # simplification: full horizon, since largest RR often expires

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    min_conf = cfg["output"]["min_confluence"]
    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = list(dict.fromkeys(static_pairs + top_pairs))
    timeframes = cfg["timeframes"]
    jobs = [(s, tf) for tf in timeframes for s in pairs]

    print(f"Simulating up to {len(pairs)} pairs x {len(timeframes)} timeframes "
          f"({args.workers} workers, target: {args.max_trades} entry-opportunities x {len(RR_CANDIDATES)} R:R each)...",
          flush=True)

    all_rows = []
    n_opportunities = 0
    errors = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon, min_conf): (s, tf) for s, tf in jobs}
        done = 0
        for future in as_completed(futures):
            s, tf = futures[future]
            done += 1
            try:
                rows = future.result()
            except Exception as e:
                errors.append(f"{s} {tf}: {e}")
                continue
            all_rows.extend(rows)
            n_opportunities = len(all_rows) // len(RR_CANDIDATES)
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {n_opportunities} entry-opportunities so far "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
            if n_opportunities >= args.max_trades:
                print(f"  reached {n_opportunities} opportunities, stopping early", flush=True)
                for f in futures:
                    f.cancel()
                break

    print(f"\nDone: {n_opportunities} entry-opportunities ({len(all_rows)} rows across "
          f"{len(RR_CANDIDATES)} R:R candidates each) from {done}/{len(jobs)} series in {time.time()-t0:.0f}s",
          flush=True)
    if errors:
        print(f"{len(errors)} series failed: {errors[:5]}")

    # --- R:R sweep per detector/direction ---
    decided = [r for r in all_rows if r["outcome"] in ("win", "loss")]
    groups = defaultdict(list)
    for r in decided:
        groups[(r["detector"], r["direction"], r["rr"])].append(r)

    print(f"\n=== R:R sweep (expectancy in R per trade; positive = real edge) ===")
    by_detector = defaultdict(dict)
    for (name, direction, rr), rows in groups.items():
        n = len(rows)
        wins = sum(1 for r in rows if r["outcome"] == "win")
        wr = wins / n if n else 0
        expectancy = wr * (1 + rr) - 1
        by_detector[(name, direction)][rr] = {"n": n, "win_rate": round(wr, 3), "expectancy": round(expectancy, 3)}

    hdr = f"{'detector/direction':<26}" + "".join(f"{f'RR={rr}':>16}" for rr in RR_CANDIDATES)
    print(hdr)
    print("-" * len(hdr))
    for (name, direction), by_rr in sorted(by_detector.items()):
        line = f"{name+'/'+direction:<26}"
        for rr in RR_CANDIDATES:
            cell = by_rr.get(rr)
            cell_str = "n/a" if not cell else f"{cell['expectancy']:+.2f}R(n={cell['n']})"
            line += f"{cell_str:>16}"
        print(line)

    # --- confluence strength vs win rate, fixed at RR=2.0 ---
    base = [r for r in decided if r["rr"] == 2.0]
    by_strength = defaultdict(list)
    for r in base:
        by_strength[r["strength"]].append(r)
    print(f"\n=== Confluence strength vs win rate (fixed at R:R=2.0) ===")
    print(f"{'strength':>10}{'n':>8}{'win_rate':>10}{'expectancy':>12}")
    for strength in sorted(by_strength):
        rows = by_strength[strength]
        n = len(rows)
        if n < 15:
            continue
        wr = sum(1 for r in rows if r["outcome"] == "win") / n
        exp = wr * 3 - 1
        print(f"{strength:>10}{n:>8}{wr:>10.1%}{exp:>+11.2f}R")

    out = Path(__file__).parent / "rr_sweep_results.json"
    out.write_text(json.dumps({
        "n_opportunities": n_opportunities,
        "by_detector_rr": {f"{k[0]}/{k[1]}": v for k, v in by_detector.items()},
        "rows": all_rows,
    }, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

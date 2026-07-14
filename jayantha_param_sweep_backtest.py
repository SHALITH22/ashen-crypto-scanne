"""
Parameter sweep for jayantha_b2b, reusing jayantha_realistic_backtest.py's
exact simulate_pair_tf (same real stop/target walk-forward methodology) -
only the config passed to it changes between runs.

Motivation: jayantha_realistic_backtest.py's first real-data run found
jayantha_b2b/bullish losing (25.7% win rate against a 30.8% breakeven,
596 trades) while jayantha_b2b/bearish was only marginally profitable
(35.4% vs 33.0% breakeven). stop_ma_period=100 was chosen from a
qualitative "100 or 200" hint in the source material, never empirically
validated - this sweep checks whether a different stop level (or the
entry_ma_period pullback target) changes that picture, and separately
whether the bullish/bearish gap looks like a parameter problem or a
market-regime effect (i.e. this historical window trending down more
than up, which no B2B parameter choice would fix).

Does NOT write to realistic_backtest_results.json - this is exploratory,
not the file main.py reads live. Prints a comparison table only.

Usage:
  python jayantha_param_sweep_backtest.py
  python jayantha_param_sweep_backtest.py --max-trades 400
"""

import argparse
import copy
import time
from collections import defaultdict

from realistic_backtest import load_config
from jayantha_realistic_backtest import simulate_pair_tf
from scanner.data import get_top_pairs_by_volume
from concurrent.futures import ProcessPoolExecutor, as_completed

# (label, overrides applied to cfg["jayantha"]["b2b"])
SWEEP_CONFIGS = [
    ("stop_ma=50",  {"stop_ma_period": 50}),
    ("stop_ma=100 (current)", {"stop_ma_period": 100}),
    ("stop_ma=150", {"stop_ma_period": 150}),
    ("stop_ma=200", {"stop_ma_period": 200}),
    ("entry_ma=20", {"entry_ma_period": 20}),
    ("entry_ma=100", {"entry_ma_period": 100}),
]


def run_sweep_config(label: str, overrides: dict, base_cfg: dict, pairs: list[str], timeframes: list[str],
                     max_trades: int, horizon: int, workers: int, min_conf: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["jayantha"]["b2b"].update(overrides)

    jobs = [(s, tf) for tf in timeframes for s in pairs]
    all_trades = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, horizon, min_conf): (s, tf) for s, tf in jobs}
        for future in as_completed(futures):
            try:
                trades = future.result()
            except Exception:
                continue
            all_trades.extend(trades)
            if len(all_trades) >= max_trades:
                for f in futures:
                    f.cancel()
                break

    decided = [t for t in all_trades if t["outcome"] in ("win", "loss")]
    groups = defaultdict(list)
    for t in decided:
        groups[t["direction"]].append(t)

    result = {"label": label, "overrides": overrides, "elapsed": time.time() - t0, "by_direction": {}}
    for direction, ts in groups.items():
        n = len(ts)
        wins = sum(1 for t in ts if t["outcome"] == "win")
        wr = wins / n if n else 0.0
        avg_rr = sum(t["risk_reward"] for t in ts if t["risk_reward"]) / n if n else 0.0
        expectancy = wr * (1 + avg_rr) - 1
        breakeven = 1 / (1 + avg_rr) if avg_rr else 1.0
        result["by_direction"][direction] = {
            "n": n, "win_rate": round(wr, 3), "avg_rr": round(avg_rr, 2),
            "expectancy": round(expectancy, 3), "breakeven": round(breakeven, 3),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=500,
                    help="per config, per sweep run - kept lower than the main backtest since "
                         "6 configs run sequentially")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    base_cfg = load_config()
    min_conf = base_cfg["output"]["min_confluence"]
    static_pairs = list(base_cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(base_cfg.get("top_n_pairs", 100))
    pairs = list(dict.fromkeys(static_pairs + top_pairs))
    timeframes = base_cfg["timeframes"]

    print(f"Sweeping {len(SWEEP_CONFIGS)} configs across up to {len(pairs)} pairs x {len(timeframes)} tfs, "
          f"target {args.max_trades} trades/config...\n", flush=True)

    results = []
    for label, overrides in SWEEP_CONFIGS:
        print(f"--- {label} ({overrides}) ---", flush=True)
        r = run_sweep_config(label, overrides, base_cfg, pairs, timeframes,
                             args.max_trades, args.horizon, args.workers, min_conf)
        results.append(r)
        for direction, stats in r["by_direction"].items():
            flag = "  <-- losing" if stats["win_rate"] < stats["breakeven"] and stats["n"] >= 20 else ""
            print(f"  {direction:<10} n={stats['n']:<5} win_rate={stats['win_rate']:.1%}  "
                  f"avg_rr={stats['avg_rr']:.2f}  expectancy={stats['expectancy']:+.3f}R  "
                  f"breakeven={stats['breakeven']:.1%}{flag}")
        print(f"  ({r['elapsed']:.0f}s)\n", flush=True)

    print("=" * 78)
    print("SUMMARY (sorted by combined bullish+bearish expectancy, best first)")
    print("=" * 78)

    def combined_expectancy(r):
        vals = [s["expectancy"] * s["n"] for s in r["by_direction"].values()]
        weights = [s["n"] for s in r["by_direction"].values()]
        return sum(vals) / sum(weights) if sum(weights) else -999

    for r in sorted(results, key=combined_expectancy, reverse=True):
        ce = combined_expectancy(r)
        parts = ", ".join(f"{d}={s['expectancy']:+.3f}R (n={s['n']})" for d, s in r["by_direction"].items())
        print(f"{r['label']:<24} combined={ce:+.3f}R   {parts}")


if __name__ == "__main__":
    main()

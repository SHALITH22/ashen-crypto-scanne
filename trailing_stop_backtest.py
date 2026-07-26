"""
Backtests a trailing stop/target rule against the REAL historical trade
population (candle-by-candle replay, not just before/after outcome
comparison) before it ever touches live position management.

Rule (design confirmed with user 2026-07-25):
  - Trigger: once price reaches 50% of the way from entry to the
    ORIGINAL target (a fixed threshold, not swept - user's explicit
    choice).
  - Action (one-time, not continuous): move the stop to breakeven + a
    small buffer, AND extend the target further out by
    extension_fraction * the original reward distance. Both parameters
    ARE swept here, since the user asked for the buffer/extension size
    specifically to be evidence-based, not guessed.
  - Applies uniformly to all 5 strategies (user's explicit choice) - a
    percentage-based rule naturally scales to each strategy's own
    target distance regardless of how tight or wide it is.

Conservative convention for same-candle ambiguity (no intra-candle tick
data available): within any single candle, the ACTIVE stop/target are
checked BEFORE checking whether this candle newly crosses the 50%
trigger - matches the existing check_open_entries loop's own
stop-before-target convention, doesn't let a single candle both trigger
the adjustment AND immediately benefit from the newly-loosened target
in the same step.

Usage: python trailing_stop_backtest.py [--max-trades N]
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from audit_trades import load_config, _fetch_enriched

BUFFER_PCTS = [0.05, 0.1, 0.2, 0.3]
EXTENSION_FRACTIONS = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0]
TRIGGER_PROGRESS = 0.5


def simulate_trailing(after, entry: float, stop: float, target: float, bias: str,
                      buffer_pct: float, extension_fraction: float,
                      trigger_progress: float = TRIGGER_PROGRESS):
    """
    Walk-forward candle replay. Returns (outcome, exit_price, r_multiple,
    was_adjusted) - r_multiple computed against the ORIGINAL risk
    distance so results are comparable across trades regardless of
    whether they got adjusted.
    """
    original_risk = abs(entry - stop)
    original_reward = abs(target - entry)
    if original_risk <= 0 or original_reward <= 0:
        return None

    active_stop, active_target = stop, target
    adjusted = False

    for row in after.itertuples():
        if bias == "bullish":
            if row.low <= active_stop:
                r = (active_stop - entry) / original_risk
                return "loss" if r < 0 else "win", active_stop, r, adjusted
            if row.high >= active_target:
                r = (active_target - entry) / original_risk
                return "win", active_target, r, adjusted
            if not adjusted:
                progress = (row.high - entry) / original_reward
                if progress >= trigger_progress:
                    active_stop = entry * (1 + buffer_pct / 100)
                    active_target = target + original_reward * extension_fraction
                    adjusted = True
        else:
            if row.high >= active_stop:
                r = (entry - active_stop) / original_risk
                return "loss" if r < 0 else "win", active_stop, r, adjusted
            if row.low <= active_target:
                r = (entry - active_target) / original_risk
                return "win", active_target, r, adjusted
            if not adjusted:
                progress = (entry - row.low) / original_reward
                if progress >= trigger_progress:
                    active_stop = entry * (1 - buffer_pct / 100)
                    active_target = target - original_reward * extension_fraction
                    adjusted = True

    if len(after) == 0:
        return None
    last_close = float(after.iloc[-1]["close"])
    r = ((last_close - entry) if bias == "bullish" else (entry - last_close)) / original_risk
    return "expired", last_close, r, adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    entries = [json.loads(l) for l in open("journal.jsonl", encoding="utf-8") if l.strip()]
    resolved = [e for e in entries if e["status"] in ("win", "loss", "expired") and e.get("checked_at")]
    if args.max_trades:
        resolved = resolved[-args.max_trades:]
    print(f"{len(resolved)} resolved trades to replay")

    needed = defaultdict(set)
    for e in resolved:
        needed[e["symbol"]].add(e["timeframe"])
    fetch_jobs = [(sym, tf) for sym, tfs in needed.items() for tf in tfs]
    print(f"fetching {len(fetch_jobs)} unique symbol/timeframe series...")

    def _job(job):
        sym, tf = job
        return job, _fetch_enriched(sym, tf, cfg)

    dfs = {}
    with ThreadPoolExecutor(max_workers=cfg.get("scan_concurrency", 8)) as ex:
        for i, (job, df) in enumerate(ex.map(_job, fetch_jobs)):
            dfs[job] = df
            if (i + 1) % 25 == 0:
                print(f"  fetched {i+1}/{len(fetch_jobs)}")

    # Baseline: actual historical R-multiple (what really happened, no trailing)
    baseline_r = []
    for e in resolved:
        risk = abs(e["entry"] - e["stop"])
        if risk <= 0 or e.get("outcome_price") is None:
            continue
        move = (e["outcome_price"] - e["entry"]) if e["bias"] == "bullish" else (e["entry"] - e["outcome_price"])
        baseline_r.append(move / risk)
    baseline_avg = sum(baseline_r) / len(baseline_r) if baseline_r else 0
    baseline_wins = sum(1 for r in baseline_r if r > 0)
    print(f"\nBASELINE (actual history, no trailing): n={len(baseline_r)}, "
          f"avg_R={baseline_avg:+.4f}, win_rate={baseline_wins/len(baseline_r)*100:.1f}%")

    print(f"\n{'buffer%':>8}{'ext_frac':>10}{'n':>7}{'avg_R':>10}{'win_rate':>10}{'adjusted%':>11}")
    best = None
    for buffer_pct in BUFFER_PCTS:
        for ext_frac in EXTENSION_FRACTIONS:
            r_values = []
            adjusted_count = 0
            for e in resolved:
                df = dfs.get((e["symbol"], e["timeframe"]))
                if df is None:
                    continue
                logged_at = datetime.fromisoformat(e["logged_at"])
                after = df[df["open_time"] > logged_at].reset_index(drop=True)
                if after.empty:
                    continue
                result = simulate_trailing(after, e["entry"], e["stop"], e["target"], e["bias"],
                                           buffer_pct, ext_frac)
                if result is None:
                    continue
                outcome, exit_price, r, adjusted = result
                r_values.append(r)
                if adjusted:
                    adjusted_count += 1
            if not r_values:
                continue
            avg_r = sum(r_values) / len(r_values)
            wins = sum(1 for r in r_values if r > 0)
            wr = wins / len(r_values) * 100
            adj_pct = adjusted_count / len(r_values) * 100
            marker = ""
            if best is None or avg_r > best[0]:
                best = (avg_r, buffer_pct, ext_frac, len(r_values), wr)
                marker = "  <-- best so far"
            print(f"{buffer_pct:>7.2f}%{ext_frac:>10.2f}{len(r_values):>7}{avg_r:>+9.4f}{wr:>9.1f}%{adj_pct:>10.1f}%{marker}")

    if best:
        avg_r, buffer_pct, ext_frac, n, wr = best
        print(f"\nBest: buffer={buffer_pct}%, extension_fraction={ext_frac} -> "
              f"avg_R={avg_r:+.4f} (n={n}, win_rate={wr:.1f}%) "
              f"vs baseline avg_R={baseline_avg:+.4f}")


if __name__ == "__main__":
    main()

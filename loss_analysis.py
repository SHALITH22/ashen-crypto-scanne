"""
Deep-dive into every LOSING live trade: not just "which strategy/symbol
lost" (the dashboard already shows that) but WHY - three concrete
questions per loss:

  1. Would the newly-enabled BTC/ETH market-disagreement filter
     (risk.py's MARKET_FILTER_NAMES) have BLOCKED this specific loss, had
     it been active at the time? (only meaningful for vwap_breakout_ashen
     and ma_cross_ashen, the two strategies the filter now covers)
  2. Maximum favorable excursion (MFE): did price move toward target at
     all before reversing and hitting the stop ("near miss"), or did it
     go straight to the stop with no favorable move at all ("wrong from
     the start")? Re-fetches the real candles between logged_at and
     checked_at to answer this - the journal itself only stores the
     final outcome, not the path price took to get there.
  3. Basic breakdown by strategy/direction/symbol/timeframe, for context.

Usage: python loss_analysis.py
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from audit_trades import load_config, _fetch_enriched
from scanner.journal import entries_since
from ashen_realistic_backtest import trend_lookup

RESULTS_PATH = Path(__file__).parent / "loss_analysis_results.json"
FILTERED_STRATEGIES = {"vwap_breakout_ashen", "ma_cross_ashen"}  # the two MARKET_FILTER_NAMES now covers


def nearest_trend(trend: dict, at: datetime):
    """Trend lookup's keys are candle open_times - find the latest one at or before `at`."""
    candidates = [t for t in trend if t <= at]
    if not candidates:
        return None
    return trend[max(candidates)]


def compute_mfe(df, entry_time: datetime, exit_time: datetime, bias: str, entry: float, stop: float, target: float):
    """
    Max favorable excursion as a fraction of the entry->target distance:
    1.0 = price actually reached target at some point (shouldn't happen
    for a recorded loss, but a useful sanity bound), 0.0 = price never
    moved favorably from entry at all, negative = price never even
    matched entry (immediate adverse move).
    """
    window = df[(df["open_time"] >= entry_time) & (df["open_time"] <= exit_time)]
    if window.empty:
        return None
    target_dist = abs(target - entry)
    if target_dist <= 0:
        return None
    if bias == "bullish":
        best = window["high"].max()
        return round((best - entry) / target_dist, 3)
    else:
        best = window["low"].min()
        return round((entry - best) / target_dist, 3)


def main():
    cfg = load_config()
    entries = entries_since(None)
    losses = [e for e in entries if e["status"] == "loss"]
    all_decided = [e for e in entries if e["status"] in ("win", "loss")]
    print(f"{len(losses)} losses out of {len(all_decided)} decided trades ({len(losses)/len(all_decided):.1%})\n")

    # --- 1. Basic breakdown ---
    def breakdown(rows, key_fn, label):
        counts = defaultdict(int)
        for r in rows:
            counts[key_fn(r)] += 1
        print(f"=== Losses by {label} ===")
        for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<30}{n}")
        print()

    breakdown(losses, lambda e: f"{e['based_on']}/{e['bias']}", "strategy/direction")
    breakdown(losses, lambda e: e["symbol"], "symbol (top 15)")
    breakdown(losses, lambda e: e["timeframe"], "timeframe")

    # --- 2. Would the new BTC/ETH filter have blocked this loss? ---
    timeframes_needed = {e["timeframe"] for e in losses if e["based_on"] in FILTERED_STRATEGIES}
    print(f"=== 2. Would the new BTC/ETH market-disagreement filter have blocked these losses? ===")
    if timeframes_needed:
        btc_trends, eth_trends = {}, {}
        for tf in timeframes_needed:
            btc_df = _fetch_enriched("BTCUSDT", tf, cfg)
            eth_df = _fetch_enriched("ETHUSDT", tf, cfg)
            btc_trends[tf] = trend_lookup(btc_df) if btc_df is not None else {}
            eth_trends[tf] = trend_lookup(eth_df) if eth_df is not None else {}

        filtered_losses = [e for e in losses if e["based_on"] in FILTERED_STRATEGIES]
        would_have_blocked = 0
        would_have_allowed = 0
        unknown = 0
        for e in filtered_losses:
            logged_dt = datetime.fromisoformat(e["logged_at"])
            btc_bull = nearest_trend(btc_trends.get(e["timeframe"], {}), logged_dt)
            eth_bull = nearest_trend(eth_trends.get(e["timeframe"], {}), logged_dt)
            if btc_bull is None or eth_bull is None:
                unknown += 1
                continue
            trade_is_bullish = e["bias"] == "bullish"
            # market_disagrees requires BOTH BTC and ETH to disagree with the trade direction
            market_disagrees = (btc_bull != trade_is_bullish) and (eth_bull != trade_is_bullish)
            if market_disagrees:
                would_have_allowed += 1  # filter requires disagreement to trade - this WOULD have been allowed
            else:
                would_have_blocked += 1  # market agreed with the trade - filter would have refused it
        print(f"  {len(filtered_losses)} losses from vwap_breakout_ashen/ma_cross_ashen (the filtered strategies)")
        print(f"  Would have been BLOCKED by the new filter (market agreed with trade): {would_have_blocked}")
        print(f"  Would still have been ALLOWED (market disagreed, filter wouldn't stop it): {would_have_allowed}")
        if unknown:
            print(f"  Could not determine ({unknown} - insufficient BTC/ETH history at that point)")
    else:
        print("  No losses yet from the two filtered strategies.")
    print()

    # --- 3. Maximum favorable excursion ---
    print("=== 3. Maximum favorable excursion (did the loss ever move toward target first?) ===")
    needed = defaultdict(set)
    for e in losses:
        needed[e["symbol"]].add(e["timeframe"])
    fetch_jobs = [(sym, tf) for sym, tfs in needed.items() for tf in tfs]

    def _job(job):
        sym, tf = job
        return job, _fetch_enriched(sym, tf, cfg)

    dfs = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for job, df in ex.map(_job, fetch_jobs):
            dfs[job] = df

    mfe_results = []
    for e in losses:
        df = dfs.get((e["symbol"], e["timeframe"]))
        if df is None:
            continue
        entry_dt = datetime.fromisoformat(e["logged_at"])
        exit_dt = datetime.fromisoformat(e["checked_at"]) if e["checked_at"] else entry_dt
        mfe = compute_mfe(df, entry_dt, exit_dt, e["bias"], e["entry"], e["stop"], e["target"])
        if mfe is not None:
            mfe_results.append({**{k: e[k] for k in
                                   ("symbol", "timeframe", "based_on", "bias", "logged_at")}, "mfe": mfe})

    if mfe_results:
        near_miss = [r for r in mfe_results if r["mfe"] >= 0.5]     # got at least halfway to target
        modest_move = [r for r in mfe_results if 0.1 <= r["mfe"] < 0.5]
        no_favorable_move = [r for r in mfe_results if r["mfe"] < 0.1]  # basically went straight to the stop
        n = len(mfe_results)
        print(f"  {n} losses analyzed")
        print(f"  Near-miss (reached >=50% of the way to target before reversing): {len(near_miss)} ({len(near_miss)/n:.1%})")
        print(f"  Modest favorable move (10-50% of the way): {len(modest_move)} ({len(modest_move)/n:.1%})")
        print(f"  No favorable move at all (<10% - wrong from the start): {len(no_favorable_move)} ({len(no_favorable_move)/n:.1%})")
        avg_mfe = sum(r["mfe"] for r in mfe_results) / n
        print(f"  Average MFE across all losses: {avg_mfe:.1%} of the way to target")

        by_strategy = defaultdict(list)
        for r in mfe_results:
            by_strategy[r["based_on"]].append(r["mfe"])
        print("\n  Average MFE by strategy:")
        for s, mfes in sorted(by_strategy.items(), key=lambda kv: -len(kv[1])):
            print(f"    {s:<25}avg_mfe={sum(mfes)/len(mfes):.1%}  (n={len(mfes)})")

    Path(RESULTS_PATH).write_text(json.dumps(mfe_results, indent=2))
    print(f"\nFull MFE detail written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

"""
Tests whether Binance's account long/short ratio and open interest history
add real edge, using the REAL live journal (journal.jsonl) rather than a
synthetic walk-forward like funding_rate_backtest.py / fear_greed_backtest.py
use.

Why a different methodology than those two: Binance only retains ~30 days
of this history (a hard exchange-side limit, not something this script
controls) - nowhere near enough for a proper multi-year walk-forward. But
the live journal itself only spans ~8 days so far (see journal.jsonl), which
comfortably fits inside that 30-day window. So instead of simulating years
of hypothetical trades, this replays what the system ACTUALLY traded and
checks what long/short ratio and open interest looked like at each real
entry - smaller sample, but it's evidence about the exact live population,
not a proxy for it.

Two signals tested, both self-relative (comparing each symbol's current
reading to ITS OWN recent rolling baseline, not a fixed cross-symbol
number - raw long/short ratio and OI scale wildly differently symbol to
symbol):
  1. Long/short account ratio skew - classify_long_short_skew in
     scanner/risk.py (mirrors classify_funding's with/against-crowd shape).
  2. Open interest momentum - rising/falling/flat OI over a short lookback
     at entry time, cross-tabbed against trade direction (rising OI +
     trend-following trade = "confirmed", falling OI = "unconfirmed" -
     the standard reading, but tested here rather than assumed).

Deliberately NOT wired into any live filter by this script - only
building evidence. Whether to deploy either signal is a follow-up
decision once the numbers are in, same discipline as every other filter
in this codebase (funding, BTC/ETH agreement, VWAP structure clearance).

Usage: python oi_long_short_backtest.py
"""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scanner.data import get_long_short_ratio, get_open_interest_hist
from scanner.risk import classify_long_short_skew, LONG_SHORT_ROLLING_WINDOW as ROLLING_WINDOW

OI_LOOKBACK = 6        # ~1 day of 4h bars, for the OI momentum read
OI_MOVE_THRESHOLD = 5.0  # % change over OI_LOOKBACK to count as "rising"/"falling" rather than "flat"


def r_multiple(entry: dict) -> float | None:
    """Same formula as paper_trading.py's r_multiple - kept independent (not imported) so this script has no side effects on the live paper account replay."""
    risk = abs(entry["entry"] - entry["stop"])
    if risk <= 0 or entry.get("outcome_price") is None:
        return None
    move = ((entry["outcome_price"] - entry["entry"]) if entry["bias"] == "bullish"
            else (entry["entry"] - entry["outcome_price"]))
    return move / risk


def lookup_ratio_and_baseline(ls_df, ts) -> tuple[float | None, float | None]:
    if ls_df is None or ls_df.empty:
        return None, None
    idx = np.searchsorted(ls_df["timestamp"].to_numpy(), np.datetime64(ts), side="right") - 1
    if idx < ROLLING_WINDOW:  # not enough trailing history to form a baseline without lookahead
        return None, None
    ratio = float(ls_df["longShortRatio"].iloc[idx])
    baseline = float(ls_df["longShortRatio"].iloc[idx - ROLLING_WINDOW:idx].mean())
    return ratio, baseline


def lookup_oi_momentum(oi_df, ts) -> str | None:
    if oi_df is None or oi_df.empty:
        return None
    idx = np.searchsorted(oi_df["timestamp"].to_numpy(), np.datetime64(ts), side="right") - 1
    if idx < OI_LOOKBACK:
        return None
    now_oi = float(oi_df["sumOpenInterest"].iloc[idx])
    then_oi = float(oi_df["sumOpenInterest"].iloc[idx - OI_LOOKBACK])
    if then_oi <= 0:
        return None
    pct_change = (now_oi - then_oi) / then_oi * 100
    if pct_change > OI_MOVE_THRESHOLD:
        return "rising"
    if pct_change < -OI_MOVE_THRESHOLD:
        return "falling"
    return "flat"


def main():
    entries = [json.loads(l) for l in open("journal.jsonl", encoding="utf-8") if l.strip()]
    resolved = [e for e in entries if e["status"] in ("win", "loss", "expired") and e.get("outcome_price") is not None]
    print(f"{len(resolved)} resolved trades to test")

    symbols = sorted({e["symbol"] for e in resolved})
    print(f"fetching long/short ratio + open interest history for {len(symbols)} symbols...")

    def _fetch(sym):
        return sym, get_long_short_ratio(sym, period="4h", limit=200), get_open_interest_hist(sym, period="4h", limit=200)

    ls_data, oi_data = {}, {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (sym, ls_df, oi_df) in enumerate(ex.map(_fetch, symbols)):
            ls_data[sym] = ls_df
            oi_data[sym] = oi_df
            if (i + 1) % 25 == 0:
                print(f"  fetched {i+1}/{len(symbols)}")

    skew_buckets = defaultdict(list)
    raw_skew_buckets = defaultdict(list)
    oi_buckets = defaultdict(list)
    n_ls_available, n_oi_available = 0, 0

    for e in resolved:
        rm = r_multiple(e)
        if rm is None:
            continue

        ratio, baseline = lookup_ratio_and_baseline(ls_data.get(e["symbol"]), e["logged_at"])
        if ratio is not None:
            n_ls_available += 1
            skew = classify_long_short_skew(ratio, baseline, e["bias"])
            if skew:
                skew_buckets[skew].append(rm)
            deviation = (ratio - baseline) / baseline
            raw_bucket = ("skewed_long" if deviation > 0.2 else "skewed_short" if deviation < -0.2 else "balanced")
            raw_skew_buckets[raw_bucket].append(rm)

        momentum = lookup_oi_momentum(oi_data.get(e["symbol"]), e["logged_at"])
        if momentum:
            n_oi_available += 1
            oi_buckets[momentum].append(rm)

    def report(title, buckets):
        print(f"\n=== {title} ===")
        for label, rs in sorted(buckets.items()):
            n = len(rs)
            if n < 10:
                print(f"{label:<22}n={n:<5}(too few to trust)")
                continue
            avg_r = sum(rs) / n
            wins = sum(1 for r in rs if r > 0)
            print(f"{label:<22}n={n:<5}avg_R={avg_r:>+.4f}  win_rate={wins/n*100:>5.1f}%")

    print(f"\nlong/short ratio data available for {n_ls_available}/{len(resolved)} trades")
    print(f"open interest data available for {n_oi_available}/{len(resolved)} trades")

    report("Long/short skew vs trade direction (with/against crowd)", skew_buckets)
    report("Raw long/short skew (unconditional on direction)", raw_skew_buckets)
    report("Open interest momentum at entry (unconditional)", oi_buckets)

    out_path = "oi_long_short_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_resolved": len(resolved),
            "n_ls_available": n_ls_available,
            "n_oi_available": n_oi_available,
            "skew_buckets": {k: {"n": len(v), "avg_r": sum(v)/len(v) if v else None} for k, v in skew_buckets.items()},
            "oi_buckets": {k: {"n": len(v), "avg_r": sum(v)/len(v) if v else None} for k, v in oi_buckets.items()},
        }, f, indent=2)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()

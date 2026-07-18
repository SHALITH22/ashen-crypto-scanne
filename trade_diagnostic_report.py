"""
Full per-trade diagnostic for every RESOLVED (win or loss) journal entry -
answers, per trade, the four things a strict rule-based system should be
able to answer for itself:

  1. Entry/order correct? Re-fetch the real historical candles as they
     stood at logged_at and re-run the actual detector function (same
     mechanism as audit_trades.py) - if it reproduces the same direction
     and stop, every gate the strategy's rules impose (body ratio, wick
     ratio, extension filter, MA regime, dominance multiplier, etc.) was
     genuinely satisfied, not just assumed. This also validates timing:
     the recompute only ever sees candles that had already CLOSED by
     logged_at (no lookahead), so a pass proves the signal existed at
     that exact moment, not before or after.
  2. BTC/ETH checked? Structural fact from risk.MARKET_FILTER_NAMES, not
     computed per trade: true only for ma_cross_ashen/vwap_breakout_ashen.
     For those two, also report what BTC/ETH's own trend actually was at
     that moment (trend_lookup, same method as ashen_realistic_backtest.py
     and loss_analysis.py already use).
  3. How did price actually move? MFE (fraction of the entry->target
     distance reached before exit - loss_analysis.py's existing metric)
     AND its mirror MAE (fraction of the entry->stop distance reached
     before exit) - together these describe the whole path for BOTH wins
     ("how close a call was this win?") and losses ("did it get partway
     there first, or go straight to the stop?"), not just losses.
  4. Outcome + R-multiple, for the summary table.

Usage: python trade_diagnostic_report.py [--out trade_diagnostic.json]
Standalone report generator - not on any schedule, run on demand.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from audit_trades import load_config, _fetch_enriched, audit_entry
from scanner.journal import entries_since
from scanner.risk import MARKET_FILTER_NAMES
from ashen_realistic_backtest import trend_lookup
from loss_analysis import compute_mfe, nearest_trend

OUT_DEFAULT = Path(__file__).parent / "trade_diagnostic.json"


def compute_mae(df, entry_time, exit_time, bias, entry, stop, target):
    """Mirror of loss_analysis.compute_mfe - fraction of entry->stop distance reached (0=never moved against, 1.0=hit stop exactly)."""
    window = df[(df["open_time"] >= entry_time) & (df["open_time"] <= exit_time)]
    if window.empty:
        return None
    stop_dist = abs(stop - entry)
    if stop_dist <= 0:
        return None
    if bias == "bullish":
        worst = window["low"].min()
        return round((entry - worst) / stop_dist, 3)
    else:
        worst = window["high"].max()
        return round((worst - entry) / stop_dist, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--since", default=None, help="only entries logged at/after this ISO timestamp")
    args = ap.parse_args()

    cfg = load_config()
    entries = entries_since(None)
    resolved = [e for e in entries if e["status"] in ("win", "loss")]
    if args.since:
        resolved = [e for e in resolved if e["logged_at"] >= args.since]
    print(f"{len(resolved)} resolved trades to diagnose "
          f"({sum(1 for e in resolved if e['status']=='win')} wins, "
          f"{sum(1 for e in resolved if e['status']=='loss')} losses)")

    htf_pairing = cfg.get("ashen", {}).get("vwap_breakout", {}).get("htf_pairing", {})

    needed: dict[str, set[str]] = defaultdict(set)
    for e in resolved:
        needed[e["symbol"]].add(e["timeframe"])
        if e["based_on"] == "vwap_breakout_ashen":
            htf_tf = htf_pairing.get(e["timeframe"])
            if htf_tf:
                needed[e["symbol"]].add(htf_tf)
    needed["BTCUSDT"].update(e["timeframe"] for e in resolved if e["based_on"] in MARKET_FILTER_NAMES)
    needed["ETHUSDT"].update(e["timeframe"] for e in resolved if e["based_on"] in MARKET_FILTER_NAMES)

    fetch_jobs = [(sym, tf) for sym, tfs in needed.items() for tf in tfs]
    print(f"Fetching {len(fetch_jobs)} unique symbol/timeframe candle series...")

    def _job(job):
        sym, tf = job
        return job, _fetch_enriched(sym, tf, cfg)

    dfs = {}
    with ThreadPoolExecutor(max_workers=cfg.get("scan_concurrency", 8)) as ex:
        for i, (job, df) in enumerate(ex.map(_job, fetch_jobs)):
            dfs[job] = df
            if (i + 1) % 25 == 0:
                print(f"  fetched {i+1}/{len(fetch_jobs)}")

    btc_trend = {tf: trend_lookup(dfs[("BTCUSDT", tf)]) for tf in needed.get("BTCUSDT", set()) if dfs.get(("BTCUSDT", tf)) is not None}
    eth_trend = {tf: trend_lookup(dfs[("ETHUSDT", tf)]) for tf in needed.get("ETHUSDT", set()) if dfs.get(("ETHUSDT", tf)) is not None}

    rows = []
    for e in resolved:
        df = dfs.get((e["symbol"], e["timeframe"]))
        htf_df = None
        if e["based_on"] == "vwap_breakout_ashen":
            htf_tf = htf_pairing.get(e["timeframe"])
            htf_df = dfs.get((e["symbol"], htf_tf)) if htf_tf else None

        audit = audit_entry(e, df, htf_df, cfg)

        market_gated = e["based_on"] in MARKET_FILTER_NAMES
        btc_agree = eth_agree = None
        if market_gated:
            logged_dt = datetime.fromisoformat(e["logged_at"])
            btc_bull = nearest_trend(btc_trend.get(e["timeframe"], {}), logged_dt)
            eth_bull = nearest_trend(eth_trend.get(e["timeframe"], {}), logged_dt)
            trade_bullish = e["bias"] == "bullish"
            btc_agree = None if btc_bull is None else (btc_bull == trade_bullish)
            eth_agree = None if eth_bull is None else (eth_bull == trade_bullish)

        mfe = mae = None
        if df is not None:
            entry_dt = datetime.fromisoformat(e["logged_at"])
            exit_dt = datetime.fromisoformat(e["checked_at"]) if e["checked_at"] else entry_dt
            mfe = compute_mfe(df, entry_dt, exit_dt, e["bias"], e["entry"], e["stop"], e["target"])
            mae = compute_mae(df, entry_dt, exit_dt, e["bias"], e["entry"], e["stop"], e["target"])

        risk = abs(e["entry"] - e["stop"])
        r_multiple = None
        if risk > 0 and e.get("outcome_price") is not None:
            move = (e["outcome_price"] - e["entry"]) if e["bias"] == "bullish" else (e["entry"] - e["outcome_price"])
            r_multiple = round(move / risk, 3)

        rows.append({
            "id": e["id"], "symbol": e["symbol"], "timeframe": e["timeframe"],
            "based_on": e["based_on"], "bias": e["bias"], "status": e["status"],
            "logged_at": e["logged_at"], "checked_at": e.get("checked_at"),
            "r_multiple": r_multiple,
            "rule_check": audit["status"], "rule_reason": audit["reason"],
            "btc_eth_gated": market_gated, "btc_agrees": btc_agree, "eth_agrees": eth_agree,
            "mfe": mfe, "mae": mae,
        })

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # quick summary
    print("\n=== Rule-check summary (entry/order/timing correctness) ===")
    by_strat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_strat[r["based_on"]][0 if r["rule_check"] == "pass" else 1] += 1
    for s, (p, f) in sorted(by_strat.items()):
        print(f"  {s:<22} pass={p:4d} fail={f:4d}")

    print("\n=== BTC/ETH agreement (only meaningful for gated strategies) ===")
    gated_rows = [r for r in rows if r["btc_eth_gated"]]
    both_agree = sum(1 for r in gated_rows if r["btc_agrees"] and r["eth_agrees"])
    print(f"  {len(gated_rows)} trades from gated strategies; both BTC+ETH agreed at entry: {both_agree}")


if __name__ == "__main__":
    main()

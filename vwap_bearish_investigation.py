"""
Scoping investigation for vwap_breakout_ashen (both directions, bearish
emphasis since it's the higher-volume side): loss_analysis.py already
established losses here go wrong IMMEDIATELY (avg MFE -12.8%), a different
signature than marubozu's near-miss pattern - meaning a smaller target
(marubozu's fix) won't help here. This script's job is narrower: for every
resolved trade, recompute the SAME entry-time features the detector itself
used (scanner/ashen_vwap_breakout.py) and compare wins vs losses on each -
not to fix anything yet, just to find which factor(s) most separate a real
breakout from a fakeout, so any fix that follows targets the right lever
instead of guessing.

Features recomputed per trade (all derived from the real historical
candles at logged_at, same no-lookahead truncation as audit_trades.py):
  - vwap_distance_pct: how far the ENTRY candle closed beyond VWAP (bigger
    = more decisive breakout, smaller = marginal/borderline)
  - confirm_distance_pct: same distance but for the PRIOR (confirmation)
    candle - a weak confirmation might mean the "breakout" was already
    fading by the time this codebase's entry candle fired
  - atr_pct: ATR as a % of price at entry - volatility regime
  - structure_distance_pct: how far price sits from the HTF swing high/low
    structure level Ashen's rule requires a "retest" near - a large
    distance suggests this wasn't really a retest at all
  - funding_class: with_crowd / against_crowd / neutral (risk.classify_funding)
  - btc_agrees / eth_agrees: reused from trade_diagnostic.json if present

Usage: python vwap_bearish_investigation.py [--direction bearish|bullish|both]
Standalone report - not on any schedule.
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

from audit_trades import load_config, _fetch_enriched
from scanner.journal import entries_since
from scanner.ashen_vwap_breakout import _rolling_vwap, _atr, _htf_trend_and_structure, _DEFAULT_CFG
from scanner.risk import classify_funding

OUT_PATH = Path(__file__).parent / "vwap_investigation.json"


def _truncate_to_time(df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    return df[df["close_time"] <= cutoff].reset_index(drop=True)


def recompute_features(entry: dict, df: pd.DataFrame | None, htf_df: pd.DataFrame | None, cfg: dict) -> dict | None:
    if df is None or htf_df is None:
        return None
    vcfg = cfg.get("ashen", {}).get("vwap_breakout", {})
    swing_lookback = vcfg.get("htf_swing_lookback", _DEFAULT_CFG["htf_swing_lookback"])
    vwap_window = vcfg.get("vwap_window", _DEFAULT_CFG["vwap_window"])
    atr_period = vcfg.get("atr_period", _DEFAULT_CFG["atr_period"])

    logged_at = datetime.fromisoformat(entry["logged_at"])
    df_hist = _truncate_to_time(df, logged_at)
    htf_hist = _truncate_to_time(htf_df, logged_at)
    if len(df_hist) < vwap_window + 5 or len(htf_hist) < swing_lookback + 5:
        return None

    vwap = _rolling_vwap(df_hist, vwap_window)
    atr = _atr(df_hist, atr_period)
    if pd.isna(vwap.iloc[-1]) or pd.isna(vwap.iloc[-2]) or pd.isna(atr.iloc[-1]):
        return None

    structure = _htf_trend_and_structure(htf_hist, swing_lookback)
    entry_candle = df_hist.iloc[-1]
    prior_candle = df_hist.iloc[-2]
    close = float(entry_candle["close"])

    vwap_distance_pct = (close - vwap.iloc[-1]) / vwap.iloc[-1] * 100
    confirm_distance_pct = (prior_candle["close"] - vwap.iloc[-2]) / vwap.iloc[-2] * 100
    atr_pct = float(atr.iloc[-1]) / close * 100
    structure_level = structure["swing_low"] if entry["bias"] == "bearish" else structure["swing_high"]
    structure_distance_pct = abs(close - structure_level) / close * 100

    funding_class = None  # funding rate isn't stored historically per-entry; left None, noted in report

    return {
        "vwap_distance_pct": round(vwap_distance_pct, 4),
        "confirm_distance_pct": round(confirm_distance_pct, 4),
        "atr_pct": round(atr_pct, 4),
        "structure_distance_pct": round(structure_distance_pct, 4),
        "htf_trend_matches_bias": structure["bullish"] == (entry["bias"] == "bullish"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="both", choices=["bearish", "bullish", "both"])
    args = ap.parse_args()

    cfg = load_config()
    entries = entries_since(None)
    trades = [e for e in entries if e["based_on"] == "vwap_breakout_ashen" and e["status"] in ("win", "loss")]
    if args.direction != "both":
        trades = [e for e in trades if e["bias"] == args.direction]
    print(f"{len(trades)} resolved vwap_breakout_ashen trades to investigate")

    htf_pairing = cfg.get("ashen", {}).get("vwap_breakout", {}).get("htf_pairing", {})
    needed = defaultdict(set)
    for e in trades:
        needed[e["symbol"]].add(e["timeframe"])
        htf_tf = htf_pairing.get(e["timeframe"])
        if htf_tf:
            needed[e["symbol"]].add(htf_tf)

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

    rows = []
    for e in trades:
        df = dfs.get((e["symbol"], e["timeframe"]))
        htf_tf = htf_pairing.get(e["timeframe"])
        htf_df = dfs.get((e["symbol"], htf_tf)) if htf_tf else None
        feats = recompute_features(e, df, htf_df, cfg)
        if feats is None:
            continue
        rows.append({
            "symbol": e["symbol"], "timeframe": e["timeframe"], "bias": e["bias"],
            "status": e["status"], "logged_at": e["logged_at"], **feats,
        })

    Path(OUT_PATH).write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {OUT_PATH}")

    # --- compare wins vs losses on each feature ---
    numeric_features = ["vwap_distance_pct", "confirm_distance_pct", "atr_pct", "structure_distance_pct"]
    for bias in (["bearish", "bullish"] if args.direction == "both" else [args.direction]):
        subset = [r for r in rows if r["bias"] == bias]
        wins = [r for r in subset if r["status"] == "win"]
        losses = [r for r in subset if r["status"] == "loss"]
        print(f"\n=== vwap_breakout_ashen / {bias}: {len(wins)} wins, {len(losses)} losses ===")
        for feat in numeric_features:
            wv = [abs(r[feat]) for r in wins if r[feat] is not None]
            lv = [abs(r[feat]) for r in losses if r[feat] is not None]
            if not wv or not lv:
                continue
            w_avg = sum(wv) / len(wv)
            l_avg = sum(lv) / len(lv)
            flag = "  <-- notably different" if abs(w_avg - l_avg) / max(w_avg, l_avg, 1e-9) > 0.15 else ""
            print(f"  {feat:24s} win_avg={w_avg:7.4f}  loss_avg={l_avg:7.4f}{flag}")

        htf_mismatch_wins = sum(1 for r in wins if not r["htf_trend_matches_bias"])
        htf_mismatch_losses = sum(1 for r in losses if not r["htf_trend_matches_bias"])
        print(f"  htf_trend_mismatch (should always be False by construction): "
              f"wins={htf_mismatch_wins}/{len(wins)}  losses={htf_mismatch_losses}/{len(losses)}")


if __name__ == "__main__":
    main()

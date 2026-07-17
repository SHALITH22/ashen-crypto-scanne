"""
Realistic backtest for the four Ashen strategies (b2b_ashen, ma_cross_ashen,
marubozu_ashen, vwap_breakout_ashen) - mirrors jayantha_realistic_backtest.py's
exact methodology (replay historical candles through the same detector ->
risk-plan pipeline main.py uses live, walk forward candle-by-candle to see
whether the calculated stop or target actually got hit first) but against
run_ashen_detectors instead of run_jayantha_detectors, plus two structural
differences these strategies need:

  1. vwap_breakout_ashen needs a second (HTF) timeframe series, unlike
     jayantha (single-timeframe only). At each walk-forward step, the HTF
     view is truncated to close_time <= the entry timeframe's own current
     candle's close_time, so it never leaks future HTF candles - the same
     no-lookahead principle audit_trades.py already established for a
     different purpose (reconstructing a historical view for auditing).
  2. Uses setup_risk_plans (plural, current live logic) instead of
     setup_risk_plan (singular) - jayantha_realistic_backtest.py predates
     this session's multi-strategy independence refactor and still uses
     the old singular/bias-based selection, which main.py no longer
     actually runs live. Since these four strategies fire independently
     in production, the backtest needs the same independent walk-forward:
     each step can produce multiple plans, each gets its own
     blocked_until[based_on] tracking.

This exists because these four strategies went live with ZERO forward-
tested history - only jayantha_b2b has a real historical backtest. Live
data already shows a strong bearish >> bullish pattern for these
strategies (confirmed via the live journal), but ~50-190 live trades from
the last day can't distinguish "structurally weaker bullish setups" from
"the market has been down for a few days" - this script gets a much
larger, longer-history answer at scale instead.

Also tags every simulated trade with whether BTC's/ETH's own trend agreed
with the trade's direction (mirrors confluence_btc_backtest.py's exact
methodology) - a measurement pass only, not a live filter change. The
live market_disagrees/funding_ok filter is bypassed during generation
(True for both directions) specifically so this can measure whether that
filter WOULD help, rather than assuming it and filtering data selectively.

Results are MERGED into realistic_backtest_results.json - ashen_* entries
are added/replaced, jayantha_*/retired-detector entries are left
untouched. This is the SAME file main.py already reads
(load_realistic_backtest_expectancy/load_realistic_backtest_win_rate), so
no other code needs to change for these numbers to take effect on the
next scheduled scan.

Usage:
  python ashen_realistic_backtest.py
  python ashen_realistic_backtest.py --max-trades 2000 --horizon 60
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from realistic_backtest import load_config, validate_trade, WARMUP
from scanner.data import get_klines, get_top_pairs_by_volume
from scanner.indicators import enrich
from scanner.ashen_detectors import run_ashen_detectors
from scanner.risk import attach_atr_risk, setup_risk_plans

RESULTS_PATH = Path(__file__).parent / "realistic_backtest_results.json"
BTC_AGREEMENT_PATH = Path(__file__).parent / "ashen_btc_agreement_results.json"


def trend_lookup(df: pd.DataFrame) -> dict:
    """open_time -> bool (close above EMA20 = bullish market leader trend). Copied from confluence_btc_backtest.py."""
    enriched = df.copy()
    enriched["ema20"] = enriched["close"].ewm(span=20, adjust=False).mean()
    return dict(zip(enriched["open_time"], enriched["close"] > enriched["ema20"]))


def _truncate_to_time(df: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Same no-lookahead truncation audit_trades.py uses - keep only candles whose close_time is <= cutoff."""
    return df[df["close_time"] <= cutoff].reset_index(drop=True)


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int,
                     btc_trend: dict, eth_trend: dict) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)

    htf_pairing = cfg.get("ashen", {}).get("vwap_breakout", {}).get("htf_pairing", {})
    htf_tf = htf_pairing.get(tf)
    htf_df = None
    if htf_tf:
        htf_raw = get_klines(symbol, htf_tf, 1000)
        if htf_raw is not None and len(htf_raw) >= 60:
            htf_df = enrich(htf_raw, cfg)

    risk_cfg = cfg.get("risk", {})
    # Same per-strategy override main.py's live scan_pair() builds - without
    # this, marubozu_ashen's deliberately smaller target (see
    # ashen_marubozu.py) would be tested against the flat 1.0 floor sized
    # for the other four strategies' bigger targets, silently generating
    # ZERO marubozu candidates here instead of the actually-live population.
    min_rr_overrides = risk_cfg.get("min_risk_reward_overrides")
    min_risk_reward = ({**min_rr_overrides, "default": risk_cfg.get("min_risk_reward", 1.0)}
                       if min_rr_overrides else risk_cfg.get("min_risk_reward", 1.0))
    trades = []
    blocked_until: dict[str, int] = {}

    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[:i + 1]
        htf_window = None
        if htf_df is not None:
            cutoff = window["close_time"].iloc[-1]
            candidate = _truncate_to_time(htf_df, cutoff)
            if len(candidate) >= 60:
                htf_window = candidate

        signals = run_ashen_detectors(window, cfg, htf_window)
        if not signals:
            continue
        close = float(window["close"].iloc[-1])
        atr = float(window["atr"].iloc[-1])
        signals = attach_atr_risk(signals, close, atr,
                                  risk_cfg.get("atr_multiplier", 1.5),
                                  risk_cfg.get("reward_risk_ratio", 2.0),
                                  risk_cfg.get("max_stop_pct"))
        # market_disagrees_by_direction/funding_ok_by_direction both bypass
        # the (unproven-for-ashen, and already-excluded-in-risk.py) BTC/ETH
        # beta filter - this measures RAW baseline performance, and
        # separately tags btc_agrees/eth_agrees below so that filter's
        # value can be tested afterward instead of assumed.
        plans = setup_risk_plans(signals, close, min_risk_reward,
                                 market_disagrees_by_direction={"bullish": True, "bearish": True},
                                 funding_ok_by_direction={"bullish": True, "bearish": True},
                                 target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not plans:
            continue

        open_time = window["open_time"].iloc[-1]
        btc_bull = btc_trend.get(open_time)
        eth_bull = eth_trend.get(open_time)

        for risk in plans:
            key = risk["based_on"]
            if blocked_until.get(key, -1) >= i:
                continue
            bias = risk["direction"]

            btc_agrees = btc_bull if bias == "bullish" else (not btc_bull if btc_bull is not None else None)
            eth_agrees = eth_bull if bias == "bullish" else (not eth_bull if eth_bull is not None else None)

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
                outcome = "expired"
                resolved_at = end - 1
                outcome_price = float(df["close"].iloc[resolved_at])

            trade = {
                "symbol": symbol, "timeframe": tf, "based_on": risk["based_on"],
                "direction": bias, "entry": risk["entry"], "stop": risk["stop"],
                "target": risk["target"], "risk_reward": risk["risk_reward"],
                "outcome": outcome,
                "outcome_pct": round((outcome_price - risk["entry"]) / risk["entry"] * 100, 3),
                "btc_agrees": btc_agrees, "eth_agrees": eth_agrees,
            }
            trade["validation_error"] = validate_trade(trade)
            trades.append(trade)
            blocked_until[key] = resolved_at

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=300)
    ap.add_argument("--max-trades", type=int, default=2500)
    ap.add_argument("--horizon", type=int, default=60,
                    help="matches config/settings.yaml's journal.horizon_candles (60), not "
                         "realistic_backtest.py's 20, so these numbers are directly comparable "
                         "to what the live journal will actually produce")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()

    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = list(dict.fromkeys(static_pairs + top_pairs))
    timeframes = cfg["timeframes"]

    print("Fetching BTC/ETH trend reference per timeframe...", flush=True)
    btc_trends, eth_trends = {}, {}
    for tf in timeframes:
        btc_df = get_klines("BTCUSDT", tf, 1000)
        eth_df = get_klines("ETHUSDT", tf, 1000)
        btc_trends[tf] = trend_lookup(btc_df) if btc_df is not None else {}
        eth_trends[tf] = trend_lookup(eth_df) if eth_df is not None else {}

    jobs = [(s, tf) for tf in timeframes for s in pairs]

    print(f"Simulating 4 Ashen strategies across up to {len(pairs)} pairs x {len(timeframes)} timeframes "
          f"({args.workers} worker processes, target: {args.min_trades}-{args.max_trades} trades, "
          f"horizon={args.horizon} candles)...", flush=True)

    all_trades = []
    errors = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon,
                                   btc_trends[tf], eth_trends[tf]): (s, tf) for s, tf in jobs}
        done = 0
        for future in as_completed(futures):
            s, tf = futures[future]
            done += 1
            try:
                trades = future.result()
            except Exception as e:
                errors.append(f"{s} {tf}: {e}")
                continue
            all_trades.extend(trades)
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {len(all_trades)} trades so far ({time.time()-t0:.0f}s elapsed)",
                      flush=True)
            if len(all_trades) >= args.max_trades:
                print(f"  reached {len(all_trades)} trades (>= --max-trades {args.max_trades}), "
                      f"stopping early - cancelling remaining jobs", flush=True)
                for f in futures:
                    f.cancel()
                break

    print(f"\nDone: {len(all_trades)} simulated trades from {done}/{len(jobs)} pair/timeframe series "
          f"in {time.time()-t0:.0f}s", flush=True)
    if errors:
        print(f"{len(errors)} pair/timeframe series failed to fetch/simulate:")
        for e in errors[:10]:
            print(f"  [error] {e}")

    bad = [t for t in all_trades if t["validation_error"]]
    print(f"\nOrder/stop/target integrity check: {len(all_trades) - len(bad)}/{len(all_trades)} trades correct.")
    if bad:
        print(f"{len(bad)} trades FAILED validation:")
        for t in bad[:10]:
            print(f"  [bad] {t['symbol']} {t['timeframe']} {t['based_on']}: {t['validation_error']}")

    decided = [t for t in all_trades if t["outcome"] in ("win", "loss")]
    print(f"\n{len(decided)} decided (win/loss), {len(all_trades)-len(decided)} expired (no clean resolution)")

    groups = defaultdict(list)
    for t in decided:
        groups[(t["based_on"], t["direction"])].append(t)

    print(f"\n{'detector/direction':<28}{'n':>6}{'win_rate':>10}{'avg_rr':>8}{'avg_pct':>9}")
    print("-" * 61)
    new_summary = {}
    for key, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(ts)
        wins = sum(1 for t in ts if t["outcome"] == "win")
        wr = wins / n
        avg_rr = sum(t["risk_reward"] for t in ts if t["risk_reward"]) / n
        avg_pct = sum(t["outcome_pct"] for t in ts) / n
        name = f"{key[0]}/{key[1]}"
        flag = "  <-- proven losing (< breakeven)" if wr < 1 / (1 + avg_rr) and n >= 20 else ""
        print(f"{name:<28}{n:>6}{wr:>10.1%}{avg_rr:>8.2f}{avg_pct:>9.2f}{flag}")
        new_summary[name] = {"n": n, "win_rate": round(wr, 3), "avg_risk_reward": round(avg_rr, 2),
                             "avg_outcome_pct": round(avg_pct, 3), "breakeven_win_rate": round(1 / (1 + avg_rr), 3)}

    # --- BTC/ETH market-agreement breakdown (measurement only - see module docstring) ---
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

    print("\n=== BTC/ETH market-agreement breakdown, per Ashen strategy/direction ===")
    print("(measurement only - does NOT change what's live; see risk.py's MARKET_FILTER_NAMES)")
    for key, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        name = f"{key[0]}/{key[1]}"
        print(f"\n-- {name} --")
        report("  BTC agrees with trade direction", [t for t in ts if t["btc_agrees"] is True])
        report("  BTC disagrees with trade direction", [t for t in ts if t["btc_agrees"] is False])
        report("  ETH agrees with trade direction", [t for t in ts if t["eth_agrees"] is True])
        report("  ETH disagrees with trade direction", [t for t in ts if t["eth_agrees"] is False])

    # Merge into the existing file rather than overwrite it: ashen_* keys
    # are unique (never collide with jayantha_*/retired-detector names),
    # so old reference data is preserved untouched while the numbers that
    # actually matter to the live strategies get added. This is the exact
    # file main.py's load_realistic_backtest_expectancy/
    # load_realistic_backtest_win_rate already read every scan.
    existing = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
    ashen_names = ("b2b_ashen", "ma_cross_ashen", "marubozu_ashen", "vwap_breakout_ashen")
    merged_summary = existing.get("summary", {})
    merged_summary.update(new_summary)
    merged_trades = [t for t in existing.get("trades", []) if not t["based_on"].startswith(ashen_names)]
    merged_trades.extend(all_trades)
    merged_decided = [t for t in merged_trades if t["outcome"] in ("win", "loss")]

    RESULTS_PATH.write_text(json.dumps({
        "n_trades": len(merged_trades),
        "n_decided": len(merged_decided),
        "n_validation_failures": sum(1 for t in merged_trades if t.get("validation_error")),
        "summary": merged_summary,
        "trades": merged_trades,
    }, indent=2))
    print(f"\nashen_* results merged into {RESULTS_PATH} "
          f"(jayantha_*/retired-detector entries untouched, {len(merged_trades)} total trades in file)")

    # Separate, measurement-only file (never read by main.py) so this data
    # can't accidentally feed the live pipeline before a deliberate
    # decision is made about MARKET_FILTER_NAMES.
    BTC_AGREEMENT_PATH.write_text(json.dumps({"n_trades": len(all_trades), "trades": all_trades}, indent=2))
    print(f"BTC/ETH agreement detail written to {BTC_AGREEMENT_PATH}")


if __name__ == "__main__":
    main()

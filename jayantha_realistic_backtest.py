"""
Realistic backtest for the Jayantha B2B strategy - mirrors
realistic_backtest.py's exact methodology (replay historical candles
through the same detector -> risk-plan pipeline main.py uses live, walk
forward candle-by-candle to see whether the calculated stop or target
actually got hit first) but against run_jayantha_detectors instead of the
retired run_all_detectors.

This exists because the Jayantha strategy went live with zero forward-
tested history - main.py's win_probability display and
min_detector_expectancy blacklist are only as honest as the data behind
them, and waiting weeks for the live journal to accumulate a few hundred
resolved trades organically is a much slower way to find out whether this
translation of Jayantha's rules actually has real edge.

Results are MERGED into realistic_backtest_results.json - jayantha_b2b/
jayantha_trend/jayantha_confirmation entries are added/replaced, any
existing entries for the retired detector set (ema_stack, stochrsi, etc.)
are left untouched. This is the SAME file main.py already reads
(load_realistic_backtest_expectancy / load_realistic_backtest_win_rate in
main.py), so no other code needs to change for these numbers to take
effect on the next scheduled scan.

Usage:
  python jayantha_realistic_backtest.py
  python jayantha_realistic_backtest.py --min-trades 300 --horizon 60
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from realistic_backtest import load_config, confluence_score, validate_trade, WARMUP
from scanner.data import get_klines, get_top_pairs_by_volume
from scanner.indicators import enrich
from scanner.jayantha_detectors import run_jayantha_detectors
from scanner.risk import attach_atr_risk, setup_risk_plan

RESULTS_PATH = Path(__file__).parent / "realistic_backtest_results.json"


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    trades = []
    blocked_until: dict[str, int] = {}

    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[:i + 1]
        signals = run_jayantha_detectors(window, cfg)
        if not signals:
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
        # market_disagrees=True bypasses the (unproven-for-jayantha, and
        # already-excluded-in-risk.py) BTC/ETH beta filter - same reasoning
        # as the original script: this measures RAW baseline performance,
        # not performance filtered by a rule that's never been validated
        # for this detector.
        risk = setup_risk_plan(signals, bias, close, risk_cfg.get("min_risk_reward", 1.0),
                               market_disagrees=True,
                               target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not risk:
            continue
        key = risk["based_on"]
        if blocked_until.get(key, -1) >= i:
            continue

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
        }
        trade["validation_error"] = validate_trade(trade)
        trades.append(trade)
        blocked_until[key] = resolved_at

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=300,
                    help="jayantha_b2b fires far more rarely than the old 16-detector ensemble, "
                         "so this defaults much lower than realistic_backtest.py's 1000")
    ap.add_argument("--max-trades", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=60,
                    help="matches config/settings.yaml's journal.horizon_candles (60), not "
                         "realistic_backtest.py's 20, so these numbers are directly comparable "
                         "to what the live journal will actually produce")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    min_conf = cfg["output"]["min_confluence"]

    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = list(dict.fromkeys(static_pairs + top_pairs))
    timeframes = cfg["timeframes"]

    jobs = [(s, tf) for tf in timeframes for s in pairs]

    print(f"Simulating jayantha_b2b across up to {len(pairs)} pairs x {len(timeframes)} timeframes "
          f"({args.workers} worker processes, target: {args.min_trades}-{args.max_trades} trades, "
          f"horizon={args.horizon} candles)...", flush=True)

    all_trades = []
    errors = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon, min_conf): (s, tf) for s, tf in jobs}
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

    # Merge into the existing file rather than overwrite it: jayantha_*
    # keys are unique (never collide with the retired ema_stack/stochrsi/
    # etc. names), so old reference data is preserved untouched while the
    # numbers that actually matter to the live strategy get added. This is
    # the exact file main.py's load_realistic_backtest_expectancy /
    # load_realistic_backtest_win_rate already read every scan.
    existing = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
    merged_summary = existing.get("summary", {})
    merged_summary.update(new_summary)
    merged_trades = [t for t in existing.get("trades", []) if not t["based_on"].startswith("jayantha_")]
    merged_trades.extend(all_trades)
    merged_decided = [t for t in merged_trades if t["outcome"] in ("win", "loss")]

    RESULTS_PATH.write_text(json.dumps({
        "n_trades": len(merged_trades),
        "n_decided": len(merged_decided),
        "n_validation_failures": sum(1 for t in merged_trades if t.get("validation_error")),
        "summary": merged_summary,
        "trades": merged_trades,
    }, indent=2))
    print(f"\njayantha_* results merged into {RESULTS_PATH} "
          f"(retired-detector entries untouched, {len(merged_trades)} total trades in file)")


if __name__ == "__main__":
    main()

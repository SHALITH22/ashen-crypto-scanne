"""
Realistic backtest: replays historical klines through the EXACT same
detector -> risk-plan pipeline main.py uses live, then walks forward
candle-by-candle to see whether the calculated stop or target actually got
hit first - the same way journal.check_open_entries resolves real trades.

This is a different (and much harder) question than backtest.py answers.
backtest.py checks "was price higher/lower N candles later" with no stop
involved at all - a detector can look profitable there while actually
losing money once a realistic stop-loss is respected (this is exactly what
happened live: ema_stack/bearish and stochrsi/bearish looked ~56% in
backtest.py but were 16% and 0% respectively against real stop/target
outcomes). This script exists to get that harder, more honest number at
scale (1000+ simulated trades) instead of waiting weeks for the live
journal to accumulate enough samples on its own.

No calibration and no reliability blacklist are applied during simulation
- this pass exists to MEASURE those inputs, not assume them, so every
generated trade uses the plain geometric/ATR or structural risk plan with a
fixed reward:risk, exactly like a brand-new detector with no track record
yet would get live.

Usage:
  python realistic_backtest.py
  python realistic_backtest.py --min-trades 1500
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
from scanner.risk import attach_atr_risk, setup_risk_plan

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
WARMUP = 210  # candles before signals count (EMA200 needs history)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def confluence_score(signals: list[dict]) -> tuple[str, float]:
    """Unweighted - this pass doesn't use backtest_results.json's weights, since that's one of the things being validated."""
    bull = sum(1.0 for s in signals if s["direction"] == "bullish")
    bear = sum(1.0 for s in signals if s["direction"] == "bearish")
    if bull > bear:
        return "bullish", round(bull, 2)
    if bear > bull:
        return "bearish", round(bear, 2)
    return "mixed", round(max(bull, bear), 2)


def validate_trade(t: dict) -> str | None:
    """
    Sanity-check the entry/stop/target relationship and R:R math on a
    simulated trade - answers the user's "is the order request/stop/target
    actually correct" question directly, on every single trade, not just a
    hand-picked few.
    """
    entry, stop, target = t["entry"], t["stop"], t["target"]
    if t["direction"] == "bullish":
        if not (stop < entry < target):
            return f"bullish trade with stop/entry/target out of order: {stop} / {entry} / {target}"
    else:
        if not (stop > entry > target):
            return f"bearish trade with stop/entry/target out of order: {stop} / {entry} / {target}"
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return "zero-distance stop"
    implied_rr = round(reward / risk, 2)
    if t["risk_reward"] is not None and abs(implied_rr - t["risk_reward"]) > 0.05:
        return f"stated R:R {t['risk_reward']} doesn't match entry/stop/target math ({implied_rr})"
    return None


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    trades = []
    blocked_until: dict[str, int] = {}  # based_on -> candle index it's occupied through

    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[:i + 1]
        signals = run_all_detectors(window, cfg)
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
        # market_disagrees=True bypasses the live-only BTC/ETH filter - this
        # script exists to measure a detector's RAW baseline performance
        # (that's what discovered the filter's edge in the first place), so
        # it must not have that filter silently applied via the default.
        # target_fraction IS passed through (unlike market_disagrees) - this
        # measures what's actually live, not a stale 100%-target scenario.
        # Missing this was a real bug: after target_fraction=0.85 went live,
        # this backtest kept measuring the harder, unscaled 100% target,
        # understating every detector's real current win rate.
        risk = setup_risk_plan(signals, bias, close, risk_cfg.get("min_risk_reward", 1.0),
                               market_disagrees=True,
                               target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not risk:
            continue
        key = risk["based_on"]
        if blocked_until.get(key, -1) >= i:
            continue  # a simulated trade on this detector is still "open" here

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
    ap.add_argument("--min-trades", type=int, default=1000)
    ap.add_argument("--max-trades", type=int, default=4000,
                    help="stop dispatching new jobs once at least this many trades are collected")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    min_conf = cfg["output"]["min_confluence"]

    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = list(dict.fromkeys(static_pairs + top_pairs))  # union, static list first, de-duped, order preserved
    timeframes = cfg["timeframes"]

    # Interleaved (timeframe-major within pair-major round-robin) so an early
    # stop still samples a spread of timeframes instead of exhausting one
    # pair's 9 timeframes before moving to the next pair.
    jobs = [(s, tf) for tf in timeframes for s in pairs]

    print(f"Simulating up to {len(pairs)} pairs x {len(timeframes)} timeframes "
          f"({args.workers} worker processes, target: {args.min_trades}-{args.max_trades} trades)...", flush=True)

    all_trades = []
    errors = []
    t0 = time.time()

    # ProcessPoolExecutor, not threads - this workload is CPU-bound (each
    # job re-runs the full detector suite on a growing historical window),
    # and threads don't parallelize CPU-bound Python work at all (the GIL
    # serializes it) - a first attempt with threads ran for over an hour on
    # a fraction of the grid because of exactly this.
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
            if done % 10 == 0 or done == len(jobs):
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
    summary = {}
    for key, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(ts)
        wins = sum(1 for t in ts if t["outcome"] == "win")
        wr = wins / n
        avg_rr = sum(t["risk_reward"] for t in ts if t["risk_reward"]) / n
        avg_pct = sum(t["outcome_pct"] for t in ts) / n
        name = f"{key[0]}/{key[1]}"
        flag = "  <-- proven losing (< breakeven)" if wr < 1 / (1 + avg_rr) and n >= 20 else ""
        print(f"{name:<28}{n:>6}{wr:>10.1%}{avg_rr:>8.2f}{avg_pct:>9.2f}{flag}")
        summary[name] = {"n": n, "win_rate": round(wr, 3), "avg_risk_reward": round(avg_rr, 2),
                         "avg_outcome_pct": round(avg_pct, 3), "breakeven_win_rate": round(1 / (1 + avg_rr), 3)}

    out = Path(__file__).parent / "realistic_backtest_results.json"
    out.write_text(json.dumps({
        "n_trades": len(all_trades), "n_decided": len(decided),
        "n_validation_failures": len(bad), "summary": summary,
        "trades": all_trades,
    }, indent=2))
    print(f"\nFull detail (all {len(all_trades)} trades + summary) written to {out}")


if __name__ == "__main__":
    main()

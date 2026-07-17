"""
Tests whether taking partial profit early - exiting marubozu_ashen trades
at some fraction of the full target instead of waiting for the whole
move - improves on the current 1:1.2 full-target rule.

Motivated directly by loss_analysis.py's finding: marubozu_ashen losses
have a POSITIVE average MFE (+8.4%) - unlike vwap_breakout_ashen (-12.8%,
wrong almost immediately), marubozu losses tend to get PARTWAY to target
before reversing. That shape suggests some of those losses could become
wins (or smaller losses) if profit were banked earlier, rather than
holding out for the full 1.2R target every time.

Methodology: identical walk-forward generation to
ashen_realistic_backtest.py (same detector, same attach_atr_risk/
setup_risk_plans qualification gates, same historical candles, same
WARMUP/horizon) - so every candidate trade here is the exact same
population that already exists live. The ONLY thing that changes is the
EXIT RULE: instead of one full-target resolution per trade, each trade is
independently re-resolved against several candidate "take partial profit
at fraction f of the full target" rules (f=1.0 reproduces the current
live behavior exactly, as a baseline sanity check). Stop is always
checked first within a candle (same tie-break convention every other
backtest script here uses), so a partial-profit rule can only convert an
existing LOSS into a smaller win/loss by exiting earlier - it can never
turn a genuine winner into a loser, only cap its upside.

Usage:
  python marubozu_partial_profit_backtest.py
  python marubozu_partial_profit_backtest.py --fractions 0.4,0.5,0.6,0.7,0.8,1.0
"""

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from realistic_backtest import load_config, validate_trade, WARMUP
from scanner.data import get_klines, get_top_pairs_by_volume
from scanner.indicators import enrich
from scanner.ashen_marubozu import detect_signals as detect_marubozu
from scanner.risk import attach_atr_risk, setup_risk_plans


def _resolve(df, i: int, horizon_candles: int, bias: str, stop: float, entry: float,
            partial_target: float) -> tuple[str, float, int]:
    """
    Same stop-first-within-a-candle tie-break every script here uses.
    Returns (outcome, exit_price, resolved_at_index).
    """
    end = min(i + 1 + horizon_candles, len(df))
    for j in range(i + 1, end):
        candle = df.iloc[j]
        if bias == "bullish":
            if candle["low"] <= stop:
                return "loss", stop, j
            if candle["high"] >= partial_target:
                return "win", partial_target, j
        else:
            if candle["high"] >= stop:
                return "loss", stop, j
            if candle["low"] <= partial_target:
                return "win", partial_target, j
    return "expired", float(df["close"].iloc[end - 1]), end - 1


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int,
                     fractions: list[float]) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    trades = []
    blocked_until = -1  # marubozu is single-detector here, one clock is enough

    for i in range(WARMUP, len(df) - 1):
        if blocked_until >= i:
            continue
        window = df.iloc[:i + 1]
        signals = detect_marubozu(window, cfg)
        if not signals:
            continue
        close = float(window["close"].iloc[-1])
        atr = float(window["atr"].iloc[-1])
        signals = attach_atr_risk(signals, close, atr,
                                  risk_cfg.get("atr_multiplier", 1.5),
                                  risk_cfg.get("reward_risk_ratio", 2.0),
                                  risk_cfg.get("max_stop_pct"))
        plans = setup_risk_plans(signals, close, risk_cfg.get("min_risk_reward", 1.0),
                                 market_disagrees_by_direction={"bullish": True, "bearish": True},
                                 funding_ok_by_direction={"bullish": True, "bearish": True},
                                 target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not plans:
            continue
        risk = plans[0]  # only marubozu_ashen signals are fed in, so at most one plan
        bias = risk["direction"]
        entry, stop, target = risk["entry"], risk["stop"], risk["target"]
        base_risk = abs(entry - stop)
        base_rr = risk["risk_reward"]

        per_fraction = {}
        max_resolved_at = i
        for f in fractions:
            partial_target = entry + (target - entry) * f
            outcome, exit_price, resolved_at = _resolve(df, i, horizon_candles, bias, stop, entry, partial_target)
            realized_r = (f * base_rr) if outcome == "win" else (
                -1.0 if outcome == "loss" else (exit_price - entry) / base_risk * (1 if bias == "bullish" else -1))
            per_fraction[f] = {"outcome": outcome, "realized_r": round(realized_r, 4)}
            max_resolved_at = max(max_resolved_at, resolved_at)

        trade = {
            "symbol": symbol, "timeframe": tf, "direction": bias,
            "entry": entry, "stop": stop, "target": target, "base_rr": base_rr,
            "by_fraction": per_fraction,
        }
        trade["validation_error"] = validate_trade({"direction": bias, "entry": entry, "stop": stop,
                                                     "target": target, "risk_reward": base_rr})
        trades.append(trade)
        # Block on the SLOWEST fraction's resolution (f=1.0, the full target) so
        # every fraction gets a fair, non-overlapping trial on the same setup -
        # a shorter partial-profit exit isn't given an unfair frequency advantage
        # over the full-target baseline by being allowed to re-enter sooner.
        blocked_until = max_resolved_at

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=str, default="0.4,0.5,0.6,0.7,0.8,1.0")
    ap.add_argument("--max-trades", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fractions = [float(f) for f in args.fractions.split(",")]

    cfg = load_config()
    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = list(dict.fromkeys(static_pairs + top_pairs))
    timeframes = cfg["timeframes"]
    jobs = [(s, tf) for tf in timeframes for s in pairs]

    print(f"Simulating marubozu_ashen partial-profit exits ({fractions}) across up to "
          f"{len(pairs)} pairs x {len(timeframes)} timeframes ({args.workers} workers, "
          f"target: {args.max_trades} trades)...", flush=True)

    all_trades = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon, fractions): (s, tf) for s, tf in jobs}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                trades = future.result()
            except Exception as e:
                print(f"  [error] {futures[future]}: {e}", flush=True)
                continue
            all_trades.extend(trades)
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {len(all_trades)} trades so far ({time.time()-t0:.0f}s)", flush=True)
            if len(all_trades) >= args.max_trades:
                print(f"  reached {len(all_trades)} trades, stopping early", flush=True)
                for f in futures:
                    f.cancel()
                break

    print(f"\nDone: {len(all_trades)} trades from {done}/{len(jobs)} series in {time.time()-t0:.0f}s\n", flush=True)

    bad = [t for t in all_trades if t["validation_error"]]
    print(f"Order/stop/target integrity check: {len(all_trades) - len(bad)}/{len(all_trades)} correct.\n")

    def report(label_prefix, rows):
        print(f"{'fraction':<10}{'n':>6}{'win_rate':>10}{'avg_R':>10}")
        print("-" * 36)
        for f in fractions:
            decided = [t["by_fraction"][f] for t in rows if t["by_fraction"][f]["outcome"] in ("win", "loss")]
            n = len(decided)
            if n == 0:
                print(f"{f:<10}{'no trades':>6}")
                continue
            wins = sum(1 for d in decided if d["outcome"] == "win")
            wr = wins / n
            avg_r = sum(d["realized_r"] for d in decided) / n
            label = f"{f:.2f}{' (baseline)' if f == 1.0 else ''}"
            print(f"{label:<10}{n:>6}{wr:>10.1%}{avg_r:>10.3f}")
        print()

    print("=== ALL directions pooled ===")
    report("all", all_trades)

    for direction in ("bullish", "bearish"):
        rows = [t for t in all_trades if t["direction"] == direction]
        print(f"=== {direction.upper()} only (n={len(rows)} candidates) ===")
        report(direction, rows)

    print("Note: 'avg_R' is the average realized R per decided trade (including the -1R "
          "full-stop losses) - this IS the per-trade expectancy for that exit rule, directly "
          "comparable across fractions since every fraction is resolved against the exact "
          "same set of candidate trades.")


if __name__ == "__main__":
    main()

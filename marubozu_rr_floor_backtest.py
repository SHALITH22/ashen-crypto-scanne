"""
Option A for improving marubozu_ashen (compare against Option B in
marubozu_partial_profit_backtest.py): instead of a genuine partial-profit/
scale-out mechanism, just configure a SMALLER target from the start
(ashen_marubozu.py's reward_risk_ratio), paired with a correspondingly
lower per-strategy min_risk_reward qualification floor - the live pipeline
currently requires every trade's FULL target to clear a global 1.0 R:R
before it's even allowed to open, so a target this small would otherwise
be rejected outright, not just resolved differently.

This is a genuinely DIFFERENT mechanism from the partial-profit backtest,
not just another way of measuring the same thing: it changes the ENTRY
GEOMETRY itself (a smaller stop-to-target distance from the moment the
trade opens), not when profit gets banked on an otherwise-unchanged plan.
Whether the two converge to the same answer is exactly what this
comparison is for - the math says they should (same candle population,
equivalent target levels once target_fraction=0.85's existing global trim
is accounted for), but confirm, don't assume.

Ratio <-> fraction mapping: setup_risk_plans applies the existing global
target_fraction (0.85) on top of whatever reward_risk_ratio is configured
here, so raw reward_risk_ratio=1.2*f reproduces the exact same EFFECTIVE
target marubozu_partial_profit_backtest.py's fraction f tested (since
(1.2*f)*0.85 = 1.02*f, and 1.02 is marubozu's current effective R:R after
today's trim). Default ratios below are exactly 1.2 times Option B's
tested fractions [0.4, 0.5, 0.6, 0.7, 0.8, 1.0], for a direct comparison.

Usage:
  python marubozu_rr_floor_backtest.py
  python marubozu_rr_floor_backtest.py --ratios 0.48,0.6,0.72,0.84,0.96,1.2
"""

import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from realistic_backtest import load_config, validate_trade, WARMUP
from scanner.data import get_klines, get_top_pairs_by_volume
from scanner.indicators import enrich
from scanner.ashen_marubozu import detect_signals as detect_marubozu
from scanner.risk import attach_atr_risk, setup_risk_plans


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int,
                     reward_risk_ratio: float, min_risk_reward: float) -> list[dict]:
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})

    # Deep copy so parallel workers testing different ratios never share
    # (and corrupt) each other's config mutation.
    test_cfg = copy.deepcopy(cfg)
    test_cfg["ashen"]["marubozu"]["reward_risk_ratio"] = reward_risk_ratio

    trades = []
    blocked_until = -1

    for i in range(WARMUP, len(df) - 1):
        if blocked_until >= i:
            continue
        window = df.iloc[:i + 1]
        signals = detect_marubozu(window, test_cfg)
        if not signals:
            continue
        close = float(window["close"].iloc[-1])
        atr = float(window["atr"].iloc[-1])
        signals = attach_atr_risk(signals, close, atr,
                                  risk_cfg.get("atr_multiplier", 1.5),
                                  risk_cfg.get("reward_risk_ratio", 2.0),
                                  risk_cfg.get("max_stop_pct"))
        plans = setup_risk_plans(signals, close, min_risk_reward,
                                 market_disagrees_by_direction={"bullish": True, "bearish": True},
                                 funding_ok_by_direction={"bullish": True, "bearish": True},
                                 target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not plans:
            continue
        risk = plans[0]
        bias = risk["direction"]
        entry, stop, target = risk["entry"], risk["stop"], risk["target"]
        base_risk = abs(entry - stop)

        end = min(i + 1 + horizon_candles, len(df))
        outcome, exit_price, resolved_at = "expired", float(df["close"].iloc[end - 1]), end - 1
        for j in range(i + 1, end):
            candle = df.iloc[j]
            if bias == "bullish":
                if candle["low"] <= stop:
                    outcome, exit_price, resolved_at = "loss", stop, j
                    break
                if candle["high"] >= target:
                    outcome, exit_price, resolved_at = "win", target, j
                    break
            else:
                if candle["high"] >= stop:
                    outcome, exit_price, resolved_at = "loss", stop, j
                    break
                if candle["low"] <= target:
                    outcome, exit_price, resolved_at = "win", target, j
                    break

        realized_r = (risk["risk_reward"] if outcome == "win" else
                     -1.0 if outcome == "loss" else
                     (exit_price - entry) / base_risk * (1 if bias == "bullish" else -1))
        trade = {
            "symbol": symbol, "timeframe": tf, "direction": bias,
            "entry": entry, "stop": stop, "target": target, "risk_reward": risk["risk_reward"],
            "outcome": outcome, "realized_r": round(realized_r, 4),
        }
        trade["validation_error"] = validate_trade(trade)
        trades.append(trade)
        blocked_until = resolved_at

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=str, default="0.48,0.6,0.72,0.84,0.96,1.2")
    ap.add_argument("--min-risk-reward", type=float, default=0.3,
                    help="permissive floor so smaller-target variants aren't rejected outright - "
                         "this IS the per-strategy override Option A would need live")
    ap.add_argument("--max-trades", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ratios = [float(r) for r in args.ratios.split(",")]

    cfg = load_config()
    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = list(dict.fromkeys(static_pairs + top_pairs))
    timeframes = cfg["timeframes"]

    results_by_ratio = {}
    for ratio in ratios:
        print(f"\n=== Testing reward_risk_ratio={ratio} (min_risk_reward floor={args.min_risk_reward}) ===", flush=True)
        jobs = [(s, tf) for tf in timeframes for s in pairs]
        all_trades = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon, ratio, args.min_risk_reward): (s, tf)
                      for s, tf in jobs}
            done = 0
            for future in as_completed(futures):
                done += 1
                try:
                    trades = future.result()
                except Exception as e:
                    print(f"  [error] {futures[future]}: {e}", flush=True)
                    continue
                all_trades.extend(trades)
                if len(all_trades) >= args.max_trades:
                    for f in futures:
                        f.cancel()
                    break
        print(f"  {len(all_trades)} trades from {done}/{len(jobs)} series in {time.time()-t0:.0f}s", flush=True)
        results_by_ratio[ratio] = all_trades

    print(f"\n{'ratio':<10}{'n':>6}{'win_rate':>10}{'avg_R':>10}  (validation failures)")
    print("-" * 60)
    for ratio, trades in results_by_ratio.items():
        bad = sum(1 for t in trades if t["validation_error"])
        decided = [t for t in trades if t["outcome"] in ("win", "loss")]
        n = len(decided)
        if n == 0:
            print(f"{ratio:<10}no trades")
            continue
        wins = sum(1 for t in decided if t["outcome"] == "win")
        wr = wins / n
        avg_r = sum(t["realized_r"] for t in decided) / n
        print(f"{ratio:<10}{n:>6}{wr:>10.1%}{avg_r:>10.3f}  ({bad})")

        for direction in ("bullish", "bearish"):
            d_decided = [t for t in decided if t["direction"] == direction]
            dn = len(d_decided)
            if dn == 0:
                continue
            dwins = sum(1 for t in d_decided if t["outcome"] == "win")
            dwr = dwins / dn
            davg_r = sum(t["realized_r"] for t in d_decided) / dn
            print(f"  {direction:<8}{dn:>6}{dwr:>10.1%}{davg_r:>10.3f}")


if __name__ == "__main__":
    main()

"""
Tests whether cutting a losing trade early - before it reaches the full
stop-loss - actually improves results, or just cuts real winners short.

For every simulated trade, tracks the maximum adverse excursion (MAE) at
each candle as a fraction of the full stop distance (0.0 = at entry,
1.0 = at the stop price), using only that candle's own high/low - no
lookahead, this is the same information available in real time. Then
compares, for several candidate "soft-stop" thresholds, what would have
happened if the trade had been exited the moment MAE first crossed that
threshold, against simply waiting for the full stop.

The real risk this checks for: a trade that dips deep against you before
reversing to hit the real target is common (this is exactly what "shaking
out weak hands" describes) - an aggressive early-exit threshold can turn
real winners into realized losses. This only recommends a threshold if it
demonstrably improves expectancy across the full sample, accounting for
BOTH effects (losses cut smaller AND winners potentially cut short).

Runs on the same proven, BTC/ETH-filtered detector set as
confluence_btc_backtest.py, since that's what's actually going live.

Usage:
  python early_exit_backtest.py --max-trades 2000
"""

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from scanner.data import get_klines, get_top_pairs_by_volume
from scanner.indicators import enrich
from scanner.patterns import run_all_detectors
from scanner.risk import attach_atr_risk, setup_risk_plan, STRUCTURAL_NAMES

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
WARMUP = 210
PROVEN_NAMES = STRUCTURAL_NAMES | {"ema_stack"}
MAE_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # fraction of stop distance


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def confluence_score(signals: list[dict]) -> tuple[str, float]:
    bull = sum(1.0 for s in signals if s["direction"] == "bullish")
    bear = sum(1.0 for s in signals if s["direction"] == "bearish")
    if bull > bear:
        return "bullish", round(bull, 2)
    if bear > bull:
        return "bearish", round(bear, 2)
    return "mixed", round(max(bull, bear), 2)


def trend_lookup(df: pd.DataFrame) -> dict:
    enriched = df.copy()
    enriched["ema20"] = enriched["close"].ewm(span=20, adjust=False).mean()
    return dict(zip(enriched["open_time"], enriched["close"] > enriched["ema20"]))


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int,
                     btc_trend: dict, eth_trend: dict) -> list[dict]:
    """
    Same trade generation as confluence_btc_backtest.py (proven detectors,
    BTC+ETH must both disagree), but records the FULL path (MAE at each
    candle up to resolution) instead of just the final outcome.
    """
    df = get_klines(symbol, tf, 1000)
    if df is None or len(df) < WARMUP + horizon_candles + 10:
        return []
    df = enrich(df, cfg)
    risk_cfg = cfg.get("risk", {})
    trades = []
    blocked_until: dict[str, int] = {}

    for i in range(WARMUP, len(df) - 1):
        window = df.iloc[:i + 1]
        signals = run_all_detectors(window, cfg)
        if not signals:
            continue
        proven_signals = [s for s in signals if s["name"] in PROVEN_NAMES]
        if not proven_signals:
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
        proven_bias_signals = [s for s in proven_signals if s["direction"] == bias]
        if not proven_bias_signals:
            continue
        risk = setup_risk_plan(signals, bias, close, risk_cfg.get("min_risk_reward", 1.0),
                               market_disagrees=True,
                               target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not risk or risk["based_on"] not in PROVEN_NAMES:
            continue
        key = risk["based_on"]
        if blocked_until.get(key, -1) >= i:
            continue

        open_time = window["open_time"].iloc[-1]
        btc_bull = btc_trend.get(open_time)
        eth_bull = eth_trend.get(open_time)
        if btc_bull is None or eth_bull is None:
            continue
        trade_is_bullish = bias == "bullish"
        if not ((btc_bull != trade_is_bullish) and (eth_bull != trade_is_bullish)):
            continue  # BTC/ETH don't both disagree - wouldn't have alerted live

        stop_distance = abs(risk["entry"] - risk["stop"])
        if stop_distance <= 0:
            continue

        outcome, outcome_price, resolved_at = None, None, None
        mae_cross_at = {t: None for t in MAE_THRESHOLDS}  # first candle index MAE reached this threshold
        end = min(i + 1 + horizon_candles, len(df))
        for j in range(i + 1, end):
            candle = df.iloc[j]
            if bias == "bullish":
                adverse_px = candle["low"]
                mae = (risk["entry"] - adverse_px) / stop_distance
            else:
                adverse_px = candle["high"]
                mae = (adverse_px - risk["entry"]) / stop_distance
            mae = max(0.0, mae)
            for t in MAE_THRESHOLDS:
                if mae_cross_at[t] is None and mae >= t:
                    mae_cross_at[t] = j

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
            resolved_at = end - 1
            outcome = "expired"
            outcome_price = float(df["close"].iloc[resolved_at])

        trades.append({
            "symbol": symbol, "timeframe": tf, "based_on": risk["based_on"], "direction": bias,
            "risk_reward": risk["risk_reward"], "outcome": outcome,
            "outcome_pct": round((outcome_price - risk["entry"]) / risk["entry"] * 100, 3),
            "resolved_at_offset": resolved_at - i,
            "mae_cross_offset": {str(t): (mae_cross_at[t] - i if mae_cross_at[t] is not None else None)
                                 for t in MAE_THRESHOLDS},
        })
        blocked_until[key] = resolved_at

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    min_conf = cfg["output"]["min_confluence"]
    static_pairs = list(cfg["pairs"])
    top_pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
    pairs = [p for p in dict.fromkeys(static_pairs + top_pairs) if p not in ("BTCUSDT", "ETHUSDT")]
    timeframes = cfg["timeframes"]

    print("Fetching BTC/ETH trend reference per timeframe...", flush=True)
    btc_trends, eth_trends = {}, {}
    for tf in timeframes:
        btc_df = get_klines("BTCUSDT", tf, 1000)
        eth_df = get_klines("ETHUSDT", tf, 1000)
        btc_trends[tf] = trend_lookup(btc_df) if btc_df is not None else {}
        eth_trends[tf] = trend_lookup(eth_df) if eth_df is not None else {}

    jobs = [(s, tf) for tf in timeframes for s in pairs]
    print(f"Simulating up to {len(pairs)} pairs x {len(timeframes)} timeframes "
          f"({args.workers} workers, target: {args.max_trades} trades)...", flush=True)

    all_trades = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_pair_tf, s, tf, cfg, args.horizon, min_conf,
                                   btc_trends[tf], eth_trends[tf]): (s, tf) for s, tf in jobs}
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
                print(f"  [{done}/{len(jobs)}] {len(all_trades)} trades ({time.time()-t0:.0f}s)", flush=True)
            if len(all_trades) >= args.max_trades:
                print(f"  reached {len(all_trades)} trades, stopping early", flush=True)
                for f in futures:
                    f.cancel()
                break

    print(f"\nDone: {len(all_trades)} trades from {done}/{len(jobs)} series in {time.time()-t0:.0f}s\n", flush=True)

    decided = [t for t in all_trades if t["outcome"] in ("win", "loss")]
    n = len(decided)
    baseline_wins = sum(1 for t in decided if t["outcome"] == "win")
    baseline_wr = baseline_wins / n if n else 0
    avg_rr = sum(t["risk_reward"] for t in decided if t["risk_reward"]) / n if n else 0
    baseline_exp = baseline_wr * (1 + avg_rr) - 1
    print(f"=== Baseline (wait for full stop/target), n={n} ===")
    print(f"win_rate={baseline_wr:.1%}  avg_rr={avg_rr:.2f}  expectancy={baseline_exp:+.3f}R\n")

    print(f"{'MAE threshold':<16}{'winners cut short':>20}{'losses reduced':>18}{'net expectancy':>18}{'vs baseline':>14}")
    print("-" * 86)
    for t in MAE_THRESHOLDS:
        key = str(t)
        r_values = []
        winners_cut_short = 0
        losses_reduced = 0
        for tr in decided:
            cross_offset = tr["mae_cross_offset"][key]
            resolved_offset = tr["resolved_at_offset"]
            if cross_offset is not None and cross_offset < resolved_offset:
                # Would have exited early at -t R, regardless of what actually happened later
                r_values.append(-t)
                if tr["outcome"] == "win":
                    winners_cut_short += 1
                else:
                    losses_reduced += 1
            else:
                # Never crossed the threshold before resolution - same outcome as baseline
                if tr["outcome"] == "win":
                    r_values.append(tr["risk_reward"] if tr["risk_reward"] else 0)
                else:
                    r_values.append(-1.0)
        net_exp = sum(r_values) / len(r_values) if r_values else 0
        delta = net_exp - baseline_exp
        flag = "  <-- WORSE" if delta < -0.005 else ("  BETTER" if delta > 0.005 else "")
        print(f"{t:<16}{winners_cut_short:>20}{losses_reduced:>18}{net_exp:>+17.3f}R{delta:>+13.3f}R{flag}")

    out = Path(__file__).parent / "early_exit_results.json"
    out.write_text(json.dumps({"n_trades": len(all_trades), "baseline_expectancy": baseline_exp,
                               "trades": all_trades}, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

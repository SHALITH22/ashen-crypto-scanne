"""
Refines the detectors already proven profitable (ascending_triangle,
descending_triangle, rising_wedge, ema_stack/bullish) by testing two
specific hypotheses instead of re-litigating the ones already shown to
lose money:

  1. Confluence: does a trade backed by MORE agreeing signals (not just the
     structural pattern alone) actually perform better? The earlier sweep
     tested this across generic detectors only and found no benefit - this
     tests it specifically for our proven winners.
  2. BTC/ETH alignment: altcoins are heavily correlated with BTC and ETH.
     Does an altcoin trade perform better when BTC's (and ETH's) own trend
     agrees with the trade's direction, vs when it's fighting the market
     leader's trend?

Usage:
  python confluence_btc_backtest.py --max-trades 2000
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
PROVEN_NAMES = STRUCTURAL_NAMES | {"ema_stack"}  # the detectors worth refining, not re-litigating


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
    """open_time -> bool (close above EMA20 = bullish market leader trend)."""
    enriched = df.copy()
    enriched["ema20"] = enriched["close"].ewm(span=20, adjust=False).mean()
    return dict(zip(enriched["open_time"], enriched["close"] > enriched["ema20"]))


def simulate_pair_tf(symbol: str, tf: str, cfg: dict, horizon_candles: int, min_confluence: int,
                     btc_trend: dict, eth_trend: dict) -> list[dict]:
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
            continue  # confluence winner isn't one of our proven detectors this candle
        # market_disagrees=True bypasses the live filter this script itself
        # measures - it needs the raw trade generated regardless of BTC/ETH
        # alignment, then checks that alignment against the outcome afterward.
        risk = setup_risk_plan(signals, bias, close, risk_cfg.get("min_risk_reward", 1.0),
                               market_disagrees=True,
                               target_fraction=risk_cfg.get("target_fraction", 1.0))
        if not risk or risk["based_on"] not in PROVEN_NAMES:
            continue
        key = risk["based_on"]
        if blocked_until.get(key, -1) >= i:
            continue

        open_time = window["open_time"].iloc[-1]
        btc_agrees = btc_trend.get(open_time)
        eth_agrees = eth_trend.get(open_time)
        if btc_agrees is not None:
            btc_agrees = btc_agrees if bias == "bullish" else not btc_agrees
        if eth_agrees is not None:
            eth_agrees = eth_agrees if bias == "bullish" else not eth_agrees

        outcome, outcome_price = None, None
        end = min(i + 1 + horizon_candles, len(df))
        for j in range(i + 1, end):
            candle = df.iloc[j]
            if bias == "bullish":
                if candle["low"] <= risk["stop"]:
                    outcome, outcome_price = "loss", risk["stop"]; break
                if candle["high"] >= risk["target"]:
                    outcome, outcome_price = "win", risk["target"]; break
            else:
                if candle["high"] >= risk["stop"]:
                    outcome, outcome_price = "loss", risk["stop"]; break
                if candle["low"] <= risk["target"]:
                    outcome, outcome_price = "win", risk["target"]; break
        resolved_at = j if outcome else end - 1
        if outcome is None:
            outcome, outcome_price = "expired", float(df["close"].iloc[resolved_at])

        trades.append({
            "symbol": symbol, "timeframe": tf, "based_on": risk["based_on"], "direction": bias,
            "strength": strength, "risk_reward": risk["risk_reward"], "outcome": outcome,
            "outcome_pct": round((outcome_price - risk["entry"]) / risk["entry"] * 100, 3),
            "btc_agrees": btc_agrees, "eth_agrees": eth_agrees,
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
    print(f"Simulating up to {len(pairs)} pairs x {len(timeframes)} timeframes for proven detectors only "
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

    def report(label, rows):
        n = len(rows)
        if n < 15:
            print(f"{label:<40}n={n:<6}(too few to trust)")
            return
        wins = sum(1 for r in rows if r["outcome"] == "win")
        wr = wins / n
        avg_rr = sum(r["risk_reward"] for r in rows if r["risk_reward"]) / n
        exp = wr * (1 + avg_rr) - 1
        print(f"{label:<40}n={n:<6}win_rate={wr:>6.1%}  avg_rr={avg_rr:>5.2f}  expectancy={exp:+.3f}R")

    print("=== Confluence strength (proven detectors only) ===")
    by_strength = defaultdict(list)
    for t in decided:
        by_strength[t["strength"]].append(t)
    for s in sorted(by_strength):
        report(f"strength={s}", by_strength[s])

    print("\n=== BTC trend alignment ===")
    report("BTC agrees with trade direction", [t for t in decided if t["btc_agrees"] is True])
    report("BTC disagrees with trade direction", [t for t in decided if t["btc_agrees"] is False])

    print("\n=== ETH trend alignment ===")
    report("ETH agrees with trade direction", [t for t in decided if t["eth_agrees"] is True])
    report("ETH disagrees with trade direction", [t for t in decided if t["eth_agrees"] is False])

    print("\n=== BTC+ETH both agree vs both disagree ===")
    report("Both BTC and ETH agree", [t for t in decided if t["btc_agrees"] and t["eth_agrees"]])
    report("Both BTC and ETH disagree", [t for t in decided if t["btc_agrees"] is False and t["eth_agrees"] is False])

    out = Path(__file__).parent / "confluence_btc_results.json"
    out.write_text(json.dumps({"n_trades": len(all_trades), "trades": all_trades}, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

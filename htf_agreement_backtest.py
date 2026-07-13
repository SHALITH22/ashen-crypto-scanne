"""
Tests whether the HTF (higher-timeframe) agreement requirement - a signal's
own higher timeframes must not conflict with its bias, see scanner/mtf.py -
actually improves outcomes on the proven, BTC/ETH-filtered detector set, or
whether it's just an extra filter stacking on top of everything else
without adding real value.

Unlike the earlier backtests (which walk one timeframe per pair
independently), this needs ALL of a symbol's timeframes fetched together,
since HTF agreement is a cross-timeframe check: at the exact moment a
lower-timeframe signal fires, what was each higher timeframe's OWN
confluence bias at that same point in real time? This mirrors
scanner/mtf.py's annotate_htf exactly, just evaluated once per backtest
opportunity instead of once per live scan.

Usage:
  python htf_agreement_backtest.py --max-trades 2000
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
PROVEN_NAMES = STRUCTURAL_NAMES | {"ema_stack", "qqe_cross"}


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


def htf_bias_at(higher_df: pd.DataFrame, cfg: dict, ts) -> str | None:
    """
    The higher timeframe's own confluence bias using only candles closed
    at/before ts - exactly what was "known" at that moment, no lookahead.
    """
    idx = higher_df["open_time"].searchsorted(ts, side="right") - 1
    if idx < WARMUP:
        return None
    window = higher_df.iloc[:idx + 1]
    signals = run_all_detectors(window, cfg)
    if not signals:
        return None
    bias, _ = confluence_score(signals)
    return bias if bias != "mixed" else None


def simulate_symbol(symbol: str, cfg: dict, min_confluence: int,
                    btc_trend: dict, eth_trend: dict) -> list[dict]:
    timeframes = cfg["timeframes"]
    dfs = {}
    for tf in timeframes:
        df = get_klines(symbol, tf, 1000)
        if df is None or len(df) < WARMUP + 30:
            continue  # this timeframe lacks history (e.g. a newer coin has no 1M/1w depth) - skip only it
        dfs[tf] = enrich(df, cfg)
    if not dfs:
        return []

    risk_cfg = cfg.get("risk", {})
    trades = []

    for ti, tf in enumerate(timeframes):
        if tf not in dfs:
            continue
        df = dfs[tf]
        higher_tfs = [h for h in timeframes[ti + 1:] if h in dfs]
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
            # btc_trend/eth_trend are the full {timeframe: {timestamp: bool}}
            # dicts - index by the CURRENT timeframe first, then timestamp.
            btc_bull = btc_trend.get(tf, {}).get(open_time)
            eth_bull = eth_trend.get(tf, {}).get(open_time)
            if btc_bull is None or eth_bull is None:
                continue
            trade_is_bullish = bias == "bullish"
            if not ((btc_bull != trade_is_bullish) and (eth_bull != trade_is_bullish)):
                continue

            # HTF agreement - exactly mirrors scanner/mtf.py's annotate_htf
            htf_conflict = False
            htf_data_available = False
            for htf in higher_tfs:
                htf_bias = htf_bias_at(dfs[htf], cfg, open_time)
                if htf_bias is None:
                    continue
                htf_data_available = True
                if htf_bias != bias:
                    htf_conflict = True
            htf_agrees = True if not htf_data_available else (not htf_conflict)

            outcome, outcome_price, resolved_at = None, None, None
            end = min(i + 1 + 60, len(df))
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
                resolved_at = end - 1
                outcome, outcome_price = "expired", float(df["close"].iloc[resolved_at])

            trades.append({
                "symbol": symbol, "timeframe": tf, "based_on": risk["based_on"], "direction": bias,
                "risk_reward": risk["risk_reward"], "outcome": outcome,
                "outcome_pct": round((outcome_price - risk["entry"]) / risk["entry"] * 100, 3),
                "htf_agrees": htf_agrees, "htf_data_available": htf_data_available,
            })
            blocked_until[key] = resolved_at

    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trades", type=int, default=2000)
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

    print(f"Simulating up to {len(pairs)} symbols (all their timeframes fetched together for HTF checks) "
          f"({args.workers} workers, target: {args.max_trades} trades)...", flush=True)

    all_trades = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(simulate_symbol, s, cfg, min_conf, btc_trends, eth_trends): s for s in pairs}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                trades = future.result()
            except Exception as e:
                print(f"  [error] {futures[future]}: {e}", flush=True)
                continue
            all_trades.extend(trades)
            if done % 10 == 0 or done == len(pairs):
                print(f"  [{done}/{len(pairs)}] {len(all_trades)} trades ({time.time()-t0:.0f}s)", flush=True)
            if len(all_trades) >= args.max_trades:
                print(f"  reached {len(all_trades)} trades, stopping early", flush=True)
                for f in futures:
                    f.cancel()
                break

    print(f"\nDone: {len(all_trades)} trades from {done}/{len(pairs)} symbols in {time.time()-t0:.0f}s\n", flush=True)

    decided = [t for t in all_trades if t["outcome"] in ("win", "loss")]

    def report(label, rows):
        n = len(rows)
        if n < 20:
            print(f"{label:<30}n={n:<6}(too few to trust)")
            return
        wins = sum(1 for r in rows if r["outcome"] == "win")
        wr = wins / n
        avg_rr = sum(r["risk_reward"] for r in rows if r["risk_reward"]) / n
        exp = wr * (1 + avg_rr) - 1
        print(f"{label:<30}n={n:<6}win_rate={wr:>6.1%}  avg_rr={avg_rr:>5.2f}  expectancy={exp:+.3f}R")

    print("=== HTF agreement requirement ===")
    report("HTF agrees (current requirement)", [t for t in decided if t["htf_agrees"]])
    report("HTF conflicts (currently blocked)", [t for t in decided if not t["htf_agrees"]])
    report("All trades combined (no HTF filter)", decided)

    out = Path(__file__).parent / "htf_agreement_results.json"
    out.write_text(json.dumps({"n_trades": len(all_trades), "trades": all_trades}, indent=2))
    print(f"\nFull detail written to {out}")


if __name__ == "__main__":
    main()

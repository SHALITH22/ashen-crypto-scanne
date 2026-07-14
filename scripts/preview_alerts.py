"""
One-off preview script: runs the exact production scan (main.load_config,
main.scan_pair, all configured pairs x timeframes) against live Binance
data, then renders each qualifying setup with notify.py's own
format_setup() - the literal function that builds the real Telegram
message text - WITHOUT ever calling send_telegram. Nothing is sent
anywhere; this only prints to the console.

Not part of the live pipeline - a manual verification tool, run once and
safe to delete afterward.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import load_config, load_detector_weights, scan_pair, get_market_trend, \
    load_realistic_backtest_expectancy, load_realistic_backtest_win_rate, \
    combined_detector_expectancy, combined_detector_win_rate
from scanner.journal import detector_expectancy, detector_avg_return, detector_reliability
from scanner.mtf import annotate_htf
from scanner.notify import format_setup

cfg = load_config()
pairs = cfg["pairs"]
timeframes = cfg["timeframes"]
weights = load_detector_weights(cfg)
risk_cfg = cfg.get("risk", {})
min_n = risk_cfg.get("min_reliable_n", 10)
min_expectancy = risk_cfg.get("min_detector_expectancy", 0.0)
expectancy = combined_detector_expectancy(load_realistic_backtest_expectancy(min_n), detector_expectancy(min_n))
unreliable = {k for k, exp in expectancy.items() if exp < min_expectancy}
win_rates = combined_detector_win_rate(load_realistic_backtest_win_rate(min_n), detector_reliability(min_n))
avg_returns = detector_avg_return(min_n)

print(f"Fetching BTC/ETH market trend across {len(timeframes)} timeframes...")
btc_trend = get_market_trend("BTCUSDT", timeframes, cfg)
eth_trend = get_market_trend("ETHUSDT", timeframes, cfg)

print(f"Scanning {len(pairs)} pairs x {len(timeframes)} timeframes (live Binance data)...\n")

tg_cfg = cfg.get("notify", {}).get("telegram", {})
min_strength = tg_cfg.get("min_strength", 3)
only_agreeing = tg_cfg.get("only_htf_agreeing", True)

t0 = time.time()
results = []
errors = 0

def _scan(symbol):
    try:
        return symbol, scan_pair(symbol, timeframes, cfg, weights, avg_returns, unreliable,
                                 btc_trend, eth_trend, win_rates, use_proxy=False)
    except Exception:
        return symbol, None

with ThreadPoolExecutor(max_workers=cfg.get("scan_concurrency", 8)) as ex:
    futures = {ex.submit(_scan, p): p for p in pairs}
    done = 0
    for fut in as_completed(futures):
        symbol, res = fut.result()
        done += 1
        if done % 20 == 0:
            print(f"  ...{done}/{len(pairs)} pairs scanned ({time.time()-t0:.0f}s elapsed)")
        if res is None:
            errors += 1
            continue
        if not res["timeframes"]:
            continue
        res = annotate_htf(res, timeframes)
        results.append(res)

elapsed = time.time() - t0
print(f"\nScan complete in {elapsed:.0f}s. {len(results)}/{len(pairs)} pairs returned data, {errors} errored.\n")
print("=" * 70)
print("PREVIEW: messages that WOULD be sent to Telegram (none actually sent)")
print("=" * 70)

alert_count = 0
for res in results:
    for tf, data in res["timeframes"].items():
        if data["strength"] < min_strength:
            continue
        if only_agreeing and not data.get("htf_agrees", True):
            continue
        if not data.get("risk"):
            continue
        alert_count += 1
        print(f"\n--- Message {alert_count} ---")
        print(format_setup(res["symbol"], tf, data))

print(f"\n{'=' * 70}")
print(f"Pairs with any live Jayantha detection: {len(results)}")
print(f"Setups that would clear the Telegram gate (strength>={min_strength}, HTF agrees, has risk plan): {alert_count}")

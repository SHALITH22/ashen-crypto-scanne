"""
Regenerates crypto_universe.json and prints a fresh, curated top-100
`pairs:` list for config/settings.yaml.

MUST be run from a network that can actually reach binance.com - GitHub
Actions' IPs are entirely blocked there (confirmed, see
scanner/data.py's module docstring), which is exactly why
crypto_universe.json and the static `pairs:` list exist as committed
files instead of being fetched live every run (see
top_volume_lightweight in main.py, which also depends on a live
binance.com call and silently falls through to `pairs:` every time until
that block lifts).

The printed pairs list is restricted to symbols also listed on
Binance.US, since that's get_klines' fallback exchange when binance.com
is blocked (see scanner/data.py). A high-volume real Binance pair that
ISN'T on Binance.US (e.g. EVAAUSDT, LABUSDT, TAGUSDT as of 2026-07-12)
can't be reached at all from GitHub Actions even via that fallback -
scanning it wastes a slot that could go to a pair that actually resolves.
Without this curation, ~46/100 pairs failed with "no data" on GitHub
Actions in practice.

Run this occasionally (weekly/monthly is plenty - new listings/delistings
are the only thing that goes stale) from a local machine or any non-cloud
network, then commit + push the updated crypto_universe.json and paste
the printed pairs list into config/settings.yaml.

Usage:
  python scripts/refresh_universe.py
"""

import json
from pathlib import Path

import requests

FUTURES_URL = "https://fapi.binance.com"
BINANCE_US_URL = "https://api.binance.us"
STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "PYUSD",
    "USDS", "USDD", "GUSD", "FRAX", "LUSD", "SUSD", "USDP", "EURT", "EURC",
}

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "crypto_universe.json"


def main():
    exch = requests.get(f"{FUTURES_URL}/fapi/v1/exchangeInfo", timeout=15).json()
    # Real crypto USDT perpetuals only - excludes TRADIFI_PERPETUAL
    # (tokenized stocks/commodities like XAUUSDT, TSLAUSDT, NVDAUSDT),
    # quarterly contracts, delisted/non-trading symbols, and stable pairs.
    universe = sorted([
        s["symbol"] for s in exch["symbols"]
        if s["symbol"].endswith("USDT")
        and s.get("status") == "TRADING"
        and s.get("contractType") == "PERPETUAL"
        and s["symbol"][:-4] not in STABLE_SYMBOLS
    ])
    UNIVERSE_PATH.write_text(json.dumps(universe, indent=2))
    print(f"Wrote {len(universe)} symbols to {UNIVERSE_PATH}")

    us_exch = requests.get(f"{BINANCE_US_URL}/api/v3/exchangeInfo", timeout=15).json()
    us_usdt = {s["symbol"] for s in us_exch["symbols"]
              if s["symbol"].endswith("USDT") and s.get("status") == "TRADING"}
    dual_listed = set(universe) & us_usdt
    print(f"{len(dual_listed)}/{len(universe)} universe symbols are also on Binance.US "
          f"(usable as the static fallback pairs list from GitHub Actions)")

    ticker = requests.get(f"{FUTURES_URL}/fapi/v1/ticker/24hr", timeout=15).json()
    rows = [t for t in ticker if t["symbol"] in dual_listed]
    rows.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    top100 = [t["symbol"] for t in rows[:100]]

    print("\nPaste this into config/settings.yaml's `pairs:` list:\n")
    for s in top100:
        print(f"  - {s}")


if __name__ == "__main__":
    main()

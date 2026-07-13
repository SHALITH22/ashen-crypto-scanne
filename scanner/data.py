"""
Binance market data layer.

IMPORTANT: Public market data (klines/candles) requires NO API key.
You only need API keys later, for account data or placing orders.

Binance.com (both spot and futures) hard-blocks US-region IPs, which is
exactly what GitHub Actions runners use - confirmed via a live run where
every single request came back 451. Binance.US is a separate platform
built for US users and serves the same public kline format, so it's used
as a fallback when binance.com is geo-blocked. Locally (non-US IPs) the
binance.com endpoints work directly and Binance.US is never touched.

Binance.US only covers ~157 of our ~528 real Binance USDT perpetuals, so
that fallback alone can't reach the remaining ~371 ("long-tail") symbols
from GitHub Actions at all. BINANCE_PROXY_URL (set via env var, see
.github/workflows/scan.yml) routes those specific calls through a small
Cloudflare Worker relay instead, which fetches from binance.com using
Cloudflare's edge IPs (not blocked the same way). Opt-in per call
(use_proxy=True) rather than global, because the proxy has a real request
budget (Cloudflare's free tier caps at 100k/day) - only the long-tail
pairs that genuinely need it should route through it; the 157
Binance.US-covered pairs keep using the existing free, unlimited direct
path.
"""

import os
import time
import requests
import pandas as pd

BASE_URL = "https://api.binance.com"
FUTURES_URL = "https://fapi.binance.com"
US_URL = "https://api.binance.us"
PROXY_URL = os.environ.get("BINANCE_PROXY_URL", "").rstrip("/")
PROXY_SECRET = os.environ.get("BINANCE_PROXY_SECRET")

# Diagnostic counters, not behavior - the 403/exchangeInfo saga showed that
# "falls through to the direct chain and shows no_data" looks identical
# whether the proxy is misconfigured, unreachable, or actively rejecting
# requests, and job console logs aren't readable via the API (confirmed:
# GET /actions/jobs/{id}/logs needs admin rights this token doesn't have).
# main.py writes this into scan_health.json - public, no auth needed - so
# the real failure reason is visible without guessing.
_proxy_stats = {"configured": False, "attempts": 0, "successes": 0,
                "last_status": None, "last_error": None}


def get_proxy_stats() -> dict:
    return dict(_proxy_stats)


def _proxy_get(path: str, params: dict, timeout: int) -> requests.Response | None:
    """
    Single attempt through the Cloudflare Worker proxy. Returns None (never
    raises) on any failure - callers fall through to the normal direct/
    Binance.US chain, so a proxy outage degrades gracefully instead of
    breaking the whole run.
    """
    if not PROXY_URL:
        return None
    _proxy_stats["configured"] = True
    _proxy_stats["attempts"] += 1
    headers = {"X-Proxy-Secret": PROXY_SECRET} if PROXY_SECRET else {}
    try:
        resp = requests.get(f"{PROXY_URL}{path}", params=params, headers=headers, timeout=timeout)
        _proxy_stats["last_status"] = resp.status_code
        if resp.ok:
            _proxy_stats["successes"] += 1
        else:
            _proxy_stats["last_error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return resp
    except requests.exceptions.RequestException as e:
        _proxy_stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
        return None


KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def _endpoint_chain(futures: bool, kind: str, include_us: bool = True) -> list[str]:
    """
    Ordered list of URLs to try: requested endpoint, its binance.com
    sibling, then Binance.US - deduped in case futures=False collapses
    the first two.

    include_us=False restricts the chain to binance.com only. Binance.US is
    a separate, smaller exchange with a partially different symbol catalog
    (confirmed: some symbols exist on one but not the other) - fine as a
    last-resort DATA source for a symbol we already know is a real global
    Binance pair, but never safe as a source for deciding WHICH symbols to
    scan in the first place. Pair-discovery functions must not use it.
    """
    path = f"fapi/v1/{kind}" if futures else f"api/v3/{kind}"
    sibling_path = f"api/v3/{kind}" if futures else f"fapi/v1/{kind}"
    chain = [
        f"{FUTURES_URL if futures else BASE_URL}/{path}",
        f"{BASE_URL if futures else FUTURES_URL}/{sibling_path}",
    ]
    if include_us:
        chain.append(f"{US_URL}/api/v3/{kind}")
    seen = set()
    return [u for u in chain if not (u in seen or seen.add(u))]


def get_all_usdt_pairs(futures: bool = True) -> list[str]:
    """
    Fetch every actively trading USDT pair from the real global Binance
    (futures, falling back to spot on geo-block - never Binance.US, which
    has a different symbol catalog and would return the wrong universe).
    Returns an empty list if binance.com is unreachable (geo-blocked, or a
    network-level failure - timeout/connection error) rather than silently
    substituting a different exchange's pairs. A caller MUST treat an empty
    return as "discovery failed", not "zero pairs currently trading" - see
    main.py's scan_all fallback, added after a run silently scanned zero
    pairs for 24+ hours (every run "succeeded" while doing nothing) because
    nothing distinguished this from a legitimately quiet market.
    """
    for url in _endpoint_chain(futures, "exchangeInfo", include_us=False):
        try:
            resp = requests.get(url, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"[warn] get_all_usdt_pairs: {url} failed ({e}) - trying next endpoint")
            continue
        if resp.status_code == 451:
            futures = False  # try the sibling before giving up
            continue
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"[warn] get_all_usdt_pairs: {url} returned {resp.status_code} ({e}) - trying next endpoint")
            continue
        symbols = resp.json()["symbols"]
        return [
            s["symbol"] for s in symbols
            if s["symbol"].endswith("USDT")
            and s.get("status", s.get("contractStatus")) == "TRADING"
            # futures exchangeInfo also lists delivery contracts - keep perpetuals only
            and (not futures or s.get("contractType") == "PERPETUAL")
        ]
    return []


# Stable-to-stable pairs (e.g. USDCUSDT) aren't meaningful for pattern
# scanning - near-zero volatility by design, not a "trade".
STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "PYUSD",
    "USDS", "USDD", "GUSD", "FRAX", "LUSD", "SUSD", "USDP", "EURT", "EURC",
}


def get_top_pairs_by_volume(n: int = 100, futures: bool = True) -> list[str]:
    """
    Top N USDT perpetuals by 24h quote volume, fetched fresh on every call -
    which coins are liquid enough to scan genuinely shifts day to day, so
    this is recomputed each run rather than read from a static list.
    One request returns every symbol's volume at once, so this costs a
    single extra API call regardless of N.

    Only ever ranks against the real global Binance (never Binance.US -
    see _endpoint_chain). Returns an empty list if binance.com is
    unreachable; the caller falls back to the static pairs list rather
    than getting a silently wrong set of coins.
    """
    valid = set(get_all_usdt_pairs(futures))
    if not valid:
        return []
    ticker = None
    for url in _endpoint_chain(futures, "ticker/24hr", include_us=False):
        resp = requests.get(url, timeout=15)
        if resp.status_code == 451:
            continue
        resp.raise_for_status()
        ticker = resp.json()
        break
    if not ticker:
        return []

    rows = [t for t in ticker if t["symbol"] in valid
            and t["symbol"][:-4] not in STABLE_SYMBOLS]
    rows.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in rows[:n]]


def get_top_pairs_by_volume_lightweight(n: int = 100, universe: set[str] | None = None,
                                        futures: bool = True) -> list[str]:
    """
    Same ranking as get_top_pairs_by_volume, but ranks against a pre-built
    static universe (crypto_universe.json) instead of a live exchangeInfo
    call. This exists because binance.com hard-blocks GitHub Actions' IPs
    entirely (see module docstring) - exchangeInfo AND ticker/24hr are both
    binance.com-only by design (never Binance.US - a discovery source needs
    the real global catalog, not a smaller/different one used only as a
    last-resort klines fallback for symbols already known to scan). With no
    working fallback, get_all_usdt_pairs()/get_top_pairs_by_volume() return
    [] on every single GitHub Actions run - confirmed via scan_health.json
    silently reporting total_pairs: 0 for 24+ hours before this was found.

    universe should be refreshed periodically from a network that can
    actually reach binance.com (see scripts/refresh_universe.py) - it goes
    stale as new pairs list/delist, but that's a much smaller problem than
    scanning zero pairs every run. Still attempts a live ticker/24hr call
    (cheap, no live discovery dependency) for genuinely current volume
    ranking; only the "what symbols exist" part is static. Returns []
    if even this ticker call is blocked, same contract as the non-lightweight
    version - the caller's existing static-pairs-list fallback still applies.
    """
    if not universe:
        return []
    ticker = None
    for url in _endpoint_chain(futures, "ticker/24hr", include_us=False):
        try:
            resp = requests.get(url, timeout=15)
        except requests.exceptions.RequestException:
            continue
        if resp.status_code == 451:
            continue
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            continue
        ticker = resp.json()
        break
    if not ticker:
        return []

    rows = [t for t in ticker if t["symbol"] in universe
            and t["symbol"][:-4] not in STABLE_SYMBOLS]
    rows.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return [t["symbol"] for t in rows[:n]]


def get_current_price(symbol: str, futures: bool = True, use_proxy: bool = False) -> float | None:
    """
    Live current price - separate from get_klines, which always drops the
    still-forming candle (correct for pattern detection, but its "close"
    can be up to a full candle-period stale - up to 24h on a 1d timeframe).
    Alerts should quote this for the entry/display price, not the closed
    candle's close, so the number shown actually matches the market.

    use_proxy=True: see get_klines' docstring - same rationale, tries the
    Cloudflare Worker proxy first for long-tail symbols before falling
    through to the normal chain.
    """
    if use_proxy:
        resp = _proxy_get("/fapi/v1/ticker/price", {"symbol": symbol}, timeout=10)
        if resp is not None and resp.ok:
            return float(resp.json()["price"])

    for url in _endpoint_chain(futures, "ticker/price"):
        resp = requests.get(url, params={"symbol": symbol}, timeout=10)
        if resp.status_code == 451:
            continue
        if not resp.ok:
            return None
        return float(resp.json()["price"])
    return None


def get_funding_rate(symbol: str, limit: int = 100) -> pd.DataFrame | None:
    """
    Historical funding rate (perpetual futures only - no spot equivalent,
    so no Binance.US fallback exists for this). Extreme funding readings
    reflect how one-sided leveraged positioning currently is - the closest
    free, exchange-native equivalent to what a service like Coinglass
    surfaces. Returns a DataFrame with funding_time/funding_rate, or None
    on failure.
    """
    try:
        resp = requests.get(f"{FUTURES_URL}/fapi/v1/fundingRate",
                            params={"symbol": symbol, "limit": limit}, timeout=15)
        if not resp.ok:
            return None
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"])
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        return df
    except requests.RequestException:
        return None


def get_open_interest_hist(symbol: str, period: str = "4h", limit: int = 180) -> pd.DataFrame | None:
    """
    Historical open interest - only ~30 days retained by Binance regardless
    of limit requested (a hard exchange-side limit, not something this
    function controls), unlike kline history which goes back much further.
    period must be one of Binance's supported bucket sizes: 5m/15m/30m/1h/2h/4h/6h/12h/1d.
    """
    try:
        resp = requests.get(f"{FUTURES_URL}/futures/data/openInterestHist",
                            params={"symbol": symbol, "period": period, "limit": limit}, timeout=15)
        if not resp.ok:
            return None
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except requests.RequestException:
        return None


def get_long_short_ratio(symbol: str, period: str = "4h", limit: int = 180,
                         top_traders: bool = False) -> pd.DataFrame | None:
    """
    Long/short account ratio - global (all accounts) by default, or top-
    trader accounts only when top_traders=True (Binance's own "smart
    money" cohort - the accounts with the largest positions). Same ~30 day
    retention as open interest history.
    """
    endpoint = "topLongShortAccountRatio" if top_traders else "globalLongShortAccountRatio"
    try:
        resp = requests.get(f"{FUTURES_URL}/futures/data/{endpoint}",
                            params={"symbol": symbol, "period": period, "limit": limit}, timeout=15)
        if not resp.ok:
            return None
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data)
        df["longShortRatio"] = pd.to_numeric(df["longShortRatio"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except requests.RequestException:
        return None


def _klines_response_to_df(data: list) -> pd.DataFrame | None:
    if not data:
        return None
    df = pd.DataFrame(data, columns=KLINE_COLUMNS)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    # Drop the still-forming candle: only closed candles are valid for signals
    return df.iloc[:-1].reset_index(drop=True)


def get_klines(symbol: str, interval: str, limit: int = 300,
               futures: bool = True, max_retries: int = 3,
               use_proxy: bool = False) -> pd.DataFrame | None:
    """
    Fetch OHLCV candles for a symbol/interval.
    Tries the requested endpoint first (futures by default); on a 451
    (geo-blocked) response, moves to the next endpoint in the chain
    immediately instead of wasting retries on a domain that will never
    succeed for this IP. See module docstring for the fallback chain.
    Returns a DataFrame with numeric columns, or None on failure.

    use_proxy=True tries the Cloudflare Worker proxy (BINANCE_PROXY_URL)
    FIRST, before the normal chain - for long-tail symbols not on
    Binance.US, the direct chain will always 451/fail anyway, so trying
    the proxy first avoids wasting those requests. Falls through to the
    normal chain if the proxy is unconfigured or fails, so a proxy outage
    degrades gracefully rather than losing the symbol entirely.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    if use_proxy:
        resp = _proxy_get("/fapi/v1/klines", params, timeout=15)
        if resp is not None and resp.ok:
            df = _klines_response_to_df(resp.json())
            if df is not None:
                return df

    for url in _endpoint_chain(futures, "klines"):
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code in (451, 400):
                    # 451 = geo-blocked here; 400 = this symbol isn't valid
                    # on this specific market (e.g. spot-only, no futures
                    # contract) - both are permanent for this URL, retrying
                    # won't help, move straight to the next endpoint instead
                    # of wasting 3 retries with backoff on a dead end.
                    break
                if resp.status_code == 429:  # rate limited - back off
                    wait = int(resp.headers.get("Retry-After", 5))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return _klines_response_to_df(resp.json())

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"  [data] {symbol} {interval} failed on {url}: {e}")
                    break
                time.sleep(2 ** attempt)
    return None

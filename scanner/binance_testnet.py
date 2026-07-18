"""
Binance Futures TESTNET client - fake-money order placement and position
monitoring against testnet.binancefuture.com, a separate sandboxed
exchange from real Binance (different account, different API keys,
different base URL). Mirrors the real Futures API's request shape, so
this is genuine exchange-order mechanics (fills, rejections, position
lifecycle), not a local math simulation like paper_trading.py.

Auth: every account/trading endpoint needs the API key in the
X-MBX-APIKEY header plus a `signature` query param - HMAC-SHA256 of the
full query string, keyed by the API secret. Credentials are read ONLY
from environment variables (BINANCE_TESTNET_API_KEY /
BINANCE_TESTNET_API_SECRET) - never hardcoded, never logged, never
written to any file this module touches.

IMPORTANT (see the conversation this came from): this module only
places/queries orders - it does not decide WHEN to check for stop/target
hits. That decision loop needs to run on an always-on process, not
GitHub Actions' scheduled cron, which has been confirmed (2026-07-18) to
run on an effective ~55-90 minute cadence regardless of configured
interval - unsuitable for timely stop-loss management even on testnet
fake money, since the whole point of testnet numbers is that they
reflect what the real strategy would actually do.
"""

import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

import requests

BASE_URL = "https://testnet.binancefuture.com"

API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET")


def _sign(params: dict) -> str:
    query = urlencode(params)
    return hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()


def _signed_request(method: str, path: str, params: dict | None = None, timeout: int = 10) -> dict:
    if not API_KEY or not API_SECRET:
        raise RuntimeError("BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET not set in environment")
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    params["signature"] = _sign(params)
    headers = {"X-MBX-APIKEY": API_KEY}
    resp = requests.request(method, f"{BASE_URL}{path}", params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def ping() -> bool:
    """Unsigned connectivity check - no API key needed, just confirms the testnet host is reachable."""
    try:
        resp = requests.get(f"{BASE_URL}/fapi/v1/ping", timeout=10)
        return resp.ok
    except requests.exceptions.RequestException:
        return False


def get_account_info() -> dict:
    """Signed request - proves the API key/secret pair actually authenticates. Returns balance, positions, permissions."""
    return _signed_request("GET", "/fapi/v2/account")


def get_balance() -> list[dict]:
    return _signed_request("GET", "/fapi/v2/balance")

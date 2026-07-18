"""
Binance Multi-Timeframe Pattern Scanner
========================================
Run:  python main.py
No API keys needed - public market data only.

Output: console report + signals_output.json
(the JSON is what you later feed to Claude/Gemini for synthesis,
or to a Telegram notifier.)
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Windows defaults stdout/stderr to the console codepage (cp1252), which
# can't encode some Unicode characters (real crash: an 18-minute, 530-pair
# scan lost all its output to a buffered UnicodeEncodeError near the end,
# almost certainly from a coin name or pattern detail somewhere in the much
# larger symbol set). Force UTF-8 explicitly so this can't recur regardless
# of what characters show up - GitHub Actions (Ubuntu) already defaults to
# UTF-8, so this only changes behavior on Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from scanner.data import (get_klines, get_all_usdt_pairs, get_top_pairs_by_volume,
                          get_top_pairs_by_volume_lightweight, get_current_price, get_funding_rate,
                          get_proxy_stats, get_rotating_pair_slice)
from scanner.indicators import enrich
# Jayantha strategy (B2B pullback entry + confirmation-before-conviction)
# replaces the old generic multi-pattern detector set as the live signal
# source - see scanner/jayantha_detectors.py for why this is a drop-in
# replacement that needs no other changes anywhere downstream.
from scanner.jayantha_detectors import run_jayantha_detectors
# Ashen strategy (a second, independent trader's rules) runs ALONGSIDE
# Jayantha's, not instead of it - see scanner/ashen_detectors.py.
from scanner.ashen_detectors import run_ashen_detectors
from scanner.mtf import annotate_htf
from scanner.notify import notify_report
from scanner.risk import attach_atr_risk, setup_risk_plans, classify_funding
from scanner.journal import log_signals, detector_recent_form, mark_notified, detector_expectancy, detector_avg_return, detector_reliability
from scanner.regime import regime_label

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
BACKTEST_RESULTS_PATH = Path(__file__).parent / "backtest_results.json"
REALISTIC_BACKTEST_PATH = Path(__file__).parent / "realistic_backtest_results.json"
CRYPTO_UNIVERSE_PATH = Path(__file__).parent / "crypto_universe.json"


def load_realistic_backtest_expectancy(min_n: int) -> dict[tuple[str, str], tuple[float, int]]:
    """
    Real expectancy (in R) per (detector, direction) from
    realistic_backtest.py's historical simulation - the SAME stop/target-
    respecting methodology as the live journal (see
    journal.detector_expectancy), just at much larger scale (thousands of
    simulated trades vs. the live journal's slower-accumulating real ones).
    Returns (expectancy, n) - see detector_reliability_verdict for why the
    sample size matters here, not just the expectancy value.

    Expectancy, not raw win rate, is what actually determines whether a
    detector makes or loses money - a structural pattern with a wide,
    geometry-based target only needs a low win rate to be profitable (e.g.
    ascending_triangle/bullish: 29.3% win rate against a 14.96:1 average
    R:R is comfortably profitable, needing only ~6.3% to break even), while
    a 2:1 generic ATR-based detector needs ~33.3%. Comparing every detector
    against one flat win-rate bar would blacklist the system's best
    performers alongside its worst - this is what a flat threshold
    literally did on the first pass here before being caught and fixed.
    """
    if not REALISTIC_BACKTEST_PATH.exists():
        return {}
    summary = json.loads(REALISTIC_BACKTEST_PATH.read_text()).get("summary", {})
    out = {}
    for key, entry in summary.items():
        if entry["n"] < min_n:
            continue
        detector, _, direction = key.rpartition("/")
        out[(detector, direction)] = (entry["win_rate"] * (1 + entry["avg_risk_reward"]) - 1, entry["n"])
    return out


def _pool_by_sample_size(backtest: dict[tuple[str, str], tuple[float, int]],
                         live: dict[tuple[str, str], tuple[float, int]]
                         ) -> dict[tuple[str, str], float]:
    """
    Weighted-by-n pool of two (value, n) sources, instead of letting the
    live one simply override the backtest the moment it crosses min_n. A
    live-only override is wrong here: min_n=10 live trades is nowhere near
    enough to overturn a 300+ trade historical simulation, but that's
    exactly what a naive override does - this is what let
    volume_spike/bullish (backtest: -0.265R over 315 trades) get
    UN-blacklisted by just 10-15 live trades that happened to average
    +0.38R, pure small-sample noise given how far it diverges from the
    much larger sample. Weighting by n means a detector's verdict only
    flips once the live evidence is substantial enough to actually matter.
    Used for both expectancy (the blacklist) and win rate (the probability
    shown in each alert).
    """
    out = {}
    for key in set(backtest) | set(live):
        bt_val, bt_n = backtest.get(key, (0.0, 0))
        live_val, live_n = live.get(key, (0.0, 0))
        total_n = bt_n + live_n
        out[key] = (bt_val * bt_n + live_val * live_n) / total_n if total_n else 0.0
    return out


def combined_detector_expectancy(backtest: dict[tuple[str, str], tuple[float, int]],
                                 live: dict[tuple[str, str], tuple[float, int]]
                                 ) -> dict[tuple[str, str], float]:
    return _pool_by_sample_size(backtest, live)


def load_realistic_backtest_win_rate(min_n: int) -> dict[tuple[str, str], tuple[float, int]]:
    """Win rate (not expectancy) per (detector, direction) from realistic_backtest.py, for display."""
    if not REALISTIC_BACKTEST_PATH.exists():
        return {}
    summary = json.loads(REALISTIC_BACKTEST_PATH.read_text()).get("summary", {})
    out = {}
    for key, entry in summary.items():
        if entry["n"] < min_n:
            continue
        detector, _, direction = key.rpartition("/")
        out[(detector, direction)] = (entry["win_rate"], entry["n"])
    return out


def combined_detector_win_rate(backtest: dict[tuple[str, str], tuple[float, int]],
                               live: dict[tuple[str, str], tuple[float, int]]
                               ) -> dict[tuple[str, str], float]:
    return _pool_by_sample_size(backtest, live)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_detector_weights(cfg: dict) -> dict:
    """
    Turn backtest_results.json into a {(detector, direction): weight} map so
    strength reflects each detector's actual backtested edge instead of
    counting every signal equally. weight = 1.0 + scale * (detector's win
    rate - baseline win rate for that direction), clamped to a sane range.
    Falls back to an empty map (neutral weight 1.0 everywhere) if the
    backtest hasn't been run yet or weighting is turned off.
    """
    conf_cfg = cfg.get("confluence", {})
    if not conf_cfg.get("use_backtest_weights", True) or not BACKTEST_RESULTS_PATH.exists():
        return {}
    summary = json.loads(BACKTEST_RESULTS_PATH.read_text()).get("summary", {})
    horizon = conf_cfg.get("horizon", "h10")
    min_signals = conf_cfg.get("min_signals", 30)
    scale = conf_cfg.get("weight_scale", 5.0)
    lo, hi = conf_cfg.get("weight_clamp", [0.2, 2.5])

    baseline = {}
    for direction in ("bullish", "bearish"):
        entry = summary.get(f"ALL/{direction}", {})
        if horizon in entry:
            baseline[direction] = entry[horizon]["win_rate"]

    weights = {}
    for key, entry in summary.items():
        if key.startswith("ALL/") or horizon not in entry:
            continue
        detector, _, direction = key.rpartition("/")
        if direction not in baseline or entry["signals"] < min_signals:
            continue
        edge = entry[horizon]["win_rate"] - baseline[direction]
        weights[(detector, direction)] = max(lo, min(hi, 1.0 + scale * edge))
    return weights


def confluence_score(signals: list[dict], weights: dict | None = None) -> tuple[str, float]:
    """Net directional bias and strength from a signal list, weighted by backtested edge."""
    weights = weights or {}
    bull = sum(weights.get((s["name"], "bullish"), 1.0) for s in signals if s["direction"] == "bullish")
    bear = sum(weights.get((s["name"], "bearish"), 1.0) for s in signals if s["direction"] == "bearish")
    if bull > bear:
        return "bullish", round(bull, 2)
    if bear > bull:
        return "bearish", round(bear, 2)
    return "mixed", round(max(bull, bear), 2)


def get_market_trend(symbol: str, timeframes: list[str], cfg: dict) -> dict[str, bool | None]:
    """
    Current trend (close > EMA20) per timeframe for a market-leader symbol
    (BTC/ETH) - True=bullish, False=bearish, None if data unavailable.
    Computed once per run and shared across every pair's scan (see
    risk.setup_risk_plan's market_disagrees param for why this matters).
    """
    trend = {}
    for tf in timeframes:
        df = get_klines(symbol, tf, cfg["candle_limit"])
        if df is None or len(df) < 25:
            trend[tf] = None
            continue
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        trend[tf] = bool(df["close"].iloc[-1] > ema20.iloc[-1])
    return trend


def scan_pair(symbol: str, timeframes: list[str], cfg: dict, weights: dict,
             avg_returns: dict | None = None, unreliable: set | None = None,
             btc_trend: dict | None = None, eth_trend: dict | None = None,
             win_rates: dict | None = None, use_proxy: bool = False) -> dict:
    win_rates = win_rates or {}
    result = {"symbol": symbol, "timeframes": {}, "errors": []}
    # One live-price call shared across all timeframes for this symbol - the
    # closed candle's close can be up to a full candle-period stale (up to
    # 24h on 1d), so alerts quote this instead for entry/display, while
    # detection still correctly uses only closed candles (no lookahead).
    live_price = get_current_price(symbol, use_proxy=use_proxy)
    is_market_leader = symbol in ("BTCUSDT", "ETHUSDT")
    # Funding rate has no Binance.US equivalent (perpetuals-only, no spot
    # market) and no proxy route either - skipped entirely for long-tail
    # (use_proxy) symbols to stay inside the proxy's free-tier request
    # budget. classify_funding(None, ...) already returns None here, which
    # main.py's `!= "against_crowd"` check treats as funding_ok=True - the
    # same as if funding were checked and came back neutral, not a special
    # case needing extra code.
    current_funding = None
    if not use_proxy:
        fr_df = get_funding_rate(symbol, limit=1)
        current_funding = float(fr_df["fundingRate"].iloc[-1]) if fr_df is not None and not fr_df.empty else None

    # Fetched into a dict first (rather than processed inline per timeframe,
    # as before Ashen's vwap_breakout was added) so that detector can look
    # up its paired higher timeframe's data - per config's
    # ashen.vwap_breakout.htf_pairing, that HTF is always itself one of this
    # same `timeframes` list, so this avoids a duplicate fetch/API call for
    # it. time.sleep(0.15) stays here, at the actual fetch step, not moved
    # to the detection loop below (which makes no API calls at all).
    dfs: dict[str, "pd.DataFrame"] = {}
    for tf in timeframes:
        raw = get_klines(symbol, tf, cfg["candle_limit"], use_proxy=use_proxy)
        if raw is None or len(raw) < 60:
            if raw is None:
                print(f"  [skip] {symbol} {tf}: no data (bad symbol or API error)")
                result["errors"].append({"timeframe": tf, "reason": "no_data"})
            time.sleep(0.15)  # be polite to the API
            continue
        dfs[tf] = enrich(raw, cfg)
        time.sleep(0.15)  # be polite to the API

    htf_pairing = cfg.get("ashen", {}).get("vwap_breakout", {}).get("htf_pairing", {})
    for tf in timeframes:
        if tf not in dfs:
            continue
        df = dfs[tf]
        htf_df = dfs.get(htf_pairing.get(tf))
        signals = run_jayantha_detectors(df, cfg) + run_ashen_detectors(df, cfg, htf_df)
        if signals:
            close = live_price if live_price is not None else float(df["close"].iloc[-1])
            atr = float(df["atr"].iloc[-1])
            risk_cfg = cfg.get("risk", {})
            # A flat float unless config has min_risk_reward_overrides set,
            # in which case _resolve_min_rr (risk.py) looks up a
            # strategy-specific floor - added for marubozu_ashen, whose
            # deliberately smaller target (see ashen_marubozu.py's
            # reward_risk_ratio) would otherwise fail the same 1.0 bar
            # sized for the other four strategies' bigger targets.
            min_rr_overrides = risk_cfg.get("min_risk_reward_overrides")
            min_risk_reward = ({**min_rr_overrides, "default": risk_cfg.get("min_risk_reward", 1.0)}
                               if min_rr_overrides else risk_cfg.get("min_risk_reward", 1.0))
            signals = attach_atr_risk(signals, close, atr,
                                      risk_cfg.get("atr_multiplier", 1.5),
                                      risk_cfg.get("reward_risk_ratio", 2.0),
                                      risk_cfg.get("max_stop_pct"))
            # bias/strength are still computed as a descriptive summary of
            # the raw signal mix on this timeframe (used for console context
            # and as the HTF-agreement reference other timeframes' plans get
            # checked against - see mtf.annotate_htf) - no longer used to
            # gate which signals become tracked trades, since each of the
            # five live strategies is now logged independently whenever IT
            # fires (see setup_risk_plans below), not only when the combined
            # signal mix clears a shared confluence bar.
            bias, strength = confluence_score(signals, weights)
            # Regime is descriptive context, not a hard filter - a choppy
            # reading doesn't block an alert, but tells you to size down /
            # be more skeptical of continuation-style signals right now.
            regime = regime_label(df["close"].values, len(df) - 1)
            # BTC/ETH trend agreeing with THIS trade's direction means the
            # setup is more likely just beta (see risk.py's market_disagrees
            # docstring) - BTC/ETH themselves are the market leaders, so the
            # filter doesn't apply to trading them against themselves.
            # Evaluated per direction (not once for a single shared bias)
            # since independent strategies can now fire bullish AND bearish
            # on the same symbol/timeframe - each needs its own verdict.
            btc_bull = None if is_market_leader else (btc_trend or {}).get(tf)
            eth_bull = None if is_market_leader else (eth_trend or {}).get(tf)
            market_disagrees_by_direction = {}
            funding_ok_by_direction = {}
            for direction in ("bullish", "bearish"):
                if is_market_leader:
                    market_disagrees_by_direction[direction] = True
                elif btc_bull is None or eth_bull is None:
                    market_disagrees_by_direction[direction] = None
                else:
                    trade_is_bullish = direction == "bullish"
                    market_disagrees_by_direction[direction] = (
                        (btc_bull != trade_is_bullish) and (eth_bull != trade_is_bullish))
                # Confirmed via funding_rate_backtest.py (8000+ trades): trading
                # AGAINST an extreme funding reading loses money here - the
                # opposite of the classic "fade the crowd" assumption - so only
                # that specific bucket is excluded (see risk.classify_funding).
                funding_ok_by_direction[direction] = classify_funding(current_funding, direction) != "against_crowd"
            risk_plans = setup_risk_plans(signals, close, min_risk_reward,
                                          avg_returns, risk_cfg.get("min_calibrated_move_pct", 0.3),
                                          risk_cfg.get("account_size"), risk_cfg.get("account_risk_pct", 1.0),
                                          unreliable, market_disagrees_by_direction, funding_ok_by_direction,
                                          risk_cfg.get("target_fraction", 1.0),
                                          risk_cfg.get("min_stop_pct", 0.5))
            for plan in risk_plans:
                plan["recent_form"] = detector_recent_form(plan["based_on"], plan["direction"],
                                                           cfg.get("journal", {}).get("form_lookback", 5))
                # Real win probability for this exact detector/direction -
                # pooled from backtest + live data the same way the
                # blacklist is, not a black-box replica of any paid tool.
                plan["win_probability"] = win_rates.get((plan["based_on"], plan["direction"]))
            result["timeframes"][tf] = {
                "close": close,
                "bias": bias,
                "strength": strength,
                "regime": regime,
                "signals": signals,
                "risk_plans": risk_plans,
            }
    return result


def main():
    cfg = load_config()
    used_pair_discovery_fallback = False
    pair_discovery_method = "static"
    pair_rotation_slot = None
    pair_rotation_total_slots = None
    if cfg["scan_all"]:
        pairs = get_all_usdt_pairs()
        pair_discovery_method = "full_exchange_info"
        if not pairs:
            # get_all_usdt_pairs() returns [] on purpose when binance.com is
            # briefly unreachable (see its docstring - never silently
            # substitutes a different exchange's symbol catalog). Without
            # this fallback, that produced a run that "succeeded" while
            # scanning zero pairs - invisible before scan_health.json
            # started reporting total_pairs, which is how this was found.
            print("[warn] could not fetch full USDT pair list this run - falling back to static pairs list")
            pairs = cfg["pairs"]
            used_pair_discovery_fallback = True
            pair_discovery_method = "static_fallback"
    elif cfg.get("pairs_mode") == "top_volume":
        # Refetched fresh on every run (not cached/daily) - which coins are
        # liquid enough to scan shifts constantly, so this must be as
        # current as the scan itself, not a snapshot from hours/days ago.
        pairs = get_top_pairs_by_volume(cfg.get("top_n_pairs", 100))
        pair_discovery_method = "top_volume"
        if not pairs:
            print("[warn] could not fetch top-volume pairs this run - falling back to static pairs list")
            pairs = cfg["pairs"]
            used_pair_discovery_fallback = True
            pair_discovery_method = "static_fallback"
    elif cfg.get("pairs_mode") == "top_volume_lightweight":
        # binance.com hard-blocks GitHub Actions' IPs entirely (see
        # scanner/data.py module docstring) - exchangeInfo has no working
        # fallback for GH Actions (Binance.US isn't safe to use for
        # discovery, only for klines on symbols already known), so
        # scan_all/top_volume return [] on every single GH Actions run.
        # This mode ranks a live ticker/24hr call (cheap, still attempted)
        # against a pre-built static universe (crypto_universe.json,
        # refreshed periodically from a network that can reach binance.com
        # - see scripts/refresh_universe.py) instead of requiring the
        # blocked discovery call at all.
        universe = None
        if CRYPTO_UNIVERSE_PATH.exists():
            universe = set(json.loads(CRYPTO_UNIVERSE_PATH.read_text()))
        pairs = get_top_pairs_by_volume_lightweight(cfg.get("top_n_pairs", 100), universe)
        pair_discovery_method = "top_volume_lightweight"
        if not pairs:
            # Live ranking is blocked on every GH Actions run in practice
            # (confirmed via scan_health.json: used_pair_discovery_fallback
            # true on every one of the last 15+ runs) - falling back to the
            # SAME static cfg["pairs"] list every time meant ~371 of
            # crypto_universe.json's ~528 real pairs were never scanned at
            # all, silently, for as long as this has been running. Rotates
            # a disjoint slice of the FULL universe instead (see
            # get_rotating_pair_slice's docstring) - most of those ~371 are
            # genuinely not on Binance.US and will still show no_data from
            # GitHub Actions regardless of rotation (a real exchange-
            # coverage limit, not a scheduling one), but this at minimum
            # auto-discovers any pairs Binance.US adds later without a
            # manual crypto_universe.json/pairs: refresh, and spreads scan
            # attempts across the whole universe instead of hammering one
            # fixed subset every single run.
            full_universe = sorted(universe) if universe else cfg["pairs"]
            rotation_cfg = cfg.get("pair_rotation", {})
            pairs, slot, num_slices = get_rotating_pair_slice(
                full_universe,
                slice_size=rotation_cfg.get("slice_size", cfg.get("top_n_pairs", 157)),
                interval_minutes=rotation_cfg.get("interval_minutes", 5),
            )
            print(f"[warn] could not fetch live volume ranking this run - "
                  f"rotating slice {slot + 1}/{num_slices} of the full universe instead")
            used_pair_discovery_fallback = True
            pair_discovery_method = "static_fallback_rotating"
            pair_rotation_slot = slot
            pair_rotation_total_slots = num_slices
    else:
        pairs = cfg["pairs"]

    # Long-tail: the ~371 real Binance pairs not on Binance.US, unreachable
    # from GitHub Actions via the normal fallback chain at all (see
    # scanner/data.py module docstring). Routed through BINANCE_PROXY_URL
    # (a Cloudflare Worker relay) instead, on a reduced timeframe set to
    # stay inside the proxy's free-tier request budget (~80k/day at 20-min
    # cadence with 2 timeframes x ~371 pairs, vs. Cloudflare's 100k/day cap
    # - see .github/workflows/scan.yml for where BINANCE_PROXY_URL is set).
    # No-op (empty list) if crypto_universe.json is missing or long_tail is
    # disabled - existing `pairs` coverage is unaffected either way.
    long_tail_cfg = cfg.get("long_tail", {})
    long_tail_pairs = []
    long_tail_timeframes = long_tail_cfg.get("timeframes", ["4h", "1d"])
    if long_tail_cfg.get("enabled") and CRYPTO_UNIVERSE_PATH.exists():
        full_universe = set(json.loads(CRYPTO_UNIVERSE_PATH.read_text()))
        long_tail_pairs = sorted(full_universe - set(pairs))

    timeframes = cfg["timeframes"]
    weights = load_detector_weights(cfg)
    risk_cfg = cfg.get("risk", {})
    # Detectors proven to lose money at the actual stop/target sizing used
    # here - a much harder, more honest bar than backtest_results.json's
    # plain forward-return check (see risk.py's setup_risk_plan docstring).
    # Two sources, pooled by sample size (not a live-overrides-backtest
    # rule - see combined_detector_expectancy's docstring for why that was
    # wrong): the large-scale historical simulation (realistic_backtest.py,
    # thousands of trades) supplies most of the weight until the live
    # journal has accumulated enough real trades of its own to meaningfully
    # shift the verdict. Either way, refused as a trade basis entirely, no
    # matter how tight the stop looks.
    min_n = risk_cfg.get("min_reliable_n", 10)
    min_expectancy = risk_cfg.get("min_detector_expectancy", 0.0)
    expectancy = combined_detector_expectancy(load_realistic_backtest_expectancy(min_n), detector_expectancy(min_n))
    unreliable = {k for k, exp in expectancy.items() if exp < min_expectancy}
    # Real win probability per detector/direction, pooled the same way, for
    # display in each alert - not a black-box replica of any paid signal
    # tool, just our own actually-measured numbers surfaced honestly.
    win_rates = combined_detector_win_rate(load_realistic_backtest_win_rate(min_n), detector_reliability(min_n))
    # Target calibration also uses the live journal (real, stop-respecting
    # outcome_pct), not backtest_results.json's naive forward return - same
    # reason as the blacklist above, and same min_n gate so a target can't
    # get calibrated off a handful of early results either.
    avg_returns = detector_avg_return(min_n)
    # BTC/ETH's own current trend per timeframe, fetched once and shared
    # across every pair's scan - see risk.setup_risk_plan's market_disagrees
    # docstring for why an altcoin setup needs to disagree with both to count.
    btc_trend = get_market_trend("BTCUSDT", timeframes, cfg)
    eth_trend = get_market_trend("ETHUSDT", timeframes, cfg)

    total_pair_count = len(pairs) + len(long_tail_pairs)
    print(f"Scanning {len(pairs)} pairs x {len(timeframes)} timeframes"
          + (f", plus {len(long_tail_pairs)} long-tail pairs x {len(long_tail_timeframes)} timeframes via proxy"
             if long_tail_pairs else "") + "...")
    if weights:
        print(f"(strength weighted by backtested edge - {len(weights)} detector/direction weights loaded)\n")
    else:
        print("(no backtest_results.json found - strength is unweighted; run backtest.py to enable weighting)\n")
    if unreliable:
        print(f"(blacklisted as trade basis - negative expectancy at their own R:R: "
              f"{', '.join(f'{n}/{d} ({expectancy[(n,d)]:+.2f}R)' for n, d in sorted(unreliable))})\n")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "results": []}

    # Pairs are scanned concurrently (each pair's own timeframes are still
    # fetched sequentially inside scan_pair) - this is what makes a 100-pair
    # x 9-timeframe run fit inside a 15-minute schedule instead of taking
    # ~13 minutes serially. get_klines already retries with backoff on 429s,
    # so a burst of concurrent requests self-throttles rather than failing.
    concurrency = cfg.get("scan_concurrency", 8)

    # (symbol, timeframes-to-scan, use_proxy) - fast pairs get the full
    # config timeframe list direct/via-Binance.US as before; long-tail
    # pairs get the reduced timeframe list via the proxy (see long_tail_cfg
    # above). btc_trend/eth_trend already cover every config timeframe
    # (fetched with the full `timeframes` list above), so annotate_htf and
    # market_disagrees work correctly for long-tail pairs' 4h/1d too,
    # without needing a separate fetch.
    scan_jobs = [(s, timeframes, False) for s in pairs] + \
                [(s, long_tail_timeframes, True) for s in long_tail_pairs]

    def _scan(job):
        symbol, tfs, use_proxy = job
        try:
            return symbol, scan_pair(symbol, tfs, cfg, weights, avg_returns, unreliable,
                                     btc_trend, eth_trend, win_rates, use_proxy=use_proxy)
        except Exception as e:
            print(f"  [error] {symbol}: {e}")
            return symbol, None

    # Per-symbol health, captured independently of whether the symbol
    # produced any tradeable setup - a symbol failing on every timeframe
    # (e.g. delisted/bad ticker) previously vanished silently here (the
    # `not res["timeframes"]` skip below), so its errors never reached
    # anyone. This is what the PWA dashboard's "Pair Health Status" panel
    # reads (scan_health.json), so it needs every symbol's outcome, not
    # just the ones that happened to produce a signal.
    pair_health: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_scan, job) for job in scan_jobs]
        for future in as_completed(futures):
            symbol, res = future.result()
            if res is None:
                pair_health[symbol] = [{"timeframe": "*", "reason": "exception"}]
                continue
            if res.get("errors"):
                pair_health[symbol] = res["errors"]
            if not res["timeframes"]:
                continue
            res = annotate_htf(res, timeframes)
            res["max_strength"] = max(d["strength"] for d in res["timeframes"].values())
            report["results"].append(res)

            # console output: one block per independently-qualifying risk
            # plan, not gated by the tf's combined strength - each of the
            # five live strategies is shown whenever IT produced a plan.
            for tf, data in res["timeframes"].items():
                for r in data["risk_plans"]:
                    if cfg.get("mtf", {}).get("require_agreement", False) and not r["htf_agrees"]:
                        continue
                    print(f"{symbol} [{tf}]  {r['direction'].upper()} close={data['close']:.6g}  "
                          f"[HTF: {r['htf_note']}]  [regime: {data['regime']}]")
                    for s in data["signals"]:
                        if s["name"] == r["based_on"] or s["direction"] == r["direction"]:
                            print(f"    - {s['name']}: {s['detail']}")
                    rr = f"{r['risk_reward']}:1" if r["risk_reward"] else "n/a"
                    print(f"    risk: entry={r['entry']:.6g} stop={r['stop']:.6g} "
                          f"target={r['target']:.6g} (R:R {rr}, based on {r['based_on']}, "
                          f"target: {r['target_basis']})")
                    if r.get("position"):
                        p = r["position"]
                        print(f"    position: risk {p['account_risk_pct']}% (${p['dollar_risk']}) "
                              f"-> {p['units']:g} units (~${p['position_value']})")
                    if r.get("recent_form"):
                        f = r["recent_form"]
                        print(f"    recent form for {r['based_on']}/{r['direction']}: "
                              f"{f['wins']}W-{f['losses']}L (last {f['n']})")
                    print()

    report["results"].sort(key=lambda r: r["max_strength"], reverse=True)

    if cfg["output"]["mode"] in ("json", "both"):
        out = Path(__file__).parent / cfg["output"]["json_path"]
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nFull report written to {out}")

    # Overwritten every run (not append-only like journal.jsonl) - this is
    # current pair health, not a history, so a symbol that comes back
    # healthy next run should stop showing as errored. Public/token-free so
    # the PWA dashboard can read it straight off raw.githubusercontent.com.
    health_path = Path(__file__).parent / "scan_health.json"
    health_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_pairs": total_pair_count,
        "fast_pairs": len(pairs),
        "long_tail_pairs": len(long_tail_pairs),
        "pairs_with_errors": len(pair_health),
        # True when full pair discovery (scan_all) failed this run and we fell
        # back to the small static list - distinguishes "everything's fine,
        # few/no errors" from "discovery itself broke, only scanned a handful
        # of pairs", which otherwise look identical in pairs_with_errors alone.
        "used_pair_discovery_fallback": used_pair_discovery_fallback,
        "pair_discovery_method": pair_discovery_method,
        # Which disjoint slice of the full universe this run scanned, and
        # how many slices exist in the current rotation cycle - null when
        # not rotating (live ranking succeeded, or a different pairs_mode
        # is configured). See scanner.data.get_rotating_pair_slice.
        "pair_rotation_slot": pair_rotation_slot,
        "pair_rotation_total_slots": pair_rotation_total_slots,
        # Diagnoses proxy failures that otherwise look identical to
        # ordinary no_data errors - "configured": false means
        # BINANCE_PROXY_URL never reached this run at all (secret not set
        # or workflow env misconfigured); "attempts" vs "successes" and
        # "last_status"/"last_error" show exactly what happened on the
        # wire (e.g. 403 = secret mismatch, network error = Worker/DNS
        # issue) - see scanner.data.get_proxy_stats.
        "proxy_stats": get_proxy_stats(),
        "errors": pair_health,
    }, indent=2))
    if pair_health:
        print(f"\n{len(pair_health)}/{total_pair_count} pairs had errors this run - see scan_health.json")

    new_keys = None
    if cfg.get("journal", {}).get("enabled", True):
        reopen_cooldown = cfg.get("journal", {}).get("reopen_cooldown_hours", 6.0)
        logged = log_signals(report, cfg, reopen_cooldown_hours=reopen_cooldown)
        if logged:
            print(f"Logged {len(logged)} new setup(s) to journal.jsonl")
        # Only these are genuinely new setups this run - notify_report uses
        # this to skip re-alerting a setup that's already open and unchanged.
        new_keys = {(e["symbol"], e["timeframe"], e["based_on"]) for e in logged}

    sent, sent_keys = notify_report(report, cfg, new_keys=new_keys)
    if sent:
        print(f"Sent {sent} Telegram alert(s)")
        mark_notified(sent_keys)


if __name__ == "__main__":
    main()

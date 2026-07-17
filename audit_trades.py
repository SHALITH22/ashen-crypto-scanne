"""
Independent trade audit: for every NEW journal entry since the last audit
checkpoint, re-fetch real historical candles and re-run the ACTUAL detector
functions (the same run_jayantha_detectors/run_ashen_detectors main.py
calls) to confirm the logged direction/stop is genuinely what the
strategy's own rules say for that candle - not just "the code says so."
Catches bugs like the Marubozu stop bug fixed earlier this session (a
wrong reference price silently reaching every future trade) automatically,
from now on.

Scope is deliberately "new entries only, going forward" (not the existing
backlog) - see the checkpoint logic below. This also matches a hard
constraint from scanner.data.get_klines, which has no startTime/endTime
support and can only ever fetch "the most recent N candles up to now" - as
long as this runs reasonably soon after a trade is logged (daily is
comfortably soon enough), that trade's own candle is still well within the
most-recent window fetched here.

Usage: python audit_trades.py
Run on its own schedule, separate from main.py/journal_check.py (see
.github/workflows/audit.yml) - this isn't time-critical, daily is enough.

IMPORTANT for anyone testing this locally: results are only meaningful when
run from the SAME network environment main.py's live scan used for that
entry. GitHub Actions IPs are geo-blocked from binance.com (see
scanner/data.py's module docstring) and fall through get_klines's endpoint
chain to Binance.US SPOT for the ~157 "fast" pairs; a local machine with an
unblocked IP succeeds on binance.com FUTURES on the first try instead - a
genuinely different venue/market type with its own price. Confirmed while
building this: BTCUSDT/ETHUSDT (deep liquidity, near-zero spot/futures
basis) matched cleanly in a local test, while smaller alts (larger,
more volatile basis) showed real-looking but spurious "mismatches" purely
from comparing spot vs. futures prices for the same symbol - not an actual
detector bug. This script is designed to run on GitHub Actions
(.github/workflows/audit.yml), which hits the same endpoint chain scan.yml
already used to log the entry in the first place, so this isn't an issue
in production - just don't trust a local run's PASS/FAIL numbers, only its
plumbing (does it run, does it fetch/enrich/match correctly).
"""

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from scanner.data import get_klines
from scanner.indicators import enrich
from scanner.jayantha_detectors import run_jayantha_detectors
from scanner.ashen_detectors import run_ashen_detectors
from scanner.journal import entries_since

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"
AUDIT_STATE_PATH = Path(__file__).parent / "audit_state.json"
AUDIT_RESULTS_PATH = Path(__file__).parent / "audit_results.jsonl"

# Needs enough trailing history AFTER truncating to the entry's own
# historical candle for the slowest indicator (200-period MA/EMA) to
# compute cleanly, plus margin - comfortably inside Binance's per-request
# cap (1000). main.py's live scan only needs candle_limit=300 since it
# always truncates to "right now"; this needs more headroom since it
# truncates further back, to whenever the entry was actually logged.
AUDIT_KLINE_LIMIT = 700

# Stop is never calibrated in risk.py (calibrate_target only ever touches
# target), so a recomputed stop should match the stored one almost exactly
# if nothing is wrong - this tolerance only absorbs float rounding, not
# real drift. Target is deliberately NOT scored the same way (see
# audit_entry's docstring) since its calibration inputs legitimately
# change over time.
STOP_TOLERANCE_PCT = 0.1


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_audit_state() -> dict:
    if AUDIT_STATE_PATH.exists():
        return json.loads(AUDIT_STATE_PATH.read_text())
    return {"last_audited_logged_at": None}


def save_audit_state(state: dict) -> None:
    AUDIT_STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _truncate_to_time(df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """
    Keep only candles whose close_time is <= cutoff - reconstructs exactly
    what df.iloc[-1] was at the moment the trade was actually logged, not
    today's latest candle. Mirrors journal.check_open_entries's existing
    df[df["open_time"] > logged_at] pattern, just inverted (before/at,
    not after) and on close_time (the entry's own last-closed candle must
    have fully closed by logged_at, not merely opened by then).
    """
    return df[df["close_time"] <= cutoff].reset_index(drop=True)


def _fetch_enriched(symbol: str, timeframe: str, cfg: dict) -> pd.DataFrame | None:
    raw = get_klines(symbol, timeframe, AUDIT_KLINE_LIMIT)
    if raw is None or len(raw) < 60:
        return None
    return enrich(raw, cfg)


def _base_result(entry: dict) -> dict:
    return {
        "symbol": entry["symbol"],
        "timeframe": entry["timeframe"],
        "based_on": entry["based_on"],
        "bias": entry["bias"],
        "logged_at": entry["logged_at"],
        "audited_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "stored_stop": entry.get("stop"),
        "stored_target": entry.get("target"),
        "recomputed_stop": None,
        "recomputed_target": None,
        "status": None,
        "reason": None,
    }


def audit_entry(entry: dict, df: pd.DataFrame | None, htf_df: pd.DataFrame | None, cfg: dict) -> dict:
    """
    Re-run the real detector pipeline against the reconstructed historical
    view and compare against what got logged. Scores direction + stop
    (both should be exactly deterministic given the same candles and
    config - a mismatch here is a genuine bug signal). Target is reported
    but never scored: risk.setup_risk_plans can replace a detector's raw
    geometric target with calibrate_target's historical-average-move
    override, whose inputs (journal.detector_avg_return) legitimately
    drift as more trades resolve - scoring target pass/fail would produce
    false-positive noise as that calibration data changes over time, not
    real bugs.
    """
    result = _base_result(entry)
    if df is None:
        result["status"] = "fail"
        result["reason"] = "could not fetch candles to audit against"
        return result

    logged_at = datetime.fromisoformat(entry["logged_at"])
    df_hist = _truncate_to_time(df, logged_at)
    if len(df_hist) < 60:
        result["status"] = "fail"
        result["reason"] = f"not enough historical candles to reconstruct ({len(df_hist)} available before logged_at)"
        return result

    htf_hist = _truncate_to_time(htf_df, logged_at) if htf_df is not None else None

    signals = run_jayantha_detectors(df_hist, cfg) + run_ashen_detectors(df_hist, cfg, htf_hist)
    match = next((s for s in signals if s["name"] == entry["based_on"] and s["direction"] == entry["bias"]), None)

    if match is None:
        result["status"] = "fail"
        result["reason"] = f"{entry['based_on']} did not fire {entry['bias']} against the reconstructed candles"
        return result

    result["recomputed_stop"] = match.get("stop")
    result["recomputed_target"] = match.get("target")

    stored_stop = entry.get("stop")
    recomputed_stop = match.get("stop")
    if recomputed_stop is None or not stored_stop:
        result["status"] = "fail"
        result["reason"] = "missing stop value to compare"
        return result

    stop_diff_pct = abs(recomputed_stop - stored_stop) / abs(stored_stop) * 100
    if stop_diff_pct > STOP_TOLERANCE_PCT:
        result["status"] = "fail"
        result["reason"] = (f"stop mismatch: stored {stored_stop:.6g} vs recomputed {recomputed_stop:.6g} "
                            f"({stop_diff_pct:.3f}% apart, tolerance {STOP_TOLERANCE_PCT}%)")
        return result

    result["status"] = "pass"
    result["reason"] = "detector fired with matching direction and stop"
    return result


def main():
    cfg = load_config()
    state = load_audit_state()
    checkpoint = state.get("last_audited_logged_at")

    if checkpoint is None:
        # First-ever run: establish the checkpoint at "now" and audit
        # nothing retroactively - deliberate scope, see module docstring.
        existing_count = len(entries_since(None))
        print(f"First run - no prior checkpoint. Establishing checkpoint at now, "
              f"auditing nothing retroactively ({existing_count} existing entries skipped).")
        save_audit_state({"last_audited_logged_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()})
        return

    new_entries = entries_since(checkpoint)
    if not new_entries:
        print(f"No new entries since last audit checkpoint ({checkpoint}).")
        return

    print(f"Auditing {len(new_entries)} new entr{'y' if len(new_entries) == 1 else 'ies'} "
          f"logged since {checkpoint}...")

    htf_pairing = cfg.get("ashen", {}).get("vwap_breakout", {}).get("htf_pairing", {})

    # Fetch/enrich each unique (symbol, timeframe) - and any HTF pairing a
    # vwap_breakout_ashen entry needs - once, shared across every entry
    # that touches it, rather than refetching per-entry.
    needed: dict[str, set[str]] = defaultdict(set)
    for e in new_entries:
        needed[e["symbol"]].add(e["timeframe"])
        if e["based_on"] == "vwap_breakout_ashen":
            htf_tf = htf_pairing.get(e["timeframe"])
            if htf_tf:
                needed[e["symbol"]].add(htf_tf)

    fetch_jobs = [(sym, tf) for sym, tfs in needed.items() for tf in tfs]

    def _job(job):
        sym, tf = job
        return job, _fetch_enriched(sym, tf, cfg)

    dfs: dict[tuple, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=cfg.get("scan_concurrency", 8)) as executor:
        for job, df in executor.map(_job, fetch_jobs):
            dfs[job] = df

    results = []
    for e in new_entries:
        df = dfs.get((e["symbol"], e["timeframe"]))
        htf_df = None
        if e["based_on"] == "vwap_breakout_ashen":
            htf_tf = htf_pairing.get(e["timeframe"])
            htf_df = dfs.get((e["symbol"], htf_tf)) if htf_tf else None
        results.append(audit_entry(e, df, htf_df, cfg))

    with open(AUDIT_RESULTS_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    newest_logged_at = max(e["logged_at"] for e in new_entries)
    save_audit_state({"last_audited_logged_at": newest_logged_at})

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    print(f"\nAudit complete: {passed} passed, {failed} failed (of {len(results)} audited)")
    if failed:
        print("\nFAILED entries:")
        for r in results:
            if r["status"] == "fail":
                print(f"  {r['symbol']} [{r['timeframe']}] {r['based_on']}/{r['bias']} "
                      f"logged {r['logged_at']}: {r['reason']}")


if __name__ == "__main__":
    main()

"""
Live signal journal - a forward-tested track record.

Backtests are retrospective and can be (even unintentionally) overfit to
the exact data they were tuned on. This journal logs every real setup the
scanner surfaces, then later checks what price actually did, building up
real forward performance over time - the only thing that actually proves
a system works going forward.

Storage: append-only JSONL (one JSON object per line) at journal.jsonl.
No database dependency, diff-friendly, trivial to append to from a
scheduled/cloud run.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from scanner.data import get_klines

JOURNAL_PATH = Path(__file__).resolve().parent.parent / "journal.jsonl"
FAILED_PATH = Path(__file__).resolve().parent.parent / "failed_trades.jsonl"


def _load(path: Path = JOURNAL_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _save_all(entries: list[dict], path: Path = JOURNAL_PATH) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + ("\n" if entries else ""))


def _append(entry: dict, path: Path = JOURNAL_PATH) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_signals(report: dict, cfg: dict, path: Path = JOURNAL_PATH,
                reopen_cooldown_hours: float = 6.0) -> list[dict]:
    """
    Log every setup meeting min_confluence. Skips symbol/timeframe/pattern
    combos that already have an unresolved ("open") journal entry - without
    this, a persistent state signal (e.g. ema_stack, which fires on every
    run while the trend holds) would spam a fresh row every single scan
    instead of one row per real occurrence.

    Also skips combos whose most recent entry (any status) resolved within
    reopen_cooldown_hours. Without this, a persistent signal that JUST
    resolved (win/loss/expired) but is still geometrically true on the very
    next scan re-opens instantly - in practice this looked like the same
    symbol/detector alerting every scan cycle (e.g. ARBUSDT qqe_cross,
    ETHUSDT ema_stack firing every ~8-20 min back-to-back), which is really
    one persistent condition being counted as many separate "trades" rather
    than a genuinely new occurrence. This does not affect backtest scripts
    (they use their own blocked_until logic operating on horizon_candles,
    not wall-clock time) - it only governs the live journal.

    Returns the list of entries newly appended this run (not just a count) -
    callers use this to know exactly which setups are genuinely new, e.g.
    to only Telegram-alert on first occurrence instead of re-sending the
    same open setup every scan.
    """
    min_conf = cfg["output"]["min_confluence"]
    existing = _load(path)
    open_keys = {(e["symbol"], e["timeframe"], e["based_on"]) for e in existing if e["status"] == "open"}
    now_dt = datetime.now(timezone.utc).replace(tzinfo=None)
    last_resolved_at: dict[tuple, datetime] = {}
    for e in existing:
        if e["status"] == "open":
            continue
        key = (e["symbol"], e["timeframe"], e["based_on"])
        resolved_at = datetime.fromisoformat(e["checked_at"] or e["logged_at"])
        if key not in last_resolved_at or resolved_at > last_resolved_at[key]:
            last_resolved_at[key] = resolved_at
    next_id = max((e["id"] for e in existing), default=0) + 1
    now = now_dt.isoformat()

    logged = []
    for res in report["results"]:
        for tf, data in res["timeframes"].items():
            if data["strength"] < min_conf or not data.get("risk"):
                continue
            r = data["risk"]
            key = (res["symbol"], tf, r["based_on"])
            if key in open_keys:
                continue
            last_resolved = last_resolved_at.get(key)
            if last_resolved is not None:
                cooldown_elapsed = (now_dt - last_resolved).total_seconds() / 3600
                if cooldown_elapsed < reopen_cooldown_hours:
                    continue
            entry = {
                "id": next_id,
                "logged_at": now,
                "symbol": res["symbol"],
                "timeframe": tf,
                "bias": data["bias"],
                "strength": data["strength"],
                "entry": r["entry"],
                "stop": r["stop"],
                "target": r["target"],
                "based_on": r["based_on"],
                "signals": [s["name"] for s in data["signals"]],
                "status": "open",
                "checked_at": None,
                "outcome_price": None,
                "outcome_pct": None,
                # Set True only once this specific setup actually goes out as
                # a Telegram alert (main.py calls mark_notified after
                # notify_report) - logging happens at a lower confluence bar
                # than alerting does, so not every journal row was ever sent.
                # Reminders must only fire for setups the user was actually told about.
                "notified": False,
                "last_reminded_at": None,
            }
            _append(entry, path)
            open_keys.add(key)
            next_id += 1
            logged.append(entry)
    return logged


def mark_notified(keys: set, path: Path = JOURNAL_PATH) -> None:
    """
    Flag journal entries that actually went out as a Telegram alert.
    Logging happens at a lower confluence bar than alerting does, so not
    every journal row was ever sent - reminders (get_due_reminders) must
    only fire for setups the user was actually notified about, otherwise
    a reminder would reference an alert they never received.
    """
    if not keys:
        return
    entries = _load(path)
    changed = False
    for e in entries:
        if (e["symbol"], e["timeframe"], e["based_on"]) in keys:
            e["notified"] = True
            changed = True
    if changed:
        _save_all(entries, path)


def get_due_reminders(cooldown_hours: float = 4.0, path: Path = JOURNAL_PATH) -> list[dict]:
    """
    Open, already-notified entries that haven't had a reminder (or the
    original alert) within cooldown_hours - a lightweight "still open" ping
    so a setup doesn't go silent for hours while it remains live, without
    re-sending the full alert every scan the way the old notifier did.
    """
    entries = _load(path)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    due = []
    for e in entries:
        if e["status"] != "open" or not e.get("notified"):
            continue
        last = e.get("last_reminded_at") or e["logged_at"]
        elapsed_hours = (now - datetime.fromisoformat(last)).total_seconds() / 3600
        if elapsed_hours >= cooldown_hours:
            due.append(e)
    return due


def mark_reminded(reminded: list[dict], path: Path = JOURNAL_PATH) -> None:
    if not reminded:
        return
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    ids = {e["id"] for e in reminded}
    entries = _load(path)
    for e in entries:
        if e["id"] in ids:
            e["last_reminded_at"] = now
    _save_all(entries, path)


def check_open_entries(path: Path = JOURNAL_PATH, horizon_candles: int = 20,
                       kline_limit: int = 500, concurrency: int = 8) -> list[dict]:
    """
    For every open entry, fetch candles since it was logged and see whether
    stop or target was hit first. If neither hits within horizon_candles,
    mark it "expired" and record what price actually did - a signal that
    never resolves either way is still useful information (it means the
    stop/target distances were too wide, or the setup just chopped).

    Returns the entries that were resolved this run (win/loss/expired) so
    callers can push a close-out notice (e.g. Telegram) - otherwise a setup
    that already hit its stop keeps looking "live" to anyone who only reads
    the original alert message.

    Candle fetches run concurrently (this runs in the same scheduled window
    as main.py's scan, so it needs to fit the same cadence) - the actual
    outcome logic below stays sequential since it only mutates each entry's
    own dict, no shared state to worry about.
    """
    entries = _load(path)
    if not entries:
        return []
    open_entries = [e for e in entries if e["status"] == "open"]
    if not open_entries:
        return []
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    resolved_entries = []

    def _fetch(e):
        return e, get_klines(e["symbol"], e["timeframe"], kline_limit)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        fetched = list(executor.map(_fetch, open_entries))

    for e, df in fetched:
        if df is None:
            continue
        logged_at = datetime.fromisoformat(e["logged_at"])
        after = df[df["open_time"] > logged_at]
        if after.empty:
            continue

        outcome, outcome_price = None, None
        for _, candle in after.iterrows():
            if e["bias"] == "bullish":
                if candle["low"] <= e["stop"]:
                    outcome, outcome_price = "loss", e["stop"]
                    break
                if candle["high"] >= e["target"]:
                    outcome, outcome_price = "win", e["target"]
                    break
            else:
                if candle["high"] >= e["stop"]:
                    outcome, outcome_price = "loss", e["stop"]
                    break
                if candle["low"] <= e["target"]:
                    outcome, outcome_price = "win", e["target"]
                    break

        if outcome is None and len(after) >= horizon_candles:
            outcome, outcome_price = "expired", float(after["close"].iloc[-1])

        if outcome:
            e["status"] = outcome
            e["checked_at"] = now
            e["outcome_price"] = outcome_price
            e["outcome_pct"] = round((outcome_price - e["entry"]) / e["entry"] * 100, 3)
            resolved_entries.append(e)

    if resolved_entries:
        _save_all(entries, path)
    return resolved_entries


def log_failed_trades(resolved: list[dict], path: Path = FAILED_PATH) -> int:
    """
    Pull every non-win outcome (loss or expired) out into its own file,
    separate from the main journal - a dedicated place to actually review
    what didn't work and why, instead of it being buried among wins in
    journal.jsonl. Each row carries everything needed for a post-mortem:
    the detector/signals that fired, the exact entry/stop/target, and how
    price actually moved (outcome_price/outcome_pct) - a "loss" means price
    hit the stop, an "expired" means it never reached either level within
    the horizon (often a sign the stop/target distances were miscalibrated
    for that setup, not that the direction was wrong).

    Returns the number of rows appended.
    """
    failed = [e for e in resolved if e["status"] in ("loss", "expired")]
    if not failed:
        return 0
    with open(path, "a") as f:
        for e in failed:
            row = dict(e)
            row["note"] = (f"stop hit - price moved {e['outcome_pct']:+.2f}% against the {e['bias']} entry"
                           if e["status"] == "loss" else
                           f"expired unresolved - price ended {e['outcome_pct']:+.2f}% from entry, "
                           f"never reached stop or target")
            f.write(json.dumps(row) + "\n")
    return len(failed)


def detector_reliability(min_n: int = 10, path: Path = JOURNAL_PATH) -> dict[tuple[str, str], tuple[float, int]]:
    """
    Real win rate per (detector, direction) from the FULL resolved (win/loss)
    forward-tested history - not backtest_results.json, which measures a much
    easier bar (was price higher/lower N candles later, ignoring the stop
    entirely) and can show a detector as strongly edged while it actually
    loses money once a realistic stop-loss is respected. Returns (win_rate,
    n) - used for the "win probability" shown in each alert (see
    main.combined_detector_win_rate), pooled against realistic_backtest.py's
    larger sample the same way detector_expectancy is, so a handful of
    early live results can't swing a displayed probability on their own.

    Only returned once there are at least min_n decided trades, so a
    handful of early results doesn't permanently blacklist a detector.
    """
    entries = [e for e in _load(path) if e["status"] in ("win", "loss")]
    groups: dict[tuple[str, str], list[str]] = {}
    for e in entries:
        groups.setdefault((e["based_on"], e["bias"]), []).append(e["status"])
    return {key: (statuses.count("win") / len(statuses), len(statuses))
            for key, statuses in groups.items() if len(statuses) >= min_n}


def detector_expectancy(min_n: int = 10, path: Path = JOURNAL_PATH) -> dict[tuple[str, str], tuple[float, int]]:
    """
    Real expectancy (in R, i.e. multiples of risk) per (detector, direction)
    from the full resolved (win/loss) live history: win_rate * (1 + avg R:R)
    - 1. Zero at breakeven, positive is real edge. Returns (expectancy, n)
    - the sample size is included because a handful of live trades is not
    a reliable enough basis to override a larger historical dataset (see
    main.py, which pools this against realistic_backtest.py's much bigger
    sample weighted by n, rather than letting a small live sample simply
    win outright the moment it crosses min_n).

    detector_reliability's flat win-rate comparison is WRONG for detectors
    whose R:R varies a lot - e.g. a structural pattern with a 15:1 R:R only
    needs a ~6% win rate to be profitable, so comparing its win rate against
    a flat threshold tuned for a 2:1 setup would blacklist a genuinely
    excellent detector. Expectancy is the correct, R:R-aware comparison:
    it's negative if and only if the detector actually loses money at its
    own real reward:risk, regardless of what that ratio happens to be.
    """
    entries = [e for e in _load(path) if e["status"] in ("win", "loss")]
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in entries:
        groups.setdefault((e["based_on"], e["bias"]), []).append(e)
    out = {}
    for key, es in groups.items():
        if len(es) < min_n:
            continue
        wins = sum(1 for e in es if e["status"] == "win")
        win_rate = wins / len(es)
        rrs = [abs(e["target"] - e["entry"]) / abs(e["entry"] - e["stop"])
              for e in es if e["entry"] != e["stop"]]
        avg_rr = sum(rrs) / len(rrs) if rrs else 0
        out[key] = (win_rate * (1 + avg_rr) - 1, len(es))
    return out


def detector_avg_return(min_n: int = 10, path: Path = JOURNAL_PATH) -> dict[tuple[str, str], float]:
    """
    Real average % move per (detector, direction) from the full resolved
    (win/loss/expired) live history. outcome_pct already reflects what price
    actually did against the REAL stop/target - unlike backtest_results.json's
    avg_ret_pct, which is a naive N-candle-later return with no stop-loss
    involved at all (see detector_reliability's docstring for why that's a
    materially different, easier-to-satisfy number).

    This is what setup_risk_plan uses to calibrate a target - using the
    backtest version here let a detector's target get pulled from the same
    unrealistic metric that inflated its apparent edge, which is how a trade
    with only 3 live decided results and a losing recent streak still ended
    up with a calibrated (not just geometric) target. Same min_n gate as
    detector_reliability, so a handful of early results can't calibrate
    anything either.
    """
    entries = [e for e in _load(path) if e["status"] in ("win", "loss", "expired")]
    groups: dict[tuple[str, str], list[float]] = {}
    for e in entries:
        groups.setdefault((e["based_on"], e["bias"]), []).append(e["outcome_pct"])
    return {key: sum(vals) / len(vals) for key, vals in groups.items() if len(vals) >= min_n}


def detector_recent_form(detector_name: str, direction: str, n: int = 5,
                         path: Path = JOURNAL_PATH) -> dict | None:
    """
    Win/loss record from the last n decided (win/loss - excludes still-open
    and inconclusive/expired) journal entries for this specific detector +
    direction. Surfaced in alerts so a detector on a current losing streak
    doesn't get acted on with the same confidence as one that's been
    working lately - the point isn't to hide it, it's to make "this method
    hasn't been working recently" visible instead of trading on blind faith
    in a backtested average that may not reflect current conditions.
    Returns None if there isn't enough resolved history yet to be meaningful.
    """
    entries = [e for e in _load(path)
              if e["based_on"] == detector_name and e["bias"] == direction
              and e["status"] in ("win", "loss")]
    if len(entries) < 2:
        return None
    entries.sort(key=lambda e: e["checked_at"] or e["logged_at"], reverse=True)
    recent = entries[:n]
    wins = sum(1 for e in recent if e["status"] == "win")
    return {"n": len(recent), "wins": wins, "losses": len(recent) - wins}


def summarize(path: Path = JOURNAL_PATH) -> dict:
    """Win rate and average return per detector, from resolved journal entries only."""
    entries = [e for e in _load(path) if e["status"] in ("win", "loss", "expired")]
    by_detector: dict[str, list[dict]] = {}
    for e in entries:
        by_detector.setdefault(e["based_on"], []).append(e)

    summary = {}
    for name, rows in sorted(by_detector.items()):
        wins = sum(1 for r in rows if r["status"] == "win")
        losses = sum(1 for r in rows if r["status"] == "loss")
        decided = wins + losses
        summary[name] = {
            "n": len(rows),
            "wins": wins,
            "losses": losses,
            "expired": len(rows) - decided,
            "win_rate": round(wins / decided, 3) if decided else None,
            "avg_outcome_pct": round(sum(r["outcome_pct"] for r in rows) / len(rows), 3),
        }
    return summary

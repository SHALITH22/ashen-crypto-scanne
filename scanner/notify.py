"""
Telegram notifier.

Keys are NEVER hardcoded. Two ways to supply them, checked in this order:
  1. Process environment variables (e.g. GitHub Actions secrets passed via
     the workflow's `env:` block) - preferred for cloud/scheduled runs,
     never touches disk.
  2. A local `.env` file (gitignored) - for running on your own machine:
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=123456789

Get a token from @BotFather; get your chat id from @userinfobot.
If neither source has the values, the notifier silently no-ops (scanner still runs).
"""

import os
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
ENV_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")

SLT_OFFSET = timedelta(hours=5, minutes=30)  # Sri Lanka time, UTC+5:30 - the timezone this is tuned around


def to_slt_clock(iso_ts: str) -> str | None:
    """UTC ISO timestamp -> 'HH:MM:SS' in Sri Lanka local time."""
    try:
        dt = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    slt = dt.astimezone(timezone.utc).replace(tzinfo=None) + SLT_OFFSET
    return slt.strftime("%H:%M:%S")


def load_env(path: Path = ENV_PATH) -> dict:
    env = {k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())  # env vars take priority over .env
    return env


def send_telegram(text: str, env: dict | None = None) -> bool:
    env = env if env is not None else load_env()
    token, chat_id = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text,
                                     "parse_mode": "HTML",
                                     "disable_web_page_preview": True}, timeout=15)
        return r.ok
    except requests.RequestException:
        return False


def format_setup(symbol: str, tf: str, data: dict, generated_at: str | None = None) -> str:
    lines = [f"<b>{escape(symbol)}</b> [{escape(tf)}] {escape(data['bias'].upper())} "
             f"(strength {data['strength']})  close={data['close']:.6g}"]
    if generated_at:
        # Price moves between this scan and whenever you actually read the
        # message - this timestamp lets you judge how stale it might be by
        # the time you act, rather than assuming the quoted price is "now".
        lines.append(f"Scanned: {escape(generated_at)} SLT")
    if "htf_note" in data:
        lines.append(f"HTF: {escape(data['htf_note'])}")
    if data.get("regime"):
        lines.append(f"Regime: {escape(data['regime'])}"
                     + (" - lower conviction, market is choppy right now" if data["regime"] == "choppy" else ""))
    for s in data["signals"]:
        lines.append(f"- {escape(s['name'])}: {escape(s['detail'])}")
    if data.get("risk"):
        r = data["risk"]
        rr = f"{r['risk_reward']}:1" if r["risk_reward"] else "n/a"
        lines.append(f"Risk: entry {r['entry']:.6g} / stop {r['stop']:.6g} / "
                     f"target {r['target']:.6g} (R:R {rr}, based on {escape(r['based_on'])}, "
                     f"target: {escape(r['target_basis'])})")
        if r.get("win_probability") is not None:
            # Real, measured win rate for this exact detector/direction
            # (pooled from backtest + live data) - not a black-box "AI
            # confidence score", the same number that decides whether this
            # detector is even allowed to alert at all.
            lines.append(f"Win probability: {r['win_probability']:.0%} "
                         f"(measured, {escape(r['based_on'])}/{escape(data['bias'])})")
        if r.get("position"):
            p = r["position"]
            lines.append(f"Position size: risk {p['account_risk_pct']}% (${p['dollar_risk']}) "
                         f"-&gt; {p['units']:g} units (~${p['position_value']})")
        if r.get("recent_form"):
            f = r["recent_form"]
            lines.append(f"Recent form for {escape(r['based_on'])}/{escape(data['bias'])}: "
                         f"{f['wins']}W-{f['losses']}L (last {f['n']})")
    return "\n".join(lines)


def notify_report(report: dict, cfg: dict, new_keys: set | None = None) -> tuple[int, set]:
    """
    Send one message per qualifying setup. Returns (number sent, keys sent)
    - the keys let the caller flag exactly which journal entries actually
    went out as an alert (see journal.mark_notified), since reminders must
    only fire for setups the user was actually told about.

    Two gates beyond the old strength/HTF-agreement check, each fixing a
    specific way the old alerts were unusable in practice:
      - Requires an actual risk plan (data["risk"]). Without this, a setup
        with no candidate clearing min_risk_reward still fired an alert with
        a bias and strength but no entry/stop/target - unreadable as a
        trade, indistinguishable from a bug.
      - `new_keys` (symbol, timeframe, based_on): when provided, only alerts
        on setups that are genuinely new this run (per the journal's
        open-entry tracking), instead of re-sending the same still-open
        setup - with an unchanged plan - every single scan. Ongoing "still
        open" updates are handled separately by a lightweight reminder
        (see journal.get_due_reminders / notify_reminders), not a repeat
        of the full alert.

    An optional `timeframes` allowlist under notify.telegram can still
    restrict which timeframes alert, but by default (unset) every timeframe
    that clears the other gates is eligible.
    """
    tg = cfg.get("notify", {}).get("telegram", {})
    if not tg.get("enabled", False):
        return 0, set()
    env = load_env()
    if not env.get("TELEGRAM_BOT_TOKEN"):
        print("  [notify] telegram enabled but TELEGRAM_BOT_TOKEN not found in "
              "environment or .env - skipping")
        return 0, set()
    min_strength = tg.get("min_strength", 3)
    only_agreeing = tg.get("only_htf_agreeing", True)
    allowed_tfs = tg.get("timeframes")  # None = no filtering (all timeframes)
    generated_at = report.get("generated_at", "")
    scan_time = to_slt_clock(generated_at)
    sent = 0
    sent_keys = set()
    for res in report["results"]:
        for tf, data in res["timeframes"].items():
            if data["strength"] < min_strength:
                continue
            if only_agreeing and not data.get("htf_agrees", True):
                continue
            if allowed_tfs is not None and tf not in allowed_tfs:
                continue
            risk = data.get("risk")
            if not risk:
                continue  # no entry/stop/target cleared min R:R - not an actionable alert
            key = (res["symbol"], tf, risk["based_on"])
            if new_keys is not None and key not in new_keys:
                continue  # already alerted on this open setup - avoid re-sending unchanged
            if send_telegram(format_setup(res["symbol"], tf, data, scan_time), env):
                sent += 1
                sent_keys.add(key)
    return sent, sent_keys


def format_outcome(entry: dict) -> str:
    """Close-out notice for a journal entry that just resolved (win/loss/expired)."""
    icon = {"win": "✅", "loss": "❌", "expired": "⏳"}.get(entry["status"], "")
    label = {"win": "TARGET HIT", "loss": "STOP HIT", "expired": "EXPIRED (no clean resolution)"}[entry["status"]]
    sign = "+" if entry["outcome_pct"] >= 0 else ""
    return (f"{icon} <b>{escape(entry['symbol'])}</b> [{escape(entry['timeframe'])}] "
            f"{escape(entry['bias'].upper())} - {label}\n"
            f"entry {entry['entry']:.6g} -&gt; {entry['outcome_price']:.6g} "
            f"({sign}{entry['outcome_pct']}%)\n"
            f"based on {escape(entry['based_on'])}")


def format_reminder(entry: dict) -> str:
    """Short still-open ping - not a repeat of the full alert, just enough to say 'this is still live'."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    logged = datetime.fromisoformat(entry["logged_at"])
    open_hours = (now - logged).total_seconds() / 3600
    return (f"\U0001f514 <b>{escape(entry['symbol'])}</b> [{escape(entry['timeframe'])}] "
            f"{escape(entry['bias'].upper())} - still open ({open_hours:.1f}h)\n"
            f"entry {entry['entry']:.6g} / stop {entry['stop']:.6g} / target {entry['target']:.6g}")


def notify_reminders(due: list[dict], cfg: dict) -> int:
    """Push a lightweight reminder for every open, already-notified entry past its reminder cooldown."""
    tg = cfg.get("notify", {}).get("telegram", {})
    if not tg.get("enabled", False) or not due:
        return 0
    env = load_env()
    if not env.get("TELEGRAM_BOT_TOKEN"):
        return 0
    sent = 0
    for entry in due:
        if send_telegram(format_reminder(entry), env):
            sent += 1
    return sent


def notify_outcomes(resolved: list[dict], cfg: dict) -> int:
    """Push a close-out message for every journal entry resolved this run."""
    tg = cfg.get("notify", {}).get("telegram", {})
    if not tg.get("enabled", False) or not resolved:
        return 0
    env = load_env()
    if not env.get("TELEGRAM_BOT_TOKEN"):
        return 0
    sent = 0
    for entry in resolved:
        if send_telegram(format_outcome(entry), env):
            sent += 1
    return sent

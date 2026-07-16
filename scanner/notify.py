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

from scanner.journal import total_resolved

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


def format_setup(symbol: str, tf: str, plan: dict, close: float, regime: str | None,
                 generated_at: str | None = None) -> str:
    """
    One message per independently-qualifying risk plan (see
    risk.setup_risk_plans) - each of the five live strategies gets its own
    alert when it fires, instead of one message per symbol/timeframe
    bundling whichever strategies happened to agree. `plan` carries its own
    direction/htf_agrees/htf_note/win_probability/recent_form now, rather
    than these living on a shared per-timeframe `data` dict.
    """
    lines = [f"<b>{escape(symbol)}</b> [{escape(tf)}] {escape(plan['direction'].upper())} "
             f"close={close:.6g}"]
    if generated_at:
        # Price moves between this scan and whenever you actually read the
        # message - this timestamp lets you judge how stale it might be by
        # the time you act, rather than assuming the quoted price is "now".
        lines.append(f"Scanned: {escape(generated_at)} SLT")
    if plan.get("htf_note"):
        lines.append(f"HTF: {escape(plan['htf_note'])}")
    if regime:
        lines.append(f"Regime: {escape(regime)}"
                     + (" - lower conviction, market is choppy right now" if regime == "choppy" else ""))
    rr = f"{plan['risk_reward']}:1" if plan["risk_reward"] else "n/a"
    lines.append(f"Strategy: {escape(plan['based_on'])}")
    lines.append(f"Risk: entry {plan['entry']:.6g} / stop {plan['stop']:.6g} / "
                 f"target {plan['target']:.6g} (R:R {rr}, "
                 f"target: {escape(plan['target_basis'])})")
    if plan.get("win_probability") is not None:
        # Real, measured win rate for this exact detector/direction
        # (pooled from backtest + live data) - not a black-box "AI
        # confidence score", the same number that decides whether this
        # detector is even allowed to alert at all.
        lines.append(f"Win probability: {plan['win_probability']:.0%} "
                     f"(measured, {escape(plan['based_on'])}/{escape(plan['direction'])})")
    if plan.get("position"):
        p = plan["position"]
        lines.append(f"Position size: risk {p['account_risk_pct']}% (${p['dollar_risk']}) "
                     f"-&gt; {p['units']:g} units (~${p['position_value']})")
    if plan.get("recent_form"):
        f = plan["recent_form"]
        lines.append(f"Recent form for {escape(plan['based_on'])}/{escape(plan['direction'])}: "
                     f"{f['wins']}W-{f['losses']}L (last {f['n']})")
    return "\n".join(lines)


def _telegram_ready(cfg: dict, verbose: bool = False) -> dict | None:
    """
    Shared gate for all three send functions: Telegram must be enabled AND
    a real token/chat id configured AND the live journal must have
    accumulated at least min_resolved_trades resolved (win/loss/expired)
    outcomes across all five strategies combined, before any alert goes
    out - the same "prove it over N real trades before trusting it"
    discipline used for the original binance-scanner, now automated rather
    than requiring a manual flip once the count is eyeballed. Counts every
    strategy together (not per-strategy) since this gate is about trusting
    the journal/risk-plan/notify pipeline end-to-end, not any one
    strategy's individual edge - see journal.total_resolved. Returns the
    loaded env dict once every gate passes, or None if any gate blocks
    sending.
    """
    tg = cfg.get("notify", {}).get("telegram", {})
    if not tg.get("enabled", False):
        return None
    env = load_env()
    if not env.get("TELEGRAM_BOT_TOKEN"):
        if verbose:
            print("  [notify] telegram enabled but TELEGRAM_BOT_TOKEN not found in "
                  "environment or .env - skipping")
        return None
    min_resolved = tg.get("min_resolved_trades", 1000)
    resolved = total_resolved()
    if resolved < min_resolved:
        if verbose:
            print(f"  [notify] telegram paused until {min_resolved} resolved trades "
                  f"accumulate ({resolved}/{min_resolved} so far) - proving the live "
                  f"strategies are profitable before alerting on them")
        return None
    return env


def notify_report(report: dict, cfg: dict, new_keys: set | None = None) -> tuple[int, set]:
    """
    Send one message per independently-qualifying risk plan (see
    risk.setup_risk_plans) - each of the five live strategies that fired
    gets its own alert, matching the journal's one-row-per-strategy
    tracking, instead of bundling whichever strategies happened to agree
    into a single per-symbol/timeframe message. Returns (number sent, keys
    sent) - the keys let the caller flag exactly which journal entries
    actually went out as an alert (see journal.mark_notified), since
    reminders must only fire for setups the user was actually told about.

    A plan already only exists here because it cleared setup_risk_plans's
    own qualification bars (R:R, stop width, blacklist, market/funding
    filters) - unlike the old per-timeframe flow, there's no separate
    "no risk plan" case to check for, since risk_plans only ever contains
    tradeable candidates.

    `new_keys` (symbol, timeframe, based_on): when provided, only alerts on
    plans that are genuinely new this run (per the journal's open-entry
    tracking), instead of re-sending the same still-open setup - with an
    unchanged plan - every single scan. Ongoing "still open" updates are
    handled separately by a lightweight reminder (see
    journal.get_due_reminders / notify_reminders), not a repeat of the full
    alert.

    An optional `timeframes` allowlist under notify.telegram can still
    restrict which timeframes alert, but by default (unset) every timeframe
    that clears the other gates is eligible.
    """
    tg = cfg.get("notify", {}).get("telegram", {})
    env = _telegram_ready(cfg, verbose=True)
    if env is None:
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
            # min_strength still uses the tf's combined signal-mix strength
            # (confluence_score across ALL raw signals present, jayantha's
            # bonus signals included) as a display-context bar for alerting
            # specifically - unlike the journal, which now logs every
            # independently-qualifying plan regardless of what else agreed.
            if data["strength"] < min_strength or (allowed_tfs is not None and tf not in allowed_tfs):
                continue
            for plan in data.get("risk_plans", []):
                if only_agreeing and not plan.get("htf_agrees", True):
                    continue
                key = (res["symbol"], tf, plan["based_on"])
                if new_keys is not None and key not in new_keys:
                    continue  # already alerted on this open setup - avoid re-sending unchanged
                text = format_setup(res["symbol"], tf, plan, data["close"], data.get("regime"), scan_time)
                if send_telegram(text, env):
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
    if not due:
        return 0
    env = _telegram_ready(cfg)
    if env is None:
        return 0
    sent = 0
    for entry in due:
        if send_telegram(format_reminder(entry), env):
            sent += 1
    return sent


def notify_outcomes(resolved: list[dict], cfg: dict) -> int:
    """Push a close-out message for every journal entry resolved this run."""
    if not resolved:
        return 0
    env = _telegram_ready(cfg)
    if env is None:
        return 0
    sent = 0
    for entry in resolved:
        if send_telegram(format_outcome(entry), env):
            sent += 1
    return sent

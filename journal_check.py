"""
Check open journal entries against real price action and print the
current forward-tested track record.

Usage: python journal_check.py
Run this periodically (e.g. daily) - separately from main.py's scanning -
so open setups get resolved once their stop/target/horizon plays out.
"""

import sys
from pathlib import Path

import yaml

from scanner.journal import check_open_entries, summarize, JOURNAL_PATH, get_due_reminders, mark_reminded, log_failed_trades
from scanner.notify import notify_outcomes, notify_reminders

# See main.py for why - Windows' console codepage can't encode some
# Unicode characters that show up in symbol names or pattern details.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


def main():
    cfg = yaml.safe_load(open(CONFIG_PATH))
    journal_cfg = cfg.get("journal", {})
    horizon = journal_cfg.get("horizon_candles", 20)

    if not JOURNAL_PATH.exists():
        print("No journal.jsonl yet - run main.py first to log some setups.")
        return

    resolved = check_open_entries(horizon_candles=horizon, concurrency=cfg.get("scan_concurrency", 8))
    print(f"Resolved {len(resolved)} entr{'y' if len(resolved) == 1 else 'ies'}.\n")

    # Close-out notice - without this, a setup that already hit its stop or
    # target keeps looking "live" to anyone who only saw the original alert.
    sent = notify_outcomes(resolved, cfg)
    if sent:
        print(f"Sent {sent} outcome notice(s)")

    # Losses/expired pulled into their own file for review - "what didn't
    # work and why", separate from the main journal's mix of wins and losses.
    failed = log_failed_trades(resolved)
    if failed:
        print(f"Logged {failed} failed trade(s) to failed_trades.jsonl for review")

    # Lightweight "still open" ping for setups that were actually alerted on
    # and remain open past the cooldown - not a repeat of the full alert.
    cooldown = cfg.get("notify", {}).get("telegram", {}).get("reminder_cooldown_hours", 4.0)
    due = get_due_reminders(cooldown_hours=cooldown)
    reminded = notify_reminders(due, cfg)
    if reminded:
        print(f"Sent {reminded} reminder(s)")
        mark_reminded(due)

    summary = summarize()
    if not summary:
        print("No resolved entries yet.")
        return

    hdr = f"{'detector':<28}{'n':>5}{'win_rate':>10}{'avg%':>8}{'expired':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in summary.items():
        wr = f"{s['win_rate']:.1%}" if s["win_rate"] is not None else "n/a"
        print(f"{name:<28}{s['n']:>5}{wr:>10}{s['avg_outcome_pct']:>8.2f}{s['expired']:>9}")


if __name__ == "__main__":
    main()

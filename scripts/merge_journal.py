#!/usr/bin/env python3
"""
Git custom merge driver for journal.jsonl.

Why this exists: journal.jsonl is append-only, and multiple scheduled scan
runs can genuinely overlap (this got worse once the schedule moved from
every 30 min to every 15 min, plus manual "Run workflow" triggers) - two
runs each appending new rows and pushing around the same time is a normal
race, not a real conflict. Git's default line-based merge can't tell "both
sides added different new rows" from an actual conflict and fails the
whole `git pull --rebase` outright, which is exactly what was crashing the
"Commit journal updates" step (and losing that run's journal writes,
including which setups had already been Telegram-notified).

This driver merges by CONTENT instead of line position: each entry's
(symbol, timeframe, based_on, logged_at) is a stable natural key (the
scanner's own dedup guarantees only one open entry per symbol/tf/detector
at a time, and logged_at is a microsecond timestamp), so entries from both
sides can be unioned safely, preferring whichever version of a shared
entry is more resolved (a real win/loss/expired beats a stale "open").

Configured via .gitattributes (`journal.jsonl merge=journal-merge`) and a
`git config merge.journal-merge.driver` line in the workflow (see scan.yml)
- git then calls this as `merge_journal.py <base> <ours> <theirs>` for any
merge/rebase touching journal.jsonl, and expects the resolved file written
back to <ours> in place. Exit 0 means "resolved cleanly, no conflict".
"""

import json
import sys


def load(path: str) -> list[dict]:
    try:
        return [json.loads(line) for line in open(path) if line.strip()]
    except FileNotFoundError:
        return []


def key(e: dict) -> tuple:
    return (e["symbol"], e["timeframe"], e["based_on"], e["logged_at"])


def more_resolved(a: dict, b: dict) -> dict:
    """Whichever of two same-key entries is the more complete/authoritative record."""
    if a["status"] != "open" and b["status"] == "open":
        return a
    if b["status"] != "open" and a["status"] == "open":
        return b
    if a["status"] != "open" and b["status"] != "open":
        return a if (a.get("checked_at") or "") >= (b.get("checked_at") or "") else b
    return a  # both still open - identical either way


def main() -> int:
    # git invokes as: merge_journal.py %O %A %B  (base, ours, theirs)
    _base_path, ours_path, theirs_path = sys.argv[1], sys.argv[2], sys.argv[3]
    ours = load(ours_path)
    theirs = load(theirs_path)

    by_key: dict[tuple, dict] = {}
    for e in ours + theirs:
        k = key(e)
        by_key[k] = more_resolved(by_key[k], e) if k in by_key else e

    merged = list(by_key.values())
    merged.sort(key=lambda e: (e["logged_at"], e["symbol"], e["timeframe"]))
    for i, e in enumerate(merged, start=1):
        e["id"] = i
        e.setdefault("notified", False)
        e.setdefault("last_reminded_at", None)

    with open(ours_path, "w") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

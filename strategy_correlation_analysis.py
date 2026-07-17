"""
Strategy correlation/overlap analysis: are the five live strategies
actually five INDEPENDENT bets, or do they fire together often enough -
and win/lose together often enough - that the real diversification is
much less than "5 strategies x 1% risk each" implies?

Three questions, answered directly from journal.jsonl (real timestamps,
real outcomes - not a backtest simulation, since only the live journal has
true wall-clock overlap between different strategies' trades):

  1. Co-occurrence: how often do two strategies have OVERLAPPING open
     windows on the same symbol at all (regardless of direction)?
  2. Direction agreement: when they do overlap, do they agree (same bias)
     or disagree?
  3. Outcome correlation: for overlapping pairs where BOTH resolved
     (win/loss), does knowing one won/lost tell you anything about the
     other, beyond each strategy's own baseline win rate ("lift")?

Also reports the peak number of strategies concurrently open on the same
symbol at any single point in time - the real answer to "what's the worst
case simultaneous risk on one coin," which the flat 1%-per-trade sizing
model doesn't account for on its own.

Usage: python strategy_correlation_analysis.py
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

JOURNAL_PATH = Path(__file__).parent / "journal.jsonl"

# Below this many overlapping DECIDED pairs, a correlation number is noise -
# reported as "insufficient data" rather than a misleadingly precise figure.
MIN_PAIRS_FOR_CORRELATION = 15


def load_journal() -> list[dict]:
    entries = []
    for line in JOURNAL_PATH.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


def intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and b_start <= a_end


def main():
    entries = load_journal()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for e in entries:
        e["_start"] = parse_dt(e["logged_at"])
        e["_end"] = parse_dt(e["checked_at"]) or now

    strategies = sorted({e["based_on"] for e in entries})
    print(f"Loaded {len(entries)} journal entries across {len(strategies)} strategies: {', '.join(strategies)}\n")

    # Each strategy's own baseline win rate (decided trades only) - the
    # reference point "lift" is measured against.
    baseline_win_rate = {}
    for s in strategies:
        decided = [e for e in entries if e["based_on"] == s and e["status"] in ("win", "loss")]
        if decided:
            wins = sum(1 for e in decided if e["status"] == "win")
            baseline_win_rate[s] = (wins / len(decided), len(decided))

    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_symbol[e["symbol"]].append(e)

    co_occur = defaultdict(int)
    same_direction = defaultdict(int)
    opposite_direction = defaultdict(int)
    contingency = defaultdict(lambda: {"both_win": 0, "both_loss": 0, "a_win_b_loss": 0, "a_loss_b_win": 0})
    max_concurrent_by_symbol = {}

    for symbol, group in by_symbol.items():
        for i, e1 in enumerate(group):
            for e2 in group[i + 1:]:
                if e1["based_on"] == e2["based_on"]:
                    continue  # only cross-strategy overlap is the question here
                if not intervals_overlap(e1["_start"], e1["_end"], e2["_start"], e2["_end"]):
                    continue
                pair = tuple(sorted([e1["based_on"], e2["based_on"]]))
                co_occur[pair] += 1
                if e1["bias"] == e2["bias"]:
                    same_direction[pair] += 1
                else:
                    opposite_direction[pair] += 1

                if e1["status"] in ("win", "loss") and e2["status"] in ("win", "loss"):
                    a, b = pair
                    e_a = e1 if e1["based_on"] == a else e2
                    e_b = e2 if e1["based_on"] == a else e1
                    if e_a["status"] == "win" and e_b["status"] == "win":
                        contingency[pair]["both_win"] += 1
                    elif e_a["status"] == "loss" and e_b["status"] == "loss":
                        contingency[pair]["both_loss"] += 1
                    elif e_a["status"] == "win" and e_b["status"] == "loss":
                        contingency[pair]["a_win_b_loss"] += 1
                    else:
                        contingency[pair]["a_loss_b_win"] += 1

        # Peak concurrent open count on this symbol (any status, "open"
        # trades count as ongoing through `now`).
        events = []
        for e in group:
            events.append((e["_start"], 1))
            events.append((e["_end"], -1))
        events.sort(key=lambda x: (x[0], -x[1]))  # opens before closes at the same instant
        cur = peak = 0
        for _, delta in events:
            cur += delta
            peak = max(peak, cur)
        max_concurrent_by_symbol[symbol] = peak

    print("=== 1+2. Co-occurrence and direction agreement (same symbol, overlapping open windows) ===")
    print(f"{'strategy pair':<45}{'co-occur':>10}{'same-dir':>10}{'opp-dir':>10}")
    print("-" * 75)
    all_pairs = sorted(co_occur.keys(), key=lambda p: -co_occur[p])
    for pair in all_pairs:
        n = co_occur[pair]
        sd = same_direction[pair]
        od = opposite_direction[pair]
        print(f"{pair[0] + ' / ' + pair[1]:<45}{n:>10}{sd:>10}{od:>10}")
    if not all_pairs:
        print("(no cross-strategy overlaps found in this journal yet)")

    print("\n=== 3. Outcome correlation (lift over each strategy's own baseline win rate) ===")
    for pair in all_pairs:
        c = contingency[pair]
        n_decided_pairs = c["both_win"] + c["both_loss"] + c["a_win_b_loss"] + c["a_loss_b_win"]
        a, b = pair
        print(f"\n-- {a} / {b} --  ({n_decided_pairs} overlapping decided pairs)")
        print(f"   both won: {c['both_win']}  both lost: {c['both_loss']}  "
              f"{a} won/{b} lost: {c['a_win_b_loss']}  {a} lost/{b} won: {c['a_loss_b_win']}")
        if n_decided_pairs < MIN_PAIRS_FOR_CORRELATION:
            print(f"   -> insufficient data (< {MIN_PAIRS_FOR_CORRELATION} pairs) for a trustworthy correlation figure")
            continue
        # P(B wins | A wins) vs B's own unconditional baseline win rate.
        a_wins = c["both_win"] + c["a_win_b_loss"]
        b_win_given_a_win = c["both_win"] / a_wins if a_wins else None
        a_losses = c["both_loss"] + c["a_loss_b_win"]
        b_win_given_a_loss = c["a_loss_b_win"] / a_losses if a_losses else None
        b_baseline = baseline_win_rate.get(b, (None, 0))[0]
        if b_win_given_a_win is not None and b_baseline is not None:
            lift = b_win_given_a_win - b_baseline
            print(f"   P({b} wins | {a} wins) = {b_win_given_a_win:.1%}  vs  {b}'s own baseline {b_baseline:.1%}  "
                  f"(lift {lift:+.1%})")
        if b_win_given_a_loss is not None and b_baseline is not None:
            lift = b_win_given_a_loss - b_baseline
            print(f"   P({b} wins | {a} loses) = {b_win_given_a_loss:.1%}  vs  {b}'s own baseline {b_baseline:.1%}  "
                  f"(lift {lift:+.1%})")

    print("\n=== 4. Peak concurrent strategies open on the same symbol ===")
    overall_peak = max(max_concurrent_by_symbol.values(), default=0)
    print(f"Overall peak across all symbols: {overall_peak} strategies simultaneously open on one symbol "
          f"(= up to {overall_peak}% of account concurrently at risk on that single coin, at 1% risk/trade)")
    top_symbols = sorted(max_concurrent_by_symbol.items(), key=lambda kv: -kv[1])[:10]
    print(f"\n{'symbol':<15}{'peak concurrent':>18}")
    for symbol, peak in top_symbols:
        print(f"{symbol:<15}{peak:>18}")

    print(f"\n=== Each strategy's own baseline win rate (for reference) ===")
    for s in strategies:
        if s in baseline_win_rate:
            wr, n = baseline_win_rate[s]
            print(f"  {s:<25}{wr:>8.1%}  (n={n})")
        else:
            print(f"  {s:<25}{'no decided trades yet':>25}")


if __name__ == "__main__":
    main()

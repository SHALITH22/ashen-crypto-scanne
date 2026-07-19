"""
Sweeps candidate minimum-structure-clearance thresholds for
vwap_breakout_ashen against the real historical trade population
(vwap_bearish_investigation.py's already-computed structure_distance_pct
per trade - see that script for the finding this is testing: wins
average 5.04% clearance past the broken structure level, losses average
only 3.49%).

Simulates "what if a trade only fired when structure_distance_pct was at
least X%" by filtering the REAL logged trades and their REAL outcomes -
not a fresh walk-forward resimulation, since we already have the
ground-truth clearance value and outcome for every trade. This directly
answers: does raising the bar actually lift win rate, and at what cost
in trade volume?

Usage: python vwap_structure_clearance_sweep.py
"""

import json
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "vwap_investigation.json"
REWARD_RISK = 1.5  # vwap_breakout_ashen's configured reward_risk_ratio
BREAKEVEN_WIN_RATE = 1 / (1 + REWARD_RISK)


def main():
    rows = json.load(open(RESULTS_PATH, encoding="utf-8"))
    rows = [r for r in rows if r.get("structure_distance_pct") is not None]
    print(f"{len(rows)} trades with structure_distance_pct available")

    baseline_wins = sum(1 for r in rows if r["status"] == "win")
    baseline_wr = baseline_wins / len(rows)
    print(f"\nBaseline (no filter): n={len(rows)}, win_rate={baseline_wr:.1%}, "
          f"breakeven={BREAKEVEN_WIN_RATE:.1%}")

    print(f"\n{'threshold':>10}{'n':>8}{'win_rate':>10}{'vs_breakeven':>14}{'expectancy(R)':>16}{'kept%':>8}")
    for threshold in [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]:
        kept = [r for r in rows if r["structure_distance_pct"] >= threshold]
        if not kept:
            continue
        wins = sum(1 for r in kept if r["status"] == "win")
        wr = wins / len(kept)
        expectancy = wr * REWARD_RISK - (1 - wr)
        kept_pct = len(kept) / len(rows) * 100
        marker = " <-- profitable" if expectancy > 0 else ""
        print(f"{threshold:>9.1f}%{len(kept):>8}{wr:>9.1%}{wr - BREAKEVEN_WIN_RATE:>+13.1%}{expectancy:>+15.3f}{kept_pct:>7.1f}%{marker}")

    # Direction split - the earlier investigation flagged bearish and
    # bullish may respond differently (bullish win sample was thin).
    for bias in ("bearish", "bullish"):
        sub = [r for r in rows if r["bias"] == bias]
        print(f"\n=== {bias} only (n={len(sub)}) ===")
        print(f"{'threshold':>10}{'n':>8}{'win_rate':>10}{'expectancy(R)':>16}")
        for threshold in [0.0, 3.0, 4.0, 4.5, 5.0, 6.0]:
            kept = [r for r in sub if r["structure_distance_pct"] >= threshold]
            if not kept:
                continue
            wins = sum(1 for r in kept if r["status"] == "win")
            wr = wins / len(kept)
            expectancy = wr * REWARD_RISK - (1 - wr)
            print(f"{threshold:>9.1f}%{len(kept):>8}{wr:>9.1%}{expectancy:>+15.3f}")


if __name__ == "__main__":
    main()

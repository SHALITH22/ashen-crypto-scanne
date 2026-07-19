"""
Full comprehensive loss report: for every resolved trade, categorizes
WHY it won or lost into one of four buckets, using real evidence rather
than guessing:

  1. TECHNICAL/RULE FAILURE - trade_diagnostic.json's independent re-
     verification (audit_trades.py's mechanism) found the detector's own
     rules did NOT actually produce this trade when re-run against real
     historical candles, or a gated strategy's BTC/ETH agreement
     requirement was somehow violated despite the live gate requiring it.
     A genuine system defect, not a normal loss.
  2. MARKET STRUCTURAL HEADWIND - for the 3 strategies NOT gated on
     BTC/ETH agreement (marubozu_ashen, b2b_ashen, jayantha_b2b, excluded
     per risk.py's _UNPROVEN_MARKET_FILTER_EXCLUSIONS), BTC/ETH's trend
     agreed WITH the trade's direction at entry - the same condition the
     PROVEN filter treats as unfavorable for the 2 gated strategies,
     though not backtested specifically for these 3. A hypothesis worth
     surfacing, not a proven cause for these strategies specifically.
  3. NORMAL PROBABILISTIC LOSS - rules followed correctly, no adverse
     market signal - the setup was executed exactly as designed and
     still didn't work, which is expected some fraction of the time for
     any <100%-win-rate strategy. Sub-categorized by MFE (loss_analysis.py's
     established method): "wrong immediately" (never moved favorably) vs
     "near-miss" (got most of the way to target first) vs "modest".
  4. NOT YET INDEPENDENTLY VERIFIED - trade_diagnostic.json doesn't cover
     this trade yet (diagnostic runs daily, may lag very recent trades).

Usage: python loss_categorization_report.py
Reads journal.jsonl + trade_diagnostic.json (both must be reasonably
fresh - see trade_diagnostic.yml). Writes loss_categorization.json for
the Monitor and prints a summary table + examples to the console.
"""

import json
from collections import defaultdict
from pathlib import Path

JOURNAL_PATH = Path(__file__).parent / "journal.jsonl"
DIAGNOSTIC_PATH = Path(__file__).parent / "trade_diagnostic.json"
OUT_PATH = Path(__file__).parent / "loss_categorization.json"

# Mirrors risk.py's _UNPROVEN_MARKET_FILTER_EXCLUSIONS - kept in sync
# manually since this is a standalone script, not importing risk.py's
# live config-dependent value directly.
UNGATED_STRATEGIES = {"jayantha_b2b", "b2b_ashen", "marubozu_ashen"}


def categorize(entry: dict, diag: dict | None) -> dict:
    result = {
        "id": entry["id"], "symbol": entry["symbol"], "timeframe": entry["timeframe"],
        "based_on": entry["based_on"], "bias": entry["bias"], "status": entry["status"],
        "logged_at": entry["logged_at"],
    }

    if diag is None:
        result["category"] = "not_yet_verified"
        result["reason"] = "trade_diagnostic.json doesn't cover this trade yet"
        return result

    result["rule_check"] = diag.get("rule_check")
    result["mfe"] = diag.get("mfe")
    result["mae"] = diag.get("mae")
    result["btc_agrees"] = diag.get("btc_agrees")
    result["eth_agrees"] = diag.get("eth_agrees")

    if diag.get("rule_check") == "fail":
        result["category"] = "technical_rule_failure"
        result["reason"] = diag.get("rule_reason", "rule check failed, reason not recorded")
        return result

    # NOT the same confidence level as rule_check above. The live gate
    # computes BTC/ETH trend ONCE per scan run (main.py's get_market_trend -
    # a single snapshot at scan start, shared across every pair). The
    # diagnostic reconstructs a full historical trend series and looks up
    # the value at the trade's exact logged_at instead. These are DIFFERENT
    # measurements of "what was BTC/ETH doing" - a mismatch here found this
    # affecting wins and losses in roughly equal proportion (70% of gated
    # trades either way), which is inconsistent with a live enforcement
    # failure (that would skew toward losses) and more consistent with a
    # genuine methodology difference between the two computations. Kept
    # separate from technical_rule_failure rather than asserted as a bug -
    # see the conversation this came from for the full investigation.
    if diag.get("btc_eth_gated"):
        both_disagreed = diag.get("btc_agrees") is False and diag.get("eth_agrees") is False
        if not both_disagreed and diag.get("btc_agrees") is not None:
            result["category"] = "btc_eth_recompute_mismatch"
            result["reason"] = "diagnostic's historical BTC/ETH recompute disagrees with what the live once-per-scan gate should have required - unresolved measurement discrepancy, not confirmed as a live enforcement failure"
            return result

    if entry["based_on"] in UNGATED_STRATEGIES:
        both_agreed = diag.get("btc_agrees") is True and diag.get("eth_agrees") is True
        if both_agreed:
            result["category"] = "market_structural_headwind"
            result["reason"] = "BTC/ETH both trending WITH this trade's direction at entry (the same condition the proven filter treats as unfavorable for other strategies - not backtested for this one specifically)"
            return result

    mfe = diag.get("mfe")
    result["category"] = "normal_probabilistic"
    if entry["status"] == "loss":
        if mfe is None:
            result["reason"] = "rules followed correctly, market alignment neutral - normal loss"
        elif mfe < 0.1:
            result["reason"] = f"wrong immediately (MFE {mfe:.0%}) - never moved favorably before hitting the stop"
        elif mfe >= 0.5:
            result["reason"] = f"near-miss (MFE {mfe:.0%}) - got most of the way to target before reversing"
        else:
            result["reason"] = f"modest favorable move (MFE {mfe:.0%}) before reversing"
    else:
        result["reason"] = "rules followed correctly, market alignment favorable/neutral - normal win"
    return result


def main():
    entries = [json.loads(l) for l in open(JOURNAL_PATH, encoding="utf-8") if l.strip()]
    resolved = [e for e in entries if e["status"] in ("win", "loss")]

    diag_rows = json.load(open(DIAGNOSTIC_PATH, encoding="utf-8")) if DIAGNOSTIC_PATH.exists() else []
    diag_by_id = {d["id"]: d for d in diag_rows}

    results = [categorize(e, diag_by_id.get(e["id"])) for e in resolved]

    OUT_PATH.write_text(json.dumps(results, indent=2))

    losses = [r for r in results if r["status"] == "loss"]
    wins = [r for r in results if r["status"] == "win"]

    def summarize(rows, label):
        print(f"\n=== {label} ({len(rows)} total) ===")
        by_cat = defaultdict(int)
        for r in rows:
            by_cat[r["category"]] += 1
        for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:28s} {n:4d}  ({n/len(rows)*100:.1f}%)")

    summarize(losses, "LOSSES")
    summarize(wins, "WINS")

    print("\n=== Loss reason breakdown (normal_probabilistic only) ===")
    normal_losses = [r for r in losses if r["category"] == "normal_probabilistic"]
    wrong_immediately = sum(1 for r in normal_losses if r.get("mfe") is not None and r["mfe"] < 0.1)
    near_miss = sum(1 for r in normal_losses if r.get("mfe") is not None and r["mfe"] >= 0.5)
    modest = sum(1 for r in normal_losses if r.get("mfe") is not None and 0.1 <= r["mfe"] < 0.5)
    unknown_mfe = sum(1 for r in normal_losses if r.get("mfe") is None)
    print(f"  wrong immediately (MFE<10%): {wrong_immediately}")
    print(f"  near-miss (MFE>=50%):        {near_miss}")
    print(f"  modest move (10-50%):        {modest}")
    print(f"  MFE unavailable:             {unknown_mfe}")

    print("\n=== Technical/rule-failure losses (sample) ===")
    tech_losses = [r for r in losses if r["category"] == "technical_rule_failure"]
    for r in tech_losses[:10]:
        print(f"  {r['symbol']:12s} {r['based_on']:20s} {r['bias']:8s} - {r['reason']}")

    print("\n=== Market structural headwind losses (sample) ===")
    market_losses = [r for r in losses if r["category"] == "market_structural_headwind"]
    for r in market_losses[:10]:
        print(f"  {r['symbol']:12s} {r['based_on']:20s} {r['bias']:8s}")

    print(f"\nFull detail written to {OUT_PATH}")


if __name__ == "__main__":
    main()

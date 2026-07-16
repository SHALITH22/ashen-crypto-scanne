"""
Multi-timeframe confluence filter.

A lower-timeframe signal is more trustworthy when higher timeframes agree.
annotate_htf() adds, to every risk plan in a scan result:
  htf_agrees: True  - no directional higher-TF disagrees (or no higher TF data)
              False - at least one directional higher TF has the opposite bias
  htf_note:   human-readable summary, e.g. "4h,1d agree" / "1d conflicts"

Timeframe order is taken from the config's `timeframes` list, which must be
sorted from lowest to highest.

Evaluated per PLAN (against that plan's own direction), not once per
timeframe against a single shared bias - since independent strategies can
now fire in opposite directions on the same symbol/timeframe (e.g.
jayantha_b2b bullish + marubozu_ashen bearish), each needs its own
independent HTF-agreement verdict rather than sharing one tf-level answer
that only makes sense for a single consolidated direction. `data["bias"]`
(each higher timeframe's own overall confluence_score across all its raw
signals) is still what a plan is checked against - that stays a useful
"what does the bigger picture lean toward" summary, just no longer used as
the plan's own direction.
"""


def annotate_htf(result: dict, timeframes: list[str]) -> dict:
    tfs = result["timeframes"]
    for tf, data in tfs.items():
        higher = [h for h in timeframes[timeframes.index(tf) + 1:]
                  if h in tfs and tfs[h]["bias"] != "mixed"]
        for plan in data.get("risk_plans", []):
            direction = plan["direction"]
            if not higher:
                plan["htf_agrees"] = True
                plan["htf_note"] = "no higher-TF data"
                continue
            agree = [h for h in higher if tfs[h]["bias"] == direction]
            conflict = [h for h in higher if tfs[h]["bias"] != direction]
            plan["htf_agrees"] = not conflict
            parts = []
            if agree:
                parts.append(",".join(agree) + " agree")
            if conflict:
                parts.append(",".join(conflict) + " conflict")
            plan["htf_note"] = "; ".join(parts)
    return result

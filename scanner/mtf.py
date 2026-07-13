"""
Multi-timeframe confluence filter.

A lower-timeframe signal is more trustworthy when higher timeframes agree.
annotate_htf() adds, to every timeframe entry in a scan result:
  htf_agrees: True  - no directional higher-TF disagrees (or no higher TF data)
              False - at least one directional higher TF has the opposite bias
  htf_note:   human-readable summary, e.g. "4h,1d agree" / "1d conflicts"

Timeframe order is taken from the config's `timeframes` list, which must be
sorted from lowest to highest.
"""


def annotate_htf(result: dict, timeframes: list[str]) -> dict:
    tfs = result["timeframes"]
    for tf, data in tfs.items():
        if data["bias"] == "mixed":
            data["htf_agrees"] = True
            data["htf_note"] = "n/a (mixed bias)"
            continue
        higher = [h for h in timeframes[timeframes.index(tf) + 1:]
                  if h in tfs and tfs[h]["bias"] != "mixed"]
        if not higher:
            data["htf_agrees"] = True
            data["htf_note"] = "no higher-TF data"
            continue
        agree = [h for h in higher if tfs[h]["bias"] == data["bias"]]
        conflict = [h for h in higher if tfs[h]["bias"] != data["bias"]]
        data["htf_agrees"] = not conflict
        parts = []
        if agree:
            parts.append(",".join(agree) + " agree")
        if conflict:
            parts.append(",".join(conflict) + " conflict")
        data["htf_note"] = "; ".join(parts)
    return result

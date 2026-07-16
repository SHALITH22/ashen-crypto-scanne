"""
Ashen strategy detector layer - orchestrates the four "Trading with Ashen"
detectors (VWAP breakout, MA cross + dominant candle, Marubozu, B2B
counter-candle) into main.py's live signal pipeline, ALONGSIDE
run_jayantha_detectors (not replacing it) - this is a second, independent
trader's rules added as more detectors feeding the same signals list.

Each sub-detector already returns signals in the shape
{"name", "direction", "detail", "stop", "target"} - the same contract
patterns.py's retired detectors and jayantha_detectors.py use - so no
other code downstream (attach_atr_risk, confluence_score, setup_risk_plan,
journal, notify) needs to change for these to work. scanner/risk.py's
STRUCTURAL_NAMES includes all four Ashen detector names (they carry their
own geometry, same as jayantha_b2b) and excludes them from
MARKET_FILTER_NAMES (that beta filter is unproven for anything outside
the original retired detector set - see risk.py's comments).
"""

import pandas as pd

from scanner import ashen_b2b, ashen_ma_cross, ashen_marubozu, ashen_vwap_breakout


def run_ashen_detectors(df: pd.DataFrame, cfg: dict, htf_df: pd.DataFrame | None = None) -> list[dict]:
    """
    Drop-in alongside run_jayantha_detectors - same signature shape
    (signals list, same shared contract), called from the same spot in
    main.py's scan_pair().

    htf_df: the paired higher-timeframe candles for THIS entry timeframe,
    per config/settings.yaml's ashen.vwap_breakout.htf_pairing - None if
    no pairing is configured (or the paired timeframe wasn't fetched this
    run), in which case ashen_vwap_breakout simply doesn't fire and every
    other Ashen detector is unaffected.
    """
    acfg = cfg.get("ashen", {})
    if not acfg.get("enabled", True):
        return []

    signals: list[dict] = []
    signals += ashen_b2b.detect_signals(df, cfg)
    signals += ashen_ma_cross.detect_signals(df, cfg)
    signals += ashen_marubozu.detect_signals(df, cfg)
    signals += ashen_vwap_breakout.detect_signals(df, htf_df, cfg)
    return signals

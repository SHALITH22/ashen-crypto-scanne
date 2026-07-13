"""
Jayantha strategy detector layer - REPLACES the generic multi-pattern
detector set (patterns.py's run_all_detectors) as main.py's live signal
source.

This is the only file that changes what main.py actually trades on. Every
downstream stage - attach_atr_risk, confluence_score, setup_risk_plan,
the live journal, Telegram notify, console/JSON output - is untouched and
keeps working exactly as before, because this module returns signals in
the identical shape patterns.py always did:
    {"name": ..., "direction": "bullish"/"bearish", "detail": ...,
     "stop": <optional>, "target": <optional>}

Strategy source: Jayantha Ukwatta's B2B ("buy near the moving average,
sell away from it") entry technique, combined with his "confirmation
before conviction" rule - never trade an approaching level, only a
CLOSED candle that has actually confirmed it - as extracted from ~594
hours of his trading commentary (see the project's crypto knowledge base).

Three real conditions are folded into up to three confluent signals per
detected setup, so the existing min_confluence gate (settings.yaml,
unchanged) stays meaningful for this strategy exactly as it did for the
old one:

  1. jayantha_b2b        - the pullback/rally itself (always present when
                            a setup fires; carries its own geometry-based
                            stop/target, same contract as the old
                            "structural" pattern detectors)
  2. jayantha_trend       - EMA stack alignment (a precondition for
                            jayantha_b2b firing at all, surfaced as its
                            own confluent signal rather than hidden)
  3. jayantha_confirmation - only added when the bounce/rejection off the
                            MA is CLEANLY confirmed by a closed candle
                            (confirmation score >= min_confirmation_bonus),
                            not just a marginal close

Setups where the confirmation score falls below min_confirmation_to_fire
are discarded entirely - a wick-only touch or barely-there close never
reaches the console, journal, or Telegram, matching Jayantha's explicit
rule that an unconfirmed level is not a trade.
"""

import pandas as pd

from scanner.jayantha_b2b import B2BDetector
from scanner.jayantha_confirmation import ConfirmationValidator

# Defaults used if config/settings.yaml has no `jayantha:` section at all
# (keeps this module safe to import/call even against an older config).
_DEFAULT_B2B_CFG = {
    "ma_periods": [20, 50, 100, 200],
    "entry_ma_period": 50,
    "trend_ma_period": 200,
    "stop_ma_period": 100,
    "pullback_tolerance_pct": 1.5,
    "min_pullback_depth_pct": 2.0,
}

_DEFAULT_CONFIRMATION_CFG = {
    "min_close_beyond_level_pct": 0.2,
    "fakeout_detection_candles": 5,
    "require_closed_candle": True,
    # Setup is discarded entirely below this (wick-only / barely-touched)
    "min_confirmation_to_fire": 0.3,
    # Only above this does the setup earn the extra jayantha_confirmation
    # confluence signal (a clean, well-confirmed close)
    "min_confirmation_bonus": 0.65,
}


def _hashable(value):
    """List-valued config (e.g. ma_periods) isn't hashable as-is - needed
    since this feeds a dict key below, not because config.yaml itself
    requires it."""
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def _detector_cache_key(cfg_b2b: dict) -> tuple:
    return tuple(sorted(((k, _hashable(v)) for k, v in cfg_b2b.items()), key=lambda kv: kv[0]))


# B2BDetector/ConfirmationValidator are pure/stateless given their config,
# so they're built once per distinct config rather than once per candle
# (main.py calls this once per symbol x timeframe, many times a run).
_b2b_cache: dict[tuple, B2BDetector] = {}
_confirmation_cache: dict[tuple, ConfirmationValidator] = {}


def _get_b2b_detector(b2b_cfg: dict) -> B2BDetector:
    key = _detector_cache_key(b2b_cfg)
    if key not in _b2b_cache:
        _b2b_cache[key] = B2BDetector(
            ma_periods=b2b_cfg.get("ma_periods", _DEFAULT_B2B_CFG["ma_periods"]),
            entry_ma_period=b2b_cfg.get("entry_ma_period", _DEFAULT_B2B_CFG["entry_ma_period"]),
            trend_ma_period=b2b_cfg.get("trend_ma_period", _DEFAULT_B2B_CFG["trend_ma_period"]),
            stop_ma_period=b2b_cfg.get("stop_ma_period", _DEFAULT_B2B_CFG["stop_ma_period"]),
            pullback_tolerance_pct=b2b_cfg.get("pullback_tolerance_pct", _DEFAULT_B2B_CFG["pullback_tolerance_pct"]),
            min_pullback_depth_pct=b2b_cfg.get("min_pullback_depth_pct", _DEFAULT_B2B_CFG["min_pullback_depth_pct"]),
        )
    return _b2b_cache[key]


def _get_confirmation_validator(conf_cfg: dict) -> ConfirmationValidator:
    key = _detector_cache_key({k: v for k, v in conf_cfg.items()
                               if k in ("min_close_beyond_level_pct", "fakeout_detection_candles",
                                        "require_closed_candle")})
    if key not in _confirmation_cache:
        _confirmation_cache[key] = ConfirmationValidator(
            min_close_beyond_level_pct=conf_cfg.get(
                "min_close_beyond_level_pct", _DEFAULT_CONFIRMATION_CFG["min_close_beyond_level_pct"]),
            fakeout_detection_candles=conf_cfg.get(
                "fakeout_detection_candles", _DEFAULT_CONFIRMATION_CFG["fakeout_detection_candles"]),
            require_closed_candle=conf_cfg.get(
                "require_closed_candle", _DEFAULT_CONFIRMATION_CFG["require_closed_candle"]),
        )
    return _confirmation_cache[key]


def run_jayantha_detectors(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """
    Drop-in replacement for patterns.run_all_detectors - same signature,
    same return shape, called from the same spot in main.py's scan_pair().

    Returns [] if Jayantha strategy is disabled, no B2B setup exists on
    this timeframe, or the setup exists but fails confirmation entirely.
    """
    jcfg = cfg.get("jayantha", {})
    if not jcfg.get("enabled", True):
        return []

    b2b_cfg = jcfg.get("b2b", {})
    conf_cfg = jcfg.get("confirmation", {})

    detector = _get_b2b_detector(b2b_cfg)
    validator = _get_confirmation_validator(conf_cfg)

    setup = detector.detect_b2b_setup(df, direction="both")
    if not setup["b2b_found"]:
        return []

    direction = "bullish" if setup["setup_type"] == "bullish_pullback" else "bearish"
    details = setup["details"]
    # details["entry_ma"] is the MA *period* (e.g. 50), not its price -
    # recompute the actual price level to hand to the confirmation validator.
    ma_periods = b2b_cfg.get("ma_periods", _DEFAULT_B2B_CFG["ma_periods"])
    entry_ma_period = b2b_cfg.get("entry_ma_period", _DEFAULT_B2B_CFG["entry_ma_period"])
    entry_ma_price = df["close"].ewm(span=entry_ma_period, adjust=False).mean().iloc[-1]

    conf_direction = "up" if direction == "bullish" else "down"
    conf_score = validator.get_confirmation_score(df, entry_ma_price, conf_direction)

    min_to_fire = conf_cfg.get("min_confirmation_to_fire", _DEFAULT_CONFIRMATION_CFG["min_confirmation_to_fire"])
    if conf_score < min_to_fire:
        # Wick-only touch or unconfirmed close - Jayantha's rule is this
        # never becomes a trade, so it never becomes an alert either.
        return []

    signals = [{
        "name": "jayantha_b2b",
        "direction": direction,
        "detail": f"{setup['rationale']} (confirmation {conf_score:.0%})",
        "stop": setup["stop_price"],
        "target": setup["target_price"],
    }]

    if details.get("ema_stack_aligned"):
        # Bullish stack means the SHORTEST period has the HIGHEST value
        # (EMA20 > EMA50 > EMA100 > EMA200), so periods are listed ascending;
        # bearish is the mirror (EMA200 > ... > EMA20), periods descending.
        stack_order = " > ".join(str(p) for p in sorted(ma_periods, reverse=(direction == "bearish")))
        signals.append({
            "name": "jayantha_trend",
            "direction": direction,
            "detail": f"EMA stack aligned {direction} ({stack_order})",
        })

    min_bonus = conf_cfg.get("min_confirmation_bonus", _DEFAULT_CONFIRMATION_CFG["min_confirmation_bonus"])
    if conf_score >= min_bonus:
        signals.append({
            "name": "jayantha_confirmation",
            "direction": direction,
            "detail": f"Close cleanly confirmed beyond {entry_ma_period}MA pullback zone (score {conf_score:.0%})",
        })

    return signals

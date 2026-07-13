"""
Jayantha Confirmation-Before-Conviction Rule Validator

Philosophy: "Never treat approaching pattern as tradeable; wait for CLOSE beyond level."

The "closed door, not the door that looks like it might open" rule.

Jayantha's exact sequence: Break → Fakeout → Retest → Real Move

Key rules:
1. Only trade CLOSED candles beyond levels, never wick touches
2. Multiple fakeouts expected before real move (patience!)
3. Whipsaw through obvious round levels is common (stop-hunt behavior)
4. Weekly/monthly closes matter more than intraday
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List


class ConfirmationValidator:
    """
    Validates Jayantha's confirmation-before-conviction rule

    A setup is only tradeable AFTER:
    1. Price closes beyond the level (not just touches via wick)
    2. Enough candles have closed to avoid fakeouts
    3. Prior fakeouts are acknowledged (expect bounces before real move)
    """

    def __init__(self,
                 min_close_beyond_level_pct: float = 0.2,
                 fakeout_detection_candles: int = 5,
                 require_closed_candle: bool = True,
                 min_closed_candles_after_break: int = 1):
        """
        Initialize confirmation validator

        Args:
            min_close_beyond_level_pct: Close must be at least this far beyond level (%)
            fakeout_detection_candles: Look back this many candles for prior fakeouts
            require_closed_candle: Only trade if close confirmed (never wick-only)
            min_closed_candles_after_break: How many candles after break before trading?
        """
        self.min_close_beyond_level_pct = min_close_beyond_level_pct
        self.fakeout_detection_candles = fakeout_detection_candles
        self.require_closed_candle = require_closed_candle
        self.min_closed_candles_after_break = min_closed_candles_after_break

    def validate_breakout_closure(self,
                                   candles: pd.DataFrame,
                                   breakout_level: float,
                                   breakout_direction: str = "up") -> Dict[str, Any]:
        """
        Is the breakout CONFIRMED by a closed candle beyond the level?

        Args:
            candles: OHLCV candles
            breakout_level: Level that was supposed to break
            breakout_direction: "up" or "down"

        Returns:
            {
                "breakout_confirmed": bool,
                "close_beyond_level_pct": float,
                "candles_since_close_beyond": int,
                "wick_touched_first": bool,  # Did it touch via wick before close?
                "rejection_type": str or None,  # Why not confirmed?
                "confirmation_strength": float (0-1),
                "rationale": str
            }
        """
        result = {
            "breakout_confirmed": False,
            "close_beyond_level_pct": 0.0,
            "candles_since_close_beyond": -1,
            "wick_touched_first": False,
            "rejection_type": None,
            "confirmation_strength": 0.0,
            "rationale": ""
        }

        if len(candles) < 2:
            result["rejection_type"] = "not_enough_candles"
            result["rationale"] = "Not enough candles to validate"
            return result

        last_candle = candles.iloc[-1]
        close = last_candle["close"]
        high = last_candle["high"]
        low = last_candle["low"]

        # Check if wick touched the level
        if breakout_direction == "up":
            wick_touched = high >= breakout_level
            close_beyond = close > breakout_level
            distance_beyond = ((close - breakout_level) / breakout_level) * 100
        else:  # down
            wick_touched = low <= breakout_level
            close_beyond = close < breakout_level
            distance_beyond = ((breakout_level - close) / breakout_level) * 100

        # Scenario 1: Wick-only touch (not confirmed by close)
        if wick_touched and not close_beyond:
            result["wick_touched_first"] = True
            result["rejection_type"] = "wick_only_touch"
            result["rationale"] = "Wick touched level but close did NOT confirm (wick-only). Likely fakeout."
            result["confirmation_strength"] = 0.1  # Very low confidence
            return result

        # Scenario 2: Close beyond, but too small
        if close_beyond and distance_beyond < self.min_close_beyond_level_pct:
            result["rejection_type"] = "close_too_small"
            result["close_beyond_level_pct"] = distance_beyond
            result["rationale"] = (
                f"Close is beyond level, but only by {distance_beyond:.3f}%. "
                f"Minimum required: {self.min_close_beyond_level_pct}%. Possible stop-hunt."
            )
            result["confirmation_strength"] = 0.3
            return result

        # Scenario 3: Confirmed!
        if close_beyond and distance_beyond >= self.min_close_beyond_level_pct:
            result["breakout_confirmed"] = True
            result["close_beyond_level_pct"] = distance_beyond
            result["candles_since_close_beyond"] = 1
            result["confirmation_strength"] = min(1.0, 0.6 + (distance_beyond / 2.0) * 0.1)
            result["rationale"] = (
                f"Breakout CONFIRMED: Close {distance_beyond:.3f}% beyond level. "
                f"This is a real breakout, not a wick-hunt."
            )
            return result

        # Scenario 4: Price didn't break at all
        if not wick_touched:
            result["rejection_type"] = "no_break_yet"
            result["rationale"] = "Price hasn't even touched the level yet. No breakout to confirm."
            return result

        return result

    def validate_pattern_closure(self,
                                  candles: pd.DataFrame,
                                  pattern_type: str,
                                  pattern_high: float,
                                  pattern_low: float,
                                  breakout_direction: str = "up") -> Dict[str, Any]:
        """
        Validate chart pattern closure (flag, triangle, wedge, etc.)

        Pattern breakout only 'counts' if:
        1. Closes beyond the pattern boundary (not wick-only)
        2. With sufficient distance beyond
        3. With bullish/bearish candle confirmation

        Args:
            candles: OHLCV
            pattern_type: "flag", "triangle", "wedge", "cup_handle", etc.
            pattern_high: Top of consolidation
            pattern_low: Bottom of consolidation
            breakout_direction: "up" (bullish flag) or "down" (bearish flag)
        """
        result = {
            "pattern_confirmed": False,
            "breakout_level": pattern_high if breakout_direction == "up" else pattern_low,
            "closed_beyond": False,
            "candle_confirmation": False,
            "confirmation_strength": 0.0,
            "rejection_type": None,
            "rationale": ""
        }

        if len(candles) < 2:
            result["rejection_type"] = "insufficient_data"
            result["rationale"] = "Not enough candles"
            return result

        last_candle = candles.iloc[-1]
        breakout_level = result["breakout_level"]

        # Check 1: Close beyond pattern boundary
        if breakout_direction == "up":
            closed_beyond = last_candle["close"] > breakout_level
            distance = ((last_candle["close"] - breakout_level) / breakout_level) * 100
            bullish_candle = last_candle["close"] > last_candle["open"]
            candle_confirmation = bullish_candle
        else:  # down
            closed_beyond = last_candle["close"] < breakout_level
            distance = ((breakout_level - last_candle["close"]) / breakout_level) * 100
            bearish_candle = last_candle["close"] < last_candle["open"]
            candle_confirmation = bearish_candle

        result["closed_beyond"] = closed_beyond
        result["candle_confirmation"] = candle_confirmation

        # Validation
        if not closed_beyond:
            result["rejection_type"] = "close_inside_consolidation"
            result["rationale"] = "Close is still inside the pattern consolidation. No breakout yet."
            return result

        if distance < self.min_close_beyond_level_pct:
            result["rejection_type"] = "close_too_close_to_boundary"
            result["rationale"] = (
                f"Close only {distance:.2f}% beyond pattern boundary. "
                f"Likely trapped within consolidation noise."
            )
            result["confirmation_strength"] = 0.2
            return result

        if not candle_confirmation:
            result["rejection_type"] = "wrong_candle_direction"
            result["rationale"] = (
                f"Closed beyond pattern boundary, but candle body is {
                'bearish' if breakout_direction == 'up' else 'bullish'
                }. "
                f"Not a clean confirmation."
            )
            result["confirmation_strength"] = 0.3
            return result

        # Confirmed!
        result["pattern_confirmed"] = True
        result["confirmation_strength"] = min(1.0, 0.7 + (distance / 5.0) * 0.2)
        result["rationale"] = (
            f"{pattern_type.capitalize()} breakout CONFIRMED: "
            f"Closed {distance:.2f}% beyond boundary with "
            f"{'bullish' if breakout_direction == 'up' else 'bearish'} candle."
        )
        return result

    def detect_fakeout_pattern(self,
                                candles: pd.DataFrame,
                                prior_level: float,
                                lookback: int = None) -> Dict[str, Any]:
        """
        Has this level been broken and faked out before?

        Returns:
            {
                "prior_fakeouts": int,
                "fakeout_indices": [candle indices],
                "pattern": "multiple_fakeouts" | "first_attempt" | "retest",
                "expected_real_move_probability": float (0-1),
                "rationale": str
            }
        """
        lookback = lookback or self.fakeout_detection_candles
        result = {
            "prior_fakeouts": 0,
            "fakeout_indices": [],
            "pattern": "first_attempt",
            "expected_real_move_probability": 0.5,
            "rationale": "Level hasn't been tested before"
        }

        if len(candles) < lookback + 5:
            return result

        # Look back for prior touches of this level (±0.5%)
        tolerance = (prior_level * 0.005)  # 0.5% tolerance
        recent = candles.iloc[-lookback:]

        fakeout_count = 0
        fakeout_indices = []

        for i, (idx, candle) in enumerate(recent.iterrows()):
            high = candle["high"]
            low = candle["low"]
            close = candle["close"]

            # Did this candle touch the level?
            touched = (low <= prior_level <= high)

            if touched:
                # Was it a wick-only touch (close didn't confirm)?
                close_confirmed = abs(close - prior_level) < tolerance
                if not close_confirmed:
                    fakeout_count += 1
                    fakeout_indices.append(i)

        result["prior_fakeouts"] = fakeout_count
        result["fakeout_indices"] = fakeout_indices

        # Interpret the pattern
        if fakeout_count == 0:
            result["pattern"] = "first_attempt"
            result["expected_real_move_probability"] = 0.4
            result["rationale"] = "Level being tested for the first time"
        elif fakeout_count == 1:
            result["pattern"] = "one_fakeout"
            result["expected_real_move_probability"] = 0.6
            result["rationale"] = "One prior fakeout. Next attempt often succeeds."
        else:  # 2+
            result["pattern"] = "multiple_fakeouts"
            result["expected_real_move_probability"] = 0.75
            result["rationale"] = (
                f"{fakeout_count} prior fakeouts. Market is testing resolve. "
                f"Next real break likely imminent."
            )

        return result

    def get_confirmation_score(self,
                                candles: pd.DataFrame,
                                signal_level: float,
                                signal_direction: str) -> float:
        """
        Overall confirmation score (0-1)

        0.0 = just approaching level, no wick touch
        0.3 = wick touched but close didn't confirm
        0.5 = close confirmed by small amount (< 0.5%)
        0.7 = close confirmed cleanly
        0.9+ = close strongly confirmed + multiple candles + prior fakeouts

        Args:
            candles: OHLCV
            signal_level: Level that should be confirmed
            signal_direction: "up" (expect break above) or "down" (expect break below)

        Returns:
            float (0-1): Confirmation score
        """
        if len(candles) < 2:
            return 0.0

        last = candles.iloc[-1]
        close = last["close"]
        high = last["high"]
        low = last["low"]

        if signal_direction == "up":
            if close > signal_level:
                distance = ((close - signal_level) / signal_level) * 100
                if distance >= 0.5:
                    return 0.85
                elif distance >= 0.2:
                    return 0.65
                else:
                    return 0.35
            elif high > signal_level:
                return 0.15  # Wick touch
            else:
                return 0.0  # No touch
        else:  # down
            if close < signal_level:
                distance = ((signal_level - close) / signal_level) * 100
                if distance >= 0.5:
                    return 0.85
                elif distance >= 0.2:
                    return 0.65
                else:
                    return 0.35
            elif low < signal_level:
                return 0.15  # Wick touch
            else:
                return 0.0  # No touch

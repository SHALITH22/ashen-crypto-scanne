"""
Jayantha B2B (Buy at Moving Average) Entry Detector

Philosophy: "Buy near the moving average, sell away from it"
- Enter on pullbacks to rising MAs inside uptrends (bullish)
- Enter on rallies to falling MAs inside downtrends (bearish)
- Avoid chasing already-extended candles
- Stop just beyond a key MA - specifically the 100 MA (not the 200), a
  deliberate middle ground: the 200 MA is the trend filter, not the stop,
  since for a pullback ENTERED at the 50 MA it sits far enough away to
  produce a poor risk:reward on a normal pullback (confirmed by testing -
  a 200-MA-based stop failed this system's own min_risk_reward gate on a
  realistic crafted setup, discarding the whole trade plan)

Jayantha's exact rule:
  "Don't enter chasing green candles. Wait for pullback to MA, then enter
   when price bounces off that MA with a bullish candle."
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


class B2BDetector:
    """
    Detects Jayantha's B2B (Buy at Moving Average) pullback setups

    A B2B setup is:
    1. Established uptrend (price > 200 MA, EMA stack aligned)
    2. Recent pullback (price near 50 MA)
    3. Pullback hasn't broken the 200 MA (still in uptrend)
    4. Bounces off the MA with bullish candle (confirmation)
    5. Stop below 100 MA; target at prior swing high
    """

    def __init__(self,
                 ma_periods: list = None,
                 entry_ma_period: int = 50,
                 trend_ma_period: int = 200,
                 stop_ma_period: int = 100,
                 pullback_tolerance_pct: float = 1.5,
                 min_pullback_depth_pct: float = 2.0,
                 ema_aligned_lookback: int = 5):
        """
        Initialize B2B detector

        Args:
            ma_periods: Which MAs to calculate [20, 50, 100, 200]
            entry_ma_period: Which MA to buy near (50)
            trend_ma_period: Which MA defines trend (200) - the trend
                filter only; NOT used for the stop (see stop_ma_period)
            stop_ma_period: Which MA the stop sits just beyond (100) -
                closer than the trend MA, so the stop stays tied to the
                entry technique itself rather than the whole trend
            pullback_tolerance_pct: How close to MA = "touched" (1.5%)
            min_pullback_depth_pct: Minimum pullback depth to qualify (avoid shallow)
            ema_aligned_lookback: How many candles back to check EMA stack alignment
        """
        self.ma_periods = ma_periods or [20, 50, 100, 200]
        self.entry_ma_period = entry_ma_period
        self.trend_ma_period = trend_ma_period
        self.stop_ma_period = stop_ma_period
        self.pullback_tolerance_pct = pullback_tolerance_pct
        self.min_pullback_depth_pct = min_pullback_depth_pct
        self.ema_aligned_lookback = ema_aligned_lookback

    def detect_b2b_setup(self,
                         candles: pd.DataFrame,
                         direction: str = "both") -> Dict[str, Any]:
        """
        Detect B2B setup (bullish pullback or bearish rally)

        Args:
            candles: OHLCV candles with columns [open, high, low, close, volume]
            direction: "up" (bullish only), "down" (bearish only), "both"

        Returns:
            {
                "b2b_found": bool,
                "setup_type": "bullish_pullback" | "bearish_rally" | None,
                "entry_price": float,
                "stop_price": float,
                "target_price": float,
                "confidence": float (0-1),
                "rationale": str,
                "details": {
                    "entry_ma": int,
                    "trend_ma": int,
                    "pullback_depth_pct": float,
                    "price_near_ma_pct": float,
                    "ema_stack_aligned": bool,
                    "trend_intact": bool
                }
            }
        """
        # Initialize output
        result = {
            "b2b_found": False,
            "setup_type": None,
            "entry_price": None,
            "stop_price": None,
            "target_price": None,
            "confidence": 0.0,
            "rationale": "",
            "details": {}
        }

        # Ensure we have enough candles (robust to stop_ma_period being
        # reconfigured larger than trend_ma_period, though the defaults
        # never trigger that case)
        min_candles = max(self.trend_ma_period, self.stop_ma_period) + 10
        if len(candles) < min_candles:
            result["rationale"] = f"Not enough candles ({len(candles)} < {min_candles})"
            return result

        # Compute MAs
        mas = self._compute_moving_averages(candles)
        if not mas:
            result["rationale"] = "Failed to compute moving averages"
            return result

        # Current price and MA levels
        current_price = candles.iloc[-1]["close"]
        entry_ma = mas[self.entry_ma_period].iloc[-1]
        trend_ma = mas[self.trend_ma_period].iloc[-1]
        stop_ma = mas[self.stop_ma_period].iloc[-1]

        # Bullish B2B (pullback in uptrend)
        if direction in ["up", "both"]:
            bullish_setup = self._detect_bullish_b2b(candles, mas, current_price, entry_ma, trend_ma, stop_ma)
            if bullish_setup["found"]:
                result.update(bullish_setup["result"])
                return result

        # Bearish B2B (rally in downtrend)
        if direction in ["down", "both"]:
            bearish_setup = self._detect_bearish_b2b(candles, mas, current_price, entry_ma, trend_ma, stop_ma)
            if bearish_setup["found"]:
                result.update(bearish_setup["result"])
                return result

        result["rationale"] = "No B2B setup detected"
        return result

    def _detect_bullish_b2b(self, candles, mas, current_price, entry_ma, trend_ma, stop_ma):
        """Detect bullish B2B: pullback to MA in uptrend"""

        result = {
            "found": False,
            "result": {}
        }

        # Check 1: Trend intact (price > trend MA)
        if current_price <= trend_ma:
            return result  # No uptrend

        # Check 2: EMA stack aligned (bullish)
        ema_aligned = self._check_ema_stack_aligned_bullish(mas)
        if not ema_aligned:
            return result  # Stack not aligned

        last_candle = candles.iloc[-1]

        # Check 3: The candle's LOW actually reached down into the pullback
        # zone (at, or within tolerance above, the entry MA) - this is the
        # "touch", using the wick/low rather than the close so a pullback
        # that dipped through the MA and is now reclaiming it still counts.
        touch_distance_pct = ((last_candle["low"] - entry_ma) / entry_ma) * 100
        touched_ma_zone = touch_distance_pct <= self.pullback_tolerance_pct
        if not touched_ma_zone:
            return result  # Price never pulled back to MA

        # Check 4: Pullback depth is meaningful (at least 2%)
        pullback_high = candles["high"].iloc[-20:].max()  # Recent high
        pullback_depth = ((pullback_high - current_price) / pullback_high) * 100

        if pullback_depth < self.min_pullback_depth_pct:
            return result  # Pullback too shallow

        # Check 5: Reclaim confirmation - close is a bullish body AND has
        # closed back ABOVE the entry MA (not just touched it with a wick).
        # This is what "confirmation before conviction" actually means here:
        # a candle that dipped to the MA and closed back below it is still
        # an open question, not a trade.
        bullish_candle = last_candle["close"] > last_candle["open"]
        reclaimed_ma = current_price > entry_ma

        if not bullish_candle or not reclaimed_ma:
            return result  # No confirmed reclaim yet

        reclaim_distance_pct = ((current_price - entry_ma) / entry_ma) * 100

        # Compute stop and target
        stop_price = stop_ma * 0.995  # Slight buffer below the 100 MA (stop_ma_period)
        target_price = pullback_high * 1.02  # Prior high with slight buffer

        # Confidence scoring
        confidence = 0.7  # Base confidence
        if reclaim_distance_pct >= 0.3:  # Decisive reclaim, not a marginal one
            confidence += 0.1
        if ema_aligned:
            confidence += 0.1
        if pullback_depth >= 5.0:  # Deeper pullback = more confidence
            confidence += 0.05
        confidence = min(confidence, 1.0)

        result["found"] = True
        result["result"] = {
            "b2b_found": True,
            "setup_type": "bullish_pullback",
            "entry_price": current_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "confidence": confidence,
            "rationale": (
                f"Bullish B2B: Price pulled back into the {self.entry_ma_period} MA zone "
                f"then reclaimed it (closed {reclaim_distance_pct:.2f}% above), EMA stack "
                f"aligned, trend intact above {self.trend_ma_period} MA. Stop below "
                f"{self.stop_ma_period} MA."
            ),
            "details": {
                "entry_ma": self.entry_ma_period,
                "trend_ma": self.trend_ma_period,
                "stop_ma": self.stop_ma_period,
                "pullback_depth_pct": pullback_depth,
                "price_near_ma_pct": abs(reclaim_distance_pct),
                "ema_stack_aligned": True,
                "trend_intact": True
            }
        }
        return result

    def _detect_bearish_b2b(self, candles, mas, current_price, entry_ma, trend_ma, stop_ma):
        """Detect bearish B2B: rally to MA in downtrend"""

        result = {
            "found": False,
            "result": {}
        }

        # Check 1: Downtrend intact (price < trend MA)
        if current_price >= trend_ma:
            return result  # No downtrend

        # Check 2: EMA stack aligned (bearish)
        ema_aligned = self._check_ema_stack_aligned_bearish(mas)
        if not ema_aligned:
            return result  # Stack not aligned

        last_candle = candles.iloc[-1]

        # Check 3: The candle's HIGH actually reached up into the rally
        # zone (at, or within tolerance below, the entry MA) - the wick/high
        # rather than the close, so a rally that poked through the MA and
        # is now being rejected back down still counts.
        touch_distance_pct = ((entry_ma - last_candle["high"]) / entry_ma) * 100
        touched_ma_zone = touch_distance_pct <= self.pullback_tolerance_pct
        if not touched_ma_zone:
            return result  # Price never rallied to MA

        # Check 4: Rally depth is meaningful (at least 2%)
        rally_low = candles["low"].iloc[-20:].min()  # Recent low
        rally_depth = ((current_price - rally_low) / rally_low) * 100

        if rally_depth < self.min_pullback_depth_pct:
            return result  # Rally too shallow

        # Check 5: Rejection confirmation - close is a bearish body AND has
        # closed back BELOW the entry MA (not just poked it with a wick).
        bearish_candle = last_candle["close"] < last_candle["open"]
        rejected_at_ma = current_price < entry_ma

        if not bearish_candle or not rejected_at_ma:
            return result  # No confirmed rejection yet

        rejection_distance_pct = ((entry_ma - current_price) / entry_ma) * 100

        # Compute stop and target
        stop_price = stop_ma * 1.005  # Slight buffer above the 100 MA (stop_ma_period)
        target_price = rally_low * 0.98  # Prior low with slight buffer

        # Confidence scoring
        confidence = 0.7  # Base
        if rejection_distance_pct >= 0.3:  # Decisive rejection, not marginal
            confidence += 0.1
        if ema_aligned:
            confidence += 0.1
        if rally_depth >= 5.0:  # Deeper rally
            confidence += 0.05
        confidence = min(confidence, 1.0)

        result["found"] = True
        result["result"] = {
            "b2b_found": True,
            "setup_type": "bearish_rally",
            "entry_price": current_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "confidence": confidence,
            "rationale": (
                f"Bearish B2B: Price rallied into the {self.entry_ma_period} MA zone then "
                f"was rejected (closed {rejection_distance_pct:.2f}% below), EMA stack "
                f"aligned downward, downtrend intact below {self.trend_ma_period} MA. "
                f"Stop above {self.stop_ma_period} MA."
            ),
            "details": {
                "entry_ma": self.entry_ma_period,
                "trend_ma": self.trend_ma_period,
                "stop_ma": self.stop_ma_period,
                "pullback_depth_pct": rally_depth,
                "price_near_ma_pct": abs(rejection_distance_pct),
                "ema_stack_aligned": True,
                "trend_intact": True
            }
        }
        return result

    def _compute_moving_averages(self, candles: pd.DataFrame) -> Optional[Dict[int, pd.Series]]:
        """Compute all needed moving averages"""
        try:
            mas = {}
            for period in self.ma_periods:
                if period <= len(candles):
                    mas[period] = candles["close"].ewm(span=period, adjust=False).mean()
            return mas if mas else None
        except Exception as e:
            print(f"Error computing MAs: {e}")
            return None

    def _check_ema_stack_aligned_bullish(self, mas: Dict[int, pd.Series]) -> bool:
        """Check if EMA stack is aligned bullish: 20 > 50 > 100 > 200"""
        try:
            latest = {period: mas[period].iloc[-1] for period in self.ma_periods if period in mas}
            if len(latest) < 4:
                return False

            # Bullish: 20 > 50 > 100 > 200
            periods_sorted = sorted(latest.keys())
            for i in range(len(periods_sorted) - 1):
                if latest[periods_sorted[i]] < latest[periods_sorted[i + 1]]:
                    return False  # Reversed
            return True
        except:
            return False

    def _check_ema_stack_aligned_bearish(self, mas: Dict[int, pd.Series]) -> bool:
        """Check if EMA stack is aligned bearish: 20 < 50 < 100 < 200"""
        try:
            latest = {period: mas[period].iloc[-1] for period in self.ma_periods if period in mas}
            if len(latest) < 4:
                return False

            # Bearish: 20 < 50 < 100 < 200
            periods_sorted = sorted(latest.keys())
            for i in range(len(periods_sorted) - 1):
                if latest[periods_sorted[i]] > latest[periods_sorted[i + 1]]:
                    return False  # Reversed
            return True
        except:
            return False

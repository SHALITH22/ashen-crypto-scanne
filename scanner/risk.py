"""
Risk levels (stop-loss / target) for each signal.

Structural patterns (double top/bottom, head & shoulders, triangle/wedge,
flag/pennant) already compute geometry-based stop/target inside
patterns.py, since that's where the pivot points and necklines are known.
This module fills in a generic ATR-based stop/target for every other
signal (trend/momentum/S-R/candlestick/divergence), and picks one
consolidated entry/stop/target for the overall setup shown in the console
and Telegram alert.
"""

from math import floor, log10


def round_sig(x: float, sig: int = 6) -> float:
    """
    Round to N significant figures, not N decimal places. round(x, 6) is
    fine for a $63,000 BTC price but destroys almost all precision on a
    $0.00000428 SHIB-style price - entry/stop/target all collapsing to the
    same displayed "4e-06" is exactly that bug, not a coincidence.
    """
    if x == 0 or x != x:  # zero or NaN
        return 0.0
    digits = sig - int(floor(log10(abs(x)))) - 1
    return round(x, digits)


STRUCTURAL_NAMES = {
    "double_top", "double_bottom", "head_and_shoulders", "inverse_head_and_shoulders",
    "ascending_triangle", "descending_triangle", "rising_wedge", "falling_wedge",
    "bull_flag", "bear_flag", "bull_pennant", "bear_pennant",
    # jayantha_b2b carries its own geometry-based stop (beyond the 200 MA)
    # and target (prior swing high/low) exactly like the patterns above -
    # without this, setup_risk_plan's "pick whichever candidate has the
    # tightest stop" tie-break silently prefers the generic ATR-filled
    # stop/target that ends up attached to the jayantha_trend/
    # jayantha_confirmation bonus signals instead (they carry no geometry
    # of their own), discarding jayantha_b2b's real, deliberately-computed
    # plan in favor of a fallback one every single time.
    "jayantha_b2b",
}

EXTREME_FUNDING_THRESHOLD = 0.0005  # 0.05% per 8h funding interval - a commonly cited "hot" reading


def classify_funding(rate: float | None, bias: str) -> str | None:
    """
    Whether the current funding rate reflects the crowd positioned WITH or
    AGAINST a trade's direction. None if funding data is unavailable.

    Confirmed via an 8000+ trade backtest (funding_rate_backtest.py) that
    trading AGAINST an extreme funding reading actually loses money here
    (-0.10R) - the opposite of the classic "fade the overleveraged crowd"
    assumption - while neutral and with-the-crowd readings both stay
    profitable. Our proven detectors are trend-continuation patterns, and
    extreme funding here more often reflects a genuinely strong ongoing
    trend than imminent reversal, so fighting it tends to fight the trend.
    """
    if rate is None:
        return None
    if rate > EXTREME_FUNDING_THRESHOLD:
        return "with_crowd" if bias == "bullish" else "against_crowd"
    if rate < -EXTREME_FUNDING_THRESHOLD:
        return "with_crowd" if bias == "bearish" else "against_crowd"
    return "neutral"


def _structural_levels_valid(s: dict, close: float) -> bool:
    """
    Structural patterns (triangle/wedge/H&S/flag) fit a trendline through a
    handful of pivot points and extrapolate it to "now" - when that linear
    fit doesn't track price well, the extrapolated level can land on the
    WRONG side of the current close (e.g. a "resistance" that's actually
    below price), which silently produces a backwards stop/target: a
    bearish trade with its stop BELOW entry instead of above. A stop/target
    pair only makes sense if the stop and target actually bracket price in
    the direction implied - bullish needs stop < close < target, bearish
    needs stop > close > target.
    """
    if s["direction"] == "bullish":
        return s["stop"] < close < s["target"]
    if s["direction"] == "bearish":
        return s["stop"] > close > s["target"]
    return True


def attach_atr_risk(signals: list[dict], close: float, atr: float, atr_mult: float,
                    reward_risk: float, max_stop_pct: float | None = None) -> list[dict]:
    """
    Fill stop/target for any signal that doesn't already carry VALID
    structural levels (see _structural_levels_valid - a structural pattern
    whose geometry produced a backwards stop gets its levels discarded and
    falls through to the generic ATR-based ones below instead of silently
    handing out a broken trade plan). ATR scales with candle size, so on
    high timeframes (1w/1M) a fixed multiplier can produce a stop 30-40%
    away from price - useless as an actual risk plan. max_stop_pct caps the
    distance to a sane fraction of price regardless of how wide the raw
    ATR is.
    """
    if atr is None or atr <= 0:
        return signals
    distance = atr * atr_mult
    if max_stop_pct is not None:
        distance = min(distance, close * max_stop_pct / 100)
    for s in signals:
        if "stop" in s and "target" in s:
            if _structural_levels_valid(s, close):
                continue
            del s["stop"]
            del s["target"]
        if s["direction"] == "bullish":
            s["stop"] = close - distance
            s["target"] = close + distance * reward_risk
        elif s["direction"] == "bearish":
            s["stop"] = close + distance
            s["target"] = close - distance * reward_risk
    return signals


def calibrate_target(entry: float, detector_name: str, direction: str,
                     avg_returns: dict, min_move_pct: float = 0.3) -> float | None:
    """
    Override a geometric/ATR target with the detector's actual average
    historical forward return (from backtest_results.json) when available -
    grounds the target in what price has genuinely tended to do after this
    specific setup, instead of a fixed multiple or a "measured move"
    assumption that was never checked against real data.

    avg_returns[(name, direction)] is a raw mean forward return (can be
    negative for a bearish detector that historically worked, since price
    fell) - entry * (1 + pct/100) is correct as-is for either direction,
    no sign-flipping needed. Skipped if the move is too small to be a
    meaningful target (near-zero average moves aren't a "plan").
    """
    avg_pct = avg_returns.get((detector_name, direction))
    if avg_pct is None or abs(avg_pct) < min_move_pct:
        return None
    return entry * (1 + avg_pct / 100)


def position_size(entry: float, stop: float, account_size: float | None,
                  account_risk_pct: float) -> dict | None:
    """
    Professional risk management rule: risk a fixed small % of account
    equity per trade, sized off the actual stop distance - never a fixed
    unit count and never scaled by "how sure you feel" about the setup.
    Returns None if account_size isn't configured (we don't guess it).
    """
    if account_size is None or account_size <= 0:
        return None
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return None
    dollar_risk = account_size * account_risk_pct / 100
    units = dollar_risk / risk_per_unit
    return {
        "account_risk_pct": account_risk_pct,
        "dollar_risk": round(float(dollar_risk), 2),
        "units": round_sig(float(units), 6),
        "position_value": round(float(units * entry), 2),
    }


# jayantha_trend/jayantha_confirmation (scanner/jayantha_detectors.py) are
# confluence context only, not independent trade bases - they carry no
# geometry of their own tied to the B2B thesis, so letting them become a
# candidate here would mean falling back to an arbitrary ATR-based stop/
# target whenever jayantha_b2b's own (real, thesis-tied) geometry fails
# the R:R gate, silently misrepresenting what the alert is actually based
# on. Excluded entirely: if jayantha_b2b itself doesn't qualify, no risk
# plan is produced at all, which is the honest outcome.
CONFLUENCE_ONLY_NAMES = {"jayantha_trend", "jayantha_confirmation"}

# The market-leader-disagreement and funding filters below were proven via
# confluence_btc_backtest.py / funding_rate_backtest.py specifically on the
# original detector set - never validated against jayantha_b2b, and likely
# wrong for it: B2B setups are pullbacks *within* an established trend, and
# an established trend typically already moves with BTC/ETH rather than
# against them, so blindly inheriting this filter would reject a large
# share of otherwise-valid setups on an unproven basis. jayantha_b2b is
# added to STRUCTURAL_NAMES (for stop/target selection) but deliberately
# excluded here.
MARKET_FILTER_NAMES = (STRUCTURAL_NAMES | {"ema_stack"}) - {"jayantha_b2b"}


def setup_risk_plan(signals: list[dict], bias: str, close: float,
                    min_risk_reward: float = 1.0, avg_returns: dict | None = None,
                    min_calibrated_move_pct: float = 0.3,
                    account_size: float | None = None,
                    account_risk_pct: float = 1.0,
                    unreliable: set | None = None,
                    market_disagrees: bool | None = None,
                    funding_ok: bool = True,
                    target_fraction: float = 1.0,
                    min_stop_pct: float = 0.5) -> dict | None:
    """
    Pick one consolidated entry/stop/target for the setup: prefer a
    structural (pattern-based) level over a generic ATR one, since it's
    tied to actual chart geometry rather than a fixed multiple. The target
    is then replaced with the detector's real historical average move when
    that data is available (see calibrate_target) - the reported scenario
    reflects what has actually happened, not just what the geometry implies.

    Structural stop/target come from independent pieces of geometry though
    (e.g. a flag's stop is the whole consolidation range, its target is
    just the pole height) - nothing guarantees those two distances produce
    a favorable ratio. A candidate is only used if its reward:risk (using
    the calibrated target where available) clears min_risk_reward; if
    nothing does, this returns None rather than presenting a plan that
    risks more than it can gain. Among qualifying candidates, prefer the
    tightest (most disciplined) stop distance.

    `unreliable` is a set of (detector_name, direction) pairs to refuse
    outright, regardless of stop tightness or backtested edge - these are
    detectors the live journal has already shown to lose money at this
    stop/target sizing (see journal.detector_reliability). Without this,
    plan selection picks purely on which candidate's stop happens to be
    numerically tightest, which has nothing to do with whether that
    detector has ever actually worked.

    `market_disagrees` is whether BTC's and ETH's own current trend
    DISAGREES with `bias` - confirmed via a 2000+ trade backtest
    (confluence_btc_backtest.py) that every one of our proven detectors
    performs better - two of them flip from losing to winning - when an
    altcoin's setup goes AGAINST the market leaders rather than with them.
    A same-direction setup is more likely just beta (the whole market
    moving together) diluting the pattern's own signal, not genuine
    independent structure in that specific coin. Candidates in
    MARKET_FILTER_NAMES (the detectors this was actually tested on) are
    refused unless market_disagrees is True - None (BTC/ETH data
    unavailable) also refuses them, since the filter is unproven without it.

    `funding_ok` (default True, so callers that don't compute it aren't
    affected) is False when classify_funding says the trade is going
    AGAINST an extreme funding reading - the one funding_rate_backtest.py
    bucket that actually loses money. Unlike market_disagrees this
    defaults permissive, since most trades (neutral or with-the-crowd
    funding) are fine - only the specific proven-bad case is excluded.
    """
    unreliable = unreliable or set()
    candidates = [s for s in signals
                  if s["direction"] == bias and "stop" in s and "target" in s
                  and s["name"] not in CONFLUENCE_ONLY_NAMES
                  and (s["name"], s["direction"]) not in unreliable
                  and (s["name"] not in MARKET_FILTER_NAMES or market_disagrees is True)
                  and (s["name"] not in MARKET_FILTER_NAMES or funding_ok)]
    if not candidates:
        return None
    avg_returns = avg_returns or {}

    def effective_target(s):
        calibrated = calibrate_target(close, s["name"], s["direction"], avg_returns, min_calibrated_move_pct)
        return calibrated if calibrated is not None else s["target"]

    def rr(s):
        risk = abs(close - s["stop"])
        return abs(effective_target(s) - close) / risk if risk > 0 else 0

    def stop_pct(s):
        return abs(close - s["stop"]) / close * 100 if close > 0 else 0

    # A structural pattern's stop comes from its own chart geometry (e.g. a
    # flag's invalidation is the bottom of the consolidation) - textbook
    # correct, but a tight consolidation can put that level a fraction of a
    # percent from entry. A backtest checking candle highs/lows against an
    # exact price can't see the problem (found via win rate staying flat
    # across stop-width buckets, 27-31% everywhere including <0.2%) because
    # it assumes a perfect fill with zero bid-ask spread or slippage - in
    # real trading a stop this tight can get triggered by spread/slippage
    # alone, regardless of whether the pattern itself was ever actually
    # wrong. Reject rather than widen it, since artificially moving a
    # pattern's own invalidation level would misrepresent its real geometry.
    qualifying = [s for s in candidates if rr(s) >= min_risk_reward and stop_pct(s) >= min_stop_pct]
    if not qualifying:
        return None  # every candidate here risks more than it could gain, or has an unrealistically tight stop

    structural = [s for s in qualifying if s["name"] in STRUCTURAL_NAMES]
    pool = structural or qualifying
    pick = min(pool, key=lambda s: abs(close - s["stop"]))

    stop = pick["stop"]
    raw_target = effective_target(pick)
    calibrated = (raw_target != pick["target"])
    # Confirmed via a 6000+ trade backtest (win_rate_levers_backtest.py):
    # taking profit at 85% of the full geometric/calibrated target is a
    # small, genuine improvement over waiting for the full 100% - both win
    # rate (30.6% -> 35.5%) and expectancy (+0.440R -> +0.453R) improved,
    # unlike more aggressive fractions (e.g. 50%), which raise win rate but
    # give back more expectancy than they gain. Applied uniformly to
    # whichever candidate was already picked above - this doesn't change
    # detector/candidate selection, only how much of the move is captured.
    target = close + (raw_target - close) * target_fraction
    risk = abs(close - stop)
    reward = abs(target - close)
    return {
        "entry": round_sig(float(close), 6),
        "stop": round_sig(float(stop), 6),
        "target": round_sig(float(target), 6),
        "risk_reward": round(float(reward / risk), 2) if risk > 0 else None,
        "based_on": pick["name"],
        "target_basis": "historical avg move" if calibrated else "pattern/ATR estimate",
        "position": position_size(close, stop, account_size, account_risk_pct),
    }

"""
Ashen B2B Detector ("Trading with Ashen" Part 04)

NOT the same strategy as scanner/jayantha_b2b.py despite the shared "B2B"
name from the source material - the two are unrelated setups that
happen to share an abbreviation. Jayantha's B2B is an EMA-stack pullback
-to-the-moving-average continuation entry (see jayantha_b2b.py).
Ashen's B2B ("bulls-to-bears"/counter-candle) is: inside an aggressive,
strongly one-directional run of candles, a single opposite-colored
"counter" candle appears and then resolves back in the dominant
direction - that resolution is the entry. Kept as a fully separate
detector (`based_on` = "b2b_ashen", never "jayantha_b2b") so alerts,
the journal, and backtests never conflate the two.

Ashen's exact rule: in an aggressive downtrend (string of red candles),
a green counter-candle appearing mid-run is a short entry once price
breaks back below that green candle's low (stop above its high). Same
logic mirrored for an aggressive uptrend with a red counter-candle
(long entry above its high, stop below its low). He explicitly says to
only take the 1st or 2nd such counter-candle inside a given run - "the
third pullback rarely succeeds, only the first and second tend to
work" - which this detector enforces by counting prior isolated
counter-candles within the lookback window and refusing to fire on the
3rd or later.

The video's "aggressive" judgment is Ashen's own discretionary read of
the chart, not a mechanical rule he gives numbers for - this
implementation approximates it as "a same-colored run of at least
min_run_length candles immediately before the counter-candle", which is
the closest mechanical proxy to what he describes and is what the
config's min_run_length tunes.
"""

import pandas as pd

_DEFAULT_CFG = {
    "min_run_length": 3,     # consecutive same-colored candles required immediately before the counter-candle
    "lookback": 15,          # how far back to count prior counter-candle occurrences in this run
    "max_occurrence": 2,     # only the 1st/2nd counter-candle in a run is tradeable; 3rd+ is rejected
    "reward_risk_ratio": 1.5,
}


def _candle_color(candle) -> int:
    if candle["close"] > candle["open"]:
        return 1
    if candle["close"] < candle["open"]:
        return -1
    return 0


def _count_prior_counter_occurrences(colors: list[int], dominant: int) -> int:
    """
    Counts isolated counter-colored candles (surrounded by dominant-colored
    candles) appearing in `colors` (oldest-first, NOT including the current
    candle) - each one is a prior "counter-candle event" in this run.
    """
    count = 0
    for i in range(1, len(colors) - 1):
        if colors[i] == -dominant and colors[i - 1] == dominant:
            count += 1
    return count


def detect_signals(df: pd.DataFrame, cfg: dict) -> list[dict]:
    bcfg = cfg.get("ashen", {}).get("b2b", {})
    if not bcfg.get("enabled", True):
        return []

    min_run_length = bcfg.get("min_run_length", _DEFAULT_CFG["min_run_length"])
    lookback = bcfg.get("lookback", _DEFAULT_CFG["lookback"])
    max_occurrence = bcfg.get("max_occurrence", _DEFAULT_CFG["max_occurrence"])
    reward_risk = bcfg.get("reward_risk_ratio", _DEFAULT_CFG["reward_risk_ratio"])

    min_candles = lookback + min_run_length + 2
    if len(df) < min_candles:
        return []

    window = df.iloc[-(lookback + 1):]
    colors = [_candle_color(c) for _, c in window.iterrows()]
    if colors[-1] == 0:
        return []
    counter_candle_color = colors[-2] if len(colors) >= 2 else None
    if counter_candle_color is None or counter_candle_color == 0:
        return []
    dominant = -counter_candle_color  # the trend the counter-candle interrupted

    # Immediately-preceding run of dominant-colored candles must be long enough
    run_len = 0
    for c in reversed(colors[:-2]):
        if c == dominant:
            run_len += 1
        else:
            break
    if run_len < min_run_length:
        return []

    occurrences_before = _count_prior_counter_occurrences(colors[:-1], dominant)
    if occurrences_before >= max_occurrence:
        return []  # 3rd (or later) counter-candle in this run - Ashen's rule says skip it

    counter_candle = df.iloc[-2]
    current_candle = df.iloc[-1]
    close = float(current_candle["close"])

    if dominant == -1 and current_candle["close"] < counter_candle["low"]:
        # Downtrend resuming: price broke back below the green counter-candle's low
        stop = float(counter_candle["high"])
        target = close - (stop - close) * reward_risk
        return [{
            "name": "b2b_ashen",
            "direction": "bearish",
            "detail": (f"Counter-candle #{occurrences_before + 1} in downtrend run "
                       f"({run_len} red candles) resolved back down"),
            "stop": stop,
            "target": target,
        }]

    if dominant == 1 and current_candle["close"] > counter_candle["high"]:
        # Uptrend resuming: price broke back above the red counter-candle's high
        stop = float(counter_candle["low"])
        target = close + (close - stop) * reward_risk
        return [{
            "name": "b2b_ashen",
            "direction": "bullish",
            "detail": (f"Counter-candle #{occurrences_before + 1} in uptrend run "
                       f"({run_len} green candles) resolved back up"),
            "stop": stop,
            "target": target,
        }]

    return []

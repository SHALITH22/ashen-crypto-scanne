"""
Pattern / signal detection layer.

Each detector returns None (no signal) or a dict:
  {"name": ..., "direction": "bullish"/"bearish"/"neutral", "detail": ...}

Design rule: every signal is based on CLOSED candles only
(data.py already drops the forming candle).
"""

import numpy as np
import pandas as pd


# ---------- helpers ----------

def _last(df: pd.DataFrame, col: str, n: int = 1):
    return df[col].iloc[-n]


def _line_value(p1: tuple[int, float], p2: tuple[int, float], x: float) -> float:
    """Value of the line through two (index, price) points at position x."""
    (x1, y1), (x2, y2) = p1, p2
    if x2 == x1:
        return y2
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def find_pivots(df: pd.DataFrame, lookback: int = 150,
                order: int = 2) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Swing highs/lows within the trailing `lookback` candles.
    A pivot needs `order` bars of confirmation on each side, so results
    are always at least `order` bars behind the most recent candle -
    exactly like the rest of this module, no lookahead.
    Returns (pivot_highs, pivot_lows) as (index-in-window, price) tuples,
    oldest first.
    """
    window = df.tail(lookback).reset_index(drop=True)
    highs, lows = window["high"].values, window["low"].values
    pivot_highs, pivot_lows = [], []
    for i in range(order, len(window) - order):
        if highs[i] == max(highs[i - order:i + order + 1]):
            pivot_highs.append((i, float(highs[i])))
        if lows[i] == min(lows[i - order:i + order + 1]):
            pivot_lows.append((i, float(lows[i])))
    return pivot_highs, pivot_lows


def find_sr_levels(pivot_highs: list[tuple[int, float]], pivot_lows: list[tuple[int, float]],
                   tolerance_pct: float = 0.5, min_touches: int = 3) -> list[dict]:
    """Horizontal support/resistance from pivots that cluster within tolerance_pct."""
    pivots = np.array(sorted(p for _, p in pivot_highs + pivot_lows))
    levels = []
    used = np.zeros(len(pivots), dtype=bool)
    for i, p in enumerate(pivots):
        if used[i]:
            continue
        cluster = np.abs(pivots - p) / p * 100 <= tolerance_pct
        touches = cluster.sum()
        if touches >= min_touches:
            levels.append({"price": float(pivots[cluster].mean()), "touches": int(touches)})
            used |= cluster
    return levels


# ---------- trend / momentum detectors ----------

def detect_ema_stack(df: pd.DataFrame) -> dict | None:
    """Trend filter: bullish stack close>20>50>200, bearish reversed."""
    c, e20, e50, e200 = (_last(df, x) for x in ["close", "ema_20", "ema_50", "ema_200"])
    if c > e20 > e50 > e200:
        return {"name": "ema_stack", "direction": "bullish",
                "detail": "Price above EMA20>50>200 (uptrend structure)"}
    if c < e20 < e50 < e200:
        return {"name": "ema_stack", "direction": "bearish",
                "detail": "Price below EMA20<50<200 (downtrend structure)"}
    return None


def detect_ema_cross(df: pd.DataFrame) -> dict | None:
    """EMA20/50 cross on the last closed candle."""
    e20_now, e50_now = _last(df, "ema_20"), _last(df, "ema_50")
    e20_prev, e50_prev = _last(df, "ema_20", 2), _last(df, "ema_50", 2)
    if e20_prev <= e50_prev and e20_now > e50_now:
        return {"name": "ema_cross", "direction": "bullish", "detail": "EMA20 crossed above EMA50"}
    if e20_prev >= e50_prev and e20_now < e50_now:
        return {"name": "ema_cross", "direction": "bearish", "detail": "EMA20 crossed below EMA50"}
    return None


def detect_stochrsi_extreme(df: pd.DataFrame, oversold: float, overbought: float,
                            require_turn: bool = False) -> dict | None:
    """
    require_turn=True only fires when %K has started reversing out of the
    extreme zone (fewer, higher-quality signals - avoids catching a falling knife).
    """
    k, d = _last(df, "stochrsi_k"), _last(df, "stochrsi_d")
    k_prev = _last(df, "stochrsi_k", 2)
    if k < oversold:
        if require_turn and k <= k_prev:
            return None
        turning = " and turning up" if k > k_prev else ""
        return {"name": "stochrsi", "direction": "bullish",
                "detail": f"StochRSI oversold ({k:.1f}){turning}"}
    if k > overbought:
        if require_turn and k >= k_prev:
            return None
        turning = " and turning down" if k < k_prev else ""
        return {"name": "stochrsi", "direction": "bearish",
                "detail": f"StochRSI overbought ({k:.1f}){turning}"}
    return None


def detect_qqe_cross(df: pd.DataFrame) -> dict | None:
    """
    QQE trend flip on the last closed candle (see indicators.add_qqe) -
    RSI MA crossing its own trailing volatility band. "strong"/"weak"
    qualifier is how far RSI MA sits from the neutral 50 line at the
    moment of the flip - a flip near 50 is a marginal call, one further
    from center reflects more decisive momentum already in hand.
    """
    trend_now, trend_prev = _last(df, "qqe_trend"), _last(df, "qqe_trend", 2)
    if trend_now == trend_prev:
        return None
    rsi_ma = _last(df, "qqe_rsi_ma")
    strength = "strong" if abs(rsi_ma - 50) > 15 else "weak"
    if trend_now == 1:
        return {"name": "qqe_cross", "direction": "bullish",
                "detail": f"QQE flipped bullish ({strength}, RSI MA {rsi_ma:.1f})"}
    return {"name": "qqe_cross", "direction": "bearish",
            "detail": f"QQE flipped bearish ({strength}, RSI MA {rsi_ma:.1f})"}


def detect_volume_spike(df: pd.DataFrame, multiplier: float) -> dict | None:
    v, vma = _last(df, "volume"), _last(df, "volume_ma")
    if pd.isna(vma) or vma == 0:
        return None
    if v >= multiplier * vma:
        candle_dir = "bullish" if _last(df, "close") > _last(df, "open") else "bearish"
        return {"name": "volume_spike", "direction": candle_dir,
                "detail": f"Volume {v / vma:.1f}x its 20-MA on a {candle_dir} candle"}
    return None


# ---------- support/resistance detectors ----------

def detect_sr_break(df: pd.DataFrame, levels: list[dict]) -> dict | None:
    """Did the last closed candle close through a valid S/R level?"""
    if not levels:
        return None
    close, prev_close = _last(df, "close"), _last(df, "close", 2)
    for lvl in levels:
        p = lvl["price"]
        if prev_close < p < close:
            return {"name": "sr_break", "direction": "bullish",
                    "detail": f"Closed above resistance {p:.4g} ({lvl['touches']} touches)"}
        if prev_close > p > close:
            return {"name": "sr_break", "direction": "bearish",
                    "detail": f"Closed below support {p:.4g} ({lvl['touches']} touches)"}
    return None


def detect_sr_test(df: pd.DataFrame, levels: list[dict], tolerance_pct: float) -> dict | None:
    """Is price currently sitting AT a key level (decision zone)?"""
    if not levels:
        return None
    close = _last(df, "close")
    for lvl in levels:
        if abs(close - lvl["price"]) / lvl["price"] * 100 <= tolerance_pct:
            return {"name": "sr_test", "direction": "neutral",
                    "detail": f"Price testing level {lvl['price']:.4g} ({lvl['touches']} touches)"}
    return None


# ---------- classic chart patterns (pivot-based geometry) ----------

def detect_triangle_wedge(df: pd.DataFrame, pivot_highs: list[tuple[int, float]],
                          pivot_lows: list[tuple[int, float]], window_len: int) -> dict | None:
    """
    Fits a trendline through the last few pivot highs and pivot lows and
    classifies the shape: ascending/descending/symmetrical triangle, or
    rising/falling wedge. Requires the two lines to actually be converging.
    """
    if len(pivot_highs) < 3 or len(pivot_lows) < 3:
        return None
    hxs = np.array([p[0] for p in pivot_highs[-4:]])
    hys = np.array([p[1] for p in pivot_highs[-4:]])
    lxs = np.array([p[0] for p in pivot_lows[-4:]])
    lys = np.array([p[1] for p in pivot_lows[-4:]])
    h_slope, h_int = np.polyfit(hxs, hys, 1)
    l_slope, l_int = np.polyfit(lxs, lys, 1)

    avg_price = float(_last(df, "close"))
    x0, x1 = 0, window_len - 1
    width0 = (h_int + h_slope * x0) - (l_int + l_slope * x0)
    width1 = (h_int + h_slope * x1) - (l_int + l_slope * x1)
    if width0 <= 0 or width1 <= 0 or width1 > width0 * 0.85:
        return None  # lines aren't meaningfully converging

    h_pct = h_slope * window_len / avg_price * 100
    l_pct = l_slope * window_len / avg_price * 100
    flat = 1.0  # % move over the window below which a line counts as "flat"
    top_now = h_int + h_slope * x1
    bottom_now = l_int + l_slope * x1
    height = width1  # current gap between the two trendlines - the pattern's measured move

    if abs(h_pct) < flat and l_pct > flat:
        return {"name": "ascending_triangle", "direction": "bullish",
                "detail": "Ascending triangle: flat resistance, rising support",
                "stop": bottom_now, "target": top_now + height}
    if abs(l_pct) < flat and h_pct < -flat:
        return {"name": "descending_triangle", "direction": "bearish",
                "detail": "Descending triangle: flat support, falling resistance",
                "stop": top_now, "target": bottom_now - height}
    if h_pct < -flat and l_pct > flat:
        return {"name": "symmetrical_triangle", "direction": "neutral",
                "detail": "Symmetrical triangle: converging trendlines, breakout pending"}
    if h_pct > flat and l_pct > flat and h_pct < l_pct:
        return {"name": "rising_wedge", "direction": "bearish",
                "detail": "Rising wedge: converging upward channel (reversal risk)",
                "stop": top_now, "target": bottom_now - height}
    if h_pct < -flat and l_pct < -flat and h_pct > l_pct:
        return {"name": "falling_wedge", "direction": "bullish",
                "detail": "Falling wedge: converging downward channel (reversal risk)",
                "stop": bottom_now, "target": top_now + height}
    return None


def detect_double_top(df: pd.DataFrame, pivot_highs: list[tuple[int, float]],
                      pivot_lows: list[tuple[int, float]], tolerance_pct: float) -> dict | None:
    """Two comparable peaks with a trough between them; fires on the neckline break."""
    if len(pivot_highs) < 2:
        return None
    h1, h2 = pivot_highs[-2], pivot_highs[-1]
    if abs(h1[1] - h2[1]) / h1[1] * 100 > tolerance_pct:
        return None
    between = [l for l in pivot_lows if h1[0] < l[0] < h2[0]]
    if not between:
        return None
    trough = min(between, key=lambda l: l[1])
    if (min(h1[1], h2[1]) - trough[1]) / trough[1] * 100 < tolerance_pct:
        return None  # peaks and trough too close together to be a real pattern
    close, prev_close = _last(df, "close"), _last(df, "close", 2)
    if prev_close >= trough[1] and close < trough[1]:
        top = max(h1[1], h2[1])
        measured_move = top - trough[1]
        return {"name": "double_top", "direction": "bearish",
                "detail": f"Double top ~{(h1[1] + h2[1]) / 2:.4g}, broke neckline {trough[1]:.4g}",
                "stop": top, "target": trough[1] - measured_move}
    return None


def detect_double_bottom(df: pd.DataFrame, pivot_highs: list[tuple[int, float]],
                         pivot_lows: list[tuple[int, float]], tolerance_pct: float) -> dict | None:
    """Two comparable troughs with a peak between them; fires on the neckline break."""
    if len(pivot_lows) < 2:
        return None
    l1, l2 = pivot_lows[-2], pivot_lows[-1]
    if abs(l1[1] - l2[1]) / l1[1] * 100 > tolerance_pct:
        return None
    between = [h for h in pivot_highs if l1[0] < h[0] < l2[0]]
    if not between:
        return None
    peak = max(between, key=lambda h: h[1])
    if (peak[1] - max(l1[1], l2[1])) / peak[1] * 100 < tolerance_pct:
        return None
    close, prev_close = _last(df, "close"), _last(df, "close", 2)
    if prev_close <= peak[1] and close > peak[1]:
        bottom = min(l1[1], l2[1])
        measured_move = peak[1] - bottom
        return {"name": "double_bottom", "direction": "bullish",
                "detail": f"Double bottom ~{(l1[1] + l2[1]) / 2:.4g}, broke neckline {peak[1]:.4g}",
                "stop": bottom, "target": peak[1] + measured_move}
    return None


def _legs_symmetric(s1: tuple[int, float], head: tuple[int, float], s2: tuple[int, float],
                    max_ratio: float = 3.0) -> bool:
    """Reject wildly lopsided shoulders (one leg N times longer than the other)."""
    left_span, right_span = head[0] - s1[0], s2[0] - head[0]
    if left_span <= 0 or right_span <= 0:
        return False
    return max(left_span, right_span) / min(left_span, right_span) <= max_ratio


def detect_head_shoulders(df: pd.DataFrame, pivot_highs: list[tuple[int, float]],
                          pivot_lows: list[tuple[int, float]], window_len: int,
                          tolerance_pct: float, min_depth_pct: float) -> dict | None:
    """Shoulder-head-shoulder in highs, sloped neckline through the two troughs."""
    if len(pivot_highs) < 3:
        return None
    s1, head, s2 = pivot_highs[-3], pivot_highs[-2], pivot_highs[-1]
    # head must clearly clear both shoulders, not just edge above them (kills noise patterns)
    margin = tolerance_pct / 2 / 100
    if not (head[1] > s1[1] * (1 + margin) and head[1] > s2[1] * (1 + margin)):
        return None
    if abs(s1[1] - s2[1]) / s1[1] * 100 > tolerance_pct * 2:
        return None
    if not _legs_symmetric(s1, head, s2):
        return None
    left = [l for l in pivot_lows if s1[0] < l[0] < head[0]]
    right = [l for l in pivot_lows if head[0] < l[0] < s2[0]]
    if not left or not right:
        return None
    l1, l2 = left[-1], right[0]
    neck_avg = (l1[1] + l2[1]) / 2
    if (min(s1[1], s2[1]) - neck_avg) / neck_avg * 100 < min_depth_pct:
        return None  # too shallow to be a real pattern, likely noise
    cur_x, prev_x = window_len - 1, window_len - 2
    neck_now, neck_prev = _line_value(l1, l2, cur_x), _line_value(l1, l2, prev_x)
    close, prev_close = _last(df, "close"), _last(df, "close", 2)
    if prev_close >= neck_prev and close < neck_now:
        measured_move = head[1] - neck_now
        return {"name": "head_and_shoulders", "direction": "bearish",
                "detail": f"Head & shoulders, broke neckline {neck_now:.4g}",
                "stop": head[1], "target": neck_now - measured_move}
    return None


def detect_inverse_head_shoulders(df: pd.DataFrame, pivot_highs: list[tuple[int, float]],
                                  pivot_lows: list[tuple[int, float]], window_len: int,
                                  tolerance_pct: float, min_depth_pct: float) -> dict | None:
    """Inverse H&S in lows, sloped neckline through the two peaks."""
    if len(pivot_lows) < 3:
        return None
    s1, head, s2 = pivot_lows[-3], pivot_lows[-2], pivot_lows[-1]
    margin = tolerance_pct / 2 / 100
    if not (head[1] < s1[1] * (1 - margin) and head[1] < s2[1] * (1 - margin)):
        return None
    if abs(s1[1] - s2[1]) / s1[1] * 100 > tolerance_pct * 2:
        return None
    if not _legs_symmetric(s1, head, s2):
        return None
    left = [h for h in pivot_highs if s1[0] < h[0] < head[0]]
    right = [h for h in pivot_highs if head[0] < h[0] < s2[0]]
    if not left or not right:
        return None
    h1, h2 = left[-1], right[0]
    neck_avg = (h1[1] + h2[1]) / 2
    if (neck_avg - max(s1[1], s2[1])) / neck_avg * 100 < min_depth_pct:
        return None
    cur_x, prev_x = window_len - 1, window_len - 2
    neck_now, neck_prev = _line_value(h1, h2, cur_x), _line_value(h1, h2, prev_x)
    close, prev_close = _last(df, "close"), _last(df, "close", 2)
    if prev_close <= neck_prev and close > neck_now:
        measured_move = neck_now - head[1]
        return {"name": "inverse_head_and_shoulders", "direction": "bullish",
                "detail": f"Inverse head & shoulders, broke neckline {neck_now:.4g}",
                "stop": head[1], "target": neck_now + measured_move}
    return None


def detect_flag_pennant(df: pd.DataFrame, pole_window: int, flag_window: int,
                        min_pole_pct: float, max_range_ratio: float,
                        max_pole_pct: float = 50.0) -> dict | None:
    """
    A sharp directional move (the pole) followed by a tight consolidation,
    confirmed on a breakout that continues the pole's direction. Narrower
    second-half range within the consolidation gets called a pennant
    (converging) instead of a flag (roughly parallel channel).

    max_pole_pct rejects poles beyond a normal continuation move - a >50%
    move in one pole_window is far more likely a new-listing pump/crash or
    a data anomaly than a disciplined flagpole, and the measured-move
    projection (pole height added past the breakout) can go non-physical
    (negative price target) when the pole is that extreme.
    """
    total = pole_window + flag_window
    if len(df) < total + 2:
        return None
    w = df.tail(total + 1)
    pole, flag = w.iloc[:pole_window], w.iloc[pole_window:total]

    pole_start, pole_end = pole["close"].iloc[0], pole["close"].iloc[-1]
    pole_pct = (pole_end - pole_start) / pole_start * 100
    if abs(pole_pct) < min_pole_pct or abs(pole_pct) > max_pole_pct:
        return None
    pole_range = pole["high"].max() - pole["low"].min()
    flag_range = flag["high"].max() - flag["low"].min()
    if pole_range == 0 or flag_range / pole_range > max_range_ratio:
        return None  # consolidation not tight enough relative to the pole

    half = len(flag) // 2
    first_half_range = flag["high"].iloc[:half].max() - flag["low"].iloc[:half].min()
    second_half_range = flag["high"].iloc[half:].max() - flag["low"].iloc[half:].min()
    shape = "pennant" if second_half_range < first_half_range * 0.7 else "flag"

    flag_high, flag_low = flag["high"].max(), flag["low"].min()
    close, prev_close = _last(df, "close"), _last(df, "close", 2)
    pole_height = abs(pole_end - pole_start)

    if pole_pct > 0 and prev_close <= flag_high and close > flag_high:
        return {"name": f"bull_{shape}", "direction": "bullish",
                "detail": f"Bull {shape}: {pole_pct:.1f}% pole, broke consolidation high {flag_high:.4g}",
                "stop": flag_low, "target": flag_high + pole_height}
    if pole_pct < 0 and prev_close >= flag_low and close < flag_low:
        target = flag_low - pole_height
        if target <= 0:
            return None  # measured move would put the target at/below zero - not physical
        return {"name": f"bear_{shape}", "direction": "bearish",
                "detail": f"Bear {shape}: {pole_pct:.1f}% pole, broke consolidation low {flag_low:.4g}",
                "stop": flag_high, "target": target}
    return None


# ---------- momentum divergence ----------

def detect_rsi_divergence(df: pd.DataFrame, pivot_highs: list[tuple[int, float]],
                          pivot_lows: list[tuple[int, float]], lookback: int,
                          min_price_pct: float, min_rsi_diff: float,
                          require_extreme: bool, oversold: float, overbought: float) -> dict | None:
    """
    Regular divergence between price swing points and RSI - price and
    momentum disagree, a classic early-warning reversal signal.
    Bearish: price higher high, RSI lower high. Bullish: price lower low, RSI higher low.

    require_extreme=True only counts it when the second pivot's RSI is also
    in oversold/overbought territory - textbook usage is divergence AT an
    exhaustion zone, not any two swing points that happen to disagree
    (backtesting showed the unfiltered version has no edge - divergence
    firing mid-trend just gets run over by continuation).
    """
    if "rsi" not in df.columns:
        return None
    rsi = df["rsi"].tail(lookback).reset_index(drop=True)

    if len(pivot_highs) >= 2:
        h1, h2 = pivot_highs[-2], pivot_highs[-1]
        price_pct = (h2[1] - h1[1]) / h1[1] * 100
        rsi1, rsi2 = rsi.iloc[h1[0]], rsi.iloc[h2[0]]
        if price_pct > min_price_pct and (rsi1 - rsi2) > min_rsi_diff:
            if not require_extreme or rsi1 > overbought:
                return {"name": "rsi_divergence", "direction": "bearish",
                        "detail": f"Bearish divergence: price higher high (+{price_pct:.1f}%), "
                                  f"RSI lower high ({rsi1:.1f}->{rsi2:.1f})"}

    if len(pivot_lows) >= 2:
        l1, l2 = pivot_lows[-2], pivot_lows[-1]
        price_pct = (l1[1] - l2[1]) / l1[1] * 100
        rsi1, rsi2 = rsi.iloc[l1[0]], rsi.iloc[l2[0]]
        if price_pct > min_price_pct and (rsi2 - rsi1) > min_rsi_diff:
            if not require_extreme or rsi1 < oversold:
                return {"name": "rsi_divergence", "direction": "bullish",
                        "detail": f"Bullish divergence: price lower low (-{price_pct:.1f}%), "
                                  f"RSI higher low ({rsi1:.1f}->{rsi2:.1f})"}
    return None


# ---------- candlestick patterns ----------

def detect_engulfing(df: pd.DataFrame) -> dict | None:
    """Current candle's real body fully engulfs the prior candle's, opposite color."""
    o1, c1 = _last(df, "open", 2), _last(df, "close", 2)
    o2, c2 = _last(df, "open", 1), _last(df, "close", 1)
    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1:
        return {"name": "engulfing", "direction": "bullish",
                "detail": f"Bullish engulfing: {o2:.4g}-{c2:.4g} engulfs prior red candle"}
    if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1:
        return {"name": "engulfing", "direction": "bearish",
                "detail": f"Bearish engulfing: {c2:.4g}-{o2:.4g} engulfs prior green candle"}
    return None


def detect_hammer_shooting_star(df: pd.DataFrame, trend_lookback: int) -> dict | None:
    """
    Small body with a long wick on one side only. Only counts as a reversal
    signal in the right trend context - a hammer needs a prior downtrend,
    a shooting star needs a prior uptrend, otherwise it's just noise.
    """
    if len(df) <= trend_lookback + 1:
        return None
    o, c, h, l = _last(df, "open"), _last(df, "close"), _last(df, "high"), _last(df, "low")
    full_range = h - l
    if full_range == 0:
        return None
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    prior_close = _last(df, "close", trend_lookback + 1)
    close = _last(df, "close")

    small_body = body / full_range < 0.35
    if small_body and lower_wick >= 2 * body and upper_wick <= body * 0.5 and prior_close > close:
        return {"name": "hammer", "direction": "bullish",
                "detail": "Hammer after downtrend: long lower wick, small body"}
    if small_body and upper_wick >= 2 * body and lower_wick <= body * 0.5 and prior_close < close:
        return {"name": "shooting_star", "direction": "bearish",
                "detail": "Shooting star after uptrend: long upper wick, small body"}
    return None


# ---------- orchestrator ----------

def run_all_detectors(df: pd.DataFrame, cfg: dict) -> list[dict]:
    sig_cfg = cfg["signals"]
    lookback = sig_cfg.get("pattern_lookback", 150)
    pattern_tol = sig_cfg.get("pattern_tolerance_pct", 1.5)
    hs_min_depth = sig_cfg.get("hs_min_depth_pct", 2.0)
    flag_pole_window = sig_cfg.get("flag_pole_window", 10)
    flag_window = sig_cfg.get("flag_window", 8)
    flag_min_pole_pct = sig_cfg.get("flag_min_pole_pct", 5.0)
    flag_max_pole_pct = sig_cfg.get("flag_max_pole_pct", 50.0)
    flag_max_range_ratio = sig_cfg.get("flag_max_range_ratio", 0.5)
    candle_trend_lookback = sig_cfg.get("candle_trend_lookback", 5)
    div_min_price_pct = sig_cfg.get("divergence_min_price_pct", 0.5)
    div_min_rsi_diff = sig_cfg.get("divergence_min_rsi_diff", 3.0)
    div_require_extreme = sig_cfg.get("divergence_require_extreme", True)
    rsi_oversold = sig_cfg.get("rsi_oversold", 30)
    rsi_overbought = sig_cfg.get("rsi_overbought", 70)
    pivot_highs, pivot_lows = find_pivots(df, lookback)
    window_len = min(lookback, len(df))
    levels = find_sr_levels(pivot_highs, pivot_lows, sig_cfg["sr_tolerance_pct"], sig_cfg["sr_min_touches"])

    detectors = [
        detect_ema_stack(df),
        detect_ema_cross(df),
        detect_stochrsi_extreme(df, sig_cfg["stochrsi_oversold"], sig_cfg["stochrsi_overbought"],
                                sig_cfg.get("stochrsi_require_turn", False)),
        detect_qqe_cross(df),
        detect_volume_spike(df, sig_cfg["volume_spike_multiplier"]),
        detect_sr_break(df, levels),
        detect_sr_test(df, levels, sig_cfg["sr_tolerance_pct"]),
        detect_triangle_wedge(df, pivot_highs, pivot_lows, window_len),
        detect_double_top(df, pivot_highs, pivot_lows, pattern_tol),
        detect_double_bottom(df, pivot_highs, pivot_lows, pattern_tol),
        detect_head_shoulders(df, pivot_highs, pivot_lows, window_len, pattern_tol, hs_min_depth),
        detect_inverse_head_shoulders(df, pivot_highs, pivot_lows, window_len, pattern_tol, hs_min_depth),
        detect_flag_pennant(df, flag_pole_window, flag_window, flag_min_pole_pct,
                            flag_max_range_ratio, flag_max_pole_pct),
        detect_engulfing(df),
        detect_hammer_shooting_star(df, candle_trend_lookback),
        detect_rsi_divergence(df, pivot_highs, pivot_lows, window_len, div_min_price_pct, div_min_rsi_diff,
                              div_require_extreme, rsi_oversold, rsi_overbought),
    ]
    return [s for s in detectors if s]

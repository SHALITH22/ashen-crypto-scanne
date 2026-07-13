# Jayantha Strategy — What This Fork Actually Does

This repo is a fork of [binance-scanner](https://github.com/SHALITH22/binance-scanner)
with one change: **the trading strategy**. The old generic multi-pattern
detector set (EMA crosses, double tops, head & shoulders, triangles,
wedges, flags, candlesticks, RSI divergence — 16 detectors, weighted by
backtest) has been removed from the live pipeline and replaced with
**Jayantha Ukwatta's B2B entry technique**, extracted from ~594 hours of
his trading commentary.

Everything else — data fetching, indicators, multi-timeframe agreement,
risk/stop/target calculation, position sizing, the live journal,
Telegram notify, console/JSON output, the GitHub Actions schedule — is
**unchanged**. This was a deliberate design choice: swap only the
detector layer, keep every already-tested piece of infrastructure around
it exactly as it was.

---

## The strategy

**B2B — "buy near the moving average, sell away from it"**

Jayantha's signature rule: enter on a pullback (bullish) or rally
(bearish) to a rising/falling moving average *inside an already
established trend* — never chase a candle that's already extended away
from it.

1. **Trend filter** — price must be on the trend side of the 200 MA
   (above for bullish, below for bearish), and the EMA stack must be
   aligned (20 > 50 > 100 > 200 for bullish, reversed for bearish).
2. **Pullback/rally detection** — price must have moved to within
   `pullback_tolerance_pct` of the 50 MA, and the pullback/rally itself
   must be at least `min_pullback_depth_pct` deep (rejects noise-level
   wiggles).
3. **Confirmation before conviction** — Jayantha's other constantly
   repeated rule: never trade an *approaching* level, only a *closed*
   candle that has actually confirmed it. A setup's confirmation score
   (0–1) is computed from how cleanly the candle closed back beyond the
   MA after the touch:
   - `< 0.3` (wick-only touch, or barely there) → **setup discarded
     entirely.** It never reaches the console, the journal, or Telegram.
   - `0.3–0.65` → setup fires with 2 confluent signals (`jayantha_b2b` +
     `jayantha_trend`).
   - `>= 0.65` → setup fires with all 3 confluent signals (adds
     `jayantha_confirmation`), a cleaner, higher-conviction read.
4. **Stop/target** — geometry-based, not a flat percentage: stop just
   beyond the 200 MA (the trend invalidation level), target at the
   pullback's/rally's prior swing high/low. This flows into the same
   `attach_atr_risk` / `setup_risk_plan` machinery the old detectors
   used, so a broken/backwards geometric level still falls back safely
   to an ATR-based stop exactly as before.

## Where the code lives

| File | Role |
|---|---|
| `scanner/jayantha_b2b.py` | `B2BDetector` — trend filter, pullback/rally detection, EMA stack check, stop/target geometry |
| `scanner/jayantha_confirmation.py` | `ConfirmationValidator` — confirmation scoring (closed candle vs. wick-only touch) |
| `scanner/jayantha_detectors.py` | **The actual plug-in point.** `run_jayantha_detectors(df, cfg)` — same signature and return shape as the retired `patterns.run_all_detectors`, called from the same line in `main.py` |
| `config/settings.yaml` → `jayantha:` section | All B2B and confirmation thresholds |
| `config/settings.yaml` → `risk:` section | Unchanged machinery — now configured with `account_size: 10000`, `account_risk_pct: 1.0` (Jayantha's stated 1%-per-trade rule: even a run of 100 losing trades stays theoretically survivable, since each risks 1% of *current* equity, not a fixed amount) |

The **only** line changed in `main.py` itself is the detector call inside
`scan_pair()`:

```python
# before: signals = run_all_detectors(df, cfg)
signals = run_jayantha_detectors(df, cfg)
```

Everything downstream of that line — risk attachment, confluence
scoring, market-leader/funding filters, journal logging, Telegram
alerting — reads `signals` the same way it always did, because the
shape is identical.

## What was deliberately *not* carried over

- **Confluence checking (news catalysts, on-chain data)** — out of scope
  for this pass; would need paid API integrations (Glassnode,
  CryptoQuant, a news API). Left for a future iteration.
- **A separate risk-management module** — an earlier draft
  (`jayantha_risk_management.py`) duplicated what `scanner/risk.py`
  already does (fixed-fractional position sizing off stop distance,
  `max_stop_pct`/`min_stop_pct` guards, R:R validation). Rather than run
  two parallel risk engines, `risk.py` is reused as-is with the account
  size/risk % set to match.
- **The old detector set itself** (`scanner/patterns.py`) — left in the
  repo unchanged (not deleted) because `backtest.py` and the various
  standalone `*_backtest.py` research scripts still import it, and
  deleting it would break those without adding any value. It's simply no
  longer called from `main.py`'s live path.

## Verification

- `smoke_test.py` — offline, no network — runs `run_jayantha_detectors`
  against synthetic uptrend/downtrend/flat OHLCV data and asserts no
  exceptions, confirming the live code path is exercised end-to-end
  without needing Binance access.
- `python -m py_compile` was run across every changed/added `.py` file
  to catch syntax errors before commit.

## Known inert leftover

`.github/workflows/backtest.yml` still runs `backtest.py` weekly against
the retired detector set and commits `backtest_results.json`. This is
harmless — `main.py`'s `load_detector_weights()` just finds no matching
entries for `jayantha_b2b`/`jayantha_trend`/`jayantha_confirmation` and
defaults to neutral weighting — but it is now testing code the live
scanner doesn't use. Left in place rather than removed, since disabling
a scheduled workflow wasn't asked for and doesn't affect correctness;
worth revisiting later.

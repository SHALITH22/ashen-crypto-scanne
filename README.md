# Ashen Crypto Scanner (Jayantha B2B Strategy)

Fork of [binance-scanner](https://github.com/SHALITH22/binance-scanner)
running **Jayantha Ukwatta's B2B entry technique** instead of the original
generic multi-pattern detector set. See [JAYANTHA_STRATEGY.md](JAYANTHA_STRATEGY.md)
for exactly what changed and why — short version: only the detector layer
was swapped, every other already-tested piece (data fetching, risk/stop/
target calculation, multi-timeframe agreement, the live journal, Telegram
notify, GitHub Actions schedule) is unchanged.

Scans Binance pairs across 15m → 1M timeframes for B2B pullback/rally
setups, scores confluence, attaches a stop/target risk plan, sends
Telegram alerts, and logs every alert to a forward-tested journal. Runs
automatically in the cloud via GitHub Actions — no server or always-on PC
required.

**This is a SIGNAL/ALERT system, not a trading bot.** No orders are placed.
No API keys are needed for market data — Binance public data is free.
Only Telegram credentials (for alerts) are required, and those are never
hardcoded (see "Secrets" below).

---

## What it detects

**Jayantha's B2B — "buy near the moving average, sell away from it":**
enter on a pullback/rally to a rising/falling MA inside an already
established trend, confirmed by a closed candle (never a wick-only
touch or an approaching level). Up to three confluent signals per setup:

- **`jayantha_b2b`** — the pullback/rally itself: trend filter (price on
  the trend side of the 200 MA), EMA stack aligned, price within
  tolerance of the 50 MA, pullback/rally deep enough to be real. Carries
  its own geometry-based stop (beyond the 200 MA) and target (prior
  swing high/low).
- **`jayantha_trend`** — EMA stack alignment, surfaced as its own signal.
- **`jayantha_confirmation`** — only added when the bounce/rejection is
  *cleanly* confirmed (confirmation score ≥ 0.65), not just barely
  closed beyond the MA.

Setups with a confirmation score below 0.3 (wick-only touch, no real
close) are discarded before they ever reach the console, journal, or
Telegram — see `scanner/jayantha_detectors.py` and
[JAYANTHA_STRATEGY.md](JAYANTHA_STRATEGY.md) for the full breakdown.

## Risk plan

`jayantha_b2b`'s stop/target come from its own geometry (stop just beyond
the 200 MA, target at the prior swing high/low) — same contract the old
structural pattern detectors used. If that geometry ever comes out
backwards (rare, but possible on a bad linear fit), it falls back to an
ATR-based stop, capped at `max_stop_pct` of price so high-timeframe
candles (1w/1M) don't produce absurdly wide stops. Position size is fixed-
fractional: `risk.account_size` × `risk.account_risk_pct` (default 1%) ÷
stop distance, so every trade risks the same dollar amount regardless of
how wide or tight its stop is.

## Live journal

Every setup that clears `min_confluence` gets logged to `journal.jsonl`.
`journal_check.py` later checks real price action to see whether stop or
target was hit first, building an actual forward-tested track record —
not just a retrospective backtest.

---

## Running locally

```bash
pip install -r requirements.txt
python main.py            # one scan, prints + writes signals_output.json
python journal_check.py   # resolve open journal entries against real price
python smoke_test.py      # offline sanity check, no network needed - exercises run_jayantha_detectors directly
python backtest.py        # NOTE: measures the retired old detector set, not jayantha_b2b - see JAYANTHA_STRATEGY.md
```

Edit `config/settings.yaml` to change pairs, timeframes, and every
threshold mentioned above. Set `scan_all: true` to scan every USDT
perpetual instead of just the configured `pairs` list.

## Running in the cloud (GitHub Actions)

Two scheduled workflows in `.github/workflows/`:

- **`scan.yml`** — runs `main.py` + `journal_check.py` every 30 minutes,
  commits `journal.jsonl` back to the repo so history persists across runs.
- **`backtest.yml`** — runs `backtest.py` weekly, commits a refreshed
  `backtest_results.json` so detector weights stay current.

Both can also be triggered manually from the repo's **Actions** tab via
**Run workflow**.

### Secrets

Set these in the repo's **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | your numeric Telegram chat ID |

Locally, put the same two keys in a `.env` file (see `.env.example`) —
`scanner/notify.py` checks environment variables first (for CI), then
falls back to `.env` (for local runs). `.env` is gitignored; never commit it.

---

## Architecture

```
main.py                     orchestrator: scan -> weight -> risk -> notify -> journal
config/settings.yaml        every tunable threshold, incl. `jayantha:` section
scanner/data.py             Binance REST fetching (public, no key)
scanner/indicators.py       EMA, RSI, StochRSI, ATR, volume MA (pure pandas)
scanner/jayantha_b2b.py     B2BDetector - pullback/rally + trend/EMA-stack checks
scanner/jayantha_confirmation.py  ConfirmationValidator - closed-candle confirmation scoring
scanner/jayantha_detectors.py     the live plug-in point - same shape as the retired run_all_detectors
scanner/patterns.py         retired detector set - kept only for backtest.py/research scripts, not called from main.py
scanner/mtf.py              higher-timeframe agreement filter
scanner/risk.py             stop/target + fixed-fractional position sizing (unchanged, reused as-is)
scanner/journal.py          forward-tested track record (log + resolve)
scanner/notify.py           Telegram formatting/sending
signals_output.json         machine-readable output of the last scan
journal.jsonl               append-only log of every real alert + its outcome
```

See [JAYANTHA_STRATEGY.md](JAYANTHA_STRATEGY.md) for the full detail on
what changed, what was deliberately left as-is, and what's a known inert
leftover (`backtest.yml` still runs weekly against the retired detector
set — harmless, just not meaningful anymore).

## Honest notes on accuracy

No technical-pattern system hits 80%+ win rates on liquid markets under
real, unbiased forward testing — if it did, the edge would already be
arbitraged away. This B2B strategy has not yet been forward-tested at
scale in this repo (the live journal starts fresh); treat any specific
win-rate expectation as unproven until `journal.jsonl` accumulates enough
resolved trades to say something real. The risk plan (fixed 1% risk per
trade, 2:1 reward:risk default) matters as much as the direction call —
a lower win rate can still be profitable at a favorable R:R, and a higher
one can still lose money at a poor one.

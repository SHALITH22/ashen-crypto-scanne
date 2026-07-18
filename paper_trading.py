"""
Simulates a virtual account trading exactly what journal.jsonl actually
logged, with realistic position sizing - not just a raw win/loss count.
Answers: if you'd been paper trading this system since trade #1, with
proper risk management, where would your account actually be today?

Deliberately stateless/full-replay rather than incremental (unlike
audit_trades.py's checkpoint pattern) - journal.jsonl is small enough
(~1000 entries today) that replaying the whole thing every run is cheap
(well under a second) and eliminates an entire class of incremental-
state bugs. Output is a pure function of journal.jsonl's contents plus
the fixed STARTING_BALANCE/RISK_PCT below, so re-running never drifts.

Position sizing: each trade's dollar risk is locked in at the moment it
OPENS (logged_at), sized against whatever the account balance was AT
THAT MOMENT - never recomputed later when it resolves. This matters
because trades can be open concurrently (confirmed up to 13 at once on
one symbol - see strategy_correlation_analysis.py); sizing a trade off
the CURRENT balance at resolution time instead would let it silently
benefit from gains that other, still-open trades hadn't realized yet
when THIS trade was actually placed - producing an unrealistic,
lookahead-biased equity curve. Simulated as two chronological events per
trade (open, close) processed in timestamp order across the whole
journal at once, not trade-by-trade in isolation.

Usage: python paper_trading.py
Writes paper_account.json (current state) and paper_equity_curve.jsonl
(one row per resolved trade, for charting) - both to the repo root, same
pattern as scan_health.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from scanner.journal import entries_since

STARTING_BALANCE = 1000.0
RISK_PCT = 1.0  # % of current balance risked per trade at entry time

ACCOUNT_PATH = Path(__file__).parent / "paper_account.json"
EQUITY_CURVE_PATH = Path(__file__).parent / "paper_equity_curve.jsonl"


def r_multiple(entry: dict) -> float | None:
    """Exact realized R for a resolved trade, from its actual outcome_price - not an assumed flat R:R."""
    risk = abs(entry["entry"] - entry["stop"])
    if risk <= 0 or entry.get("outcome_price") is None:
        return None
    move = ((entry["outcome_price"] - entry["entry"]) if entry["bias"] == "bullish"
            else (entry["entry"] - entry["outcome_price"]))
    return move / risk


def simulate(entries: list[dict]) -> tuple[dict, list[dict]]:
    events = []
    for e in entries:
        events.append((datetime.fromisoformat(e["logged_at"]), 0, e["id"], "open", e))
        if e["status"] in ("win", "loss", "expired"):
            events.append((datetime.fromisoformat(e["checked_at"]), 1, e["id"], "close", e))
    # Tie-break: close (0) before open (1) at an identical timestamp - releases
    # capital before deploying it, the conventional choice for simultaneous
    # events. Negligible practical impact given microsecond timestamp precision.
    events.sort(key=lambda ev: (ev[0], ev[3] == "open", ev[2]))

    balance = STARTING_BALANCE
    peak = balance
    max_drawdown_pct = 0.0
    open_risk: dict[int, float] = {}
    equity_curve = []

    for _, _, trade_id, kind, e in events:
        if kind == "open":
            open_risk[trade_id] = balance * RISK_PCT / 100
            continue

        risk_dollars = open_risk.pop(trade_id, balance * RISK_PCT / 100)
        rm = r_multiple(e)
        if rm is None:
            continue
        pnl = risk_dollars * rm
        balance += pnl
        peak = max(peak, balance)
        drawdown_pct = (peak - balance) / peak * 100 if peak > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        equity_curve.append({
            "trade_id": trade_id,
            "closed_at": e["checked_at"],
            "symbol": e["symbol"],
            "timeframe": e["timeframe"],
            "based_on": e["based_on"],
            "bias": e["bias"],
            "status": e["status"],
            "r_multiple": round(rm, 3),
            "risk_dollars": round(risk_dollars, 2),
            "pnl_dollars": round(pnl, 2),
            "balance_after": round(balance, 2),
        })

    open_risk_dollars = sum(open_risk.values())
    account = {
        "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "starting_balance": STARTING_BALANCE,
        "risk_pct": RISK_PCT,
        "balance": round(balance, 2),
        "peak_balance": round(peak, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "total_return_pct": round((balance - STARTING_BALANCE) / STARTING_BALANCE * 100, 2),
        "trades_closed": len(equity_curve),
        "trades_currently_open": len(open_risk),
        "open_risk_dollars": round(open_risk_dollars, 2),
        "open_risk_pct_of_balance": round(open_risk_dollars / balance * 100, 2) if balance > 0 else 0.0,
    }
    return account, equity_curve


def main():
    entries = entries_since(None)
    account, equity_curve = simulate(entries)

    ACCOUNT_PATH.write_text(json.dumps(account, indent=2) + "\n")
    with open(EQUITY_CURVE_PATH, "w") as f:
        for row in equity_curve:
            f.write(json.dumps(row) + "\n")

    print(f"Starting balance: ${account['starting_balance']:.2f}")
    print(f"Current balance:  ${account['balance']:.2f}  ({account['total_return_pct']:+.2f}%)")
    print(f"Peak balance:     ${account['peak_balance']:.2f}")
    print(f"Max drawdown:     {account['max_drawdown_pct']:.2f}%")
    print(f"Trades closed:    {account['trades_closed']}")
    print(f"Trades open now:  {account['trades_currently_open']} "
          f"(${account['open_risk_dollars']:.2f} at risk, "
          f"{account['open_risk_pct_of_balance']:.2f}% of balance)")


if __name__ == "__main__":
    main()

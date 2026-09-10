"""
Paper-trading simulator: turns each ticker's buy/sell/hold suggestion (from
llm_decision.py, after passing critic.py's review) into a simulated trade
against fake money, and keeps every trade as a permanent record. No real
money moves — this exists so the agent's suggestions can be judged
retroactively against what they'd actually have earned.

Sizing: SIM_STARTING_CAPITAL (config.py) is split evenly across the
watchlist — each ticker gets a fixed allocation and trades only within it.
There is no shared cash pool across tickers, and no rebalancing if the
watchlist size changes between runs. Within a ticker's allocation, a buy
deploys a fraction of the ticker's *available cash* based on the LLM's
confidence in the suggestion (POSITION_SIZE_*_CONFIDENCE in config.py) —
high confidence uses the full available cash, lower confidence holds some
back. Leftover cash from a partial buy isn't stranded: it's still tracked
in the ticker's running cash balance and available to a later buy once the
current position is closed (see get_position's replay logic).

Exit rule: a position only closes on an explicit "sell" suggestion from the
agent on a later run, and always exits the *entire* position — no
stop-loss/take-profit, and no confidence-scaled partial sells. Partial
sells were deliberately left out: they'd require tracking multiple
cost-basis lots per ticker (e.g. FIFO or average cost) instead of the
current single-lot-per-open-position model, for a less obviously useful
signal than confidence-scaled entries (a "sell, but only somewhat" call is
a murkier concept than "buy, but only somewhat"). This otherwise matches
the rest of the pipeline's signal-driven (not price-driven) design.

State isn't stored directly — like memory.py, everything is derived by
replaying the `trades` table, so there's a single source of truth (the
trade log itself) rather than state that can drift out of sync with it.
Trades live in the same SQLite file as run history (memory.DB_PATH), so the
existing GitHub Actions persistence step already covers them.
"""
import datetime as dt

import config
import data_fetch
import memory

_VALID_ACTIONS = {"buy", "sell"}


def init_db():
    with memory._connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                shares REAL NOT NULL,
                cash_amount REAL NOT NULL,
                realized_pnl REAL,
                reason TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # Migration for DBs created before confidence-based sizing existed —
        # CREATE TABLE IF NOT EXISTS above is a no-op against an existing
        # table, so an already-created trades table needs these added
        # explicitly rather than picking them up from the schema above.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
        if "confidence" not in existing_cols:
            conn.execute("ALTER TABLE trades ADD COLUMN confidence TEXT")
        if "size_fraction" not in existing_cols:
            conn.execute("ALTER TABLE trades ADD COLUMN size_fraction REAL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades (ticker, run_date)")


def _ticker_trades(ticker: str) -> list:
    init_db()
    with memory._connect() as conn:
        rows = conn.execute(
            "SELECT action, price, shares, cash_amount, realized_pnl "
            "FROM trades WHERE ticker = ? ORDER BY id ASC",
            (ticker,),
        ).fetchall()
    return rows


def get_position(ticker: str, allocation: float) -> dict:
    """
    Replays this ticker's trade history against its starting allocation to
    get current state: available cash, open shares, and (if holding) the
    cost basis of that open position.
    """
    cash = allocation
    shares = 0.0
    cost_basis = 0.0
    for action, _price, tr_shares, cash_amount, _pnl in _ticker_trades(ticker):
        if action == "buy":
            cash -= cash_amount
            shares += tr_shares
            cost_basis = cash_amount
        elif action == "sell":
            cash += cash_amount
            shares = 0.0
            cost_basis = 0.0
    return {"cash": cash, "shares": shares, "cost_basis": cost_basis}


def record_trade(ticker: str, run_date: str, action: str, price: float, shares: float,
                  cash_amount: float, realized_pnl: float = None, reason: str = None,
                  confidence: str = None, size_fraction: float = None):
    if action not in _VALID_ACTIONS:
        raise ValueError(f"invalid trade action: {action!r}")
    init_db()
    with memory._connect() as conn:
        conn.execute(
            """INSERT INTO trades
               (run_date, ticker, action, price, shares, cash_amount, realized_pnl, reason,
                confidence, size_fraction, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_date, ticker, action, price, shares, cash_amount, realized_pnl, reason,
             confidence, size_fraction, dt.datetime.now().isoformat()),
        )


def _position_size_fraction(confidence: str) -> float:
    """Fraction of a ticker's available cash to deploy on a buy, by the
    LLM's stated confidence. Unrecognized/missing confidence is treated as
    low — conservative default for an unreadable signal."""
    return {
        "high": config.POSITION_SIZE_HIGH_CONFIDENCE,
        "medium": config.POSITION_SIZE_MEDIUM_CONFIDENCE,
        "low": config.POSITION_SIZE_LOW_CONFIDENCE,
    }.get(confidence, config.POSITION_SIZE_LOW_CONFIDENCE)


def process_suggestion(ticker: str, suggestion: str, price, allocation: float,
                        confidence: str = None, run_date: str = None, reason: str = None):
    """
    Applies a buy/sell/hold suggestion to the simulated portfolio for
    `ticker`. No-ops (returns None) on "hold", on a "buy" with no available
    cash or an already-open position, on a "sell" with no open position, or
    if `price` is missing/invalid. Returns a dict describing the trade
    executed, if any:
      {"action": "buy", "shares": ..., "cash_amount": ..., "size_fraction": ...}
      {"action": "sell", "shares": ..., "cash_amount": ..., "realized_pnl": ...}

    A buy deploys `_position_size_fraction(confidence)` of the ticker's
    available cash, not necessarily all of it — see the module docstring.
    A sell always exits the entire position, regardless of confidence.
    """
    if price is None or price <= 0:
        return None
    run_date = run_date or dt.date.today().isoformat()
    position = get_position(ticker, allocation)

    if suggestion == "buy" and position["shares"] == 0 and position["cash"] > 0:
        fraction = _position_size_fraction(confidence)
        cash_amount = position["cash"] * fraction
        shares = cash_amount / price
        record_trade(ticker, run_date, "buy", price, shares, cash_amount, reason=reason,
                      confidence=confidence, size_fraction=fraction)
        return {"action": "buy", "shares": shares, "cash_amount": cash_amount, "size_fraction": fraction}

    if suggestion == "sell" and position["shares"] > 0:
        shares = position["shares"]
        cash_amount = shares * price
        realized_pnl = cash_amount - position["cost_basis"]
        record_trade(ticker, run_date, "sell", price, shares, cash_amount,
                      realized_pnl=realized_pnl, reason=reason)
        return {"action": "sell", "shares": shares, "cash_amount": cash_amount, "realized_pnl": realized_pnl}

    return None


def portfolio_summary(watchlist: list, allocation: float, current_prices: dict) -> dict:
    """
    current_prices: {ticker: latest_price or None}, used to mark open
    positions to market. Returns aggregate totals across the whole
    watchlist: starting capital, current total equity (cash + marked
    positions), total return %, and total realized P&L from closed trades.
    """
    starting_capital = allocation * len(watchlist)
    realized_total = 0.0
    equity_total = 0.0
    open_positions = []

    for ticker in watchlist:
        position = get_position(ticker, allocation)
        for action, _price, _shares, _cash_amount, realized_pnl in _ticker_trades(ticker):
            if action == "sell" and realized_pnl is not None:
                realized_total += realized_pnl

        if position["shares"] > 0:
            mark_price = current_prices.get(ticker)
            mark_value = position["shares"] * mark_price if mark_price else position["cost_basis"]
            equity_total += mark_value
            open_positions.append({
                "ticker": ticker,
                "shares": position["shares"],
                "unrealized_pnl": mark_value - position["cost_basis"],
            })
        else:
            equity_total += position["cash"]

    return {
        "starting_capital": starting_capital,
        "total_equity": equity_total,
        "total_return_pct": ((equity_total - starting_capital) / starting_capital * 100) if starting_capital else 0.0,
        "realized_pnl": realized_total,
        "open_positions": open_positions,
    }


def buy_and_hold_benchmark(watchlist: list, allocation: float, anchor_date: str, current_prices: dict) -> dict:
    """
    What buy-and-hold would have returned since `anchor_date` (typically
    memory.get_earliest_run_date() — this bot's first-ever run), using the
    same per-ticker allocation as the live portfolio, so the two numbers
    are directly comparable. Fetches each ticker's closing price on/after
    anchor_date fresh on every call — a lightweight cost for a once-daily
    run, and avoids needing to persist a price that's fixed once set.

    A ticker whose anchor or current price can't be determined is skipped
    from both the total and its capital base (so a missing price doesn't
    misleadingly read as a 100% loss) and listed in `skipped_tickers`.
    """
    equity_total = 0.0
    skipped = []

    for ticker in watchlist:
        current_price = current_prices.get(ticker)
        if not current_price:
            skipped.append(ticker)
            continue
        try:
            anchor_price = float(data_fetch.get_price_history(ticker, start=anchor_date)["Close"].iloc[0])
        except Exception:
            skipped.append(ticker)
            continue
        shares = allocation / anchor_price
        equity_total += shares * current_price

    priced_starting_capital = allocation * (len(watchlist) - len(skipped))

    return {
        "anchor_date": anchor_date,
        "starting_capital": priced_starting_capital,
        "total_equity": equity_total,
        "total_return_pct": ((equity_total - priced_starting_capital) / priced_starting_capital * 100)
        if priced_starting_capital else 0.0,
        "skipped_tickers": skipped,
    }

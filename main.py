"""
Entry point. Run manually with `python main.py`, or schedule via cron / GitHub Actions.

Pipeline: for each ticker in WATCHLIST ->
  fetch price history + news -> compute signals -> LLM synthesis (optional,
  includes a buy/sell/hold suggestion) -> apply that suggestion to the
  paper-trading simulation (simulator.py) -> build digest -> send to Telegram
"""
import datetime as dt
import traceback

import config
import data_fetch
import analysis
import llm_decision
import notifier
import memory
import simulator


def build_ticker_report(ticker: str):
    """Returns {"signals": [...], "headlines": [...], "llm": dict|None, "latest_price": float|None}."""
    signals = []
    headlines = []
    latest_price = None

    try:
        price_df = data_fetch.get_price_history(ticker)
        signals += analysis.technical_signals(price_df)
        latest_price = float(price_df["Close"].iloc[-1])
    except Exception as e:
        print(f"[{ticker}] price fetch/analysis failed: {e}")

    try:
        sentiment = data_fetch.get_news_sentiment(ticker)
        headlines = data_fetch.get_recent_headlines(ticker)
        if sentiment:
            signals += analysis.sentiment_signals(sentiment)
        else:
            # Premium endpoint unavailable (free-tier key) — fall back to a
            # basic free keyword scan over recent headlines instead.
            signals += analysis.headline_keyword_sentiment(headlines)
    except Exception as e:
        print(f"[{ticker}] sentiment/news fetch failed: {e}")

    llm_result = llm_decision.synthesize(ticker, signals, headlines)

    return {"signals": signals, "headlines": headlines, "llm": llm_result, "latest_price": latest_price}


def run_simulation(ticker: str, report: dict, allocation: float, run_date: str):
    """
    Applies this run's LLM suggestion (if any) to the paper-trading
    simulation. No-op if the LLM layer is unconfigured/failed, or the
    suggestion was rejected/withheld by the critic (which defaults an
    unreviewable draft's suggestion to "hold").
    """
    llm = report.get("llm")
    price = report.get("latest_price")
    if not llm or price is None:
        return None
    return simulator.process_suggestion(
        ticker, llm.get("suggestion", "hold"), price, allocation,
        run_date=run_date, reason=llm.get("summary"),
    )


def format_digest(results: dict, trades: dict, portfolio: dict) -> str:
    today = dt.date.today().isoformat()
    lines = [f"*Stock Signal Digest — {today}*\n"]

    flagged = {
        t: r for t, r in results.items()
        if len(r["signals"]) >= config.MIN_SIGNALS_TO_ALERT
        or (r["llm"] and r["llm"].get("watch_worthy"))
    }

    if not flagged:
        lines.append("No notable signals today across your watchlist.")
    else:
        for ticker, report in flagged.items():
            lines.append(f"\n*{ticker}*")
            for name, direction, explanation in report["signals"]:
                arrow = {"bullish": "▲", "bearish": "▼", "neutral": "●"}.get(direction, "●")
                lines.append(f"  {arrow} {name}: {explanation}")

            llm = report["llm"]
            if llm:
                conf = llm.get("confidence", "unknown")
                lines.append(f"  🤖 [{conf} confidence] {llm.get('summary', '')}")
                lines.append(f"  📈 Suggestion: {llm.get('suggestion', 'hold').upper()}")
                if llm.get("tool_calls"):
                    lines.append(f"  🔧 Agent pulled extra data: {len(llm['tool_calls'])} tool call(s)")
                if llm.get("critic_approved") is False:
                    lines.append("  ⚠️ Flagged by safety review — see note above")

            trade = trades.get(ticker)
            if trade:
                if trade["action"] == "buy":
                    lines.append(f"  💰 Simulated BUY: {trade['shares']:.3f} sh @ ${report['latest_price']:.2f} (${trade['cash_amount']:.2f})")
                else:
                    lines.append(f"  💰 Simulated SELL: {trade['shares']:.3f} sh @ ${report['latest_price']:.2f} — realized P&L: ${trade['realized_pnl']:+.2f}")

    if portfolio:
        lines.append("\n*Paper-trading portfolio*")
        lines.append(f"  Equity: ${portfolio['total_equity']:.2f} (started ${portfolio['starting_capital']:.2f}, {portfolio['total_return_pct']:+.1f}%)")
        lines.append(f"  Realized P&L to date: ${portfolio['realized_pnl']:+.2f}")
        if portfolio["open_positions"]:
            held = ", ".join(p["ticker"] for p in portfolio["open_positions"])
            lines.append(f"  Open positions: {held}")

    lines.append("\n_Simulated trades only — not financial advice, do your own research._")
    return "\n".join(lines)


def main():
    results = {}
    trades = {}
    allocation = config.SIM_STARTING_CAPITAL / len(config.WATCHLIST) if config.WATCHLIST else 0
    run_date = dt.date.today().isoformat()

    for ticker in config.WATCHLIST:
        print(f"Processing {ticker}...")
        try:
            results[ticker] = build_ticker_report(ticker)
        except Exception:
            print(f"Unexpected failure on {ticker}:")
            traceback.print_exc()
            results[ticker] = {"signals": [], "headlines": [], "llm": None, "latest_price": None}

        try:
            memory.save_run(ticker, results[ticker])
        except Exception as e:
            # History is a nice-to-have, not load-bearing — a save failure
            # shouldn't stop the digest itself from going out.
            print(f"[{ticker}] Failed to save run history: {e}")

        try:
            trades[ticker] = run_simulation(ticker, results[ticker], allocation, run_date)
        except Exception as e:
            print(f"[{ticker}] Trading simulation failed: {e}")
            trades[ticker] = None

    current_prices = {t: r.get("latest_price") for t, r in results.items()}
    try:
        portfolio = simulator.portfolio_summary(config.WATCHLIST, allocation, current_prices)
    except Exception as e:
        print(f"Portfolio summary failed: {e}")
        portfolio = None

    digest = format_digest(results, trades, portfolio)
    notifier.send_telegram_message(digest)


if __name__ == "__main__":
    main()

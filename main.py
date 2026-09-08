"""
Entry point. Run manually with `python main.py`, or schedule via cron / GitHub Actions.

Pipeline: for each ticker in WATCHLIST ->
  fetch price history + news -> compute signals -> LLM synthesis (optional) ->
  build digest -> send to Telegram
"""
import datetime as dt
import traceback

import config
import data_fetch
import analysis
import llm_decision
import notifier
import memory


def build_ticker_report(ticker: str):
    """Returns {"signals": [...], "headlines": [...], "llm": dict|None}."""
    signals = []
    headlines = []

    try:
        price_df = data_fetch.get_price_history(ticker)
        signals += analysis.technical_signals(price_df)
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

    return {"signals": signals, "headlines": headlines, "llm": llm_result}


def format_digest(results: dict) -> str:
    today = dt.date.today().isoformat()
    lines = [f"*Stock Signal Digest — {today}*\n"]

    flagged = {
        t: r for t, r in results.items()
        if len(r["signals"]) >= config.MIN_SIGNALS_TO_ALERT
        or (r["llm"] and r["llm"].get("watch_worthy"))
    }

    if not flagged:
        lines.append("No notable signals today across your watchlist.")
        return "\n".join(lines)

    for ticker, report in flagged.items():
        lines.append(f"\n*{ticker}*")
        for name, direction, explanation in report["signals"]:
            arrow = {"bullish": "▲", "bearish": "▼", "neutral": "●"}.get(direction, "●")
            lines.append(f"  {arrow} {name}: {explanation}")

        llm = report["llm"]
        if llm:
            conf = llm.get("confidence", "unknown")
            lines.append(f"  🤖 [{conf} confidence] {llm.get('summary', '')}")
            if llm.get("tool_calls"):
                lines.append(f"  🔧 Agent pulled extra data: {len(llm['tool_calls'])} tool call(s)")
            if llm.get("critic_approved") is False:
                lines.append("  ⚠️ Flagged by safety review — see note above")

    lines.append("\n_Not financial advice — signals only, do your own research._")
    return "\n".join(lines)


def main():
    results = {}
    for ticker in config.WATCHLIST:
        print(f"Processing {ticker}...")
        try:
            results[ticker] = build_ticker_report(ticker)
        except Exception:
            print(f"Unexpected failure on {ticker}:")
            traceback.print_exc()
            results[ticker] = {"signals": [], "headlines": [], "llm": None}

        try:
            memory.save_run(ticker, results[ticker])
        except Exception as e:
            # History is a nice-to-have, not load-bearing — a save failure
            # shouldn't stop the digest itself from going out.
            print(f"[{ticker}] Failed to save run history: {e}")

    digest = format_digest(results)
    notifier.send_telegram_message(digest)


if __name__ == "__main__":
    main()

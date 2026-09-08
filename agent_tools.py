"""
Tools exposed to the Gemini agent for autonomous use during synthesis.

These are plain Python functions with type hints and docstrings — Gemini's
automatic function calling (AFC) infers the callable schema directly from
them and decides on its own whether/when to call them while reasoning about
a ticker. This is what makes the agent's information-gathering genuinely
autonomous rather than a fixed bundle of pre-fetched context: the model
chooses to pull more evidence only when it judges the initial signals or
headlines too thin to reach a confident read.

Each tool wraps a data_fetch.py call to avoid duplicating fetch logic, and
reports its own usage into a shared call-log list so the caller can see
what the agent actually chose to do (used in main.py's digest formatting).
"""
import data_fetch


def make_tools(call_log: list):
    """
    Returns a fresh pair of tool functions bound to this ticker's call_log,
    so concurrent/successive syntheses for different tickers don't share
    invocation history with each other.
    """

    def get_extended_price_history(ticker: str) -> dict:
        """Fetch a longer, 6-month price and volume history for a stock ticker,
        beyond the default ~3-month analysis window. Use this when the
        available technical signals are ambiguous, contradictory, or seem to
        lack context — e.g. to check whether a volume spike or price move is
        actually unusual relative to a longer trend, or the ticker is just
        naturally volatile. Returns summary statistics rather than raw data:
        the 6-month high and low, the average daily volume, and where the
        current price sits within that 6-month range (as a percentile).

        Args:
            ticker: The stock ticker symbol, e.g. "AAPL".
        """
        call_log.append(f"get_extended_price_history({ticker})")
        try:
            df = data_fetch.get_price_history(ticker, period="6mo")
            close = df["Close"]
            current = close.iloc[-1]
            low, high = close.min(), close.max()
            position_pct = ((current - low) / (high - low) * 100) if high > low else 50.0
            return {
                "six_month_high": round(float(high), 2),
                "six_month_low": round(float(low), 2),
                "current_price": round(float(current), 2),
                "position_in_range_pct": round(position_pct, 1),
                "average_daily_volume": int(df["Volume"].mean()),
            }
        except Exception as e:
            return {"error": f"Could not fetch extended price history: {e}"}

    def get_extended_headlines(ticker: str) -> dict:
        """Fetch a longer lookback window (14 days instead of the default 3)
        of recent company news headlines for a stock ticker. Use this when
        the initial headline set is sparse (e.g. 0-1 headlines) or seems
        insufficient to judge whether news sentiment is actually notable.

        Args:
            ticker: The stock ticker symbol, e.g. "AAPL".
        """
        call_log.append(f"get_extended_headlines({ticker})")
        try:
            headlines = data_fetch.get_recent_headlines(ticker, days=14)
            return {"headlines": headlines, "count": len(headlines)}
        except Exception as e:
            return {"error": f"Could not fetch extended headlines: {e}"}

    return [get_extended_price_history, get_extended_headlines]

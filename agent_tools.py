"""
Tools exposed to the Gemini agents for autonomous use during synthesis.

These are plain Python functions with type hints and docstrings — Gemini's
automatic function calling (AFC) infers the callable schema directly from
them and decides on its own whether/when to call them while reasoning about
a ticker. This is what makes the agents' information-gathering genuinely
autonomous rather than a fixed bundle of pre-fetched context: a model
chooses to pull more evidence only when it judges its initial evidence too
thin to reach a confident read.

Bundled per specialist domain (see llm_decision.py's technical/news
analyst split): the technical analyst only gets price-history tools, the
news analyst only gets headline tools, and both get get_recent_history —
a specialist has no legitimate use for a tool outside its own domain, so
it's simpler and safer not to offer it, rather than trust the model not to
reach for it.

Each tool wraps a data source (data_fetch.py for live evidence, memory.py
for the agent's own past judgments) to avoid duplicating logic, and reports
its own usage into a shared call-log list so the caller can see what an
agent actually chose to do (used in main.py's digest formatting).

Two layers bound tool-call cost/latency, deliberately at different scopes:
`_MAX_TOOL_CALLS` in llm_decision.py caps one specialist's own calls within
one ticker; ToolCallBudget below caps the total across an entire main.py
run (every ticker, every specialist, including generator/critic-loop
retries) — without it, several genuinely ambiguous tickers in the same
run could each max out their own per-specialist ceiling and still compound
into real, uncapped aggregate cost.
"""
import data_fetch
import memory


class ToolCallBudget:
    """Shared, mutable counter threaded from main.py (one instance per run)
    through llm_decision.py into the tool closures below. Once exhausted,
    a tool call is refused (returns an {"error": ...} dict the model sees
    and can reason around, same shape as any other tool failure) rather
    than allowed to keep running — the model still decides whether to call
    a tool at all, this only caps how many of those calls can actually
    execute across the whole run."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._warned = False

    def try_consume(self) -> bool:
        if self.used >= self.limit:
            if not self._warned:
                print(f"[budget] Tool call budget ({self.limit}) exhausted for this run — "
                      f"further tool requests will be refused.")
                self._warned = True
            return False
        self.used += 1
        return True


def _make_get_recent_history_tool(ticker: str, call_log: list, tool_budget: ToolCallBudget):
    """Shared by both specialists — see make_technical_tools/make_news_tools.
    ticker is bound here (rather than left as a model-supplied argument like
    the other tools) so the model has no way to query another ticker's
    history mid-analysis of this one."""

    def get_recent_history() -> dict:
        """Look up this same ticker's own analysis history from the last 7
        days — what it was flagged for previously, its confidence, its
        buy/sell/hold suggestion, and whether that suggestion actually
        passed critic review (critic_approved) or was rejected and why
        (critic_reason). Use this to check whether today's evidence is a
        continuation of something already noted recently (in which case
        say so explicitly rather than re-presenting it as brand new), or
        to weigh a repeat suggestion differently if it was rejected before
        for reasons that still apply today. Do not call this unless
        today's evidence gives you a specific reason to check for
        continuity — most tickers most days don't need this.
        """
        if not tool_budget.try_consume():
            return {"error": "This run's tool call budget has been used up — proceed with the evidence you already have."}
        call_log.append(f"get_recent_history({ticker})")
        try:
            history = memory.get_recent_history(ticker, days=7)
            if not history:
                return {"history": [], "note": "No prior analysis found in the last 7 days."}
            return {"history": history}
        except Exception as e:
            return {"error": f"Could not fetch recent history: {e}"}

    return get_recent_history


def make_technical_tools(ticker: str, call_log: list, tool_budget: ToolCallBudget):
    """Tools for the technical analyst: extended price history + this
    ticker's own recent history. No headline access — that's the news
    analyst's domain."""

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
        if not tool_budget.try_consume():
            return {"error": "This run's tool call budget has been used up — proceed with the evidence you already have."}
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

    return [get_extended_price_history, _make_get_recent_history_tool(ticker, call_log, tool_budget)]


def make_news_tools(ticker: str, call_log: list, tool_budget: ToolCallBudget):
    """Tools for the news analyst: extended headlines + this ticker's own
    recent history. No price-history access — that's the technical
    analyst's domain."""

    def get_extended_headlines(ticker: str) -> dict:
        """Fetch a longer lookback window (14 days instead of the default 3)
        of recent company news headlines for a stock ticker. Use this when
        the initial headline set is sparse (e.g. 0-1 headlines) or seems
        insufficient to judge whether news sentiment is actually notable.

        Args:
            ticker: The stock ticker symbol, e.g. "AAPL".
        """
        if not tool_budget.try_consume():
            return {"error": "This run's tool call budget has been used up — proceed with the evidence you already have."}
        call_log.append(f"get_extended_headlines({ticker})")
        try:
            headlines = data_fetch.get_recent_headlines(ticker, days=14)
            return {"headlines": headlines, "count": len(headlines)}
        except Exception as e:
            return {"error": f"Could not fetch extended headlines: {e}"}

    return [get_extended_headlines, _make_get_recent_history_tool(ticker, call_log, tool_budget)]

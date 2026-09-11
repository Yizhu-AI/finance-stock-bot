"""
Central configuration. All secrets come from environment variables (.env file),
never hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# Comma-separated list of tickers to watch, e.g. "AAPL,TSLA,NVDA"
WATCHLIST = [t.strip().upper() for t in os.getenv("WATCHLIST", "AAPL,TSLA,NVDA").split(",") if t.strip()]

# --- Signal thresholds (tune these as you learn what's noisy vs. useful) ---
SMA_SHORT = 20
SMA_LONG = 50
VOLUME_SPIKE_MULTIPLIER = 1.8      # today's volume vs 20-day avg volume
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
NEWS_LOOKBACK_DAYS = 3
SENTIMENT_BULLISH_THRESHOLD = 0.6  # Finnhub bullishPercent above this = notably bullish
SENTIMENT_BEARISH_THRESHOLD = 0.4  # below this = notably bearish

# MACD crossover (pandas-ta) — standard 12/26/9 periods
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands (pandas-ta) — standard 20-period, 2 std dev
BB_PERIOD = 20
BB_STD = 2.0

# Minimum number of triggered signals before a ticker is included in the digest
MIN_SIGNALS_TO_ALERT = 1

# Delay between tickers in main.py's loop when the LLM layer is configured,
# to stay under Gemini's free-tier rate limit (15 requests/min). Each ticker
# runs up to 4 Gemini calls (technical analyst, news analyst, coordinator,
# critic — see "About the LLM decision layer" in the README for why it's a
# multi-agent pipeline, not one call), each with up to _MAX_TOOL_CALLS tool
# round-trips on top — with no pacing, a 10+ ticker watchlist can blow past
# that limit within a single run even before considering retries. Not
# applied if GEMINI_API_KEY isn't set (no LLM calls happen, nothing to pace).
# Raised from 8s to 15s when the single-analyst design became a 4-call
# per-ticker pipeline, to keep the same rough peak-RPM safety margin at the
# new, higher average call volume per ticker.
LLM_REQUEST_DELAY_SECONDS = float(os.getenv("LLM_REQUEST_DELAY_SECONDS") or "15")

# Delay between each pair of a ticker's own Gemini calls (technical analyst
# -> news analyst -> coordinator -> critic), inside llm_decision.py.
# LLM_REQUEST_DELAY_SECONDS above only paces the *start* of each ticker's
# turn — it doesn't stop a single ticker's own requests from bursting out
# almost simultaneously. Those bursts, repeated every
# LLM_REQUEST_DELAY_SECONDS, are enough to peak a rolling 60-second window
# well above the steady-state pacing would suggest (observed: free-tier
# limit is 15 RPM, but Google's own dashboard showed a 22 RPM peak even
# with the single-analyst design's lighter call volume). This doesn't reach
# the automatic tool-call round-trips themselves — those happen inside the
# SDK's own function-calling loop within one generate_content() call, with
# no hook to pace between them.
INTRA_TICKER_REQUEST_DELAY_SECONDS = float(os.getenv("INTRA_TICKER_REQUEST_DELAY_SECONDS") or "2")

# Caps the TOTAL autonomous tool calls across an entire run (every ticker,
# every specialist, including generator/critic-loop retries) — see
# agent_tools.ToolCallBudget. This is a different scope than
# _MAX_TOOL_CALLS in llm_decision.py, which only bounds one specialist's
# own calls within one ticker: several genuinely ambiguous tickers in the
# same run could each hit that per-specialist ceiling and still compound
# into real, uncapped aggregate cost/latency without this. Defaults to 2
# tool calls per ticker in the watchlist — comfortably above the observed
# typical usage (0-2 total per ticker, tool use is the exception not the
# rule) while still bounding the worst case.
GLOBAL_TOOL_CALL_BUDGET = int(os.getenv("GLOBAL_TOOL_CALL_BUDGET") or (2 * len(WATCHLIST)) or 10)

# --- Paper-trading simulation ---
# Total simulated capital, split evenly across the watchlist at run time
# (SIM_STARTING_CAPITAL / len(WATCHLIST) per ticker). No real money moves —
# see simulator.py.
SIM_STARTING_CAPITAL = float(os.getenv("SIM_STARTING_CAPITAL") or "10000")

# Position sizing by suggestion confidence: fraction of a ticker's
# available cash to deploy on a "buy" — higher confidence, bigger stake.
# An unrecognized/missing confidence value falls back to the low-confidence
# fraction (conservative default). Sells always exit the full position
# regardless of confidence — see simulator.py's docstring for why.
POSITION_SIZE_HIGH_CONFIDENCE = 1.0
POSITION_SIZE_MEDIUM_CONFIDENCE = 0.6
POSITION_SIZE_LOW_CONFIDENCE = 0.3

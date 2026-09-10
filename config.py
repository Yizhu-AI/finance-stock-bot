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
# can use several requests (synthesis, up to _MAX_TOOL_CALLS tool round-trips,
# a critic check) — with no pacing, a 10+ ticker watchlist can blow past that
# limit within a single run even before considering retries. Not applied if
# GEMINI_API_KEY isn't set (no LLM calls happen, nothing to pace).
LLM_REQUEST_DELAY_SECONDS = float(os.getenv("LLM_REQUEST_DELAY_SECONDS") or "8")

# Delay between a ticker's own synthesis call and its critic review call,
# inside llm_decision.py. LLM_REQUEST_DELAY_SECONDS above only paces the
# *start* of each ticker's turn — it doesn't stop a single ticker's own
# requests (synthesis, then possibly a few automatic tool-call round-trips,
# then the critic's separate call) from bursting out almost simultaneously.
# Those bursts, repeated every LLM_REQUEST_DELAY_SECONDS, are enough to peak
# a rolling 60-second window well above the steady-state pacing would
# suggest (observed: free-tier limit is 15 RPM, but Google's own dashboard
# showed a 22 RPM peak). This doesn't reach the automatic tool-call
# round-trips themselves — those happen inside the SDK's own function-calling
# loop within one generate_content() call, with no hook to pace between them.
CRITIC_REQUEST_DELAY_SECONDS = float(os.getenv("CRITIC_REQUEST_DELAY_SECONDS") or "2")

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

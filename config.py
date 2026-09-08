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

# Minimum number of triggered signals before a ticker is included in the digest
MIN_SIGNALS_TO_ALERT = 1

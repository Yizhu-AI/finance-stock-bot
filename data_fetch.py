"""
Pulls raw data for a ticker:
  - price/volume history (yfinance — free, no API key needed)
  - news + sentiment (Finnhub — free tier API key required)
"""
import datetime as dt
import requests
import yfinance as yf

import config


def get_price_history(ticker: str, period: str = "3mo", interval: str = "1d"):
    """Return a pandas DataFrame of OHLCV data."""
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")
    # yfinance can return a NaN OHLC row for the current, still-forming
    # trading day (volume already present, prices not yet settled). Drop
    # it here rather than in every caller — every consumer of this
    # DataFrame (analysis.py's rolling indicators, main.py's latest_price,
    # backtest.py) otherwise risks silently computing against NaN.
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def get_news_sentiment(ticker: str):
    """
    Finnhub's news-sentiment endpoint returns aggregate sentiment metrics
    (bullish/bearish %, sentiment score). NOTE: as of writing, this endpoint
    requires a paid Finnhub plan — free-tier keys get a 403. We degrade
    gracefully here (return None) so callers fall back to the free
    headline-keyword sentiment in analysis.py instead of crashing.
    Docs: https://finnhub.io/docs/api/news-sentiment
    """
    if not config.FINNHUB_API_KEY:
        return None
    url = "https://finnhub.io/api/v1/news-sentiment"
    resp = requests.get(url, params={"symbol": ticker, "token": config.FINNHUB_API_KEY}, timeout=10)
    if resp.status_code == 403:
        # Free-tier key, premium-only endpoint — expected, not an error.
        return None
    resp.raise_for_status()
    return resp.json()


def get_recent_headlines(ticker: str, days: int = None):
    """Return a short list of recent company news headlines for context in the digest."""
    if not config.FINNHUB_API_KEY:
        return []
    days = days or config.NEWS_LOOKBACK_DAYS
    today = dt.date.today()
    frm = today - dt.timedelta(days=days)
    url = "https://finnhub.io/api/v1/company-news"
    resp = requests.get(
        url,
        params={
            "symbol": ticker,
            "from": frm.isoformat(),
            "to": today.isoformat(),
            "token": config.FINNHUB_API_KEY,
        },
        timeout=10,
    )
    resp.raise_for_status()
    articles = resp.json()
    # Keep it light — just headline + source, most recent first
    return [{"headline": a["headline"], "source": a["source"]} for a in articles[:5]]

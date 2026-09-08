"""
Turns raw price data + sentiment data into a set of discrete "signals" —
each signal is a (name, direction, explanation) tuple so the digest can
show *why* a ticker was flagged, not just that it was.
"""
import pandas as pd
import config


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def technical_signals(df: pd.DataFrame):
    """df must have Close and Volume columns, most recent row last."""
    signals = []

    df = df.copy()
    df["sma_short"] = df["Close"].rolling(config.SMA_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(config.SMA_LONG).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["avg_volume_20"] = df["Volume"].rolling(20).mean()

    if len(df) < config.SMA_LONG + 2:
        return signals  # not enough history yet

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Moving average crossover
    if prev["sma_short"] <= prev["sma_long"] and latest["sma_short"] > latest["sma_long"]:
        signals.append(("SMA crossover", "bullish", f"{config.SMA_SHORT}-day SMA crossed above {config.SMA_LONG}-day SMA"))
    elif prev["sma_short"] >= prev["sma_long"] and latest["sma_short"] < latest["sma_long"]:
        signals.append(("SMA crossover", "bearish", f"{config.SMA_SHORT}-day SMA crossed below {config.SMA_LONG}-day SMA"))

    # Volume spike
    if latest["avg_volume_20"] and latest["Volume"] > config.VOLUME_SPIKE_MULTIPLIER * latest["avg_volume_20"]:
        ratio = latest["Volume"] / latest["avg_volume_20"]
        signals.append(("Volume spike", "neutral", f"Volume is {ratio:.1f}x the 20-day average"))

    # RSI extremes
    if latest["rsi"] >= config.RSI_OVERBOUGHT:
        signals.append(("RSI overbought", "bearish", f"RSI at {latest['rsi']:.0f} (overbought)"))
    elif latest["rsi"] <= config.RSI_OVERSOLD:
        signals.append(("RSI oversold", "bullish", f"RSI at {latest['rsi']:.0f} (oversold)"))

    return signals


def sentiment_signals(sentiment_data: dict):
    """Uses Finnhub's premium news-sentiment endpoint, when available."""
    signals = []
    if not sentiment_data:
        return signals

    sentiment = sentiment_data.get("sentiment", {})
    bullish_pct = sentiment.get("bullishPercent")
    bearish_pct = sentiment.get("bearishPercent")

    if bullish_pct is not None and bullish_pct >= config.SENTIMENT_BULLISH_THRESHOLD:
        signals.append(("News sentiment", "bullish", f"{bullish_pct*100:.0f}% of recent news/mentions are bullish"))
    elif bearish_pct is not None and bearish_pct >= (1 - config.SENTIMENT_BEARISH_THRESHOLD):
        signals.append(("News sentiment", "bearish", f"{bearish_pct*100:.0f}% of recent news/mentions are bearish"))

    return signals


# Small, free fallback for when Finnhub's premium news-sentiment endpoint
# isn't available: a naive keyword scan over recent headlines. This is much
# cruder than a real sentiment model — it's meant to catch obviously
# lopsided news coverage, not subtle sentiment shifts.
_POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies",
    "upgrade", "upgraded", "record", "growth", "profit", "profits", "gain",
    "gains", "strong", "outperform", "bullish", "jump", "jumps", "rise", "rises",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "plunge", "plunges", "slump", "slumps", "downgrade",
    "downgraded", "loss", "losses", "decline", "declines", "weak", "underperform",
    "bearish", "drop", "drops", "fall", "falls", "lawsuit", "investigation",
    "recall", "cut", "cuts", "layoffs",
}


def headline_keyword_sentiment(headlines: list, min_headlines: int = 3):
    """headlines: list of {"headline": str, "source": str} from get_recent_headlines."""
    signals = []
    if not headlines or len(headlines) < min_headlines:
        return signals

    pos, neg = 0, 0
    for h in headlines:
        words = h["headline"].lower().split()
        pos += sum(1 for w in words if w.strip(".,!?") in _POSITIVE_WORDS)
        neg += sum(1 for w in words if w.strip(".,!?") in _NEGATIVE_WORDS)

    total = pos + neg
    if total == 0:
        return signals  # no clear keyword signal either way

    net_ratio = pos / total
    if net_ratio >= 0.7 and pos >= 2:
        signals.append(("Headline sentiment (basic)", "bullish", f"{pos} positive vs {neg} negative keyword hits across {len(headlines)} headlines"))
    elif net_ratio <= 0.3 and neg >= 2:
        signals.append(("Headline sentiment (basic)", "bearish", f"{neg} negative vs {pos} positive keyword hits across {len(headlines)} headlines"))

    return signals

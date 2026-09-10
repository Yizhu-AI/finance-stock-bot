"""
Turns raw price data + sentiment data into a set of discrete "signals" —
each signal is a (name, direction, explanation) tuple so the digest can
show *why* a ticker was flagged, not just that it was.
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # noqa: F401 — import registers the df.ta accessor used below

import config


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss == 0 makes the division above NaN — resolve it explicitly:
    # gains with zero losses is a genuine RSI of 100 (fully overbought), but
    # zero gains AND zero losses means the price hasn't moved at all, which
    # is neutral (RSI 50), not oversold. The original naive
    # "avg_loss.replace(0, tiny_epsilon)" approach collapsed both cases to
    # RSI≈0, incorrectly flagging a flat, unmoving price as oversold.
    no_loss_mask = avg_loss == 0
    rsi = rsi.where(~no_loss_mask, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi


def technical_signals(df: pd.DataFrame):
    """df must have Close and Volume columns, most recent row last."""
    signals = []

    df = df.copy()
    df["sma_short"] = df["Close"].rolling(config.SMA_SHORT).mean()
    df["sma_long"] = df["Close"].rolling(config.SMA_LONG).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["avg_volume_20"] = df["Volume"].rolling(20).mean()
    df.ta.macd(fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL, append=True)
    df.ta.bbands(length=config.BB_PERIOD, std=config.BB_STD, append=True)

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

    # MACD crossover — a second, independent trend-following read alongside
    # the SMA crossover above (different smoothing, reacts faster).
    macd_col = f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    macd_signal_col = f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    if prev[macd_col] <= prev[macd_signal_col] and latest[macd_col] > latest[macd_signal_col]:
        signals.append(("MACD crossover", "bullish", "MACD line crossed above its signal line"))
    elif prev[macd_col] >= prev[macd_signal_col] and latest[macd_col] < latest[macd_signal_col]:
        signals.append(("MACD crossover", "bearish", "MACD line crossed below its signal line"))

    # Bollinger Bands — a volatility-relative read on how stretched the
    # close is from its own recent mean, independent of RSI's fixed 0-100 scale.
    bb_lower_col = f"BBL_{config.BB_PERIOD}_{config.BB_STD}"
    bb_upper_col = f"BBU_{config.BB_PERIOD}_{config.BB_STD}"
    if latest["Close"] < latest[bb_lower_col]:
        signals.append(("Bollinger Bands", "bullish",
                         f"Close (${latest['Close']:.2f}) is below the lower band (${latest[bb_lower_col]:.2f})"))
    elif latest["Close"] > latest[bb_upper_col]:
        signals.append(("Bollinger Bands", "bearish",
                         f"Close (${latest['Close']:.2f}) is above the upper band (${latest[bb_upper_col]:.2f})"))

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

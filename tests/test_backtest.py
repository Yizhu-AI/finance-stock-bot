"""
Tests for backtest.py's pure helper functions (return math, signal
generation, formatting). run_backtest_backtrader()/run_backtest_vectorbt()
are not tested here — like main.py's live LLM/API calls, they need real
network access and a backtesting engine, so they're exercised by actually
running `python backtest.py`, not by pytest.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import backtest


def test_buy_and_hold_return_pct_gain():
    df = pd.DataFrame({"Close": [100.0, 110.0, 120.0]})
    assert backtest._buy_and_hold_return_pct(df) == 20.0


def test_buy_and_hold_return_pct_loss():
    df = pd.DataFrame({"Close": [100.0, 90.0, 80.0]})
    assert backtest._buy_and_hold_return_pct(df) == -20.0


def test_fmt_pct_formats_with_sign():
    assert backtest._fmt_pct(5.0) == "+5.0%"
    assert backtest._fmt_pct(-5.0) == "-5.0%"


def test_fmt_pct_handles_none():
    assert backtest._fmt_pct(None) == "n/a"


def test_fmt_num_formats_with_decimals():
    assert backtest._fmt_num(1.23456) == "1.23"
    assert backtest._fmt_num(1.23456, decimals=1) == "1.2"


def test_fmt_num_handles_none():
    assert backtest._fmt_num(None) == "n/a"


def test_sma_crossover_signals_fires_on_trend_reversal(monkeypatch):
    monkeypatch.setattr(backtest.config, "SMA_SHORT", 2)
    monkeypatch.setattr(backtest.config, "SMA_LONG", 4)

    # Falling, then rising (crosses up = entry), then falling again (crosses down = exit).
    close = pd.Series([20, 18, 16, 14, 12, 14, 16, 18, 20, 22, 24, 22, 20, 18, 16, 14])
    entries, exits = backtest._sma_crossover_signals(close)

    assert entries.dtype == bool and exits.dtype == bool
    assert not entries.isna().any() and not exits.isna().any()
    # warmup period (before SMA_LONG has a full window) can't signal anything
    assert not entries.iloc[: backtest.config.SMA_LONG - 1].any()
    assert not exits.iloc[: backtest.config.SMA_LONG - 1].any()
    # a bar is never simultaneously an entry and an exit
    assert not (entries & exits).any()
    assert entries.any() and exits.any()
    # the entry should land after the price trend has actually turned upward
    first_entry_idx = entries[entries].index[0]
    assert close.iloc[first_entry_idx] > close.iloc[first_entry_idx - 1]


def test_sma_crossover_signals_no_signal_on_flat_price():
    close = pd.Series([100.0] * (backtest.config.SMA_LONG + 5))
    entries, exits = backtest._sma_crossover_signals(close)
    assert not entries.any()
    assert not exits.any()

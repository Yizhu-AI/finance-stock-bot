"""
Tests for backtest.py's pure helper functions (return math, formatting).
The actual run_backtest() is not tested here — like main.py's live LLM/API
calls, it needs real network access and backtrader's engine, so it's
exercised by actually running `python backtest.py`, not by pytest.
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

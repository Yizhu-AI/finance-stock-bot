"""
Tests for memory.py — persistent run history. Uses a temporary DB file per
test so tests never touch the real signal_history.db, and each test starts
from a clean slate.
"""
import sys
import os
import datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import memory


def _days_ago(n: int) -> str:
    return (dt.date.today() - dt.timedelta(days=n)).isoformat()


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    memory.DB_PATH = str(tmp_path / "test_history.db")
    yield
    # tmp_path is cleaned up automatically by pytest between tests


def test_save_and_retrieve_single_run():
    report = {
        "signals": [("SMA crossover", "bullish", "test")],
        "llm": {"summary": "Test summary.", "confidence": "medium", "watch_worthy": True, "critic_approved": True, "tool_calls": []},
    }
    run_date = _days_ago(2)
    memory.save_run("AAPL", report, run_date=run_date)
    history = memory.get_recent_history("AAPL", days=7, exclude_today=False)
    assert len(history) == 1
    assert history[0]["date"] == run_date
    assert history[0]["summary"] == "Test summary."
    assert history[0]["was_watch_worthy"] is True


def test_history_ordered_most_recent_first():
    report = {"signals": [], "llm": None}
    memory.save_run("AAPL", report, run_date=_days_ago(6))
    memory.save_run("AAPL", report, run_date=_days_ago(2))
    memory.save_run("AAPL", report, run_date=_days_ago(4))

    history = memory.get_recent_history("AAPL", days=30, exclude_today=False)
    dates = [h["date"] for h in history]
    assert dates == sorted(dates, reverse=True)


def test_tickers_isolated_from_each_other():
    report = {"signals": [], "llm": None}
    run_date = _days_ago(2)
    memory.save_run("AAPL", report, run_date=run_date)
    memory.save_run("TSLA", report, run_date=run_date)

    aapl_history = memory.get_recent_history("AAPL", days=7, exclude_today=False)
    tsla_history = memory.get_recent_history("TSLA", days=7, exclude_today=False)
    assert len(aapl_history) == 1
    assert len(tsla_history) == 1


def test_untouched_ticker_returns_empty_list():
    history = memory.get_recent_history("NVDA", days=7)
    assert history == []


def test_run_with_no_llm_result_saves_null_fields():
    report = {"signals": [("RSI overbought", "bearish", "test")], "llm": None}
    memory.save_run("AAPL", report, run_date=_days_ago(2))
    history = memory.get_recent_history("AAPL", days=7, exclude_today=False)
    assert history[0]["summary"] is None
    assert history[0]["signal_count"] == 1


def test_days_cutoff_excludes_old_runs():
    report = {"signals": [], "llm": None}
    memory.save_run("AAPL", report, run_date="2020-01-01")  # far in the past
    history = memory.get_recent_history("AAPL", days=7, exclude_today=False)
    assert history == []

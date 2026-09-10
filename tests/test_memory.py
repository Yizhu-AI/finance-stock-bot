"""
Tests for memory.py — persistent run history. Uses a temporary DB file per
test so tests never touch the real signal_history.db, and each test starts
from a clean slate.
"""
import sys
import os
import datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

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
        "llm": {
            "summary": "Test summary.", "confidence": "medium", "watch_worthy": True,
            "critic_approved": True, "critic_reason": "grounded and plausible",
            "suggestion": "buy", "tool_calls": [], "regenerated": False,
        },
    }
    run_date = _days_ago(2)
    memory.save_run("AAPL", report, run_date=run_date)
    history = memory.get_recent_history("AAPL", days=7, exclude_today=False)
    assert len(history) == 1
    assert history[0]["date"] == run_date
    assert history[0]["summary"] == "Test summary."
    assert history[0]["was_watch_worthy"] is True
    assert history[0]["suggestion"] == "buy"
    assert history[0]["critic_approved"] is True
    assert history[0]["critic_reason"] == "grounded and plausible"
    assert history[0]["regenerated"] is False


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
    assert history[0]["suggestion"] is None
    assert history[0]["critic_approved"] is None
    assert history[0]["critic_reason"] is None
    assert history[0]["regenerated"] is None


def test_critic_rejected_run_persists_reason_and_hold_suggestion():
    # Mirrors what critic.review() returns on rejection: suggestion
    # defaults to "hold", but critic_reason explains why the original
    # call was rejected — this is the audit-trail case that motivated
    # adding these columns (a rejection's reasoning used to only exist
    # in that run's console output).
    report = {
        "signals": [("MACD crossover", "bullish", "test")],
        "llm": {
            "summary": "The automated analysis for this ticker did not pass safety review and has been withheld.",
            "confidence": "low", "watch_worthy": True, "critic_approved": False,
            "critic_reason": "suggestion contradicted by uniformly bearish evidence",
            "suggestion": "hold", "tool_calls": [],
        },
    }
    memory.save_run("MRNA", report, run_date=_days_ago(1))
    history = memory.get_recent_history("MRNA", days=7, exclude_today=False)
    assert history[0]["critic_approved"] is False
    assert history[0]["critic_reason"] == "suggestion contradicted by uniformly bearish evidence"
    assert history[0]["suggestion"] == "hold"


def test_regenerated_run_persists_flag():
    # Mirrors llm_decision.py's generator/critic loop: a draft rejected
    # once, revised, and approved on the retry still carries
    # regenerated=True even though the final critic_approved is True —
    # this is the "did the loop actually kick in" audit signal.
    report = {
        "signals": [("RSI oversold", "bullish", "test")],
        "llm": {
            "summary": "Revised summary after critic feedback.", "confidence": "medium",
            "watch_worthy": True, "critic_approved": True,
            "critic_reason": "revised suggestion now matches the evidence",
            "suggestion": "buy", "tool_calls": [], "regenerated": True,
        },
    }
    memory.save_run("GOOG", report, run_date=_days_ago(1))
    history = memory.get_recent_history("GOOG", days=7, exclude_today=False)
    assert history[0]["regenerated"] is True
    assert history[0]["critic_approved"] is True


def test_init_db_migrates_pre_existing_database_missing_new_columns():
    # Simulate a DB created before suggestion/critic_reason existed (the
    # real signal_history.db's actual starting state) by hand-creating the
    # old schema, then verify memory.init_db() adds the new columns
    # without touching existing data.
    conn = sqlite3.connect(memory.DB_PATH)
    conn.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            signal_count INTEGER NOT NULL,
            summary TEXT,
            confidence TEXT,
            watch_worthy INTEGER,
            critic_approved INTEGER,
            tool_calls TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO runs (run_date, ticker, signal_count, summary, created_at) VALUES (?, ?, ?, ?, ?)",
        ("2026-01-01", "AAPL", 1, "pre-migration row", "2026-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    memory.init_db()

    conn = sqlite3.connect(memory.DB_PATH)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    row = conn.execute("SELECT summary, suggestion, critic_reason, regenerated FROM runs WHERE ticker = 'AAPL'").fetchone()
    conn.close()

    assert {"suggestion", "critic_reason", "regenerated"} <= cols
    assert row[0] == "pre-migration row"  # existing data untouched
    assert row[1] is None
    assert row[2] is None
    assert row[3] is None


def test_days_cutoff_excludes_old_runs():
    report = {"signals": [], "llm": None}
    memory.save_run("AAPL", report, run_date="2020-01-01")  # far in the past
    history = memory.get_recent_history("AAPL", days=7, exclude_today=False)
    assert history == []


def test_get_earliest_run_date_returns_none_when_empty():
    assert memory.get_earliest_run_date() is None


def test_get_earliest_run_date_returns_min_date_across_tickers():
    report = {"signals": [], "llm": None}
    memory.save_run("AAPL", report, run_date=_days_ago(2))
    memory.save_run("TSLA", report, run_date=_days_ago(10))
    memory.save_run("AAPL", report, run_date=_days_ago(5))

    assert memory.get_earliest_run_date() == _days_ago(10)

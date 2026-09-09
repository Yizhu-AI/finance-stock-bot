"""
Tests for simulator.py — the paper-trading layer. Uses a temporary DB file
per test (same fixture pattern as test_memory.py, since simulator.py shares
memory.py's connection helper), so tests never touch the real
signal_history.db and each test starts from a clean slate.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import memory
import simulator

ALLOCATION = 1000.0


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    memory.DB_PATH = str(tmp_path / "test_history.db")
    yield


def test_get_position_starts_with_full_allocation():
    position = simulator.get_position("AAPL", ALLOCATION)
    assert position == {"cash": ALLOCATION, "shares": 0.0, "cost_basis": 0.0}


def test_buy_opens_position():
    trade = simulator.process_suggestion("AAPL", "buy", 100.0, ALLOCATION, run_date="2026-09-01")
    assert trade == {"action": "buy", "shares": 10.0, "cash_amount": ALLOCATION}

    position = simulator.get_position("AAPL", ALLOCATION)
    assert position["shares"] == 10.0
    assert position["cash"] == 0.0
    assert position["cost_basis"] == ALLOCATION


def test_buy_noop_when_already_holding():
    simulator.process_suggestion("AAPL", "buy", 100.0, ALLOCATION, run_date="2026-09-01")
    second = simulator.process_suggestion("AAPL", "buy", 120.0, ALLOCATION, run_date="2026-09-02")
    assert second is None

    position = simulator.get_position("AAPL", ALLOCATION)
    assert position["shares"] == 10.0  # unchanged by the no-op second buy


def test_sell_closes_position_with_realized_pnl():
    simulator.process_suggestion("AAPL", "buy", 100.0, ALLOCATION, run_date="2026-09-01")
    trade = simulator.process_suggestion("AAPL", "sell", 110.0, ALLOCATION, run_date="2026-09-05")

    assert trade["action"] == "sell"
    assert trade["shares"] == 10.0
    assert trade["cash_amount"] == 1100.0
    assert trade["realized_pnl"] == pytest.approx(100.0)

    position = simulator.get_position("AAPL", ALLOCATION)
    assert position == {"cash": 1100.0, "shares": 0.0, "cost_basis": 0.0}


def test_sell_noop_when_no_position():
    trade = simulator.process_suggestion("AAPL", "sell", 100.0, ALLOCATION, run_date="2026-09-01")
    assert trade is None
    assert simulator.get_position("AAPL", ALLOCATION)["cash"] == ALLOCATION


def test_hold_is_always_a_noop():
    assert simulator.process_suggestion("AAPL", "hold", 100.0, ALLOCATION) is None
    simulator.process_suggestion("AAPL", "buy", 100.0, ALLOCATION, run_date="2026-09-01")
    assert simulator.process_suggestion("AAPL", "hold", 105.0, ALLOCATION) is None


def test_process_suggestion_skips_on_missing_price():
    assert simulator.process_suggestion("AAPL", "buy", None, ALLOCATION) is None
    assert simulator.process_suggestion("AAPL", "buy", 0, ALLOCATION) is None
    assert simulator.get_position("AAPL", ALLOCATION)["shares"] == 0.0


def test_portfolio_summary_mixes_open_and_closed_positions():
    # AAPL: bought and sold at a profit (realized).
    simulator.process_suggestion("AAPL", "buy", 100.0, ALLOCATION, run_date="2026-09-01")
    simulator.process_suggestion("AAPL", "sell", 110.0, ALLOCATION, run_date="2026-09-03")
    # TSLA: still holding, marked to a current price for unrealized P&L.
    simulator.process_suggestion("TSLA", "buy", 200.0, ALLOCATION, run_date="2026-09-02")

    summary = simulator.portfolio_summary(
        ["AAPL", "TSLA"], ALLOCATION, current_prices={"AAPL": 110.0, "TSLA": 220.0},
    )

    assert summary["starting_capital"] == 2000.0
    assert summary["realized_pnl"] == pytest.approx(100.0)
    # AAPL cash (1100) + TSLA position marked at 220 (5 sh * 220 = 1100)
    assert summary["total_equity"] == pytest.approx(2200.0)
    assert summary["total_return_pct"] == pytest.approx(10.0)
    assert summary["open_positions"] == [{"ticker": "TSLA", "shares": 5.0, "unrealized_pnl": pytest.approx(100.0)}]


def test_tickers_have_independent_allocations():
    simulator.process_suggestion("AAPL", "buy", 100.0, ALLOCATION, run_date="2026-09-01")
    tsla_position = simulator.get_position("TSLA", ALLOCATION)
    assert tsla_position == {"cash": ALLOCATION, "shares": 0.0, "cost_basis": 0.0}

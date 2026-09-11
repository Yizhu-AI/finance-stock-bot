"""
Tests for agent_tools.ToolCallBudget — the only pure, self-contained logic
in agent_tools.py. The actual tool functions (get_extended_price_history,
etc.) need live data_fetch/memory calls, so like the rest of the LLM layer
they're exercised by actually running the bot rather than by pytest.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent_tools


def test_try_consume_succeeds_within_limit():
    budget = agent_tools.ToolCallBudget(3)
    assert budget.try_consume() is True
    assert budget.try_consume() is True
    assert budget.try_consume() is True
    assert budget.used == 3


def test_try_consume_refuses_once_limit_reached():
    budget = agent_tools.ToolCallBudget(1)
    assert budget.try_consume() is True
    assert budget.try_consume() is False
    assert budget.try_consume() is False  # stays refused, doesn't go negative
    assert budget.used == 1


def test_zero_limit_refuses_immediately():
    budget = agent_tools.ToolCallBudget(0)
    assert budget.try_consume() is False
    assert budget.used == 0


def test_warns_only_once_when_exhausted(capsys):
    budget = agent_tools.ToolCallBudget(1)
    budget.try_consume()  # succeeds, no warning
    budget.try_consume()  # first refusal — warns
    budget.try_consume()  # second refusal — no repeat warning

    out = capsys.readouterr().out
    assert out.count("Tool call budget") == 1


def test_budget_is_independent_per_instance():
    # Confirms it's plain instance state, not accidentally shared/global —
    # important since main.py creates exactly one per run and threads it
    # through every ticker.
    first = agent_tools.ToolCallBudget(1)
    second = agent_tools.ToolCallBudget(1)
    first.try_consume()
    assert first.used == 1
    assert second.used == 0
    assert second.try_consume() is True

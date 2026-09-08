"""
Tests for critic.py — the safety review layer. LLM-dependent behavior is
tested with the client disabled (rule-based check is the only gate in that
configuration), so these run without needing a live API key.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import critic


def setup_function(_):
    # Ensure every test runs with no live client, regardless of environment
    # .env — these tests exercise the deterministic + graceful-degradation
    # paths, not the live LLM call.
    critic._client = None


def test_rule_based_check_catches_buy_language():
    assert critic._rule_based_check("You should buy this stock now.") is False


def test_rule_based_check_catches_sell_language():
    assert critic._rule_based_check("Time to sell before it drops further.") is False


def test_rule_based_check_passes_neutral_summary():
    assert critic._rule_based_check("AAPL shows a bullish crossover with elevated volume.") is True


def test_rule_based_check_case_insensitive():
    assert critic._rule_based_check("STRONG BUY signal detected today.") is False


def test_review_passes_clean_draft_unchanged():
    draft = {"summary": "AAPL shows mixed signals with no clear direction.", "confidence": "low", "watch_worthy": False}
    result = critic.review("AAPL", [], [], draft)
    assert result["critic_approved"] is True
    assert result["summary"] == draft["summary"]


def test_review_replaces_rejected_draft_with_fallback():
    draft = {"summary": "Strong buy — load up on this stock now.", "confidence": "high", "watch_worthy": True}
    result = critic.review("AAPL", [], [], draft)
    assert result["critic_approved"] is False
    assert result["summary"] == critic._FALLBACK_SUMMARY


def test_review_preserves_watch_worthy_flag_on_rejection():
    # Even when rejected, watch_worthy should carry through so the ticker
    # can still surface in the digest (with the raw signals, sans the
    # untrustworthy summary) rather than being silently dropped.
    draft = {"summary": "You should sell immediately.", "confidence": "high", "watch_worthy": True}
    result = critic.review("AAPL", [], [], draft)
    assert result["watch_worthy"] is True


def test_review_handles_empty_draft_gracefully():
    assert critic.review("AAPL", [], [], {}) == {}
    assert critic.review("AAPL", [], [], None) is None


def test_llm_check_defaults_to_approved_when_unconfigured():
    approved, reason = critic._llm_check("AAPL", [], [], {"summary": "test", "confidence": "low"})
    assert approved is True
    assert "no critic configured" in reason

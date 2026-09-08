"""
A second, independent pass that checks the primary agent's output before
it's allowed to reach the digest. This is the generator/critic pattern:
one model (or call) produces a draft, a separate call evaluates it against
explicit constraints, and only an approved (possibly revised) version ships.

Two layers of defense, deliberately redundant:
  1. A cheap, deterministic rule-based scan for banned phrases (buy/sell/
     hold language) — catches obvious violations with zero LLM cost and
     zero chance of the critic itself being wrong.
  2. An LLM critic call that checks subtler issues: does the summary
     actually follow from the given evidence, is the confidence level
     plausible given how strong/weak the signals are, is there anything
     that reads like disguised investment advice.

If either layer fails the draft, we don't try to be clever about partial
fixes — we replace the summary with a safe, generic fallback and mark it
as flagged, since a failed safety check on autonomous, unreviewed output
should fail closed, not attempt a repair that might itself be wrong.
"""
import json
from google import genai
from google.genai import types

import config

_client = genai.Client(api_key=config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None

_BANNED_PHRASES = [
    "you should buy", "you should sell", "buy now", "sell now", "strong buy",
    "strong sell", "i recommend buying", "i recommend selling", "time to buy",
    "time to sell", "load up on", "get out of", "invest now", "don't invest",
]

_CRITIC_SYSTEM_PROMPT = """You are a strict compliance reviewer for an automated \
financial signal digest. You will be shown the raw evidence (technical signals and \
headlines) a junior analyst model was given, and the summary it produced. Your job \
is to check that summary against three rules:

1. It must not tell the reader to buy, sell, or hold — directly or by clear implication.
2. It must not state any fact, number, or event not present in the given evidence.
3. Its stated confidence level must be plausible given how strong or weak/contradictory \
the evidence actually is — flag it if the summary sounds far more certain than the \
evidence supports.

Respond with ONLY a JSON object, no other text:
{"approved": <true|false>, "reason": "<one sentence explaining your verdict>"}"""

_FALLBACK_SUMMARY = "The automated analysis for this ticker did not pass safety review and has been withheld. Raw signals are shown above."


def _rule_based_check(summary: str) -> bool:
    """Returns True if the summary passes (no banned phrases found)."""
    lowered = summary.lower()
    return not any(phrase in lowered for phrase in _BANNED_PHRASES)


def _llm_check(ticker: str, signals: list, headlines: list, draft: dict) -> bool:
    """Returns True if the LLM critic approves the draft. Fails closed (False) on any error."""
    if _client is None:
        return True  # no critic configured — rule-based check is the only gate

    signal_lines = "\n".join(f"- {name} ({direction}): {explanation}" for name, direction, explanation in signals) or "None"
    headline_lines = "\n".join(f"- {h['headline']}" for h in headlines[:5]) or "None"

    user_prompt = f"""Ticker: {ticker}

Evidence given to the analyst:
Signals:
{signal_lines}

Headlines:
{headline_lines}

Analyst's summary to review:
"{draft.get('summary', '')}"
Stated confidence: {draft.get('confidence', 'unknown')}"""

    try:
        response = _client.models.generate_content(
            model=config.LLM_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_CRITIC_SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=150,
            ),
        )
        result = json.loads(response.text)
        return bool(result.get("approved", False))
    except Exception as e:
        print(f"[{ticker}] Critic check failed to run: {e} — failing closed")
        return False


def review(ticker: str, signals: list, headlines: list, draft: dict) -> dict:
    """
    Takes the primary agent's draft result and returns a final version safe
    to ship — either the original (if it passes both checks) or a sanitized
    fallback (if either check fails).
    """
    if not draft or not draft.get("summary"):
        return draft

    passed_rules = _rule_based_check(draft["summary"])
    passed_llm = _llm_check(ticker, signals, headlines, draft) if passed_rules else False

    if passed_rules and passed_llm:
        draft["critic_approved"] = True
        return draft

    print(f"[{ticker}] Critic rejected draft summary (rules={passed_rules}, llm={passed_llm}) — using fallback")
    return {
        "summary": _FALLBACK_SUMMARY,
        "confidence": "low",
        "watch_worthy": draft.get("watch_worthy", False),
        "critic_approved": False,
    }

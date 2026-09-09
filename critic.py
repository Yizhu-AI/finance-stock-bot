"""
A second, independent pass that checks the primary agent's output before
it's allowed to reach the digest. This is the generator/critic pattern:
one model (or call) produces a draft, a separate call evaluates it against
explicit constraints, and only an approved (possibly revised) version ships.

Two layers of defense, deliberately redundant:
  1. A cheap, deterministic rule-based scan of the free-text `summary` for
     banned phrases (direct imperative buy/sell/hold language addressed at
     the reader) — catches obvious violations with zero LLM cost and zero
     chance of the critic itself being wrong. The structured `suggestion`
     field (buy/sell/hold, for simulator.py) is exempt from this scan by
     design — it's expected to say exactly that.
  2. An LLM critic call that checks subtler issues: does the summary
     actually follow from the given evidence, is the confidence level
     plausible given how strong/weak the signals are, and is the structured
     `suggestion` actually consistent with the evidence rather than
     contradicted by it.

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
financial signal digest that also drives a paper-trading simulation. You will be shown \
the raw evidence (technical signals and headlines) a junior analyst model was given, \
plus the summary and structured buy/sell/hold suggestion it produced. Your job is to \
check the draft against four rules:

1. The free-text "summary" must not directly address the reader with buy/sell/hold \
language ("you should...", "time to sell...") — it should read as analysis, not a \
directive. The "suggestion" field is allowed and expected to state buy/sell/hold.
2. It must not state any fact, number, or event not present in the given evidence.
3. Its stated confidence level must be plausible given how strong or weak/contradictory \
the evidence actually is — flag it if the summary sounds far more certain than the \
evidence supports.
4. The "suggestion" must be a reasonable read of the given evidence — flag it if it's \
clearly contradicted by the signals (e.g. "buy" on evidence that's uniformly bearish).

Respond with ONLY a JSON object, no other text:
{"approved": <true|false>, "reason": "<one sentence explaining your verdict>"}"""

_FALLBACK_SUMMARY = "The automated analysis for this ticker did not pass safety review and has been withheld. Raw signals are shown above."
_VALID_SUGGESTIONS = {"buy", "sell", "hold"}


def _rule_based_check(summary: str) -> bool:
    """Returns True if the summary passes (no banned phrases found)."""
    lowered = summary.lower()
    return not any(phrase in lowered for phrase in _BANNED_PHRASES)


def _llm_check(ticker: str, signals: list, headlines: list, draft: dict) -> tuple:
    """Returns (approved: bool, reason: str). Fails closed (False) on any error."""
    if _client is None:
        return True, "no critic configured — rule-based check is the only gate"

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
Stated confidence: {draft.get('confidence', 'unknown')}
Stated suggestion: {draft.get('suggestion', 'unknown')}"""

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
        return bool(result.get("approved", False)), result.get("reason", "no reason given")
    except Exception as e:
        return False, f"critic check failed to run: {e}"


def review(ticker: str, signals: list, headlines: list, draft: dict) -> dict:
    """
    Takes the primary agent's draft result and returns a final version safe
    to ship — either the original (if it passes both checks) or a sanitized
    fallback (if either check fails).
    """
    if not draft or not draft.get("summary"):
        return draft

    passed_rules = _rule_based_check(draft["summary"])
    if not passed_rules:
        passed_llm, reason = False, "rejected by rule-based banned-phrase scan"
    elif draft.get("suggestion") not in _VALID_SUGGESTIONS:
        # Malformed/missing suggestion from the model — fail closed rather
        # than guess, same as any other unreviewable draft.
        passed_llm, reason = False, f"invalid or missing suggestion: {draft.get('suggestion')!r}"
    else:
        passed_llm, reason = _llm_check(ticker, signals, headlines, draft)

    if passed_rules and passed_llm:
        draft["critic_approved"] = True
        draft["critic_reason"] = reason
        return draft

    print(f"[{ticker}] Critic rejected draft summary — {reason}")
    return {
        "summary": _FALLBACK_SUMMARY,
        "confidence": "low",
        "watch_worthy": draft.get("watch_worthy", False),
        # Fail closed: an unreviewable draft's suggestion isn't trusted
        # either, so the simulator gets a safe no-op instead of a trade.
        "suggestion": "hold",
        "critic_approved": False,
        "critic_reason": reason,
    }

"""
The 'agentic' layer: instead of a fixed rule combining technical + sentiment
signals, this hands the raw evidence to Gemini and asks it to reason about
what's actually notable and why, and how confident that read is.

Two things make this genuinely agentic rather than a single scripted call:
  1. Tool use (agent_tools.py) — the model can autonomously decide to pull
     more price history or more headlines mid-reasoning if it judges the
     initial evidence too thin, rather than always working from a fixed
     pre-fetched bundle.
  2. A critic pass (critic.py) — the draft this function produces is not
     shipped directly; it's independently reviewed against explicit safety
     constraints before main.py ever sees it.

In addition to its analysis, the model is asked for a structured buy/sell/hold
`suggestion`, strictly derived from the given evidence. This exists to drive
`simulator.py`'s paper-trading simulation — a bookkeeping exercise against
fake money, not investment advice to a human reader. The free-text `summary`
is still kept analytical (no direct "you should buy" language) so the
narrative and the mechanical suggestion stay clearly separated; critic.py
checks both.
"""
import json
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import config
import agent_tools
import critic

_client = genai.Client(api_key=config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None

# Gemini's own status codes for "try again, it's transient" — worth a couple
# quick retries rather than giving up on the whole ticker immediately.
_RETRYABLE_STATUS_CODES = {429, 500, 503}
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2

# Bounds how many autonomous tool calls the model can make per ticker per
# synthesis — an agent that can call tools needs an explicit ceiling, or a
# single ambiguous ticker could spiral into unbounded cost/latency.
_MAX_TOOL_CALLS = 4

_SYSTEM_PROMPT = """You are a financial signal analyst. You are given raw technical \
and news signals for a stock ticker, computed by a rules-based pipeline. Your job is \
to synthesize them into a short, plain-English read of what's going on and why it \
might matter to someone watching this stock.

You have three tools available: get_extended_price_history, get_extended_headlines, \
and get_recent_history (this ticker's own analysis from the last 7 days). Use them \
ONLY when genuinely useful:
- get_extended_price_history / get_extended_headlines: when the initial evidence is \
too thin or ambiguous to reach a confident read.
- get_recent_history: only when today's evidence gives you a specific reason to check \
whether this is a continuation of something already flagged recently, not as a routine \
check on every ticker.
Do not call a tool if the initial evidence is already sufficient to answer well; \
calling tools has a real cost and unnecessary calls should be avoided.

Rules you must follow:
- Never invent facts, numbers, or news not present in the input you're given or returned \
by a tool call.
- If the signals are weak, contradictory, or thin — even after using a tool — say so \
plainly rather than manufacturing a confident narrative.
- Be concise: 2-4 sentences.
- In the "summary" field specifically, stay analytical — describe what the evidence shows, \
don't address the reader directly or use imperative language like "you should buy/sell". \
The "suggestion" field below is the only place the buy/sell/hold call belongs.

Once you are done (whether or not you used any tools), respond with ONLY a JSON object, \
no other text, no markdown code fences, in this exact shape:
{"summary": "<2-4 sentence plain-English synthesis>", "confidence": "<low|medium|high>", \
"watch_worthy": <true|false>, "suggestion": "<buy|sell|hold>"}

"confidence" reflects how much the raw signals agree with each other and how \
strong they are individually.
"watch_worthy" should be true only if this ticker seems meaningfully more \
interesting today than an average day, based solely on the given signals.
"suggestion" is a mechanical call for a paper-trading simulation, strictly derived \
from the given signals: "buy" if the evidence leans clearly bullish, "sell" if it \
leans clearly bearish, "hold" if it's weak, mixed, or contradictory. Default to \
"hold" whenever you're not confident either direction is clearly supported."""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    # The model is instructed to avoid fences, but strip them defensively
    # in case it adds them anyway.
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return json.loads(text)


def synthesize(ticker: str, signals: list, headlines: list):
    """
    signals: list of (name, direction, explanation) tuples from analysis.py
    headlines: list of {"headline": str, "source": str}
    Returns a dict {"summary": str, "confidence": str, "watch_worthy": bool,
    "suggestion": "buy"|"sell"|"hold", "critic_approved": bool,
    "tool_calls": list} or None if the LLM layer isn't configured / the call
    fails (pipeline still works without it).
    """
    if _client is None:
        return None
    if not signals and not headlines:
        return None  # nothing for the model to reason about

    signal_lines = "\n".join(f"- {name} ({direction}): {explanation}" for name, direction, explanation in signals) or "None"
    headline_lines = "\n".join(f"- {h['headline']} ({h['source']})" for h in headlines[:5]) or "None"

    user_prompt = f"""Ticker: {ticker}

Computed signals:
{signal_lines}

Recent headlines:
{headline_lines}"""

    tool_call_log = []  # populated by agent_tools if the model chooses to call them
    tools = agent_tools.make_tools(ticker, tool_call_log)

    draft = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = _client.models.generate_content(
                model=config.LLM_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    tools=tools,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=_MAX_TOOL_CALLS,
                    ),
                    max_output_tokens=600,
                ),
            )
            draft = _parse_json_response(response.text)
            break
        except genai_errors.APIError as e:
            status_code = getattr(e, "code", None)
            if status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                wait = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"[{ticker}] Gemini {status_code} (attempt {attempt}/{_MAX_RETRIES}) — retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"[{ticker}] LLM synthesis failed: {e}")
            return None
        except Exception as e:
            print(f"[{ticker}] LLM synthesis failed: {e}")
            return None

    if draft is None:
        return None

    if tool_call_log:
        print(f"[{ticker}] Agent used {len(tool_call_log)} tool call(s): {tool_call_log}")

    final = critic.review(ticker, signals, headlines, draft)
    final["tool_calls"] = tool_call_log
    return final

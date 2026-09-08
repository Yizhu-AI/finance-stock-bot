"""
The 'agentic' layer: instead of a fixed rule combining technical + sentiment
signals, this hands the raw evidence to Gemini and asks it to reason about
what's actually notable and why, and how confident that read is.

This does NOT ask the model for a buy/sell recommendation — it asks for
analysis and a confidence-in-the-signal rating, which is a materially
different (and more honest) task than "tell me what to trade."
"""
import json
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import config

_client = genai.Client(api_key=config.GEMINI_API_KEY) if config.GEMINI_API_KEY else None

# Gemini's own status codes for "try again, it's transient" — worth a couple
# quick retries rather than giving up on the whole ticker immediately.
_RETRYABLE_STATUS_CODES = {429, 500, 503}
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2

_SYSTEM_PROMPT = """You are a financial signal analyst. You are given raw technical \
and news signals for a stock ticker, computed by a rules-based pipeline. Your job is \
to synthesize them into a short, plain-English read of what's going on and why it \
might matter to someone watching this stock.

Rules you must follow:
- You are NOT a financial advisor. Never tell the reader to buy, sell, or hold.
- Never invent facts, numbers, or news not present in the input you're given.
- If the signals are weak, contradictory, or thin, say so plainly rather than \
manufacturing a confident narrative.
- Be concise: 2-4 sentences.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"summary": "<2-4 sentence plain-English synthesis>", "confidence": "<low|medium|high>", "watch_worthy": <true|false>}

"confidence" reflects how much the raw signals agree with each other and how \
strong they are individually — not how strongly you'd act on them.
"watch_worthy" should be true only if this ticker seems meaningfully more \
interesting today than an average day, based solely on the given signals."""


def synthesize(ticker: str, signals: list, headlines: list):
    """
    signals: list of (name, direction, explanation) tuples from analysis.py
    headlines: list of {"headline": str, "source": str}
    Returns a dict {"summary": str, "confidence": str, "watch_worthy": bool} or None
    if the LLM layer isn't configured / the call fails (pipeline still works without it).
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

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = _client.models.generate_content(
                model=config.LLM_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    max_output_tokens=300,
                ),
            )
            # response_mime_type="application/json" makes Gemini return valid JSON
            # directly in response.text — no markdown-fence stripping needed.
            return json.loads(response.text)
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
            # Non-API errors (bad JSON from the model, network issues, etc.)
            # aren't worth retrying — fail this ticker's synthesis and move on.
            print(f"[{ticker}] LLM synthesis failed: {e}")
            return None

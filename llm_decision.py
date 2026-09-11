"""
The 'agentic' layer: instead of a fixed rule combining technical + sentiment
signals, this hands the raw evidence to Gemini and asks it to reason about
what's actually notable and why, and how confident that read is.

This is a multi-agent pipeline, not a single scripted call — a technical
analyst and a news analyst each reason independently over their own domain
(neither sees the other's evidence), and a coordinator reconciles their two
reads into one final judgment, paying explicit attention to whether the
specialists agree or disagree. This mirrors how a real research desk
would work: a chartist and a news analyst reach their own conclusions
first, and disagreement between them is itself a meaningful signal, not
just noise to average away — which one scripted call blending all the
evidence together can't represent (it just sees mixed evidence, not two
independent, possibly conflicting conclusions). If only one domain has
evidence for a ticker (e.g. no headlines available), the coordinator is
skipped entirely — reconciliation needs two things to reconcile.

Two things make each specialist genuinely agentic rather than a scripted
call:
  1. Tool use (agent_tools.py) — each specialist can autonomously decide to
     pull more evidence mid-reasoning if it judges its own initial evidence
     too thin, rather than always working from a fixed pre-fetched bundle.
     Bundled per domain: the technical analyst only gets price-history
     tools, the news analyst only gets headline tools. Bounded two ways:
     _MAX_TOOL_CALLS caps one specialist's own calls within one ticker, and
     the tool_budget (agent_tools.ToolCallBudget) threaded in from main.py
     caps the total across the entire run — several genuinely ambiguous
     tickers could each hit their own per-specialist ceiling and still
     compound into real, uncapped aggregate cost without the second bound.
  2. A critic pass (critic.py) — the final draft (from the coordinator, or
     directly from a lone specialist) is not shipped directly; it's
     independently reviewed against explicit safety constraints before
     main.py ever sees it. A rejection isn't necessarily final: the
     critic's specific reason is fed back to whichever component produced
     the draft for exactly one revision attempt (_MAX_REGENERATION_ATTEMPTS)
     before falling back to the generic safe message — a genuine
     generator/critic loop rather than reject-and-discard. Only that one
     retry is allowed, not an open-ended loop, for the same reason
     _MAX_TOOL_CALLS bounds tool use: an ambiguous ticker shouldn't be able
     to spiral into unbounded cost or latency chasing critic approval.

In addition to its analysis, the pipeline produces a structured buy/sell/hold
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

# Bounds how many autonomous tool calls EACH specialist can make per ticker
# — an agent that can call tools needs an explicit ceiling, or a single
# ambiguous ticker could spiral into unbounded cost/latency. Technical and
# news each get their own budget of up to this many.
_MAX_TOOL_CALLS = 4

# How many times a critic-rejected draft gets fed back for revision before
# falling back to the generic safe message. One, matching the "generator/
# critic loop" being a bounded two-pass process, not an open-ended retry
# until approval — a rejection is a legitimate outcome, not a bug to
# route around indefinitely.
_MAX_REGENERATION_ATTEMPTS = 1

_LEAN_TO_SUGGESTION = {"bullish": "buy", "bearish": "sell", "neutral": "hold"}

_TECHNICAL_SYSTEM_PROMPT = """You are the technical analyst for a stock signal bot. \
You are given ONLY the computed technical signals for a ticker (moving averages, \
RSI, MACD, Bollinger Bands, volume) — no news or headlines. Your job is to read what \
the technical picture says on its own terms, independent of sentiment or news.

You have two tools available: get_extended_price_history (6-month price context) and \
get_recent_history (this ticker's own analysis from the last 7 days). Use them ONLY \
when genuinely useful — when the initial signals are too thin, ambiguous, or \
contradictory to read confidently, or today's evidence gives you a specific reason to \
check for continuity with a past judgment. Calling a tool has a real cost; don't call \
one just because it's available.

Rules you must follow:
- Never invent facts or numbers not present in the input you're given or returned by \
a tool call.
- If the signals are weak, contradictory, or thin — even after using a tool — say so \
plainly rather than manufacturing a confident narrative.
- Be concise: 1-3 sentences.

Respond with ONLY a JSON object, no other text, no markdown code fences:
{"read": "<1-3 sentence technical read>", "lean": "<bullish|bearish|neutral>", \
"confidence": "<low|medium|high>"}

"lean" is your directional read strictly from the technical evidence itself.
"confidence" reflects how much the technical signals agree with each other and how \
strong they are individually."""

_NEWS_SYSTEM_PROMPT = """You are the news analyst for a stock signal bot. You are \
given ONLY recent headlines for a ticker — no technical/price signals. Your job is to \
read what the news/sentiment picture says on its own terms, independent of price action.

You have two tools available: get_extended_headlines (a 14-day headline lookback \
instead of 3) and get_recent_history (this ticker's own analysis from the last 7 \
days). Use them ONLY when genuinely useful — when the initial headlines are sparse or \
seem insufficient to judge notability, or today's evidence gives you a specific reason \
to check for continuity with a past judgment. Calling a tool has a real cost; don't \
call one just because it's available.

Rules you must follow:
- Never invent facts, numbers, or events not present in the input you're given or \
returned by a tool call.
- If the headlines are sparse, generic, or don't clearly bear on the stock, say so \
plainly rather than manufacturing a confident narrative.
- Be concise: 1-3 sentences.

Respond with ONLY a JSON object, no other text, no markdown code fences:
{"read": "<1-3 sentence news read>", "lean": "<bullish|bearish|neutral>", \
"confidence": "<low|medium|high>"}

"lean" is your directional read strictly from the news evidence itself.
"confidence" reflects how one-sided and substantive the headlines are — a single \
ambiguous headline is low confidence even if it's the only evidence you have."""

_COORDINATOR_SYSTEM_PROMPT = """You are the coordinator for a stock signal bot's two \
specialist analysts — a technical analyst (reasoning purely from price/volume \
signals) and a news analyst (reasoning purely from headlines). You are given both \
specialists' reads, directional leans, and confidence levels. Your job is to \
reconcile them into one final judgment, paying particular attention to whether they \
agree or disagree.

Rules you must follow:
- If both specialists lean the same direction, say so, and let that agreement support \
a correspondingly higher (but still honest) confidence.
- If they disagree, say so explicitly in your summary — do not silently pick a side \
without acknowledging the conflict. Default toward a hold-leaning suggestion when the \
specialists genuinely conflict, unless one side's case is clearly and substantially \
stronger than the other's.
- Never invent facts beyond what the two specialists reported to you.
- In the "summary" field specifically, stay analytical — describe what the two reads \
show, don't address the reader directly or use imperative language like "you should \
buy/sell". The "suggestion" field below is the only place the buy/sell/hold call belongs.
- Be concise: 2-4 sentences.

Respond with ONLY a JSON object, no other text, no markdown code fences, in this exact \
shape:
{"summary": "<2-4 sentence plain-English synthesis>", "confidence": "<low|medium|high>", \
"watch_worthy": <true|false>, "suggestion": "<buy|sell|hold>"}

"watch_worthy" should be true only if this ticker seems meaningfully more interesting \
today than an average day, based on the two specialists' reads.
"suggestion" is a mechanical call for a paper-trading simulation, strictly derived \
from the two specialists' reads: "buy" if they clearly (or on balance) lean bullish, \
"sell" if bearish, "hold" if they're weak, mixed, or in conflict. Default to "hold" \
whenever you're not confident either direction is clearly supported."""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    # The model is instructed to avoid fences, but strip them defensively
    # in case it adds them anyway.
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return json.loads(text)


def _call_gemini(ticker: str, system_prompt: str, user_prompt: str, tools=None, max_output_tokens: int = 300) -> dict:
    """Shared retry/parse logic for one agent call (specialist or
    coordinator). Returns the parsed JSON dict, or None on failure."""
    generate_config_kwargs = dict(system_instruction=system_prompt, max_output_tokens=max_output_tokens)
    if tools:
        generate_config_kwargs["tools"] = tools
        generate_config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=_MAX_TOOL_CALLS,
        )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = _client.models.generate_content(
                model=config.LLM_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(**generate_config_kwargs),
            )
            return _parse_json_response(response.text)
        except genai_errors.APIError as e:
            status_code = getattr(e, "code", None)
            if status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                wait = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"[{ticker}] Gemini {status_code} (attempt {attempt}/{_MAX_RETRIES}) — retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"[{ticker}] LLM call failed: {e}")
            return None
        except Exception as e:
            print(f"[{ticker}] LLM call failed: {e}")
            return None
    return None


def _revision_note(critic_reason: str) -> str:
    """Appended to a regeneration attempt's prompt — shared wording works
    for both a specialist (revising its read) and the coordinator (revising
    its reconciled judgment)."""
    return (f'Your previous answer was rejected by compliance review for this reason: '
            f'"{critic_reason}"\n\nRevise your answer to address this specific concern, '
            f'while still following all the rules and the JSON format above.')


def _run_technical_analyst(ticker: str, signals: list, call_log: list, tool_budget, revision_note: str = None) -> dict:
    signal_lines = "\n".join(f"- {name} ({direction}): {explanation}" for name, direction, explanation in signals)
    user_prompt = f"Ticker: {ticker}\n\nComputed technical signals:\n{signal_lines}"
    if revision_note:
        user_prompt += f"\n\n{revision_note}"
    tools = agent_tools.make_technical_tools(ticker, call_log, tool_budget)
    return _call_gemini(ticker, _TECHNICAL_SYSTEM_PROMPT, user_prompt, tools=tools, max_output_tokens=300)


def _run_news_analyst(ticker: str, headlines: list, call_log: list, tool_budget, revision_note: str = None) -> dict:
    headline_lines = "\n".join(f"- {h['headline']} ({h['source']})" for h in headlines[:5])
    user_prompt = f"Ticker: {ticker}\n\nRecent headlines:\n{headline_lines}"
    if revision_note:
        user_prompt += f"\n\n{revision_note}"
    tools = agent_tools.make_news_tools(ticker, call_log, tool_budget)
    return _call_gemini(ticker, _NEWS_SYSTEM_PROMPT, user_prompt, tools=tools, max_output_tokens=300)


def _run_coordinator(ticker: str, technical: dict, news: dict, revision_note: str = None) -> dict:
    user_prompt = f"""Ticker: {ticker}

Technical analyst:
- Read: {technical.get('read', '')}
- Lean: {technical.get('lean', 'unknown')}
- Confidence: {technical.get('confidence', 'unknown')}

News analyst:
- Read: {news.get('read', '')}
- Lean: {news.get('lean', 'unknown')}
- Confidence: {news.get('confidence', 'unknown')}"""
    if revision_note:
        user_prompt += f"\n\n{revision_note}"
    return _call_gemini(ticker, _COORDINATOR_SYSTEM_PROMPT, user_prompt, max_output_tokens=500)


def _solo_draft(specialist_result: dict, domain: str) -> dict:
    """Used when only one specialist could run (the other domain had no
    evidence at all) — skips the coordinator, since reconciling requires
    two reads and there's only one. The mapping from lean to
    summary/suggestion is deliberately mechanical rather than another LLM
    call: with nothing to reconcile, a coordinator call would only be
    paraphrasing the specialist, not adding judgment."""
    lean = specialist_result.get("lean", "neutral")
    confidence = specialist_result.get("confidence", "low")
    domain_label = "Technical" if domain == "technical" else "News"
    return {
        "summary": f"{domain_label} read only (no {'news' if domain == 'technical' else 'technical'} "
                   f"evidence available): {specialist_result.get('read', '')}",
        "confidence": confidence,
        "watch_worthy": lean != "neutral",
        "suggestion": _LEAN_TO_SUGGESTION.get(lean, "hold"),
    }


def synthesize(ticker: str, signals: list, headlines: list, tool_budget):
    """
    signals: list of (name, direction, explanation) tuples from analysis.py
    headlines: list of {"headline": str, "source": str}
    tool_budget: an agent_tools.ToolCallBudget shared across the whole
    main.py run (every ticker) — caps total tool calls in aggregate, not
    just per-ticker-per-specialist (see _MAX_TOOL_CALLS).
    Returns a dict {"summary": str, "confidence": str, "watch_worthy": bool,
    "suggestion": "buy"|"sell"|"hold", "critic_approved": bool,
    "tool_calls": list, "technical_lean"/"technical_confidence": str|None,
    "news_lean"/"news_confidence": str|None, "regenerated": bool} or None if
    the LLM layer isn't configured / no evidence exists / every call that
    ran failed (pipeline still works without it). "regenerated" is True iff
    a critic rejection triggered the one-revision generator/critic loop
    (regardless of whether the revision was ultimately approved).
    """
    if _client is None:
        return None
    if not signals and not headlines:
        return None  # nothing for either specialist to reason about

    tool_call_log = []  # shared across specialists — populated by agent_tools

    technical = _run_technical_analyst(ticker, signals, tool_call_log, tool_budget) if signals else None
    if headlines:
        if technical is not None:
            time.sleep(config.INTRA_TICKER_REQUEST_DELAY_SECONDS)
        news = _run_news_analyst(ticker, headlines, tool_call_log, tool_budget)
    else:
        news = None

    if technical and news:
        time.sleep(config.INTRA_TICKER_REQUEST_DELAY_SECONDS)
        draft = _run_coordinator(ticker, technical, news)
        source = "coordinator"
    elif technical:
        draft = _solo_draft(technical, "technical")
        source = "technical"
    elif news:
        draft = _solo_draft(news, "news")
        source = "news"
    else:
        draft = None  # every call that had evidence to work with still failed
        source = None

    if draft is None:
        return None

    if tool_call_log:
        print(f"[{ticker}] Agent used {len(tool_call_log)} tool call(s): {tool_call_log}")

    time.sleep(config.INTRA_TICKER_REQUEST_DELAY_SECONDS)
    final = critic.review(ticker, signals, headlines, draft)
    regenerated = False

    # Generator/critic loop: a rejection isn't necessarily final. Feed the
    # critic's specific reason back to whichever component produced the
    # draft for one revision attempt before accepting the fallback.
    for _ in range(_MAX_REGENERATION_ATTEMPTS):
        if final.get("critic_approved") is not False:
            break  # approved (or no verdict at all, e.g. an empty draft) — nothing to revise

        reason = final.get("critic_reason", "no reason given")
        print(f"[{ticker}] Critic rejected — attempting one revision. Reason: {reason}")
        note = _revision_note(reason)
        time.sleep(config.INTRA_TICKER_REQUEST_DELAY_SECONDS)

        revised = None
        if source == "coordinator":
            revised = _run_coordinator(ticker, technical, news, revision_note=note)
        elif source == "technical":
            revised_specialist = _run_technical_analyst(ticker, signals, tool_call_log, tool_budget, revision_note=note)
            if revised_specialist is not None:
                technical = revised_specialist  # reflect the read actually used in the digest
                revised = _solo_draft(technical, "technical")
        elif source == "news":
            revised_specialist = _run_news_analyst(ticker, headlines, tool_call_log, tool_budget, revision_note=note)
            if revised_specialist is not None:
                news = revised_specialist
                revised = _solo_draft(news, "news")

        if revised is None:
            break  # the revision call itself failed — nothing new to review, don't waste a critic call re-checking the same draft

        regenerated = True
        time.sleep(config.INTRA_TICKER_REQUEST_DELAY_SECONDS)
        final = critic.review(ticker, signals, headlines, revised)

    final["tool_calls"] = tool_call_log
    final["technical_lean"] = technical.get("lean") if technical else None
    final["technical_confidence"] = technical.get("confidence") if technical else None
    final["news_lean"] = news.get("lean") if news else None
    final["news_confidence"] = news.get("confidence") if news else None
    final["regenerated"] = regenerated
    return final

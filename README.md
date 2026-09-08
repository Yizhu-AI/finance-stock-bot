# Stock Signal Bot

An autonomous pipeline that watches a list of tickers, computes technical +
news signals, uses an LLM agent (with its own tools and a safety-review
layer) to reason about what's notable, and sends you a digest over
Telegram. This is a **research/signal tool**, not investment advice —
treat its output as one input among many.

## How it works

```
data_fetch.py     -> pulls price history (yfinance) + news (Finnhub)
analysis.py       -> turns raw data into discrete signals (SMA crossover, RSI, volume spike, sentiment)
memory.py         -> persists each run to SQLite, and serves the agent's own history back to it
agent_tools.py    -> tools the agent can call on its own: more price history, more headlines, its own past runs
llm_decision.py   -> hands signals+headlines to Gemini, which reasons (optionally using tools) to a judgment
critic.py         -> independently reviews that judgment before it's allowed to ship
main.py           -> orchestrates all of the above into a digest, saves the run, only surfaces notable tickers
notifier.py       -> sends the digest to your Telegram (one or more recipients)
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a Finnhub API key (free)
- Sign up at https://finnhub.io/register
- Free tier includes company news headlines, which is enough for this project.
- **Correction:** Finnhub's dedicated `news-sentiment` endpoint (bullish/bearish %) is
  actually **premium-only**, not free as originally stated here. The bot handles this
  automatically — when that endpoint returns a 403, it falls back to a basic free
  keyword scan over recent headlines instead (see "About the sentiment signal" below).

### 2b. Get a Gemini API key (for the LLM synthesis layer)
- Sign up / log in at https://aistudio.google.com, then create an API key
  (API keys section in the left sidebar).
- This powers the reasoning layer described below — the bot works without it too
  (it'll just skip the LLM synthesis and show raw signals only), but you lose the
  "why this matters" narrative.

### 3. Create a Telegram bot
1. Open Telegram, message **@BotFather**, run `/newbot`, follow the prompts.
2. BotFather gives you a token — save it.
3. Send your new bot any message (e.g. "hi") so it has a chat to reply to.
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser —
   find `"chat":{"id": ...}` in the JSON. That number is your `TELEGRAM_CHAT_ID`.
5. **Sending to more than one person:** `TELEGRAM_CHAT_ID` accepts a comma-separated
   list, e.g. `TELEGRAM_CHAT_ID=111111111,-222222222`. Get each person's chat ID the
   same way (they message the bot once, you look it up via `getUpdates`), then join
   them with commas. Each recipient gets the digest independently — no group chat
   needed, and one bad/blocked ID won't stop delivery to the others.

### 4. Configure
```bash
cp .env.example .env
# edit .env with your Finnhub key, Telegram token/chat id, and watchlist
```

### 5. Run it
```bash
python main.py
```
If Telegram isn't configured yet, it just prints the digest to your terminal —
useful for testing the analysis logic before wiring up notifications.

## Scheduling it

**Option A — cron (if you have an always-on machine, e.g. a Raspberry Pi or a small VPS):**
```bash
crontab -e
# Run every weekday at 9:00 AM
0 9 * * 1-5 cd /path/to/stock-signal-bot && /usr/bin/python3 main.py >> logs/run.log 2>&1
```

**Option B — GitHub Actions (free, no server needed):**

The workflow file is already included at `.github/workflows/daily.yml`. To use it:

1. Push this project to a GitHub repo (private is fine — it's your project either way).
2. In the repo, go to **Settings → Secrets and variables → Actions → New repository secret**
   and add these five secrets:
   - `FINNHUB_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `GEMINI_API_KEY`
   - `WATCHLIST` (e.g. `AAPL,TSLA,NVDA`)
3. That's it — it'll run automatically on the schedule defined in the workflow
   (default: 9:30 AM Eastern on weekdays). Edit the `cron:` line in the workflow
   file to change the time — cron schedules are in UTC.
4. To test it immediately without waiting for the schedule: go to the **Actions**
   tab in your repo → **Daily Stock Signal Digest** → **Run workflow**.

Note: GitHub's cron schedules can run a few minutes late during high-traffic
periods — fine for a personal digest, just don't rely on it for precise timing.

## About the LLM decision layer

`llm_decision.py` is what makes this an *agentic* pipeline rather than a fixed
rule engine: instead of only combining signals mechanically, it hands the raw
technical signals + recent headlines to Gemini and asks it to reason about
what's actually notable and why, with an honest confidence read (not a
buy/sell call — the model is explicitly instructed never to give one).

Two things make this genuinely *agentic* rather than a single scripted LLM
call:

**1. Autonomous tool use (`agent_tools.py`)** — the model isn't limited to a
fixed, pre-fetched bundle of evidence. It has three tools available —
`get_extended_price_history` (6-month price context), `get_extended_headlines`
(a 14-day headline lookback instead of 3), and `get_recent_history` (this
same ticker's own analysis from the last 7 days, via `memory.py`) — and
decides *on its own*, per ticker, whether the initial evidence is thin
enough, or continuity with a past judgment relevant enough, to warrant
pulling more before answering. This is bounded to at most 4 tool calls per
ticker (`_MAX_TOOL_CALLS` in `llm_decision.py`) so an ambiguous ticker can't
spiral into unbounded cost or latency. When the agent does use a tool, it
shows up in the digest as a 🔧 line.

**2. A critic / verification pass (`critic.py`)** — the model's draft output
is never shipped directly. It passes through an independent second check
first: a cheap, deterministic scan for banned buy/sell language, plus a
separate LLM call that verifies the summary is actually grounded in the
given evidence and that its stated confidence is plausible. If either check
fails, the draft is replaced with a safe fallback message rather than
attempting an automatic "fix" — a failed safety check on autonomous,
unreviewed output should fail closed. A critic-rejected ticker shows up in
the digest with a ⚠️ flag.

- Requires `GEMINI_API_KEY` in your `.env` / repo secrets. Without it, the
  bot still runs fine — it just skips the 🤖 synthesis line and shows raw
  signals only.
- `watch_worthy: true` can surface a ticker in the digest even when the rule
  engine's signal count is below threshold, if Gemini judges the combination
  of signals+headlines genuinely notable together.
- Model used is set by `LLM_MODEL` in `config.py` (defaults to `gemini-3.5-flash-lite`
  — Google's cheapest, fastest, highest-throughput tier, a good fit for this task
  since it's synthesizing a few signals into short structured JSON, not deep
  reasoning). Swap to `gemini-3.8-flash` (current flagship Flash model) if you want
  noticeably stronger reasoning and don't mind the higher cost/demand.
  Note: Google's Flash lineup moves fast (new versions roughly every few weeks) —
  if you hit a 404 "model no longer available" error, check
  https://ai.google.dev/gemini-api/docs/models for the current model ID and
  update `LLM_MODEL` (or the default here) accordingly.

## Persistent memory (`memory.py`)

Every run saves each ticker's result (signal count, LLM summary, confidence,
watch-worthy flag) to a local SQLite file, `signal_history.db`. This is what
lets the agent reflect rather than reason from scratch every time — a third
tool, `get_recent_history`, gives it access to a ticker's own last 7 days of
analysis, and the system prompt tells it to use this when today's evidence
looks like it might be a continuation of something already flagged.

**Persistent history across GitHub Actions runs:** GitHub Actions runners
are ephemeral — each scheduled run starts from a fresh checkout of the repo,
so without extra handling, `signal_history.db` would be created empty on
every single run and the memory tool would never have anything to find.
`daily.yml` handles this with a "Persist signal history" step that commits
the updated `.db` file back to the repo after each run. This is a
lightweight, honest-about-its-limits pattern — it works fine for a personal
project's cadence, but it's not what you'd use at real scale (a proper
hosted database, not a file committed to git history, would be the next
step if this needed to handle much higher write volume or multiple writers).

## Running the tests

```bash
pip install pytest  # already in requirements.txt
pytest tests/ -v
```

The test suite covers `analysis.py` (technical indicator edge cases —
including a real bug it caught: a completely flat price was originally
computed as RSI-oversold instead of neutral), `critic.py` (the rule-based
and graceful-degradation paths), `memory.py` (persistence and ticker
isolation), and `notifier.py` (message chunking). It does not test the live
LLM/API calls themselves — those need real credentials and network access,
so they're exercised by actually running the bot rather than by pytest.
`.github/workflows/tests.yml` runs this suite automatically on every push,
separately from the daily digest job.

## About the sentiment signal

Finnhub's premium `news-sentiment` endpoint gives a proper bullish/bearish % computed
from news + social mentions — but it requires a paid plan. On a free-tier key, the bot
automatically falls back to `headline_keyword_sentiment()` in `analysis.py`: a simple
word-count scan over recent headlines (words like "beats", "surges", "upgrade" vs.
"misses", "plunge", "downgrade"). This is much cruder than real sentiment analysis —
it'll catch obviously one-sided news coverage but miss anything subtle. If you later
upgrade to a paid Finnhub plan, the bot will automatically use the premium endpoint
instead — no code changes needed.

## Tuning signals

All thresholds live in `config.py` — e.g. `VOLUME_SPIKE_MULTIPLIER`,
`RSI_OVERBOUGHT/OVERSOLD`, `SENTIMENT_BULLISH_THRESHOLD`. Start conservative and
loosen them once you see how noisy your watchlist is.

## Roadmap ideas (v3+)

Already built: LLM reasoning layer, autonomous tool use, a critic/safety
review pass, and persistent memory across runs. Natural next steps from here:

- **Global tool-call budget** — right now each ticker independently gets up
  to 4 tool calls; across an 11-ticker watchlist that's a real aggregate
  cost/latency ceiling that isn't bounded across the whole run, only per-ticker.
- **Structured run logging** — write each run's decisions (flagged tickers,
  tool usage, critic verdicts and reasons) to a log file or a dedicated
  table, both as an audit trail and as richer data for the memory tool to draw on.
- **Multi-agent orchestration** — split the single analyst role into
  specialized sub-agents (technical vs. news) with a coordinator that
  reconciles disagreement between them, rather than one model reasoning
  over all evidence at once.
- **A generator/critic loop instead of a single critic pass** — currently a
  rejected draft is replaced with a fallback; a fuller version would let the
  critic's `reason` feed back into a second generation attempt before falling back.
- **Add X/Twitter data** once you're ready to deal with API cost/rate limits —
  cashtag search (`$TSLA`) is the most direct route, feeding into the same
  `analysis.py` signal format.
- **Pre-commit secret scanning** — a hook that blocks a commit containing
  anything that looks like an API key, given this project's own earlier
  `.env`-tracking near-miss.

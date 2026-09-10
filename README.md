# Stock Signal Bot

An autonomous pipeline that watches a list of tickers, computes technical +
news signals, uses an LLM agent (with its own tools and a safety-review
layer) to reason about what's notable — including a buy/sell/hold
suggestion — and paper-trades that suggestion against simulated capital,
sending you a digest over Telegram. This is a **research/simulation tool**,
not real trading and not investment advice — the buy/sell suggestions only
ever move fake money and exist to let you retroactively judge whether the
agent's calls would have been worth following; treat its output as one
input among many.

## How it works

```
data_fetch.py     -> pulls price history (yfinance) + news (Finnhub)
analysis.py       -> turns raw data into discrete signals (SMA crossover, RSI, MACD crossover, Bollinger Bands, volume spike, sentiment)
memory.py         -> persists each run to SQLite, and serves the agent's own history back to it
agent_tools.py    -> tools the agent can call on its own: more price history, more headlines, its own past runs
llm_decision.py   -> a technical analyst + news analyst reason independently (optionally using tools), a coordinator reconciles their reads into a judgment + a buy/sell/hold suggestion
critic.py         -> independently reviews that judgment (and suggestion) before it's allowed to ship
simulator.py      -> applies an approved suggestion to a paper-trading portfolio, keeps every simulated trade as a permanent record
main.py           -> orchestrates all of the above into a digest, saves the run, only surfaces notable tickers
notifier.py       -> sends the digest to your Telegram (one or more recipients)
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
If this fails while building `llvmlite`/`numba` (a `vectorbt` dependency),
see "Installing vectorbt" under Backtesting below for the fix.

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
   and add these five required secrets:
   - `FINNHUB_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `GEMINI_API_KEY`
   - `WATCHLIST` (e.g. `AAPL,TSLA,NVDA`)

   Three more are optional — all have sensible defaults in `config.py`, add
   them only if you want to override:
   - `SIM_STARTING_CAPITAL` (paper-trading capital, default $10,000)
   - `LLM_REQUEST_DELAY_SECONDS` (pacing between tickers, default 15s)
   - `INTRA_TICKER_REQUEST_DELAY_SECONDS` (pacing within a ticker's own requests, default 2s)
3. That's it — it'll run automatically on the schedule defined in the workflow
   (default: hourly, 9am-5pm Eastern/Toronto, weekdays — 9 runs/day). Edit
   the `cron:` line in the workflow file to change the times — cron
   schedules are in UTC.
4. To test it immediately without waiting for the schedule: go to the **Actions**
   tab in your repo → **Daily Stock Signal Digest** → **Run workflow**.

Note: GitHub's cron schedules can run a few minutes late during high-traffic
periods — fine for a personal digest, just don't rely on it for precise timing.

**Daylight saving time:** GitHub Actions cron is always UTC and doesn't
auto-adjust for DST, so a schedule written for Eastern time is only exactly
right for half the year — the workflow file's comment explains which UTC
hours to use for EDT vs. EST, and you'd shift every hour by 1 twice a year
to stay exact. If you'd rather not think about it twice a year, a
DST-proof alternative is to run the job every hour year-round (`cron: '0
* * * *'`) and have an early step check the actual local time and skip
the rest of the job outside your target hours:
```yaml
- name: Check if within business hours
  run: |
    hour=$(TZ='America/Toronto' date +%H)
    if [ "$hour" -lt 9 ] || [ "$hour" -gt 17 ]; then
      echo "Outside 9am-5pm Toronto time ($hour:00) — skipping."
      echo "SKIP=true" >> "$GITHUB_ENV"
    fi
```
then add `if: env.SKIP != 'true'` to the later steps (dependency install,
run signal bot). This trades a fixed schedule for correctness — the job
still starts every hour (each skip is nearly instant/free), it just does
nothing outside your window, automatically right across DST changes.

## About the LLM decision layer

`llm_decision.py` is what makes this an *agentic* pipeline rather than a fixed
rule engine — and specifically a **multi-agent** one: rather than one model
seeing all the evidence at once, a **technical analyst** and a **news
analyst** each reason independently over their own domain (neither sees the
other's evidence), and a **coordinator** reconciles their two reads into one
final judgment. This mirrors how a real research desk works — a chartist and
a news analyst reach their own conclusions first, and *disagreement between
them is itself a meaningful signal*, not just noise to blend away. A single
call given all the evidence together can't represent that: it only ever sees
one pile of mixed evidence, never two independent, possibly conflicting
conclusions. If a ticker only has evidence in one domain (e.g. no headlines
today), the coordinator step is skipped entirely — reconciling needs two
things to reconcile, so the lone specialist's read is used directly.

The final output is the same shape either way: an honest confidence read,
plus a structured `buy`/`sell`/`hold` `suggestion` strictly derived from the
evidence. That suggestion drives `simulator.py`'s paper-trading simulation
(see below) — it is never presented to you as a directive. The free-text
`summary` is kept deliberately analytical (no "you should buy" language
addressed at the reader); the mechanical call lives only in the structured
`suggestion` field, and `critic.py` enforces that separation regardless of
which path produced the draft.

Two things make each specialist genuinely *agentic* rather than a single
scripted LLM call:

**1. Autonomous tool use (`agent_tools.py`)** — a specialist isn't limited to
a fixed, pre-fetched bundle of evidence. Bundled per domain — the technical
analyst gets `get_extended_price_history` (6-month price context) and
`get_recent_history`; the news analyst gets `get_extended_headlines` (a
14-day lookback instead of 3) and `get_recent_history` — each decides *on
its own*, per ticker, whether its initial evidence is thin enough, or
continuity with a past judgment relevant enough, to warrant pulling more
before answering. Neither specialist can reach outside its own domain's
tools (the technical analyst has no way to pull headlines, and vice versa)
— not because the model would necessarily misuse it, but because there's no
need to trust that when the tool simply isn't offered. This is bounded to
at most 4 tool calls *per specialist* per ticker (`_MAX_TOOL_CALLS` in
`llm_decision.py`) so an ambiguous ticker can't spiral into unbounded cost
or latency. When either specialist uses a tool, it shows up in the digest
as a 🔧 line.

**2. A critic / verification pass (`critic.py`)** — the coordinator's (or
lone specialist's) draft output is never shipped directly. It passes
through an independent second check first: a cheap, deterministic scan of
the free-text `summary` for banned imperative buy/sell language ("you
should buy", "time to sell", etc — the structured `suggestion` field is
exempt from this scan, since it's expected to say exactly that), plus a
separate LLM call that verifies the summary is actually grounded in the
given evidence, that its stated confidence is plausible, and that the
`suggestion` is a reasonable read of the evidence rather than contradicted
by it (e.g. "buy" on uniformly bearish signals). If any check fails, the
draft is replaced with a safe fallback message and the suggestion defaults
to `hold` (a safe no-op for the simulator) rather than attempting an
automatic "fix" — a failed safety check on autonomous, unreviewed output
should fail closed. A critic-rejected ticker shows up in the digest with a
⚠️ flag. Critic rejections are an expected, healthy part of the system, not
a bug to chase to zero — it occasionally rejects a perfectly reasonable
draft on an overly literal reading, which is the correct failure direction
(fail closed) for an automated safety check.

The digest shows each specialist's own read when both ran, so you can see
agreement or disagreement directly — e.g.:
```
📊 Technical: ▼ bearish (high confidence)
📰 News: ▲ bullish (low confidence)
🤖 [low confidence] The specialists genuinely conflict, with the technical
analyst showing high confidence in a bearish reversal... the overall bias
defaults to caution.
📈 Suggestion: HOLD
```

- Requires `GEMINI_API_KEY` in your `.env` / repo secrets. Without it, the
  bot still runs fine — it just skips the specialist/synthesis lines and
  shows raw signals only.
- `watch_worthy: true` can surface a ticker in the digest even when the rule
  engine's signal count is below threshold, if the coordinator (or lone
  specialist) judges the evidence genuinely notable.
- Model used is set by `LLM_MODEL` in `config.py` (defaults to `gemini-3.5-flash-lite`
  — Google's cheapest, fastest, highest-throughput tier, a good fit for this task
  since it's synthesizing a few signals into short structured JSON, not deep
  reasoning). Swap to `gemini-3.8-flash` (current flagship Flash model) if you want
  noticeably stronger reasoning and don't mind the higher cost/demand.
  Note: Google's Flash lineup moves fast (new versions roughly every few weeks) —
  if you hit a 404 "model no longer available" error, check
  https://ai.google.dev/gemini-api/docs/models for the current model ID and
  update `LLM_MODEL` (or the default here) accordingly.
- **Rate limits on larger watchlists:** Gemini's free tier caps out at 15
  requests/min. Each ticker now runs up to 4 Gemini calls — technical
  analyst, news analyst, coordinator, critic — each with up to
  `_MAX_TOOL_CALLS` (4) tool round-trips on top, so a 10+ ticker watchlist
  processed back-to-back can exceed that limit within a single run, causing
  some tickers to silently lose a call. Three mitigations: `main.py` pauses
  `LLM_REQUEST_DELAY_SECONDS` (`.env`, default 15s — raised from 8s when
  this became a 4-call pipeline) between tickers; `llm_decision.py`
  additionally pauses `INTRA_TICKER_REQUEST_DELAY_SECONDS` (`.env`, default
  2s) between each pair of a ticker's own calls, since a single ticker's
  requests otherwise burst out back-to-back and can peak a rolling
  60-second window well above what the inter-ticker pacing alone would
  suggest (Gemini's own quota dashboard, at aistudio.google.com, reports
  *peak* RPM in a window — worth checking there if you're unsure whether
  you're actually near the limit); and every one of those calls retries
  transient 429/500/503 errors with exponential backoff before giving up.
  Neither delay reaches the automatic tool-call round-trips themselves —
  those happen inside the SDK's own function-calling loop within one API
  call, with no hook to pace between them. If you're still hitting limits
  on a large watchlist, raise either delay or move to a paid Gemini tier.

## Paper-trading simulation (`simulator.py`)

Every ticker's critic-approved `suggestion` is applied to a simulated
portfolio — no real money moves, ever. This exists so you can look back
later and judge whether the agent's calls would actually have been worth
following, rather than taking the digest's word for it.

- **Sizing:** `SIM_STARTING_CAPITAL` (`.env`, defaults to $10,000) is split
  evenly across `WATCHLIST` at run time — each ticker trades only within its
  own fixed allocation. There's no shared cash pool across tickers, and no
  rebalancing if you change the watchlist size later.
- **Entry:** a `buy` suggestion opens a position only if that ticker isn't
  already held — and only deploys a *fraction* of the ticker's available
  cash, sized by the LLM's stated confidence
  (`POSITION_SIZE_HIGH/MEDIUM/LOW_CONFIDENCE` in `config.py`, default
  100%/60%/30%). An unrecognized or missing confidence value falls back to
  the low fraction. Cash held back isn't stranded — it stays in the
  ticker's running balance and is available to a later buy once the
  current position is fully closed.
- **Exit:** a position only closes on an explicit `sell` suggestion from a
  later run, and always exits the *entire* position regardless of
  confidence — there's no stop-loss/take-profit and no partial sells.
  Partial sells were deliberately left out: they'd need tracking multiple
  cost-basis lots per ticker instead of the current single-lot model, for
  a murkier signal ("sell, but only somewhat") than confidence-scaled
  entries. This otherwise matches the rest of the pipeline's signal-driven
  (not price-driven) design.
- **Records:** every simulated trade (buy or sell, with price, share count,
  dollar amount, confidence, size fraction, and realized P&L on sells) is
  appended to a `trades` table in `signal_history.db` — nothing is ever
  overwritten or deleted, so the full history of every simulated call is
  always available. Portfolio equity and P&L are recomputed fresh from
  that trade log each run rather than stored separately, so there's a
  single source of truth.
- The digest shows each run's suggestion, any trade it triggered, and a
  running portfolio summary (total equity, return %, realized P&L, open
  positions) at the bottom.
- **Buy-and-hold benchmark:** the portfolio summary also shows what
  buy-and-hold would have returned since the bot's first-ever run
  (`memory.get_earliest_run_date()`), using the same per-ticker allocation,
  so "did following the suggestions actually help" has a real answer
  instead of a bare equity number. Unlike `backtest.py`'s benchmark (which
  has full historical data to work with), this one only has as much history
  as the bot has actually been running — it starts meaningless (day one,
  0% either way) and becomes informative over time. A ticker whose price
  can't be fetched is excluded from both sides of the comparison (noted in
  the digest) rather than misread as a loss.

## Backtesting (`backtest.py`)

Validates the mechanical SMA-crossover signal from `analysis.py` (the same
`SMA_SHORT`/`SMA_LONG` thresholds tuned in `config.py`) against historical
price data, and compares it to a naive buy-and-hold baseline over the same
window. Two independent open-source backtesting engines are supported —
[backtrader](https://www.backtrader.com/) (event-driven, one bar at a time)
and [vectorbt](https://github.com/polakowo/vectorbt) (vectorized) — so
results can be cross-checked against each other instead of trusted from a
single implementation. `--engine both` runs the identical signal through
both and prints a diff table.

**Scope, deliberately:** this backtests the deterministic technical-signal
layer only — it does not replay the LLM + critic suggestion layer itself.
Doing that would mean an actual Gemini call per historical trading day per
ticker (cost, rate limits — see "About the LLM decision layer" above for how
fast that adds up even for one live run), and the model has no point-in-time
headline archive to reason over as news looked on a past date anyway. Treat
this as a sanity baseline for the signals the LLM's suggestion is built on,
not a full backtest of the live bot end-to-end.

Sizing mirrors `simulator.py` in spirit: starting capital splits evenly
across the tickers tested, each trading within its own fixed allocation,
all-in/all-out. Two real, expected mechanical differences between the
engines mean their numbers won't match exactly — neither is a bug:

1. **Fill timing.** backtrader fills an order at the *next* bar's open
   (deliberately, to avoid lookahead bias); vectorbt, as configured here,
   fills at the *same* bar's close. A signal that lands right before a
   large overnight gap, or on the very last bar of the data window, can
   therefore execute in one engine and not the other — this is the main
   source of *differing trade counts* between the two. backtrader sizes
   each order with a small cash buffer to absorb ordinary-sized gaps, and
   logs (`[TICKER] backtrader: N order(s) rejected...`) any order a gap
   still made too expensive to fill, rather than dropping it silently.
2. **Share sizing.** backtrader buys whole shares; vectorbt sizes
   fractionally by default. This is the source of differing returns/Sharpe
   *when trade counts already match*.

`--engine both`'s comparison table flags a differing trade count inline so
you can tell which kind of difference you're looking at.

```bash
python backtest.py                          # backtrader only, all of WATCHLIST, 2-year window
python backtest.py --engine vectorbt
python backtest.py --engine both             # both engines + a comparison table
python backtest.py --tickers AAPL,TSLA --period 5y --cash 20000
```

Reports, per ticker and combined: strategy return, buy-and-hold return,
Sharpe ratio, max drawdown, trade count, and win rate.

**Installing vectorbt:** it pulls in `numba` (JIT compilation) and `plotly`,
which come with two known installability snags — `plotly>=6` renames a
trace type vectorbt's own theme-init code still references, breaking
`import vectorbt` outright (pinned `<6` in `requirements.txt`), and pip's
default resolver can pick a `numba`/`llvmlite` pairing with no prebuilt
wheel for your platform, forcing a slow/fragile source build (pinned to
known-good wheel versions). If a fresh install still tries to build from
source, use `pip install --only-binary=:all: -r requirements.txt`. Also
note it will downgrade `pandas` if you're on a very new major version it
doesn't yet support — everything in this repo is compatible with the
pandas version vectorbt settles on.

## Persistent memory (`memory.py`)

Every run saves each ticker's full result — signal count, LLM summary,
confidence, watch-worthy flag, tool usage, the buy/sell/hold `suggestion`,
and whether critic review approved it (`critic_approved`) or rejected it
and why (`critic_reason`) — to a local SQLite file, `signal_history.db`.
Persisting the critic's full reasoning, not just approved/rejected, is
what makes this a real audit trail: before, a rejection's explanation only
ever existed in that run's console output; now every day's verdict for
every ticker is queryable permanently, including the approvals (which
were never printed at all).

This is also what lets the agent reflect rather than reason from scratch
every time — a third tool, `get_recent_history`, gives it access to a
ticker's own last 7 days of analysis (suggestion and critic outcome
included), and the system prompt tells it to use this when today's
evidence looks like it might be a continuation of something already
flagged.

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
computed as RSI-oversold instead of neutral), `critic.py` (the rule-based,
suggestion-validation, retry-on-transient-error, and graceful-degradation
paths), `memory.py` (persistence and ticker isolation), `notifier.py`
(message chunking), `simulator.py` (paper-trading buy/sell/hold logic and
portfolio math), and `backtest.py`'s pure helper functions (return math,
signal generation). It does not test the live LLM/API calls, or
`backtest.py`'s/`main.py`'s actual engine runs — those need real
credentials, network access, and (for backtrader/vectorbt) a full
backtesting engine, so they're exercised by actually running the bot
rather than by pytest. `.github/workflows/tests.yml` runs this suite
automatically on every push and pull request, separately from the daily
digest job.

## Secret scanning

A [pre-commit](https://pre-commit.com/) hook (`.pre-commit-config.yaml`)
runs [detect-secrets](https://github.com/Yelp/detect-secrets) against
every commit and blocks it if something that looks like an API key,
token, or other credential is about to be committed — this project has
already had a close call with a `.env` file nearly getting tracked, and
came close again mid-development when a live GitHub token ended up
pasted into a shell command.

To enable it locally (one-time setup):
```bash
pip install pre-commit  # already in requirements.txt
pre-commit install
```
After that, `git commit` runs the scan automatically. It's also enforced
in CI (`tests.yml`'s `secret-scan` job) regardless of whether you've set
up the local hook, so a push without it still gets caught.

A genuine false positive (e.g. a test fixture that looks like a key) can
be allowlisted inline with a `# pragma: allowlist secret` comment, or by
regenerating `.secrets.baseline` with `detect-secrets scan > .secrets.baseline`
after reviewing what changed.

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
`RSI_OVERBOUGHT/OVERSOLD`, `SENTIMENT_BULLISH_THRESHOLD`, `MACD_FAST/SLOW/SIGNAL`,
`BB_PERIOD/BB_STD`, `POSITION_SIZE_HIGH/MEDIUM/LOW_CONFIDENCE`. Start
conservative and loosen them once you see how noisy your watchlist is.

**MACD crossover and Bollinger Bands** (via
[pandas-ta-classic](https://github.com/twopirllc/pandas-ta-classic)) are two
more technical signals alongside SMA crossover, RSI, and volume spike:
- MACD crossover reacts faster than the SMA crossover (different smoothing —
  EMA-based, 12/26/9 by default) and can catch trend changes SMA misses.
- Bollinger Bands flag a close outside its own recent volatility range
  (20-period, 2 std dev by default) — a relative read, unlike RSI's fixed
  0-100 scale, so it adapts to how volatile a given ticker normally is.

Not `pandas-ta`: that project's PyPI releases now require Python 3.12+, and
its last Python-3.11-compatible release predates numpy 2.0 support (it
imports the since-removed `numpy.NaN`). `pandas-ta-classic` is the actively
maintained fork with the same `df.ta.*` accessor API.

## Roadmap ideas (v3+)

Already built: a multi-agent LLM reasoning layer (technical analyst + news
analyst + reconciling coordinator), autonomous tool use per specialist, a
critic/safety review pass, persistent memory across runs with structured,
queryable run logging (suggestion and full critic verdict/reason, not
just the final summary), a buy/sell/hold suggestion that drives a
paper-trading simulation with a full trade record, confidence-based
position sizing, a buy-and-hold benchmark for that live portfolio, and
pre-commit secret scanning. Natural next steps from here:

- **Global tool-call budget** — right now each specialist independently
  gets up to 4 tool calls of its own; across an 11-ticker watchlist with
  two specialists each, that's a real aggregate cost/latency ceiling
  that isn't bounded across the whole run, only per-specialist-per-ticker.
- **A generator/critic loop instead of a single critic pass** — currently a
  rejected draft is replaced with a fallback; a fuller version would let the
  critic's `reason` feed back into a second generation attempt before falling back.
- **Add X/Twitter data** once you're ready to deal with API cost/rate limits —
  cashtag search (`$TSLA`) is the most direct route, feeding into the same
  `analysis.py` signal format.

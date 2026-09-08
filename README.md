# Stock Signal Bot (v1)

A simple pipeline that watches a list of tickers, computes technical + news-sentiment
signals, and sends you a digest over Telegram. This is a **research/signal tool**,
not investment advice — treat its output as one input among many.

## How it works

```
data_fetch.py   -> pulls price history (yfinance) + news sentiment (Finnhub)
analysis.py     -> turns raw data into discrete signals (SMA crossover, RSI, volume spike, sentiment)
main.py         -> combines everything into a digest, only surfaces tickers with signals
notifier.py     -> sends the digest to your Telegram
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a Finnhub API key (free)
- Sign up at https://finnhub.io/register
- Free tier includes company news headlines, which is enough for v1.
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
   and add these four secrets:
   - `FINNHUB_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
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

- Requires `GEMINI_API_KEY` in your `.env` / repo secrets. Without it, the
  bot still runs fine — it just skips the 🤖 synthesis line and shows raw
  signals only.
- Uses Gemini's native JSON mode (`response_mime_type="application/json"`) so
  the model returns structured JSON (`summary`, `confidence`, `watch_worthy`)
  directly — no free-text parsing or markdown-fence stripping needed.
- `watch_worthy: true` can surface a ticker in the digest even when the rule
  engine's signal count is below threshold, if Gemini judges the combination
  of signals+headlines genuinely notable together.
- Model used is set by `LLM_MODEL` in `config.py` (defaults to `gemini-2.5-flash`);
  swap to a faster/cheaper Gemini model there if you're running a large
  watchlist often, or a stronger one if you want deeper reasoning.

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

## Roadmap ideas (v2+)

- Add X/Twitter data once you're ready to deal with API cost/rate limits —
  cashtag search (`$TSLA`) is the most direct route, feeding into the same
  `analysis.py` signal format.
- Swap the rule-based signal combination for an LLM call that reads all the
  raw signals + headlines and writes a short natural-language rationale.
- Persist historical digests (SQLite) so you can back-test whether the
  signals would have been useful.

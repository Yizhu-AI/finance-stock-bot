"""
Persists each run's results to SQLite so the agent can reference its own
history — this is what turns a one-shot reasoning agent into one capable of
reflection: "I flagged this ticker as watch-worthy 3 days ago for a volume
spike — is that still relevant, or did it already play out?"

The database is a single file, created on first use. Locally this just
accumulates across runs. On GitHub Actions specifically, the runner's
filesystem is ephemeral — each scheduled run starts from a fresh checkout —
so persistence across cloud runs requires the workflow to commit the
updated .db file back to the repo after each run. See the "Persistent
history across GitHub Actions runs" section in the README for that step
and its trade-offs (it's a lightweight, honest pattern, not the "real"
answer of using an actual hosted database — worth knowing the difference
if this gets scaled up).
"""
import json
import sqlite3
import datetime as dt
from contextlib import contextmanager

DB_PATH = "signal_history.db"


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal_count INTEGER NOT NULL,
                summary TEXT,
                confidence TEXT,
                watch_worthy INTEGER,
                critic_approved INTEGER,
                tool_calls TEXT,
                created_at TEXT NOT NULL
            )
        """)
        # Migration for DBs created before structured run logging —
        # CREATE TABLE IF NOT EXISTS above is a no-op against an existing
        # table, so an already-created runs table needs these added
        # explicitly. suggestion/critic_reason are the audit-trail fields
        # that used to only exist in that run's console output.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "suggestion" not in existing_cols:
            conn.execute("ALTER TABLE runs ADD COLUMN suggestion TEXT")
        if "critic_reason" not in existing_cols:
            conn.execute("ALTER TABLE runs ADD COLUMN critic_reason TEXT")
        if "regenerated" not in existing_cols:
            conn.execute("ALTER TABLE runs ADD COLUMN regenerated INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ticker_date ON runs (ticker, run_date)")


def save_run(ticker: str, report: dict, run_date: str = None):
    """report: the same dict shape build_ticker_report() produces in main.py."""
    init_db()
    run_date = run_date or dt.date.today().isoformat()
    llm = report.get("llm") or {}

    with _connect() as conn:
        conn.execute(
            """INSERT INTO runs
               (run_date, ticker, signal_count, summary, confidence, watch_worthy,
                critic_approved, tool_calls, suggestion, critic_reason, regenerated, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_date,
                ticker,
                len(report.get("signals", [])),
                llm.get("summary"),
                llm.get("confidence"),
                int(bool(llm.get("watch_worthy"))) if llm else None,
                int(bool(llm.get("critic_approved"))) if llm else None,
                json.dumps(llm.get("tool_calls", [])) if llm else None,
                llm.get("suggestion"),
                llm.get("critic_reason"),
                int(bool(llm.get("regenerated"))) if llm else None,
                dt.datetime.now().isoformat(),
            ),
        )


def get_recent_history(ticker: str, days: int = 7, exclude_today: bool = True) -> list:
    """
    Returns past runs for a ticker, most recent first, as a list of dicts.
    Used both by main.py's own logic (not currently) and, primarily, as the
    backing implementation for the agent's get_recent_history tool — the
    suggestion/critic fields let the agent see not just what it concluded
    last time, but what it was prepared to act on and whether that call
    actually passed review.
    """
    init_db()
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    today = dt.date.today().isoformat()

    query = ("SELECT run_date, signal_count, summary, confidence, watch_worthy, "
             "suggestion, critic_approved, critic_reason, regenerated "
             "FROM runs WHERE ticker = ? AND run_date >= ?")
    params = [ticker, cutoff]
    if exclude_today:
        query += " AND run_date < ?"
        params.append(today)
    query += " ORDER BY run_date DESC"

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        {
            "date": r[0],
            "signal_count": r[1],
            "summary": r[2],
            "confidence": r[3],
            "was_watch_worthy": bool(r[4]) if r[4] is not None else None,
            "suggestion": r[5],
            "critic_approved": bool(r[6]) if r[6] is not None else None,
            "critic_reason": r[7],
            "regenerated": bool(r[8]) if r[8] is not None else None,
        }
        for r in rows
    ]


def get_earliest_run_date() -> str | None:
    """
    The date of this bot's first-ever run, across all tickers — used as the
    anchor date for benchmarking the live paper-trading portfolio against a
    buy-and-hold baseline (see simulator.buy_and_hold_benchmark). Returns
    None if no runs have been saved yet.
    """
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT MIN(run_date) FROM runs").fetchone()
    return row[0] if row and row[0] is not None else None

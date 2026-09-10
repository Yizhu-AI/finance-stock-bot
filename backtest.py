"""
Backtests the mechanical SMA-crossover signal from analysis.py (the same
SMA_SHORT/SMA_LONG thresholds tuned in config.py) against historical price
data, using backtrader and/or vectorbt — two independent backtesting
engines, so results can be cross-checked against each other rather than
trusted from a single implementation.

Scope, deliberately: this validates the deterministic technical-signal
layer the LLM's buy/sell/hold suggestion is built on top of. It does NOT
replay the LLM + critic layer itself — that would mean an actual Gemini
call per historical trading day per ticker, which isn't practical (cost,
rate limits, and the model has no way to "see" news as it looked on a past
date without a point-in-time headline archive). Treat this as a sanity
baseline for the technical signals, not a backtest of exactly what the
live bot does end-to-end. See README's "Backtesting" section.

Sizing mirrors simulator.py in spirit (SIM_STARTING_CAPITAL splits evenly
across the tickers being tested, each trading within its own fixed
allocation, all-in/all-out — no partial sizing), but the two engines don't
size identically: backtrader buys whole shares (`int(cash / price)`,
leaving small leftover cash idle), while vectorbt's default sizing is
continuous/fractional. Expect the two engines' numbers to be close but not
identical because of this — a real sizing-convention difference, not a bug
in either engine.

Run:
  python backtest.py                              # backtrader only (default)
  python backtest.py --engine vectorbt
  python backtest.py --engine both                 # side-by-side + a diff table
  python backtest.py --tickers AAPL,TSLA --period 5y --cash 20000
"""
import argparse

import backtrader as bt

import config
import data_fetch


class SmaCrossoverStrategy(bt.Strategy):
    params = dict(sma_short=config.SMA_SHORT, sma_long=config.SMA_LONG)

    def __init__(self):
        sma_short = bt.indicators.SMA(period=self.p.sma_short)
        sma_long = bt.indicators.SMA(period=self.p.sma_long)
        self.crossover = bt.indicators.CrossOver(sma_short, sma_long)
        self.rejected_orders = 0

    def next(self):
        if not self.position:
            if self.crossover > 0:
                # Orders fill at the NEXT bar's open (backtrader's default,
                # to avoid lookahead bias), not today's close used to size
                # this one — an overnight gap up can otherwise make the
                # order cost more than available cash, so it gets silently
                # rejected (Margin) and the trade is dropped. A small cash
                # buffer absorbs normal-sized gaps; notify_order below still
                # counts anything that slips past it so it's never silent.
                size = int(self.broker.getcash() * 0.99 / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        elif self.crossover < 0:
            self.close()

    def notify_order(self, order):
        if order.status in (order.Margin, order.Rejected):
            self.rejected_orders += 1


def _buy_and_hold_return_pct(df) -> float:
    first_close = float(df["Close"].iloc[0])
    last_close = float(df["Close"].iloc[-1])
    return (last_close - first_close) / first_close * 100


def _sma_crossover_signals(close):
    """Entry/exit boolean Series matching backtrader's CrossOver semantics
    (fires True exactly on the bar the crossover happens)."""
    sma_short = close.rolling(config.SMA_SHORT).mean()
    sma_long = close.rolling(config.SMA_LONG).mean()
    entries = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    exits = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
    return entries.fillna(False), exits.fillna(False)


def run_backtest_backtrader(ticker: str, period: str, cash: float) -> dict:
    df = data_fetch.get_price_history(ticker, period=period)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCrossoverStrategy)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(cash)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                         timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    start_value = cerebro.broker.getvalue()
    strat = cerebro.run()[0]
    end_value = cerebro.broker.getvalue()

    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    max_drawdown_pct = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown")
    trade_stats = strat.analyzers.trades.get_analysis()
    total_trades = trade_stats.get("total", {}).get("closed", 0)
    won_trades = trade_stats.get("won", {}).get("total", 0)
    win_rate_pct = (won_trades / total_trades * 100) if total_trades else None

    if strat.rejected_orders:
        print(f"[{ticker}] backtrader: {strat.rejected_orders} order(s) rejected "
              f"(margin/cash shortfall on next-bar-open fill) — trade(s) dropped")

    return {
        "ticker": ticker,
        "engine": "backtrader",
        "start_value": start_value,
        "end_value": end_value,
        "total_return_pct": (end_value - start_value) / start_value * 100,
        "buy_hold_return_pct": _buy_and_hold_return_pct(df),
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
    }


def run_backtest_vectorbt(ticker: str, period: str, cash: float) -> dict:
    import vectorbt as vbt  # lazy: heavy import (numba JIT), only pay for it if requested

    df = data_fetch.get_price_history(ticker, period=period)
    close = df["Close"]
    entries, exits = _sma_crossover_signals(close)

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=cash, fees=0.0, freq="1D")
    total_trades = int(pf.trades.count())

    return {
        "ticker": ticker,
        "engine": "vectorbt",
        "start_value": cash,
        "end_value": pf.final_value(),
        "total_return_pct": pf.total_return() * 100,
        "buy_hold_return_pct": _buy_and_hold_return_pct(df),
        "sharpe": pf.sharpe_ratio(),
        "max_drawdown_pct": abs(pf.max_drawdown()) * 100,
        "total_trades": total_trades,
        "win_rate_pct": (pf.trades.win_rate() * 100) if total_trades else None,
    }


_ENGINES = {
    "backtrader": run_backtest_backtrader,
    "vectorbt": run_backtest_vectorbt,
}


def _fmt_pct(value):
    return f"{value:+.1f}%" if value is not None else "n/a"


def _fmt_num(value, decimals=2):
    return f"{value:.{decimals}f}" if value is not None else "n/a"


_TABLE_HEADER = f"{'Ticker':<8}{'Strategy':>12}{'Buy&Hold':>12}{'Sharpe':>10}{'MaxDD':>10}{'Trades':>8}{'WinRate':>10}"


def _print_results_table(label: str, results: list):
    if not results:
        print(f"\n=== {label}: no results ===")
        return

    print(f"\n=== {label} ===")
    print(_TABLE_HEADER)
    print("-" * len(_TABLE_HEADER))
    for r in results:
        print(
            f"{r['ticker']:<8}"
            f"{_fmt_pct(r['total_return_pct']):>12}"
            f"{_fmt_pct(r['buy_hold_return_pct']):>12}"
            f"{_fmt_num(r['sharpe']):>10}"
            f"{_fmt_pct(-r['max_drawdown_pct'] if r['max_drawdown_pct'] is not None else None):>10}"
            f"{r['total_trades']:>8}"
            f"{_fmt_pct(r['win_rate_pct']):>10}"
        )

    total_start = sum(r["start_value"] for r in results)
    total_end = sum(r["end_value"] for r in results)
    combined_return_pct = (total_end - total_start) / total_start * 100 if total_start else 0.0
    print("-" * len(_TABLE_HEADER))
    print(f"Combined {label} return across {len(results)} ticker(s): {combined_return_pct:+.1f}% "
          f"(${total_start:.2f} -> ${total_end:.2f})")


def _print_comparison(bt_results: list, vbt_results: list):
    vbt_by_ticker = {r["ticker"]: r for r in vbt_results}
    header = f"{'Ticker':<8}{'Backtrader':>14}{'Vectorbt':>14}{'Diff (pp)':>12}"
    print("\n=== Comparison: total return % (backtrader vs vectorbt) ===")
    print(header)
    print("-" * len(header))
    for bt_r in bt_results:
        vbt_r = vbt_by_ticker.get(bt_r["ticker"])
        if vbt_r is None:
            continue
        trade_note = "" if bt_r["total_trades"] == vbt_r["total_trades"] else \
            f"  (trades: {bt_r['total_trades']} vs {vbt_r['total_trades']})"
        diff = bt_r["total_return_pct"] - vbt_r["total_return_pct"]
        print(
            f"{bt_r['ticker']:<8}"
            f"{_fmt_pct(bt_r['total_return_pct']):>14}"
            f"{_fmt_pct(vbt_r['total_return_pct']):>14}"
            f"{diff:>+11.1f}p{trade_note}"
        )
    print("\nNote: differences come from two real, expected mechanics, not a bug in")
    print("either engine — (1) fill timing: backtrader fills at the NEXT bar's open")
    print("(no lookahead), vectorbt at the SAME bar's close, so a signal near a big")
    print("overnight gap or the last bar of data can execute in one engine and not")
    print("the other; (2) sizing: backtrader buys whole shares, vectorbt fractional.")
    print("A ticker with a differing trade count above is (1); same trade count but")
    print("differing return is (2). backtrader logs any order it had to drop.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", default=",".join(config.WATCHLIST),
                         help="Comma-separated tickers (default: WATCHLIST from .env)")
    parser.add_argument("--period", default="2y",
                         help="yfinance history period, e.g. 1y/2y/5y (default: 2y)")
    parser.add_argument("--cash", type=float, default=config.SIM_STARTING_CAPITAL,
                         help="Total starting capital, split evenly across tickers (default: SIM_STARTING_CAPITAL)")
    parser.add_argument("--engine", choices=["backtrader", "vectorbt", "both"], default="backtrader",
                         help="Which backtesting engine(s) to run (default: backtrader)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    allocation = args.cash / len(tickers)
    engines = ["backtrader", "vectorbt"] if args.engine == "both" else [args.engine]

    results_by_engine = {}
    for engine in engines:
        run_fn = _ENGINES[engine]
        results = []
        for ticker in tickers:
            print(f"Backtesting {ticker} ({engine})...")
            try:
                results.append(run_fn(ticker, args.period, allocation))
            except Exception as e:
                print(f"[{ticker}] {engine} backtest failed: {e}")
        results_by_engine[engine] = results

    for engine in engines:
        _print_results_table(engine, results_by_engine[engine])

    if args.engine == "both":
        _print_comparison(results_by_engine["backtrader"], results_by_engine["vectorbt"])


if __name__ == "__main__":
    main()

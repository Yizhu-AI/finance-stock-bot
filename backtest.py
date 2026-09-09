"""
Backtests the mechanical SMA-crossover signal from analysis.py (the same
SMA_SHORT/SMA_LONG thresholds tuned in config.py) against historical price
data, using backtrader.

Scope, deliberately: this validates the deterministic technical-signal
layer the LLM's buy/sell/hold suggestion is built on top of. It does NOT
replay the LLM + critic layer itself — that would mean an actual Gemini
call per historical trading day per ticker, which isn't practical (cost,
rate limits, and the model has no way to "see" news as it looked on a past
date without a point-in-time headline archive). Treat this as a sanity
baseline for the technical signals, not a backtest of exactly what the
live bot does end-to-end. See README's "Benchmarking" section.

Sizing mirrors simulator.py: SIM_STARTING_CAPITAL splits evenly across the
tickers being tested, each trading within its own fixed allocation,
all-in/all-out (no partial sizing) — so results are comparable to the live
paper-trading portfolio's numbers.

Run: python backtest.py [--tickers AAPL,TSLA] [--period 2y] [--cash 10000]
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

    def next(self):
        if not self.position:
            if self.crossover > 0:
                size = int(self.broker.getcash() / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
        elif self.crossover < 0:
            self.close()


def _buy_and_hold_return_pct(df) -> float:
    first_close = float(df["Close"].iloc[0])
    last_close = float(df["Close"].iloc[-1])
    return (last_close - first_close) / first_close * 100


def run_backtest(ticker: str, period: str, cash: float) -> dict:
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

    return {
        "ticker": ticker,
        "start_value": start_value,
        "end_value": end_value,
        "total_return_pct": (end_value - start_value) / start_value * 100,
        "buy_hold_return_pct": _buy_and_hold_return_pct(df),
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": total_trades,
        "win_rate_pct": win_rate_pct,
    }


def _fmt_pct(value):
    return f"{value:+.1f}%" if value is not None else "n/a"


def _fmt_num(value, decimals=2):
    return f"{value:.{decimals}f}" if value is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(config.WATCHLIST),
                         help="Comma-separated tickers (default: WATCHLIST from .env)")
    parser.add_argument("--period", default="2y",
                         help="yfinance history period, e.g. 1y/2y/5y (default: 2y)")
    parser.add_argument("--cash", type=float, default=config.SIM_STARTING_CAPITAL,
                         help="Total starting capital, split evenly across tickers (default: SIM_STARTING_CAPITAL)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    allocation = args.cash / len(tickers)

    results = []
    for ticker in tickers:
        print(f"Backtesting {ticker}...")
        try:
            results.append(run_backtest(ticker, args.period, allocation))
        except Exception as e:
            print(f"[{ticker}] Backtest failed: {e}")

    if not results:
        print("No results.")
        return

    header = f"{'Ticker':<8}{'Strategy':>12}{'Buy&Hold':>12}{'Sharpe':>10}{'MaxDD':>10}{'Trades':>8}{'WinRate':>10}"
    print("\n" + header)
    print("-" * len(header))
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
    print("-" * len(header))
    print(f"Combined strategy return across {len(results)} ticker(s): {combined_return_pct:+.1f}% "
          f"(${total_start:.2f} -> ${total_end:.2f})")


if __name__ == "__main__":
    main()

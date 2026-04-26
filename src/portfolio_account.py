"""
Portfolio account simulator.

Supports four types of account activity:
  - deposit(date, amount)            : add cash (external inflow)
  - withdraw_cash(date, amount)      : remove cash (external outflow)
  - buy(date, ticker, shares)        : purchase shares using cash balance
  - sell(date, ticker, shares)       : sell shares, proceeds go to cash
  - transfer_in(date, ticker, shares) : receive shares in-kind (no cash)
  - transfer_out(date, ticker, shares): deliver shares in-kind (no cash)

Performance is computed using the Time-Weighted Return (TWR) method so that
external cash flows do not distort the return figure.

Usage
-----
    prices = pd.read_csv("data/sp500_closing_prices.csv", index_col=0, parse_dates=True)
    spy    = prices["SPY"]

    acc = PortfolioAccount(name="My Portfolio")
    acc.deposit("2024-04-26", 100_000)
    acc.buy("2024-04-26", "NVDA", shares=50)
    acc.buy("2024-04-26", "AAPL", shares=100)
    acc.deposit("2025-01-02", 10_000)           # add more cash later
    acc.withdraw_cash("2025-06-01", 5_000)      # take some cash out

    daily = acc.simulate(prices)                # DataFrame: Date x [cash, *tickers, total]
    perf  = acc.performance(prices, benchmark=spy)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import warnings

import numpy as np
import pandas as pd


# ── Transaction types ─────────────────────────────────────────────────────────

class TxType(str, Enum):
    DEPOSIT        = "deposit"
    WITHDRAW_CASH  = "withdraw_cash"
    BUY            = "buy"
    SELL           = "sell"
    TRANSFER_IN    = "transfer_in"    # shares in, no cash change
    TRANSFER_OUT   = "transfer_out"   # shares out, no cash change


@dataclass
class Transaction:
    date:   pd.Timestamp
    type:   TxType
    amount: float = 0.0          # cash amount (deposit/withdraw)
    ticker: Optional[str] = None
    shares: float = 0.0
    price:  float = 0.0          # execution price (filled from prices data if 0)

    def __repr__(self) -> str:
        if self.type in (TxType.DEPOSIT, TxType.WITHDRAW_CASH):
            return f"[{self.date.date()}] {self.type.value:>14s}  ${self.amount:,.2f}"
        return (f"[{self.date.date()}] {self.type.value:>14s}  "
                f"{self.shares:,.4f} x {self.ticker} @ ${self.price:,.2f}")


# ── Portfolio account ─────────────────────────────────────────────────────────

class PortfolioAccount:
    """
    Simulates a brokerage account: cash + equity positions.

    Parameters
    ----------
    name        : display label
    annual_fee  : management fee as a decimal (e.g. 0.0009 for 0.09%)
                  accrued daily against total portfolio value
    """

    def __init__(self, name: str = "Portfolio", annual_fee: float = 0.0):
        self.name        = name
        self.annual_fee  = annual_fee
        self._txns: list[Transaction] = []

    # ── Activity API ─────────────────────────────────────────────────────────

    def deposit(self, date: str | pd.Timestamp, amount: float) -> None:
        """Add external cash to the account."""
        if amount <= 0:
            raise ValueError(f"Deposit amount must be positive, got {amount}")
        self._txns.append(Transaction(
            date=pd.Timestamp(date), type=TxType.DEPOSIT, amount=amount))

    def withdraw_cash(self, date: str | pd.Timestamp, amount: float) -> None:
        """Remove cash from the account (external outflow)."""
        if amount <= 0:
            raise ValueError(f"Withdrawal amount must be positive, got {amount}")
        self._txns.append(Transaction(
            date=pd.Timestamp(date), type=TxType.WITHDRAW_CASH, amount=amount))

    def buy(self, date: str | pd.Timestamp, ticker: str, shares: float,
            price: float = 0.0) -> None:
        """Purchase shares; cash is debited. Price looked up from data if not given."""
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        self._txns.append(Transaction(
            date=pd.Timestamp(date), type=TxType.BUY,
            ticker=ticker.upper(), shares=shares, price=price))

    def sell(self, date: str | pd.Timestamp, ticker: str, shares: float,
             price: float = 0.0) -> None:
        """Sell shares; cash is credited. Price looked up from data if not given."""
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        self._txns.append(Transaction(
            date=pd.Timestamp(date), type=TxType.SELL,
            ticker=ticker.upper(), shares=shares, price=price))

    def transfer_in(self, date: str | pd.Timestamp, ticker: str,
                    shares: float, price: float = 0.0) -> None:
        """Receive shares in-kind (no cash impact; e.g. ACATS transfer)."""
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        self._txns.append(Transaction(
            date=pd.Timestamp(date), type=TxType.TRANSFER_IN,
            ticker=ticker.upper(), shares=shares, price=price))

    def transfer_out(self, date: str | pd.Timestamp, ticker: str,
                     shares: float, price: float = 0.0) -> None:
        """Deliver shares in-kind (no cash impact)."""
        if shares <= 0:
            raise ValueError(f"Shares must be positive, got {shares}")
        self._txns.append(Transaction(
            date=pd.Timestamp(date), type=TxType.TRANSFER_OUT,
            ticker=ticker.upper(), shares=shares, price=price))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def transactions(self) -> list[Transaction]:
        return sorted(self._txns, key=lambda t: t.date)

    def _fill_prices(self, prices: pd.DataFrame) -> list[Transaction]:
        """Return transactions with execution prices filled from the price data."""
        filled = []
        for tx in self.transactions:
            if tx.ticker and tx.price == 0.0:
                # Use the closest available price on or after the transaction date
                px_dates = prices.index[prices.index >= tx.date]
                if len(px_dates) == 0:
                    px_dates = prices.index[prices.index <= tx.date]
                px_date = px_dates[0]
                if tx.ticker not in prices.columns:
                    raise KeyError(
                        f"Ticker '{tx.ticker}' not found in price data. "
                        f"Available: {list(prices.columns[:5])} ...")
                tx = Transaction(
                    date=tx.date, type=tx.type,
                    amount=tx.amount, ticker=tx.ticker,
                    shares=tx.shares,
                    price=float(prices.loc[px_date, tx.ticker]))
            filled.append(tx)
        return filled

    def summary(self) -> None:
        """Print all transactions in chronological order."""
        print(f"\nAccount: {self.name}  (fee={self.annual_fee*100:.3f}%/yr)")
        print("-" * 55)
        for tx in self.transactions:
            print(" ", tx)
        print("-" * 55)
        print(f"  {len(self._txns)} transactions total\n")

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Replay all transactions against the price history and return a
        DataFrame with one row per trading day containing:
          - cash        : uninvested cash balance
          - <ticker>    : market value of each equity position
          - equity      : total equity value
          - total       : cash + equity (portfolio NAV)
          - fee_accrued : cumulative management fee paid
          - net_flow    : cumulative external cash flows (deposits - withdrawals)

        Prices DataFrame must have dates as index and tickers as columns.
        """
        txns  = self._fill_prices(prices)
        dates = prices.index

        # State
        cash:      float               = 0.0
        positions: dict[str, float]    = {}   # ticker -> shares
        cum_fee:   float               = 0.0
        cum_flow:  float               = 0.0
        daily_fee: float               = self.annual_fee / 252

        records = []

        # Map each transaction to the next available trading day so that
        # transactions on weekends/holidays are not silently dropped.
        txn_by_date: dict[pd.Timestamp, list[Transaction]] = {}
        for tx in txns:
            future = dates[dates >= tx.date]
            settle = future[0] if len(future) else dates[-1]
            txn_by_date.setdefault(settle, []).append(tx)

        for day in dates:
            # Apply any transactions on this day (before valuation)
            for tx in txn_by_date.get(day, []):
                if tx.type == TxType.DEPOSIT:
                    cash     += tx.amount
                    cum_flow += tx.amount

                elif tx.type == TxType.WITHDRAW_CASH:
                    if tx.amount > cash + 1e-6:
                        warnings.warn(
                            f"[{day.date()}] Insufficient cash "
                            f"(have ${cash:,.2f}, withdrawing ${tx.amount:,.2f}). "
                            f"Cash floored at 0.")
                    cash      = max(0.0, cash - tx.amount)
                    cum_flow -= tx.amount

                elif tx.type == TxType.BUY:
                    cost = tx.shares * tx.price
                    if cost > cash + 1e-6:
                        warnings.warn(
                            f"[{day.date()}] Insufficient cash to buy "
                            f"{tx.shares} x {tx.ticker} (cost ${cost:,.2f}, "
                            f"have ${cash:,.2f}). Trade skipped.")
                        continue
                    cash -= cost
                    positions[tx.ticker] = positions.get(tx.ticker, 0.0) + tx.shares

                elif tx.type == TxType.SELL:
                    held = positions.get(tx.ticker, 0.0)
                    if tx.shares > held + 1e-6:
                        warnings.warn(
                            f"[{day.date()}] Insufficient shares to sell "
                            f"{tx.shares} x {tx.ticker} (have {held:.4f}). "
                            f"Selling all held.")
                        tx = Transaction(**{**tx.__dict__, "shares": held})
                    positions[tx.ticker] = max(0.0, held - tx.shares)
                    cash += tx.shares * tx.price

                elif tx.type == TxType.TRANSFER_IN:
                    positions[tx.ticker] = positions.get(tx.ticker, 0.0) + tx.shares

                elif tx.type == TxType.TRANSFER_OUT:
                    held = positions.get(tx.ticker, 0.0)
                    if tx.shares > held + 1e-6:
                        warnings.warn(
                            f"[{day.date()}] Insufficient shares to transfer out "
                            f"{tx.shares} x {tx.ticker} (have {held:.4f}). "
                            f"Transferring all held.")
                        tx = Transaction(**{**tx.__dict__, "shares": held})
                    positions[tx.ticker] = max(0.0, held - tx.shares)

            # Mark to market
            equity_by_ticker: dict[str, float] = {}
            for tkr, shares in positions.items():
                if shares > 1e-9 and tkr in prices.columns:
                    equity_by_ticker[tkr] = shares * float(prices.loc[day, tkr])

            equity = sum(equity_by_ticker.values())
            total  = cash + equity

            # Accrue management fee
            fee_today = total * daily_fee
            cash     -= fee_today
            cum_fee  += fee_today

            row = {"Date": day, "cash": cash, "equity": equity,
                   "total": max(0.0, cash + equity),
                   "fee_accrued": cum_fee, "net_flow": cum_flow}
            row.update(equity_by_ticker)
            records.append(row)

        df = pd.DataFrame(records).set_index("Date")
        df = df.fillna(0.0)
        return df

    # ── Performance ───────────────────────────────────────────────────────────

    def performance(
        self,
        prices: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        risk_free: float = 0.045,
    ) -> dict:
        """
        Compute Time-Weighted Return (TWR) and risk metrics.

        Handles external cash flows correctly: the TWR chains sub-period
        returns between each cash flow event, so deposits/withdrawals do
        not inflate or deflate the return figure.

        Parameters
        ----------
        prices     : closing price DataFrame (index=dates, columns=tickers)
        benchmark  : optional Series of benchmark prices (e.g. prices["SPY"])
        risk_free  : annual risk-free rate for Sharpe/IR calculations

        Returns
        -------
        dict with keys: twr, ann_return, volatility, sharpe,
                        max_drawdown, [active, tracking_error, ir, benchmark_ann]
        """
        sim = self.simulate(prices)
        if sim.empty or (sim["total"] == 0).all():
            raise ValueError("No funded periods to compute performance.")

        # Drop leading zero-NAV rows (before the first deposit)
        sim = sim[sim["total"] > 0]
        nav = sim["total"]

        # Build flow map: net external cash flow on each trading day.
        # Flows are settled to the next available trading day (same logic as simulate).
        txns     = self._fill_prices(prices)
        all_dates = prices.index
        flow_map: dict[pd.Timestamp, float] = {}
        for tx in txns:
            if tx.type in (TxType.DEPOSIT, TxType.WITHDRAW_CASH):
                future = all_dates[all_dates >= tx.date]
                settle = future[0] if len(future) else all_dates[-1]
                sign   = 1.0 if tx.type == TxType.DEPOSIT else -1.0
                flow_map[settle] = flow_map.get(settle, 0.0) + sign * tx.amount

        # Time-Weighted Return via flow-adjusted daily returns.
        # On a flow day (beginning-of-day flow): r = (NAV_end - NAV_begin - F) / (NAV_begin + F)
        # On a normal day:                       r = NAV_end / NAV_begin - 1
        nav_prev   = nav.shift(1)
        daily_ret  = nav.pct_change().copy()   # normal days
        daily_ret  = daily_ret.dropna()

        for d, flow in flow_map.items():
            if d not in daily_ret.index:
                continue
            prev = float(nav_prev.loc[d])
            denom = prev + flow
            if denom > 1e-6:
                daily_ret.loc[d] = (float(nav.loc[d]) - prev - flow) / denom

        twr = float((1 + daily_ret).prod()) - 1

        n      = len(daily_ret)
        ann_r  = (1 + twr) ** (252 / n) - 1
        vol    = float(daily_ret.std() * np.sqrt(252))
        sharpe = (ann_r - risk_free) / vol if vol > 0 else np.nan

        # Max drawdown
        rolling_max = nav.cummax()
        drawdown    = (nav - rolling_max) / rolling_max
        max_dd      = float(drawdown.min())

        result = {
            "twr":         twr,
            "ann_return":  ann_r,
            "volatility":  vol,
            "sharpe":      sharpe,
            "max_drawdown": max_dd,
            "total_fees":  float(sim["fee_accrued"].iloc[-1]),
        }

        if benchmark is not None:
            bench_ret  = benchmark.pct_change().dropna().reindex(daily_ret.index)
            active_ret = daily_ret - bench_ret
            te         = float(active_ret.std() * np.sqrt(252))
            bench_ann  = (1 + bench_ret).prod() ** (252 / len(bench_ret)) - 1
            active_ann = ann_r - float(bench_ann)
            ir         = active_ann / te if te > 0 else np.nan

            result.update({
                "benchmark_ann":  float(bench_ann),
                "active_return":  active_ann,
                "tracking_error": te,
                "ir":             ir,
            })

        return result

    def print_performance(
        self,
        prices: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        risk_free: float = 0.045,
    ) -> None:
        """Pretty-print performance metrics."""
        m   = self.performance(prices, benchmark, risk_free)
        sim = self.simulate(prices)
        nav = sim["total"]

        print(f"\n{'='*52}")
        print(f"  {self.name}  —  Performance Report")
        print(f"{'='*52}")
        print(f"  Period         : {sim.index[0].date()} to {sim.index[-1].date()}")
        print(f"  Starting NAV   : ${nav.iloc[0]:>12,.2f}")
        print(f"  Ending NAV     : ${nav.iloc[-1]:>12,.2f}")
        print(f"  Net flows      : ${sim['net_flow'].iloc[-1]:>12,.2f}")
        print(f"  Fees paid      : ${m['total_fees']:>12,.2f}")
        print(f"")
        print(f"  TWR            : {m['twr']*100:>+8.2f}%")
        print(f"  Ann. Return    : {m['ann_return']*100:>+8.2f}%")
        print(f"  Volatility     : {m['volatility']*100:>8.2f}%")
        print(f"  Sharpe         : {m['sharpe']:>8.3f}")
        print(f"  Max Drawdown   : {m['max_drawdown']*100:>8.2f}%")
        if "benchmark_ann" in m:
            print(f"")
            print(f"  Benchmark Ann. : {m['benchmark_ann']*100:>+8.2f}%")
            print(f"  Active Return  : {m['active_return']*100:>+8.2f}%")
            print(f"  Tracking Error : {m['tracking_error']*100:>8.2f}%")
            print(f"  IR             : {m['ir']:>8.3f}")
        print(f"{'='*52}\n")

"""
Frec Account Activity Simulation
=================================
Defines FrecAccount with exactly four activity types:

  cash_inflow   -- add cash to the account
  cash_outflow  -- withdraw cash from the account
  stock_inflow  -- transfer shares into the account (in-kind, no cash)
  stock_outflow -- transfer shares out of the account (in-kind, no cash)
                   raises ValueError if remaining shares would go negative

Demo: top-30 Frec-weighted $100 k portfolio with one of each activity.

Run from project root:
    py simulations/frec_account_activities.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Activity types ────────────────────────────────────────────────────────────

class ActivityType(str, Enum):
    CASH_INFLOW   = "cash_inflow"    # add cash
    CASH_OUTFLOW  = "cash_outflow"   # withdraw cash
    STOCK_INFLOW  = "stock_inflow"   # transfer shares in
    STOCK_OUTFLOW = "stock_outflow"  # transfer shares out (cannot go negative)


@dataclass
class AccountActivity:
    date:   pd.Timestamp
    atype:  ActivityType
    amount: float = 0.0           # cash amount (cash activities)
    ticker: Optional[str] = None  # stock ticker (stock activities)
    shares: float = 0.0           # share count (stock activities)


# ── FrecAccount ───────────────────────────────────────────────────────────────

class FrecAccount:
    """
    Simulate a portfolio with the four account activity types above.
    Activities are scheduled at any date; they settle to the next
    available trading day if the requested date is not in prices.index.
    """

    def __init__(self, name: str = "Frec Account", annual_fee: float = 0.0):
        self.name       = name
        self.annual_fee = annual_fee
        self._activities: List[AccountActivity] = []

    # ── Schedule activities ───────────────────────────────────────────────────

    def cash_inflow(self, date, amount: float) -> None:
        """Deposit cash into the account."""
        if amount <= 0:
            raise ValueError("amount must be > 0")
        self._activities.append(
            AccountActivity(pd.Timestamp(date), ActivityType.CASH_INFLOW, amount=amount))

    def cash_outflow(self, date, amount: float) -> None:
        """
        Withdraw cash from the account.
        Raises ValueError during simulation if cash balance is insufficient.
        """
        if amount <= 0:
            raise ValueError("amount must be > 0")
        self._activities.append(
            AccountActivity(pd.Timestamp(date), ActivityType.CASH_OUTFLOW, amount=amount))

    def stock_inflow(self, date, ticker: str, shares: float) -> None:
        """Transfer shares into the account (in-kind, no cash change)."""
        if shares <= 0:
            raise ValueError("shares must be > 0")
        self._activities.append(
            AccountActivity(pd.Timestamp(date), ActivityType.STOCK_INFLOW,
                            ticker=ticker.upper(), shares=shares))

    def stock_outflow(self, date, ticker: str, shares: float) -> None:
        """
        Transfer shares out of the account (in-kind, no cash change).
        Raises ValueError during simulation if remaining shares would go negative.
        """
        if shares <= 0:
            raise ValueError("shares must be > 0")
        self._activities.append(
            AccountActivity(pd.Timestamp(date), ActivityType.STOCK_OUTFLOW,
                            ticker=ticker.upper(), shares=shares))

    # ── Simulation ────────────────────────────────────────────────────────────

    @staticmethod
    def _rebalance(
        positions: dict,
        cash: float,
        prices: pd.DataFrame,
        day: pd.Timestamp,
        new_total: float,
    ) -> tuple[dict, float]:
        """
        Redistribute new_total across current holdings in proportion to their
        current market values (cash included).  Returns (new_positions, new_cash).
        """
        mkt = {t: float(prices.loc[day, t]) * s
               for t, s in positions.items() if t in prices.columns}
        old_total = sum(mkt.values()) + cash

        if old_total <= 1e-6:
            return {}, new_total

        new_positions = {}
        for t, val in mkt.items():
            px = float(prices.loc[day, t])
            if px > 0:
                new_positions[t] = (val / old_total) * new_total / px

        new_cash = (cash / old_total) * new_total
        return new_positions, new_cash

    def simulate(
        self,
        prices: pd.DataFrame,
        initial_cash: float = 0.0,
        initial_positions: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Day-by-day simulation starting from initial_cash + initial_positions.

        On each activity date the portfolio is fully rebalanced so that weights
        stay proportional to pre-activity weights and the new total NAV equals:
          cash/stock inflow  → old_NAV + activity_value
          cash/stock outflow → old_NAV − activity_value

        Returns a DataFrame (index = trading dates) with columns:
          cash, equity, total, net_flow, fee_accrued, <ticker...>
        where each ticker column holds the daily market value of that position.
        """
        if initial_positions is None:
            initial_positions = {}

        dates     = prices.index
        daily_fee = self.annual_fee / 252

        # Settle activities to next available trading day
        act_by_date: dict[pd.Timestamp, list[AccountActivity]] = {}
        for act in self._activities:
            future = dates[dates >= act.date]
            settle = future[0] if len(future) else dates[-1]
            act_by_date.setdefault(settle, []).append(act)

        # Initial state
        cash      = float(initial_cash)
        positions = {t: float(s) for t, s in initial_positions.items() if s > 0}

        # net_flow tracks total capital put in / taken out in market-value terms
        init_pos_val = sum(
            float(prices.loc[dates[0], t]) * s
            for t, s in positions.items() if t in prices.columns
        )
        net_flow    = cash + init_pos_val
        fee_accrued = 0.0

        records: list[dict] = []

        for day in dates:
            for act in act_by_date.get(day, []):
                # Mark-to-market before activity to get current total NAV
                mkt_pre   = {t: float(prices.loc[day, t]) * s
                             for t, s in positions.items() if t in prices.columns}
                old_total = sum(mkt_pre.values()) + cash

                if act.atype == ActivityType.CASH_INFLOW:
                    new_total = old_total + act.amount
                    net_flow += act.amount
                    positions, cash = self._rebalance(positions, cash, prices, day, new_total)

                elif act.atype == ActivityType.CASH_OUTFLOW:
                    if act.amount > old_total + 1e-6:
                        raise ValueError(
                            "cash_outflow {:.2f} on {} exceeds portfolio NAV {:.2f}".format(
                                act.amount, day.date(), old_total))
                    new_total = old_total - act.amount
                    net_flow -= act.amount
                    positions, cash = self._rebalance(positions, cash, prices, day, new_total)

                elif act.atype == ActivityType.STOCK_INFLOW:
                    px = float(prices.loc[day, act.ticker]) \
                         if act.ticker in prices.columns else 0.0
                    inflow_val = act.shares * px
                    new_total  = old_total + inflow_val
                    net_flow  += inflow_val
                    positions, cash = self._rebalance(positions, cash, prices, day, new_total)

                elif act.atype == ActivityType.STOCK_OUTFLOW:
                    held = positions.get(act.ticker, 0.0)
                    if act.shares > held + 1e-6:
                        raise ValueError(
                            "stock_outflow {:.4f} shares of {} on {} "
                            "exceeds held {:.4f} -- remaining would go negative".format(
                                act.shares, act.ticker, day.date(), held))
                    px = float(prices.loc[day, act.ticker]) \
                         if act.ticker in prices.columns else 0.0
                    outflow_val = act.shares * px
                    new_total   = old_total - outflow_val
                    net_flow   -= outflow_val
                    positions, cash = self._rebalance(positions, cash, prices, day, new_total)

            # Mark to market (after any rebalance)
            mkt = {t: float(prices.loc[day, t]) * s
                   for t, s in positions.items() if t in prices.columns}
            equity = sum(mkt.values())
            total  = cash + equity

            # Accrue management fee
            fee_today    = total * daily_fee
            cash        -= fee_today
            fee_accrued += fee_today

            row = {"cash": cash, "equity": equity, "total": cash + equity,
                   "net_flow": net_flow, "fee_accrued": fee_accrued}
            row.update(mkt)
            records.append(row)

        result = pd.DataFrame(records, index=dates)
        result.index.name = "Date"
        return result

    # ── Performance ───────────────────────────────────────────────────────────

    def performance(
        self,
        sim: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        risk_free: float = 0.045,
    ) -> dict:
        """
        Performance metrics with activity dates excluded.

        On activity dates the NAV jumps/drops due to cash or stock flows, not
        market performance.  Those days are identified by a non-zero change in
        net_flow and dropped from return accumulation, drawdown, and TE so that
        the metrics reflect pure investment performance only.
        """
        nav = sim["total"]

        # Activity dates: net_flow changed (inflow or outflow event)
        is_activity = sim["net_flow"].diff().abs() > 1e-3

        # Simple daily returns; drop day-0 (no prior day) and activity dates
        all_ret    = nav.pct_change().iloc[1:]
        is_act_ret = is_activity.iloc[1:]
        clean_ret  = all_ret[~is_act_ret]

        n       = max(len(clean_ret), 1)
        twr     = float((1 + clean_ret).prod()) - 1
        ann_ret = (1 + twr) ** (252 / n) - 1
        ann_vol = float(clean_ret.std() * np.sqrt(252))
        sharpe  = (ann_ret - risk_free) / ann_vol if ann_vol > 0 else 0.0

        # Drawdown from cumulative clean returns (activity jumps not included)
        cum    = (1 + clean_ret).cumprod()
        peak   = cum.cummax()
        max_dd = float(((cum - peak) / peak).min())

        out = dict(twr=twr, ann_return=ann_ret, ann_vol=ann_vol,
                   sharpe=sharpe, max_drawdown=max_dd,
                   total_fees=float(sim["fee_accrued"].iloc[-1]))

        if benchmark is not None:
            bench_ret   = benchmark.reindex(nav.index).ffill().pct_change().iloc[1:]
            clean_bench = bench_ret[~is_act_ret].reindex(clean_ret.index)
            active_ret  = clean_ret - clean_bench
            te          = float(active_ret.std() * np.sqrt(252))
            bench_ann   = float((1 + clean_bench).prod() ** (252 / n) - 1)
            ann_act     = ann_ret - bench_ann
            ir          = ann_act / te if te > 0 else 0.0
            out.update(tracking_error=te, active_return=ann_act,
                       benchmark_ann=bench_ann, ir=ir)

        return out

    def print_performance(
        self,
        sim: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        risk_free: float = 0.045,
    ) -> None:
        p   = self.performance(sim, benchmark, risk_free)
        nav = sim["total"]
        print("\n{:=<56}".format(""))
        print("  {}".format(self.name))
        print("{:=<56}".format(""))
        print("  Period         : {} to {}".format(
              sim.index[0].date(), sim.index[-1].date()))
        print("  Final NAV      : ${:>12,.2f}".format(float(nav.iloc[-1])))
        print("  Net flows in   : ${:>12,.2f}".format(float(sim["net_flow"].iloc[-1])))
        print("  Fees paid      : ${:>12,.2f}".format(p["total_fees"]))
        print("  TWR            : {:>+9.2f}%".format(p["twr"] * 100))
        print("  Ann. Return    : {:>+9.2f}%".format(p["ann_return"] * 100))
        print("  Volatility     : {:>9.2f}%".format(p["ann_vol"] * 100))
        print("  Sharpe         : {:>9.3f}".format(p["sharpe"]))
        print("  Max Drawdown   : {:>9.2f}%".format(p["max_drawdown"] * 100))
        if "benchmark_ann" in p:
            print("  Benchmark Ann. : {:>+9.2f}%".format(p["benchmark_ann"] * 100))
            print("  Active Return  : {:>+9.2f}%".format(p["active_return"] * 100))
            print("  Track. Error   : {:>9.2f}%".format(p["tracking_error"] * 100))
            print("  IR             : {:>9.3f}".format(p["ir"]))

    def summary(self) -> None:
        print("\n  Activity log  ({} activities):".format(len(self._activities)))
        print("  {:<14} {:<12} {:<8} {:>10} {:>12}".format(
              "Type", "Date", "Ticker", "Shares", "Cash ($)"))
        print("  " + "-" * 60)
        for a in self._activities:
            print("  {:<14} {:<12} {:<8} {:>10.4f} {:>12.2f}".format(
                  a.atype.value, str(a.date.date()),
                  a.ticker or "-", a.shares, a.amount))


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    INITIAL_CASH = 100_000
    ANNUAL_FEE   = 0.0009
    TOP_N        = 30

    # Load data
    prices  = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                          index_col=0, parse_dates=True)
    weights = pd.read_csv(DATA_DIR / "frec_weights_daily.csv",
                          index_col=0, parse_dates=True)
    spy     = prices["SPY"]
    START   = prices.index[0]

    # Build initial top-30 Frec-weighted positions from $100 k
    first_w = weights.iloc[0].dropna().sort_values(ascending=False).head(TOP_N)
    first_w = first_w / first_w.sum() * 100
    px_d1   = prices.loc[START]
    share_counts: dict[str, float] = {}
    residual = INITIAL_CASH
    for tkr, pct in first_w.items():
        px = float(px_d1.get(tkr, 0))
        if px > 0:
            sh = round((pct / 100) * INITIAL_CASH / px, 4)
            share_counts[tkr] = sh
            residual -= sh * px
    initial_cash = max(0.0, residual)

    # ── Scheduled activities ──────────────────────────────────────────────────
    CASH_IN_DATE   = "2024-10-01"  # 1. cash inflow
    STOCK_IN_DATE  = "2024-12-02"  # 2. stock inflow  (+20 AAPL)
    CASH_OUT_DATE  = "2025-03-03"  # 3. cash outflow
    STOCK_OUT_DATE = "2025-06-02"  # 4. stock outflow (all NVDA from initial alloc.)

    # Dry-run to find NVDA shares held at the stock_outflow date
    acc_dry = FrecAccount(name="dry", annual_fee=ANNUAL_FEE)
    acc_dry.cash_inflow( CASH_IN_DATE,  50_000)
    acc_dry.stock_inflow(STOCK_IN_DATE, "AAPL", 20)
    acc_dry.cash_outflow(CASH_OUT_DATE, 30_000)
    sim_dry = acc_dry.simulate(prices, initial_cash, share_counts)

    nvda_settle = sim_dry.index[sim_dry.index >= pd.Timestamp(STOCK_OUT_DATE)][0]
    nvda_mktval = float(sim_dry.loc[nvda_settle].get("NVDA", 0))
    nvda_px     = float(prices.loc[nvda_settle, "NVDA"])
    nvda_shares = round(nvda_mktval / nvda_px, 4) if nvda_px > 0 else 0.0

    # Print pre-run context
    print("=" * 60)
    print("  Frec Account Activity Demo")
    print("=" * 60)
    nav_at_cout = float(sim_dry.loc[
        sim_dry.index[sim_dry.index >= pd.Timestamp(CASH_OUT_DATE)][0], "total"])
    cash_at_cout = float(sim_dry.loc[
        sim_dry.index[sim_dry.index >= pd.Timestamp(CASH_OUT_DATE)][0], "cash"])
    print("NAV at cash outflow date    : ${:,.2f}".format(nav_at_cout))
    print("Cash available              : ${:,.2f}".format(cash_at_cout))
    print("Withdrawing                 : $30,000.00")
    print("NVDA shares at outflow date : {:.4f} @ ${:.2f}".format(
          nvda_shares, nvda_px))
    print()

    # Build final account
    acc = FrecAccount(name="Frec Direct Index  (top-{})".format(TOP_N),
                      annual_fee=ANNUAL_FEE)
    acc.cash_inflow( CASH_IN_DATE,  50_000)           # 1. cash inflow
    acc.stock_inflow(STOCK_IN_DATE, "AAPL", 20)       # 2. stock inflow
    acc.cash_outflow(CASH_OUT_DATE, 30_000)            # 3. cash outflow
    if nvda_shares > 0:
        acc.stock_outflow(STOCK_OUT_DATE, "NVDA", nvda_shares)  # 4. stock outflow

    acc.summary()
    sim = acc.simulate(prices, initial_cash, share_counts)
    acc.print_performance(sim, benchmark=spy)

    # Key-date snapshot
    print("\nNAV at key dates:")
    for label, ts in [
        ("Start",              START),
        ("After cash inflow",  CASH_IN_DATE),
        ("After stock inflow", STOCK_IN_DATE),
        ("After cash outflow", CASH_OUT_DATE),
        ("After stock outflow",STOCK_OUT_DATE),
        ("End",                prices.index[-1]),
    ]:
        ts  = pd.Timestamp(ts)
        idx = sim.index[sim.index >= ts]
        if not len(idx):
            continue
        row = sim.loc[idx[0]]
        print("  {:<22} {}  NAV=${:>12,.2f}  cash=${:>10,.2f}".format(
              label, idx[0].date(), float(row["total"]), float(row["cash"])))

    # ── Chart ─────────────────────────────────────────────────────────────────
    events = [
        (pd.Timestamp(CASH_IN_DATE),   "Cash inflow +$50k",        "#f0b429"),
        (pd.Timestamp(STOCK_IN_DATE),  "Stock inflow +20 AAPL",    "#a371f7"),
        (pd.Timestamp(CASH_OUT_DATE),  "Cash outflow -$30k",       "#f85149"),
        (pd.Timestamp(STOCK_OUT_DATE), "Stock outflow all NVDA",   "#58a6ff"),
    ]

    BG       = "#0d1117"; PANEL_BG = "#161b22"; GRID_COL = "#21262d"
    PORT_COL = "#3fb950"; BENCH_COL = "#58a6ff"
    TEXT = "#e6edf3"; MUTED = "#8b949e"; ACCENT = "#f0f6fc"

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI","Arial","DejaVu Sans"],
        "text.color": TEXT, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.facecolor": BG, "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COL, "axes.grid": True,
        "grid.color": GRID_COL, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    bench_scaled = spy.reindex(sim.index).ffill()
    bench_start  = bench_scaled.iloc[0]
    port_start   = sim["total"].iloc[0]
    bench_scaled = bench_scaled / bench_start * port_start

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[4, 1.5, 1],
                   hspace=0.06, left=0.07, right=0.88, top=0.88, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    nav       = sim["total"]
    cash_line = sim["cash"]
    dates     = sim.index

    # Panel 1: NAV vs benchmark
    ax1.fill_between(dates, nav, bench_scaled,
                     where=(nav >= bench_scaled), alpha=0.10,
                     color=PORT_COL, interpolate=True)
    ax1.fill_between(dates, nav, bench_scaled,
                     where=(nav < bench_scaled), alpha=0.10,
                     color="#f85149", interpolate=True)
    ax1.plot(dates, bench_scaled, color=BENCH_COL, lw=1.6, zorder=3,
             label="SPY (scaled to same start)")
    ax1.plot(dates, nav,          color=PORT_COL,  lw=2.2, zorder=4,
             label="Portfolio NAV (net of 0.09% fee)")
    ax1.plot(dates, cash_line,    color="#f0b429", lw=1.0, ls=":", zorder=3,
             alpha=0.8, label="Cash balance")

    for evdate, evlabel, evcol in events:
        for ax in (ax1, ax2, ax3):
            ax.axvline(evdate, color=evcol, lw=1.2, ls="--", alpha=0.7, zorder=2)

    y_range = nav.max() - nav.min()
    label_yf = [0.82, 0.65, 0.48, 0.31]
    for (evdate, evlabel, evcol), yf in zip(events, label_yf):
        y_pos = nav.min() + y_range * yf
        ax1.text(evdate + pd.Timedelta(days=4), y_pos, evlabel,
                 color=evcol, fontsize=8.5, va="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=evcol, alpha=0.8))

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: "${:.0f}K".format(x / 1e3)))
    ax1.set_ylabel("Portfolio Value", labelpad=8, fontsize=11)
    ax1.tick_params(labelbottom=False)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.15,
               edgecolor=GRID_COL, fontsize=9,
               labelcolor=[BENCH_COL, PORT_COL, "#f0b429"])

    # Panel 2: stacked area — cash vs equity
    ax2.stackplot(dates, cash_line.clip(lower=0), nav - cash_line.clip(lower=0),
                  labels=["Cash", "Equity"],
                  colors=["#f0b429", PORT_COL], alpha=0.55)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: "${:.0f}K".format(x / 1e3)))
    ax2.set_ylabel("Composition", labelpad=8, fontsize=10)
    ax2.tick_params(labelbottom=False)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.15,
               edgecolor=GRID_COL, fontsize=9, labelcolor=["#f0b429", PORT_COL])

    # Panel 3: cumulative net flows
    ax3.plot(dates, sim["net_flow"], color="#f0b429", lw=1.4,
             label="Cumulative net flows")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: "${:.0f}K".format(x / 1e3)))
    ax3.set_ylabel("Net Flows $", labelpad=8, fontsize=10)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center")

    fig.text(0.07, 0.94,
             "Frec Direct Index  -  Account Activity Simulation",
             fontsize=15, fontweight="bold", color=ACCENT, va="bottom")
    start_str = dates[0].strftime("%b %Y")
    end_str   = dates[-1].strftime("%b %Y")
    fig.text(0.07, 0.91,
             "Top-{n} Frec-weighted holdings  |  0.09% annual fee  |  "
             "{s} - {e}  |  Starting value $100,000".format(
                 n=TOP_N, s=start_str, e=end_str),
             fontsize=10, color=MUTED, va="bottom")

    perf = acc.performance(sim, benchmark=spy)
    sx   = 0.89
    fig.text(sx, 0.84, "Summary", fontsize=9, color=ACCENT,
             fontweight="bold", ha="left")
    sidebar = [
        ("Final NAV",    "${:,.0f}".format(float(nav.iloc[-1])),                PORT_COL),
        ("SPY equiv.",   "${:,.0f}".format(float(bench_scaled.iloc[-1])),       BENCH_COL),
        ("Net flows",    "${:,.0f}".format(float(sim["net_flow"].iloc[-1])),    "#f0b429"),
        ("Fees paid",    "${:,.0f}".format(perf["total_fees"]),                 MUTED),
        ("TWR",          "{:+.2f}%".format(perf["twr"] * 100),                 PORT_COL),
        ("Ann. Return",  "{:+.2f}%".format(perf["ann_return"] * 100),          PORT_COL),
        ("Sharpe",       "{:.2f}".format(perf["sharpe"]),                      TEXT),
        ("Max DD",       "{:.2f}%".format(perf["max_drawdown"] * 100),         "#f85149"),
        ("Track. Err.",  "{:.2f}%".format(perf.get("tracking_error", 0) * 100), MUTED),
        ("IR",           "{:.2f}".format(perf.get("ir", 0)),                   MUTED),
    ]
    for i, (lbl, val, col) in enumerate(sidebar):
        y = 0.80 - i * 0.043
        fig.text(sx,        y, lbl, fontsize=8.5, color=MUTED, ha="left")
        fig.text(sx + 0.055, y, val, fontsize=8.5, color=col,  ha="left")

    out = DATA_DIR / "frec_account_activities.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
    print("\nChart saved ->", out)
    plt.close()


if __name__ == "__main__":
    main()

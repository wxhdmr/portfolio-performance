"""
Account Activity Simulation
============================
Demonstrates the PortfolioAccount library with four key activities on top of
a base S&P 500 NPORT-weighted portfolio:

  1. Deposit $50,000 cash
  2. Transfer in 100 shares of AAPL (in-kind, no cash)
  3. Withdraw 40% of account value as cash
  4. Transfer out all GOOG shares (in-kind, no cash)

Run from the project root:
    py simulations/account_activity_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from portfolio_account import PortfolioAccount

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CASH   = 100_000
ANNUAL_FEE     = 0.0009          # 0.09%
TOP_N          = 30              # invest in top-N NPORT holdings to keep it tractable
DATA_DIR       = Path(__file__).parent.parent / "data"

# ── Load data ─────────────────────────────────────────────────────────────────
prices  = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                      index_col=0, parse_dates=True)
weights = pd.read_csv(DATA_DIR / "sp500_weights_daily.csv",
                      index_col=0, parse_dates=True)
spy     = prices["SPY"]

START_DATE = prices.index[0]   # first trading day in our data

# ── Build initial position from earliest NPORT quarter ───────────────────────
# Use first available quarter weights to decide which stocks to buy
first_q_weights = weights.iloc[0].dropna().sort_values(ascending=False)
top_holdings    = first_q_weights.head(TOP_N)                  # top-N by weight
top_holdings    = top_holdings / top_holdings.sum() * 100      # renormalise to 100%

# Prices on day 1 — used to calculate share counts
prices_day1 = prices.loc[START_DATE]

# Allocate cash proportionally to top-N holdings
alloc         = (top_holdings / 100) * INITIAL_CASH   # $ per ticker
share_counts  = {}
for tkr, dollars in alloc.items():
    px = prices_day1.get(tkr, 0)
    if px and px > 0:
        share_counts[tkr] = round(dollars / px, 4)

# ── Scheduled activities ──────────────────────────────────────────────────────
# All activities fall on trading days; if not, PortfolioAccount settles to next.
DEPOSIT_DATE   = "2024-10-01"   # Activity 1 : +$50k cash
TRANSFER_DATE  = "2024-10-01"   # Activity 2 : +100 AAPL (same day)
WITHDRAW_DATE  = "2025-04-01"   # Activity 3 : -40% of NAV
SELL_GOOG_DATE = "2025-07-01"   # Activity 4 : transfer out all GOOG

# ── Dry-run to determine NAV and positions at withdrawal date ─────────────────
def build_base(acc: PortfolioAccount) -> None:
    """Add day-1 + activity-1 & 2 transactions to an account."""
    acc.deposit(START_DATE, INITIAL_CASH)
    for tkr, shares in share_counts.items():
        if shares > 0:
            acc.buy(START_DATE, tkr, shares)
    acc.deposit(DEPOSIT_DATE, 50_000)
    acc.transfer_in(TRANSFER_DATE, "AAPL", shares=100)

acc_dry  = PortfolioAccount(name="dry-run", annual_fee=ANNUAL_FEE)
build_base(acc_dry)
sim_dry  = acc_dry.simulate(prices)

w_settle = sim_dry.index[sim_dry.index >= pd.Timestamp(WITHDRAW_DATE)][0]
row_w    = sim_dry.loc[w_settle]
nav_at_withdraw  = float(row_w["total"])
cash_at_withdraw = float(row_w["cash"])
withdraw_target  = nav_at_withdraw * 0.40   # 40% of total NAV

# If cash is insufficient, sell positions proportionally to cover the shortfall.
# We sell from each holding in proportion to its current market value.
skip_cols   = {"cash", "equity", "total", "fee_accrued", "net_flow"}
pos_cols    = [c for c in sim_dry.columns if c not in skip_cols]
equity_vals = row_w[pos_cols].clip(lower=0)
total_equity = float(equity_vals.sum())

shortfall       = max(0.0, withdraw_target - max(0.0, cash_at_withdraw))
sell_orders: list[tuple[str, float]] = []   # (ticker, shares_to_sell)

if shortfall > 0 and total_equity > 0:
    # Sell proportionally across all equity positions
    for tkr in pos_cols:
        mkt_val = float(equity_vals.get(tkr, 0))
        if mkt_val <= 0:
            continue
        sell_frac   = min(1.0, shortfall / total_equity)   # fraction of each pos
        sell_dollars = mkt_val * sell_frac
        px = float(prices.loc[w_settle, tkr]) if tkr in prices.columns else 0
        if px > 0:
            sell_shares = round(sell_dollars / px, 4)
            if sell_shares > 0:
                sell_orders.append((tkr, sell_shares))

print(f"NAV on {WITHDRAW_DATE}    : ${nav_at_withdraw:,.2f}")
print(f"Cash available          : ${cash_at_withdraw:,.2f}")
print(f"Withdrawing 40% of NAV  : ${withdraw_target:,.2f}")
if sell_orders:
    sell_total = sum(sh * float(prices.loc[w_settle, t])
                     for t, sh in sell_orders if t in prices.columns)
    print(f"Selling positions first : ${sell_total:,.2f} across {len(sell_orders)} holdings")
print()

# Also determine GOOG shares held at transfer-out date
g_settle  = sim_dry.index[sim_dry.index >= pd.Timestamp(SELL_GOOG_DATE)][0]
goog_shares = float(sim_dry.loc[g_settle, "GOOG"]) / \
              float(prices.loc[g_settle, "GOOG"])   # market val / price
goog_shares = round(goog_shares, 4)

# ── Build final account ───────────────────────────────────────────────────────
acc = PortfolioAccount(name="S&P 500 Direct Index", annual_fee=ANNUAL_FEE)
build_base(acc)

# Activity 3a: sell proportionally to raise cash for withdrawal
for tkr, sh in sell_orders:
    acc.sell(WITHDRAW_DATE, tkr, sh)

# Activity 3b: withdraw 40% of NAV as cash
acc.withdraw_cash(WITHDRAW_DATE, withdraw_target)

# Activity 4: transfer out all GOOG shares
acc.transfer_out(SELL_GOOG_DATE, "GOOG", shares=goog_shares)

sim = acc.simulate(prices)

# ── Activity event log ────────────────────────────────────────────────────────
events = [
    (pd.Timestamp(DEPOSIT_DATE),   "Deposit $50k + 100 AAPL in-kind",   "#f0b429"),
    (pd.Timestamp(WITHDRAW_DATE),  f"Withdraw 40% (${withdraw_target:,.0f})",  "#f85149"),
    (pd.Timestamp(SELL_GOOG_DATE), "Transfer out all GOOG",              "#58a6ff"),
]

# ── Print activity summary ─────────────────────────────────────────────────────
print("=" * 60)
print("  Account Activity Log")
print("=" * 60)
acc.summary()

print("Daily NAV at key dates:")
for label, ts in [
    ("Start",             START_DATE),
    ("After deposit/AAPL", DEPOSIT_DATE),
    ("Before withdrawal", WITHDRAW_DATE),
    ("After withdrawal",  WITHDRAW_DATE),
    ("After GOOG exit",   SELL_GOOG_DATE),
    ("End",               prices.index[-1]),
]:
    ts = pd.Timestamp(ts)
    idx = sim.index[sim.index >= ts]
    if len(idx):
        row = sim.loc[idx[0]]
        goog_val = row.get("GOOG", 0)
        aapl_val = row.get("AAPL", 0)
        print(f"  {label:<22} {idx[0].date()}  NAV=${row['total']:>12,.2f}  "
              f"cash=${row['cash']:>10,.2f}  "
              f"AAPL=${aapl_val:>8,.0f}  GOOG=${goog_val:>8,.0f}")

acc.print_performance(prices, benchmark=spy)

# ── Chart ─────────────────────────────────────────────────────────────────────
BG        = "#0d1117"; PANEL_BG = "#161b22"; GRID_COL = "#21262d"
PORT_COL  = "#3fb950"; BENCH_COL = "#58a6ff"
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

# Align benchmark to sim dates, scale to same starting NAV
bench_aligned = spy.reindex(sim.index).ffill()
bench_start   = bench_aligned.iloc[0]
port_start    = sim["total"].iloc[0]
bench_scaled  = bench_aligned / bench_start * port_start

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
ax1.plot(dates, nav, color=PORT_COL, lw=2.2, zorder=4,
         label="Portfolio NAV (net of 0.09% fee)")
ax1.plot(dates, cash_line, color="#f0b429", lw=1.0, ls=":", zorder=3,
         alpha=0.8, label="Cash balance")

# Vertical event lines on all panels
for evdate, evlabel, evcol in events:
    for ax in (ax1, ax2, ax3):
        ax.axvline(evdate, color=evcol, lw=1.2, ls="--", alpha=0.7, zorder=2)

# Annotations on panel 1
y_range = nav.max() - nav.min()
label_offsets = [0.78, 0.55, 0.32]
for (evdate, evlabel, evcol), yf in zip(events, label_offsets):
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

# Panel 3: cumulative net flows vs NAV
cum_flow = sim["net_flow"]
ax3.plot(dates, cum_flow, color="#f0b429", lw=1.4, label="Cumulative net flows")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: "${:.0f}K".format(x / 1e3)))
ax3.set_ylabel("Net Flows $", labelpad=8, fontsize=10)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center")

# Titles
fig.text(0.07, 0.94,
         "S&P 500 Direct Index  -  Account Activity Simulation",
         fontsize=15, fontweight="bold", color=ACCENT, va="bottom")
fig.text(0.07, 0.91,
         "Top-30 NPORT holdings  |  0.09% annual fee  |  "
         "Apr 2024 - Apr 2026  |  Starting value $100,000",
         fontsize=10, color=MUTED, va="bottom")

# Stats sidebar
final_nav   = float(nav.iloc[-1])
final_bench = float(bench_scaled.iloc[-1])
perf        = acc.performance(prices, benchmark=spy)
sx = 0.89
fig.text(sx, 0.84, "Summary", fontsize=9, color=ACCENT,
         fontweight="bold", ha="left")
rows = [
    ("Final NAV",      "${:,.0f}".format(final_nav),                PORT_COL),
    ("SPY equiv.",     "${:,.0f}".format(final_bench),              BENCH_COL),
    ("Net flows",      "${:,.0f}".format(float(sim["net_flow"].iloc[-1])), "#f0b429"),
    ("Fees paid",      "${:,.0f}".format(perf["total_fees"]),       MUTED),
    ("TWR",            "{:+.2f}%".format(perf["twr"] * 100),        PORT_COL),
    ("Ann. Return",    "{:+.2f}%".format(perf["ann_return"] * 100), PORT_COL),
    ("Sharpe",         "{:.2f}".format(perf["sharpe"]),             TEXT),
    ("Max DD",         "{:.2f}%".format(perf["max_drawdown"] * 100), "#f85149"),
    ("Track. Err.",    "{:.2f}%".format(perf["tracking_error"] * 100), MUTED),
    ("IR",             "{:.2f}".format(perf["ir"]),                 MUTED),
]
for i, (lbl, val, col) in enumerate(rows):
    y = 0.80 - i * 0.043
    fig.text(sx, y, lbl, fontsize=8.5, color=MUTED, ha="left")
    fig.text(sx + 0.055, y, val, fontsize=8.5, color=col, ha="left")

out = DATA_DIR / "account_activity_simulation.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
print("Chart saved ->", out)
plt.close()

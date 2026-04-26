"""
Portfolio Value Comparison Chart — Frec vs SPY
================================================
Reproduces the frec.com direct-indexing dashboard style:
  - Frec portfolio value vs SPY benchmark (both starting at $100,000)
  - Cumulative outperformance in dollar and percentage terms
  - Dark-theme, 3-panel layout with stats sidebar

Frec = NVDA / GOOG / GOOGL / AVGO +1.5 pp overweight, rescaled
       0.09% annual management fee applied daily

Run from project root:
    py src/chart_portfolio_comparison.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA_DIR   = Path(__file__).parent.parent / "data"
INITIAL    = 100_000
ANNUAL_FEE = 0.0009
DAILY_FEE  = ANNUAL_FEE / 252
RISK_FREE  = 0.045

# ── Theme ─────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
SPY_COL  = "#58a6ff"
FREC_COL = "#3fb950"
NEG_COL  = "#f85149"
GROSS_COL= "#6e7681"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
ACCENT   = "#f0f6fc"


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_returns(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    spy_ret   = prices["SPY"].pct_change()
    price_ret = prices.pct_change()
    tickers   = [c for c in weights.columns if c in price_ret.columns]
    common    = weights.index.intersection(price_ret.index)
    w_lag     = weights[tickers].shift(1).loc[common] / 100
    r_day     = price_ret[tickers].loc[common]
    port_ret  = (w_lag * r_day).sum(axis=1)
    return pd.DataFrame({
        "port":   port_ret,
        "spy":    spy_ret.reindex(common),
        "active": port_ret - spy_ret.reindex(common),
    }).dropna()


def to_nav(ret: pd.Series) -> pd.Series:
    """Cumulative NAV prepended with day-0 value of INITIAL."""
    t0  = ret.index[0] - pd.tseries.offsets.BDay(1)
    nav = (1 + ret).cumprod() * INITIAL
    return pd.concat([pd.Series([INITIAL], index=[t0]), nav])


def compute_stats(port: pd.Series, spy: pd.Series) -> dict:
    act    = port - spy
    n      = len(port)
    ann_p  = (1 + port).prod() ** (252 / n) - 1
    ann_s  = (1 + spy).prod()  ** (252 / n) - 1
    te     = act.std() * np.sqrt(252)
    ir     = (ann_p - ann_s) / te if te > 0 else np.nan
    sharpe = (ann_p - RISK_FREE) / (port.std() * np.sqrt(252))
    cum    = (1 + port).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return dict(ann_return=ann_p, ann_spy=ann_s, ann_active=ann_p - ann_s,
                te=te, ir=ir, sharpe=sharpe, max_dd=max_dd,
                full=(1 + port).prod() - 1, full_spy=(1 + spy).prod() - 1)


# ── Load & compute ────────────────────────────────────────────────────────────
print("Loading data ...")
prices = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                     index_col=0, parse_dates=True)
frec_w = pd.read_csv(DATA_DIR / "frec_weights_daily.csv",
                     index_col=0, parse_dates=True)

frec_r   = compute_returns(frec_w, prices)
spy_ret  = frec_r["spy"]

frec_gross = to_nav(frec_r["port"])
frec_net   = to_nav(frec_r["port"] - DAILY_FEE)
spy_nav    = to_nav(spy_ret)

dates     = frec_net.index
spy_nav   = spy_nav.reindex(dates).ffill()
frec_gross= frec_gross.reindex(dates).ffill()

outperf_usd = frec_net - spy_nav
outperf_pct = (frec_net / spy_nav - 1) * 100

st = compute_stats(frec_r["port"] - DAILY_FEE, spy_ret)

final_net   = float(frec_net.iloc[-1])
final_gross = float(frec_gross.iloc[-1])
final_spy   = float(spy_nav.iloc[-1])
fee_drag    = final_gross - final_net
gain_usd    = final_net - final_spy
gain_pct    = (final_net / final_spy - 1) * 100

start_str = dates[1].strftime("%b %Y")
end_str   = dates[-1].strftime("%b %Y")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n{:=<60}".format(""))
print("  Frec Direct Index vs SPY")
print("{:=<60}".format(""))
print("  Final NAV (net)  : ${:>12,.2f}".format(final_net))
print("  SPY benchmark    : ${:>12,.2f}".format(final_spy))
print("  Gain vs SPY      : +${:>11,.2f}  ({:+.2f}%)".format(gain_usd, gain_pct))
print("  Fee drag         : -${:>11,.2f}".format(fee_drag))
print("  Ann. Return(net) : {:>+9.4f}%  (SPY {:>+.4f}%)".format(
      st["ann_return"] * 100, st["ann_spy"] * 100))
print("  Active Return    : {:>+9.4f}%/yr".format(st["ann_active"] * 100))
print("  Tracking Error   : {:>9.4f}%".format(st["te"] * 100))
print("  IR               : {:>9.3f}".format(st["ir"]))
print("  Sharpe           : {:>9.3f}".format(st["sharpe"]))
print("  Max Drawdown     : {:>9.4f}%".format(st["max_dd"] * 100))

# ── Chart setup ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color":        TEXT,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.grid":         True,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig = plt.figure(figsize=(14, 10), facecolor=BG)
gs  = GridSpec(2, 1, figure=fig,
               height_ratios=[4, 1.5],
               hspace=0.06,
               left=0.07, right=0.84, top=0.88, bottom=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# ── Panel 1: portfolio value ──────────────────────────────────────────────────
ax1.fill_between(dates, frec_net, spy_nav,
                 where=(frec_net >= spy_nav), interpolate=True,
                 color=FREC_COL, alpha=0.12)
ax1.fill_between(dates, frec_net, spy_nav,
                 where=(frec_net < spy_nav),  interpolate=True,
                 color=NEG_COL,  alpha=0.12)

ax1.plot(dates, spy_nav,  color=SPY_COL,  lw=1.8, zorder=3)
ax1.plot(dates, frec_net, color=FREC_COL, lw=2.4, zorder=4)

# End dots
for col, val in [(SPY_COL, final_spy), (FREC_COL, final_net)]:
    ax1.scatter([dates[-1]], [val], color=col, s=70, zorder=6)

# End-of-period value labels
ax1.annotate("SPY  ${:,.0f}".format(int(final_spy)),
             xy=(dates[-1], final_spy),
             xytext=(10, 0), textcoords="offset points",
             va="center", ha="left", fontsize=9.5, color=SPY_COL, fontweight="bold")
ax1.annotate("Frec  ${:,.0f}".format(int(final_net)),
             xy=(dates[-1], final_net),
             xytext=(10, 0), textcoords="offset points",
             va="center", ha="left", fontsize=9.5, color=FREC_COL, fontweight="bold")

# Outperformance annotation
ax1.annotate(
    "+${:,.0f}  ({:+.1f}%)".format(int(gain_usd), gain_pct),
    xy=(dates[-1], (final_net + final_spy) / 2),
    xytext=(-130, 0), textcoords="offset points",
    va="center", ha="left", fontsize=11, fontweight="bold", color=FREC_COL,
    arrowprops=dict(arrowstyle="-", color=FREC_COL, lw=0.9))

ax1.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: "${:.0f}K".format(x / 1e3)))
ax1.set_ylabel("Portfolio Value", labelpad=10, fontsize=11)
ax1.tick_params(labelbottom=False)

legend_elements = [
    Line2D([0], [0], color=SPY_COL,  lw=1.8,
           label="SPY  (buy & hold ETF)  ${:,.0f}".format(int(final_spy))),
    Line2D([0], [0], color=FREC_COL, lw=2.4,
           label="Frec  (net, 0.09% fee)  ${:,.0f}".format(int(final_net))),
]
ax1.legend(handles=legend_elements,
           loc="upper left", frameon=True, framealpha=0.15,
           edgecolor=GRID_COL, fontsize=9.5,
           labelcolor=[SPY_COL, FREC_COL])

# ── Panel 2: cumulative outperformance $ ──────────────────────────────────────
ax2.axhline(0, color=GRID_COL, lw=1.0)
ax2.fill_between(dates, outperf_usd, 0,
                 where=(outperf_usd >= 0), color=FREC_COL, alpha=0.25, interpolate=True)
ax2.fill_between(dates, outperf_usd, 0,
                 where=(outperf_usd < 0),  color=NEG_COL,  alpha=0.25, interpolate=True)
ax2.plot(dates, outperf_usd, color=FREC_COL, lw=1.8, zorder=3)

ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: "+${:.0f}K".format(x / 1e3) if x >= 0
                 else "-${:.0f}K".format(abs(x) / 1e3)))
ax2.set_ylabel("Active $", labelpad=10, fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

# ── Titles ────────────────────────────────────────────────────────────────────
fig.text(0.07, 0.94,
         "Frec Direct Index vs S&P 500 ETF (SPY)",
         fontsize=16, fontweight="bold", color=ACCENT, va="bottom")
fig.text(0.07, 0.91,
         "{s} - {e}  |  Starting value $100,000".format(
             s=start_str, e=end_str),
         fontsize=9.5, color=MUTED, va="bottom")

# ── Stats sidebar ─────────────────────────────────────────────────────────────
sx = 0.885
fig.text(sx, 0.855, "Performance", fontsize=9, color=ACCENT,
         fontweight="bold", ha="left")
sidebar = [
    ("Final NAV",    "${:,.0f}".format(int(final_net)),    FREC_COL),
    ("SPY value",    "${:,.0f}".format(int(final_spy)),    SPY_COL),
    ("Gain vs SPY",  "+${:,.0f}".format(int(gain_usd)),    FREC_COL),
    ("Active %",     "{:+.2f}%".format(gain_pct),          FREC_COL),
    ("Ann. Ret.",    "{:+.2f}%".format(st["ann_return"]*100), TEXT),
    ("SPY Ann.",     "{:+.2f}%".format(st["ann_spy"]*100),    TEXT),
    ("Track. Err.",  "{:.2f}%".format(st["te"]*100),       MUTED),
    ("IR",           "{:.2f}".format(st["ir"]),            MUTED),
    ("Sharpe",       "{:.2f}".format(st["sharpe"]),        MUTED),
    ("Max DD",       "{:.2f}%".format(st["max_dd"]*100),   NEG_COL),
]
for i, (lbl, val, col) in enumerate(sidebar):
    y = 0.810 - i * 0.047
    fig.text(sx,        y, lbl, fontsize=8.5, color=MUTED, ha="left")
    fig.text(sx + 0.06, y, val, fontsize=8.5, color=col,   ha="left")

# ── Save ──────────────────────────────────────────────────────────────────────
out = DATA_DIR / "portfolio_comparison_chart.png"
plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
print("\nChart saved ->", out)
plt.close()
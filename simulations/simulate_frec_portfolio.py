"""
simulate_frec_portfolio.py
==========================
Simulate a Frec direct-index portfolio with arbitrary account activities.
Saves a daily-holdings CSV and a Frec vs SPY comparison chart.

Activity dictionary format
--------------------------
    activities = {
        "2024-10-15": {"activitytype": "cash",  "value":  50_000},  # deposit $50 k
        "2025-01-15": {"activitytype": "cash",  "value": -20_000},  # withdraw $20 k
        "2024-11-01": {"activitytype": "AAPL",  "value":  30},      # transfer in 30 AAPL shares
        "2025-04-01": {"activitytype": "NVDA",  "value": -50},      # transfer out 50 NVDA shares
    }

  activitytype = "cash"   → cash flow;  value > 0: deposit,    value < 0: withdrawal
  activitytype = <ticker> → stock flow; value > 0: shares in,  value < 0: shares out

On each activity date the portfolio is fully rebalanced so that all holdings
maintain pre-activity weight ratios and total NAV adjusts by the activity value.

Outputs (written to data/)
--------------------------
  <output_prefix>.csv          — Date × (cash | equity | total | per-stock value) DataFrame
  <output_prefix>_chart.png    — dark-theme Frec vs SPY comparison chart

Run demo from project root:
    py simulations/simulate_frec_portfolio.py
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

sys.path.insert(0, str(Path(__file__).parent))
from frec_account_activities import FrecAccount

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

DATA_DIR  = Path(__file__).parent.parent / "data"
RISK_FREE = 0.045

# ── Theme (matches chart_portfolio_comparison.py) ─────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
SPY_COL  = "#58a6ff"
FREC_COL = "#3fb950"
NEG_COL  = "#f85149"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
ACCENT   = "#f0f6fc"
ACT_COLS = ["#f0b429", "#a371f7", "#ff7b72", "#79c0ff",
            "#56d364", "#ffa657", "#d2a8ff", "#ff9bce"]


# ── Main function ─────────────────────────────────────────────────────────────

def simulate_frec_portfolio(
    initial_value: float,
    activities: dict,
    annual_fee: float = 0.0009,
    output_prefix: str = "frec_simulation",
) -> pd.DataFrame:
    """
    Simulate a Frec direct-index portfolio with account activities.

    Parameters
    ----------
    initial_value : float
        Starting portfolio value in dollars.
    activities : dict
        {date_str: {"activitytype": "cash" | ticker, "value": float}}
        value > 0 = inflow, value < 0 = outflow.
        For cash:   value is dollars.
        For stocks: value is number of shares.
    annual_fee : float
        Annual management fee rate (default 0.09%).
    output_prefix : str
        Base name for output files saved to data/.

    Returns
    -------
    pd.DataFrame
        Daily holdings DataFrame (index = trading dates).
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    prices  = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                          index_col=0, parse_dates=True)
    weights = pd.read_csv(DATA_DIR / "frec_weights_daily.csv",
                          index_col=0, parse_dates=True)
    spy_px  = prices["SPY"]

    # ── Build initial Frec-weighted positions from initial_value ───────────────
    first_w = weights.iloc[0].dropna()
    first_w = first_w[first_w > 0]
    first_w = first_w / first_w.sum() * 100   # normalise to 100%
    px_d1   = prices.iloc[0]

    share_counts: dict[str, float] = {}
    residual = float(initial_value)
    for tkr, pct in first_w.items():
        px = float(px_d1.get(tkr, 0.0))
        if px > 0:
            sh = round((pct / 100) * initial_value / px, 4)
            share_counts[tkr] = sh
            residual -= sh * px
    initial_cash = max(0.0, residual)

    # ── Register activities ────────────────────────────────────────────────────
    acc = FrecAccount(name="Frec Direct Index", annual_fee=annual_fee)
    for date_str, info in sorted(activities.items()):
        atype = str(info["activitytype"])
        value = float(info["value"])
        if atype.lower() == "cash":
            if value > 0:
                acc.cash_inflow(date_str, value)
            elif value < 0:
                acc.cash_outflow(date_str, abs(value))
        else:
            ticker = atype.upper()
            if value > 0:
                acc.stock_inflow(date_str, ticker, value)
            elif value < 0:
                acc.stock_outflow(date_str, ticker, abs(value))

    # ── Run simulation ─────────────────────────────────────────────────────────
    print("Simulating portfolio ...")
    sim = acc.simulate(prices, initial_cash=initial_cash,
                       initial_positions=share_counts)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = DATA_DIR / f"{output_prefix}.csv"
    sim.to_csv(csv_path)
    print(f"  CSV  → {csv_path}  ({len(sim)} rows × {len(sim.columns)} cols)")

    # ── Derived series ─────────────────────────────────────────────────────────
    port_nav    = sim["total"]
    dates       = port_nav.index
    spy_aligned = spy_px.reindex(dates).ffill()
    spy_nav     = spy_aligned / float(spy_aligned.iloc[0]) * float(port_nav.iloc[0])
    outperf_usd = port_nav - spy_nav

    # ── Performance stats (TWR flow-adjusted) ──────────────────────────────────
    flow_delta = sim["net_flow"].diff().fillna(0.0)
    prev_nav   = port_nav.shift(1).fillna(port_nav.iloc[0])
    denom      = prev_nav + flow_delta.clip(lower=0)
    daily_ret  = pd.Series(0.0, index=dates)
    mask       = denom > 1e-6
    daily_ret[mask] = (port_nav - prev_nav - flow_delta)[mask] / denom[mask]
    daily_ret  = daily_ret.iloc[1:]

    spy_ret    = spy_nav.pct_change().dropna()
    active_ret = daily_ret.reindex(spy_ret.index) - spy_ret
    n          = max(len(daily_ret), 1)
    ann_port   = float((1 + daily_ret).prod() ** (252 / n) - 1)
    ann_spy    = float((1 + spy_ret).prod()   ** (252 / n) - 1)
    te         = float(active_ret.std() * np.sqrt(252))
    ir         = (ann_port - ann_spy) / te if te > 0 else 0.0
    vol        = float(daily_ret.std() * np.sqrt(252))
    sharpe     = (ann_port - RISK_FREE) / vol if vol > 0 else 0.0
    peak       = np.maximum.accumulate(port_nav.values)
    max_dd     = float(np.min((port_nav.values - peak) / np.where(peak > 0, peak, 1)))

    final_port = float(port_nav.iloc[-1])
    final_spy  = float(spy_nav.iloc[-1])
    gain_usd   = final_port - final_spy
    gain_pct   = (final_port / final_spy - 1) * 100
    total_fees = float(sim["fee_accrued"].iloc[-1])

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n{:=<58}".format(""))
    print("  Frec Portfolio vs SPY")
    print("{:=<58}".format(""))
    print("  Final NAV        : ${:>12,.2f}".format(final_port))
    print("  SPY (no flows)   : ${:>12,.2f}".format(final_spy))
    print("  Gain vs SPY      : ${:>+12,.2f}  ({:+.2f}%)".format(gain_usd, gain_pct))
    print("  Fees paid        : -${:>11,.2f}".format(total_fees))
    print("  Ann. Return(net) : {:>+9.4f}%  (SPY {:>+.4f}%)".format(
          ann_port * 100, ann_spy * 100))
    print("  Active Return    : {:>+9.4f}%/yr".format((ann_port - ann_spy) * 100))
    print("  Tracking Error   : {:>9.4f}%".format(te * 100))
    print("  IR               : {:>9.3f}".format(ir))
    print("  Sharpe           : {:>9.3f}".format(sharpe))
    print("  Max Drawdown     : {:>9.4f}%".format(max_dd * 100))

    # ── Chart setup ───────────────────────────────────────────────────────────
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

    start_str = dates[0].strftime("%b %Y")
    end_str   = dates[-1].strftime("%b %Y")

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs  = GridSpec(2, 1, figure=fig,
                   height_ratios=[4, 1.5],
                   hspace=0.06,
                   left=0.07, right=0.84, top=0.88, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ── Panel 1: NAV comparison ───────────────────────────────────────────────
    ax1.fill_between(dates, port_nav, spy_nav,
                     where=(port_nav >= spy_nav), interpolate=True,
                     color=FREC_COL, alpha=0.12)
    ax1.fill_between(dates, port_nav, spy_nav,
                     where=(port_nav <  spy_nav), interpolate=True,
                     color=NEG_COL,  alpha=0.12)
    ax1.plot(dates, spy_nav,  color=SPY_COL,  lw=1.8, zorder=3)
    ax1.plot(dates, port_nav, color=FREC_COL, lw=2.4, zorder=4)

    # End dots + value labels
    ax1.scatter([dates[-1]], [final_spy],  color=SPY_COL,  s=70, zorder=6)
    ax1.scatter([dates[-1]], [final_port], color=FREC_COL, s=70, zorder=6)
    ax1.annotate("SPY  ${:,.0f}".format(int(final_spy)),
                 xy=(dates[-1], final_spy),
                 xytext=(10, 0), textcoords="offset points",
                 va="center", ha="left", fontsize=9.5,
                 color=SPY_COL, fontweight="bold")
    ax1.annotate("Frec  ${:,.0f}".format(int(final_port)),
                 xy=(dates[-1], final_port),
                 xytext=(10, 0), textcoords="offset points",
                 va="center", ha="left", fontsize=9.5,
                 color=FREC_COL, fontweight="bold")

    col_annot = FREC_COL if gain_usd >= 0 else NEG_COL
    ax1.annotate(
        "${:+,.0f}  ({:+.1f}%)".format(int(gain_usd), gain_pct),
        xy=(dates[-1], (final_port + final_spy) / 2),
        xytext=(-130, 0), textcoords="offset points",
        va="center", ha="left", fontsize=11, fontweight="bold", color=col_annot,
        arrowprops=dict(arrowstyle="-", color=col_annot, lw=0.9))

    # Activity markers
    nav_min  = float(port_nav.min())
    nav_rng  = float(port_nav.max()) - nav_min
    act_items = [(pd.Timestamp(d), info) for d, info in sorted(activities.items())]
    for i, (act_ts, info) in enumerate(act_items):
        col   = ACT_COLS[i % len(ACT_COLS)]
        atype = str(info["activitytype"])
        value = float(info["value"])
        sign  = "+" if value >= 0 else ""
        if atype.lower() == "cash":
            lbl = "Cash {}${:,.0f}".format(sign if value >= 0 else "-", abs(value))
        else:
            lbl = "{} {}{:.2f}sh".format(atype.upper(), sign, value)
        for ax in (ax1, ax2):
            ax.axvline(act_ts, color=col, lw=1.2, ls="--", alpha=0.75, zorder=2)
        yf = 0.82 - (i % 4) * 0.17
        ax1.text(act_ts + pd.Timedelta(days=4),
                 nav_min + nav_rng * yf, lbl,
                 color=col, fontsize=8, va="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=col, alpha=0.8))

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: "${:.0f}K".format(x / 1e3)))
    ax1.set_ylabel("Portfolio Value", labelpad=10, fontsize=11)
    ax1.tick_params(labelbottom=False)

    legend_elements = [
        Line2D([0], [0], color=SPY_COL,  lw=1.8,
               label="SPY  (buy & hold ETF)  ${:,.0f}".format(int(final_spy))),
        Line2D([0], [0], color=FREC_COL, lw=2.4,
               label="Frec  (net, {:.2f}% fee)  ${:,.0f}".format(
                   annual_fee * 100, int(final_port))),
    ]
    ax1.legend(handles=legend_elements, loc="upper left",
               frameon=True, framealpha=0.15, edgecolor=GRID_COL, fontsize=9.5,
               labelcolor=[SPY_COL, FREC_COL])

    # ── Panel 2: cumulative outperformance $ ──────────────────────────────────
    ax2.axhline(0, color=GRID_COL, lw=1.0)
    ax2.fill_between(dates, outperf_usd, 0,
                     where=(outperf_usd >= 0), color=FREC_COL, alpha=0.25,
                     interpolate=True)
    ax2.fill_between(dates, outperf_usd, 0,
                     where=(outperf_usd <  0), color=NEG_COL,  alpha=0.25,
                     interpolate=True)
    ax2.plot(dates, outperf_usd, color=FREC_COL, lw=1.8, zorder=3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: "+${:.0f}K".format(x / 1e3) if x >= 0
                     else "-${:.0f}K".format(abs(x) / 1e3)))
    ax2.set_ylabel("Active $", labelpad=10, fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.07, 0.94,
             "Frec Direct Index vs S&P 500 ETF (SPY)",
             fontsize=16, fontweight="bold", color=ACCENT, va="bottom")
    fig.text(0.07, 0.91,
             "{s} - {e}  |  Starting value ${v:,.0f}".format(
                 s=start_str, e=end_str, v=initial_value),
             fontsize=9.5, color=MUTED, va="bottom")

    # ── Stats sidebar ─────────────────────────────────────────────────────────
    sx = 0.885
    fig.text(sx, 0.855, "Performance", fontsize=9, color=ACCENT,
             fontweight="bold", ha="left")
    sidebar = [
        ("Final NAV",   "${:,.0f}".format(int(final_port)),
                        FREC_COL),
        ("SPY value",   "${:,.0f}".format(int(final_spy)),
                        SPY_COL),
        ("Gain vs SPY", "${:+,.0f}".format(int(gain_usd)),
                        FREC_COL if gain_usd >= 0 else NEG_COL),
        ("Active %",    "{:+.2f}%".format(gain_pct),
                        FREC_COL if gain_pct >= 0 else NEG_COL),
        ("Ann. Ret.",   "{:+.2f}%".format(ann_port * 100),   TEXT),
        ("SPY Ann.",    "{:+.2f}%".format(ann_spy   * 100),  TEXT),
        ("Track. Err.", "{:.2f}%".format(te * 100),          MUTED),
        ("IR",          "{:.2f}".format(ir),                 MUTED),
        ("Sharpe",      "{:.2f}".format(sharpe),             MUTED),
        ("Max DD",      "{:.2f}%".format(max_dd * 100),      NEG_COL),
        ("Fees paid",   "${:,.0f}".format(int(total_fees)),   MUTED),
    ]
    for i, (lbl, val, col) in enumerate(sidebar):
        y = 0.810 - i * 0.044
        fig.text(sx,        y, lbl, fontsize=8.5, color=MUTED, ha="left")
        fig.text(sx + 0.06, y, val, fontsize=8.5, color=col,   ha="left")

    # ── Save chart ────────────────────────────────────────────────────────────
    chart_path = DATA_DIR / f"{output_prefix}_chart.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Chart → {chart_path}")
    plt.close()

    return sim


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = simulate_frec_portfolio(
        initial_value=100_000,
        activities={
            "2024-10-15": {"activitytype": "cash",  "value":  50_000},
            "2025-01-15": {"activitytype": "cash",  "value": -20_000},
            "2024-11-01": {"activitytype": "AAPL",  "value":  30},
            "2025-04-01": {"activitytype": "NVDA",  "value": -50},
        },
        output_prefix="frec_simulation",
    )
    print("\nSample output (first 3 rows):")
    print(sim[["cash", "equity", "total", "net_flow"]].head(3).to_string())
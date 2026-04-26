"""
simple_simulation.py
====================
Generates a synthetic portfolio return series that exactly hits specified
tracking error, excess return, and fee targets relative to a provided
SPY daily closing price series.

Functions
---------
    simulate_return_with_TE_ER(spy_close, target_annual_te, target_annual_er,
                                annual_fee, output_path, random_seed)
        → pd.Series  (simulated portfolio closing prices)

    validate(port_close, spy_close, annual_fee)
        → dict  (realized TE, ER, IR, Sharpe, max drawdown, etc.)

    plot_simulation(port_close, spy_close, annual_fee, chart_path)
        → saves a dark-theme comparison chart (Frec-style 2-panel layout)

Run demo from project root:
    py simulations/simple_simulation.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"


def simulate_return_with_TE_ER(
    spy_closing_prices: pd.DataFrame | pd.Series,
    target_annual_te: float,
    target_annual_er: float,
    annual_fee: float = 0.0009,
    output_path: str | Path | None = None,
    random_seed: int = 42,
) -> pd.Series:
    """
    Generate a synthetic portfolio closing-price series with exact tracking
    error, net excess return, and management fee properties.

    Parameters
    ----------
    spy_closing_prices : pd.DataFrame or pd.Series
        Either the full closing-prices DataFrame (with a "SPY" column, as
        loaded directly from sp500_closing_prices.csv) or just the SPY
        column as a Series.  The DatetimeIndex is used as the simulation
        calendar; the starting value is shared with the output series.
    target_annual_te : float
        Target annualised tracking error, e.g. 0.017 for 1.7%.
    target_annual_er : float
        Target annualised net excess return vs SPY after the management fee,
        e.g. 0.02 for +2.0%.
    annual_fee : float
        Annual management fee rate (default 0.09% = 0.0009).
    output_path : str | Path | None
        Destination CSV.  Defaults to data/simple_simulation.csv.
    random_seed : int
        NumPy seed for reproducibility (default 42).

    Returns
    -------
    pd.Series
        Simulated portfolio daily closing prices with the same DatetimeIndex
        and starting value as the SPY input.
    """
    if isinstance(spy_closing_prices, pd.DataFrame):
        spy_close = spy_closing_prices["SPY"]
    else:
        spy_close = spy_closing_prices

    spy_close = spy_close.dropna()
    n = len(spy_close)
    if n < 2:
        raise ValueError("spy_closing_prices must have at least 2 observations")

    # ── Daily parameter targets ────────────────────────────────────────────────
    daily_te          = target_annual_te / np.sqrt(252)
    daily_fee         = annual_fee / 252
    # Gross alpha must cover the fee so that net ER = target_annual_er
    daily_gross_alpha = (target_annual_er + annual_fee) / 252

    # ── Alpha: white noise rescaled to exact TE and mean ──────────────────────
    rng   = np.random.default_rng(random_seed)
    noise = rng.standard_normal(n - 1)
    noise = noise - noise.mean()                    # exact zero mean
    noise = noise / noise.std(ddof=1) * daily_te   # exact daily TE (ddof=1 matches pandas)
    alpha = noise + daily_gross_alpha         # shift to target gross alpha

    # ── SPY and portfolio daily returns ───────────────────────────────────────
    spy_ret  = spy_close.pct_change().iloc[1:].values   # shape (n-1,)
    port_ret = spy_ret + alpha - daily_fee               # net of fee

    # ── Compound into a closing-price series starting at spy_close.iloc[0] ───
    nav    = np.empty(n)
    nav[0] = float(spy_close.iloc[0])
    for i, r in enumerate(port_ret):
        nav[i + 1] = nav[i] * (1.0 + r)
    port_close = pd.Series(nav, index=spy_close.index, name="port_close")

    # ── Verify achieved metrics ────────────────────────────────────────────────
    active_ret = pd.Series(port_ret - spy_ret, index=spy_close.index[1:])
    ach_te     = float(active_ret.std() * np.sqrt(252))
    ach_er_net = float(active_ret.mean() * 252)

    print("=" * 56)
    print("  Simulated Portfolio vs SPY")
    print("=" * 56)
    print("  Period         : {} → {}".format(
          spy_close.index[0].date(), spy_close.index[-1].date()))
    print("  Trading days   : {}".format(n))
    print("  Target TE      : {:>8.4f}%   achieved {:>8.4f}%".format(
          target_annual_te * 100, ach_te * 100))
    print("  Target ER(net) : {:>+8.4f}%   achieved {:>+8.4f}%".format(
          target_annual_er * 100, ach_er_net * 100))
    print("  Annual fee     : {:>8.4f}%".format(annual_fee * 100))
    print("  Starting value : ${:>12,.4f}".format(float(spy_close.iloc[0])))
    print("  Final SPY      : ${:>12,.4f}".format(float(spy_close.iloc[-1])))
    print("  Final Port     : ${:>12,.4f}".format(float(port_close.iloc[-1])))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = DATA_DIR / "simple_simulation.csv"
    output_path = Path(output_path)

    port_ret_full   = np.concatenate([[np.nan], port_ret])
    active_ret_full = np.concatenate([[np.nan], active_ret.values])

    out = pd.DataFrame({
        "spy_close":  spy_close,
        "port_close": port_close,
        "spy_ret":    spy_close.pct_change(),
        "port_ret":   pd.Series(port_ret_full,   index=spy_close.index),
        "active_ret": pd.Series(active_ret_full, index=spy_close.index),
    })
    out.index.name = "Date"
    out.to_csv(output_path)
    print("  CSV saved      : {}".format(output_path))

    return port_close


# ── Theme ─────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
SPY_COL  = "#58a6ff"
PORT_COL = "#3fb950"
NEG_COL  = "#f85149"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
ACCENT   = "#f0f6fc"
RISK_FREE = 0.045


# ── validate ──────────────────────────────────────────────────────────────────

def validate(
    port_close: pd.Series,
    spy_close: pd.Series,
    annual_fee: float = 0.0009,
) -> dict:
    """
    Compute realized performance metrics for a simulated portfolio vs SPY.

    Parameters
    ----------
    port_close : pd.Series
        Simulated portfolio daily closing prices.
    spy_close : pd.Series
        SPY daily closing prices (same index).
    annual_fee : float
        Annual management fee used in the simulation (for reporting only).

    Returns
    -------
    dict with keys:
        ann_port_ret, ann_spy_ret, ann_er, ann_te, ir, sharpe, max_dd,
        cum_port_ret, cum_spy_ret, n_days, annual_fee
    """
    common     = port_close.index.intersection(spy_close.index)
    port       = port_close.reindex(common).dropna()
    spy        = spy_close.reindex(common).dropna()
    common     = port.index.intersection(spy.index)
    port, spy  = port.reindex(common), spy.reindex(common)

    port_ret   = port.pct_change().dropna()
    spy_ret    = spy.pct_change().dropna()
    active_ret = (port_ret - spy_ret).dropna()

    n          = len(port_ret)
    ann_port   = float((1 + port_ret).prod() ** (252 / n) - 1)
    ann_spy    = float((1 + spy_ret).prod()  ** (252 / n) - 1)
    ann_er     = float(active_ret.mean() * 252)
    ann_te     = float(active_ret.std()  * np.sqrt(252))
    ir         = ann_er / ann_te if ann_te > 0 else 0.0
    vol        = float(port_ret.std() * np.sqrt(252))
    sharpe     = (ann_port - RISK_FREE) / vol if vol > 0 else 0.0
    cum        = (1 + port_ret).cumprod()
    peak       = cum.cummax()
    max_dd     = float(((cum - peak) / peak).min())
    cum_port   = float((1 + port_ret).prod() - 1)
    cum_spy    = float((1 + spy_ret).prod()  - 1)

    stats = dict(
        ann_port_ret = ann_port,
        ann_spy_ret  = ann_spy,
        ann_er       = ann_er,
        ann_te       = ann_te,
        ir           = ir,
        sharpe       = sharpe,
        max_dd       = max_dd,
        cum_port_ret = cum_port,
        cum_spy_ret  = cum_spy,
        n_days       = n,
        annual_fee   = annual_fee,
    )

    print("=" * 56)
    print("  Realized Performance vs SPY")
    print("=" * 56)
    print("  Period         : {} → {}".format(common[0].date(), common[-1].date()))
    print("  Trading days   : {}".format(n))
    print("  Ann. Return    : {:>+8.4f}%  (SPY {:>+.4f}%)".format(
          ann_port * 100, ann_spy * 100))
    print("  Cum. Return    : {:>+8.4f}%  (SPY {:>+.4f}%)".format(
          cum_port * 100, cum_spy * 100))
    print("  Excess Return  : {:>+8.4f}%/yr".format(ann_er * 100))
    print("  Tracking Error : {:>8.4f}%/yr".format(ann_te * 100))
    print("  IR             : {:>8.3f}".format(ir))
    print("  Sharpe         : {:>8.3f}".format(sharpe))
    print("  Max Drawdown   : {:>8.4f}%".format(max_dd * 100))
    print("  Annual fee     : {:>8.4f}%".format(annual_fee * 100))

    return stats


# ── plot_simulation ───────────────────────────────────────────────────────────

def plot_simulation(
    port_close: pd.Series,
    spy_close: pd.Series,
    annual_fee: float = 0.0009,
    chart_path: str | Path | None = None,
) -> None:
    """
    Render a dark-theme 2-panel comparison chart: portfolio NAV vs SPY
    (top panel) and cumulative active return in dollars (bottom panel),
    with a stats sidebar.  Saves to chart_path (default data/simple_simulation_chart.png).
    """
    common    = port_close.index.intersection(spy_close.index)
    port_nav  = port_close.reindex(common)
    spy_nav   = spy_close.reindex(common)
    dates     = common

    port_ret   = port_nav.pct_change().dropna()
    spy_ret    = spy_nav.pct_change().dropna()
    active_ret = (port_ret - spy_ret).dropna()

    n        = len(port_ret)
    ann_port = float((1 + port_ret).prod() ** (252 / n) - 1)
    ann_spy  = float((1 + spy_ret).prod()  ** (252 / n) - 1)
    ann_er   = float(active_ret.mean() * 252)
    ann_te   = float(active_ret.std()  * np.sqrt(252))
    ir       = ann_er / ann_te if ann_te > 0 else 0.0
    vol      = float(port_ret.std() * np.sqrt(252))
    sharpe   = (ann_port - RISK_FREE) / vol if vol > 0 else 0.0
    cum      = (1 + port_ret).cumprod()
    max_dd   = float(((cum - cum.cummax()) / cum.cummax()).min())

    final_port  = float(port_nav.iloc[-1])
    final_spy   = float(spy_nav.iloc[-1])
    gain_usd    = final_port - final_spy
    gain_pct    = (final_port / final_spy - 1) * 100
    outperf_usd = port_nav - spy_nav

    start_str = dates[0].strftime("%b %Y")
    end_str   = dates[-1].strftime("%b %Y")

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
                   height_ratios=[4, 1.5], hspace=0.06,
                   left=0.07, right=0.84, top=0.88, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Panel 1: NAV comparison
    col_fill = PORT_COL if gain_usd >= 0 else NEG_COL
    ax1.fill_between(dates, port_nav, spy_nav,
                     where=(port_nav >= spy_nav), interpolate=True,
                     color=PORT_COL, alpha=0.12)
    ax1.fill_between(dates, port_nav, spy_nav,
                     where=(port_nav <  spy_nav), interpolate=True,
                     color=NEG_COL,  alpha=0.12)
    ax1.plot(dates, spy_nav,  color=SPY_COL,  lw=1.8, zorder=3)
    ax1.plot(dates, port_nav, color=PORT_COL, lw=2.4, zorder=4)

    ax1.scatter([dates[-1]], [final_spy],  color=SPY_COL,  s=70, zorder=6)
    ax1.scatter([dates[-1]], [final_port], color=PORT_COL, s=70, zorder=6)
    ax1.annotate("SPY  ${:,.2f}".format(final_spy),
                 xy=(dates[-1], final_spy), xytext=(10, 0),
                 textcoords="offset points", va="center", ha="left",
                 fontsize=9.5, color=SPY_COL, fontweight="bold")
    ax1.annotate("Port  ${:,.2f}".format(final_port),
                 xy=(dates[-1], final_port), xytext=(10, 0),
                 textcoords="offset points", va="center", ha="left",
                 fontsize=9.5, color=PORT_COL, fontweight="bold")

    col_annot = PORT_COL if gain_usd >= 0 else NEG_COL
    ax1.annotate(
        "${:+,.2f}  ({:+.2f}%)".format(gain_usd, gain_pct),
        xy=(dates[-1], (final_port + final_spy) / 2),
        xytext=(-130, 0), textcoords="offset points",
        va="center", ha="left", fontsize=11, fontweight="bold", color=col_annot,
        arrowprops=dict(arrowstyle="-", color=col_annot, lw=0.9))

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: "${:.0f}".format(x)))
    ax1.set_ylabel("Portfolio Value", labelpad=10, fontsize=11)
    ax1.tick_params(labelbottom=False)
    ax1.legend(handles=[
        Line2D([0], [0], color=SPY_COL,  lw=1.8,
               label="SPY  ${:,.2f}".format(final_spy)),
        Line2D([0], [0], color=PORT_COL, lw=2.4,
               label="Portfolio  (fee {:.2f}%)  ${:,.2f}".format(
                   annual_fee * 100, final_port)),
    ], loc="upper left", frameon=True, framealpha=0.15,
       edgecolor=GRID_COL, fontsize=9.5,
       labelcolor=[SPY_COL, PORT_COL])

    # Panel 2: cumulative active $
    ax2.axhline(0, color=GRID_COL, lw=1.0)
    ax2.fill_between(dates, outperf_usd, 0,
                     where=(outperf_usd >= 0), color=PORT_COL, alpha=0.25,
                     interpolate=True)
    ax2.fill_between(dates, outperf_usd, 0,
                     where=(outperf_usd <  0), color=NEG_COL,  alpha=0.25,
                     interpolate=True)
    ax2.plot(dates, outperf_usd, color=PORT_COL, lw=1.8, zorder=3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: "+${:.2f}".format(x) if x >= 0
                     else "-${:.2f}".format(abs(x))))
    ax2.set_ylabel("Active $", labelpad=10, fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Titles
    fig.text(0.07, 0.94, "Simulated Portfolio vs S&P 500 ETF (SPY)",
             fontsize=16, fontweight="bold", color=ACCENT, va="bottom")
    fig.text(0.07, 0.91,
             "{s} - {e}  |  Starting value ${v:,.2f}".format(
                 s=start_str, e=end_str, v=float(port_nav.iloc[0])),
             fontsize=9.5, color=MUTED, va="bottom")

    # Stats sidebar
    sx = 0.885
    fig.text(sx, 0.855, "Performance", fontsize=9, color=ACCENT,
             fontweight="bold", ha="left")
    sidebar = [
        ("Final Port",  "${:,.2f}".format(final_port),            PORT_COL),
        ("SPY value",   "${:,.2f}".format(final_spy),             SPY_COL),
        ("Gain vs SPY", "${:+,.2f}".format(gain_usd),
                        PORT_COL if gain_usd >= 0 else NEG_COL),
        ("Active %",    "{:+.2f}%".format(gain_pct),
                        PORT_COL if gain_pct >= 0 else NEG_COL),
        ("Ann. Ret.",   "{:+.2f}%".format(ann_port * 100),  TEXT),
        ("SPY Ann.",    "{:+.2f}%".format(ann_spy  * 100),  TEXT),
        ("Excess Ret.", "{:+.2f}%".format(ann_er   * 100),  PORT_COL if ann_er >= 0 else NEG_COL),
        ("Track. Err.", "{:.2f}%".format(ann_te    * 100),  MUTED),
        ("IR",          "{:.2f}".format(ir),                MUTED),
        ("Sharpe",      "{:.2f}".format(sharpe),            MUTED),
        ("Max DD",      "{:.2f}%".format(max_dd    * 100),  NEG_COL),
        ("Annual fee",  "{:.2f}%".format(annual_fee * 100), MUTED),
    ]
    for i, (lbl, val, col) in enumerate(sidebar):
        y = 0.810 - i * 0.041
        fig.text(sx,        y, lbl, fontsize=8.5, color=MUTED, ha="left")
        fig.text(sx + 0.06, y, val, fontsize=8.5, color=col,   ha="left")

    if chart_path is None:
        chart_path = DATA_DIR / "simple_simulation_chart.png"
    chart_path = Path(chart_path)
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight", facecolor=BG)
    print("  Chart saved    : {}".format(chart_path))
    plt.close()


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices = pd.read_csv(
        DATA_DIR / "sp500_closing_prices.csv",
        index_col=0, parse_dates=True)

    ANNUAL_FEE = 0.0009

    # Pass the full DataFrame — SPY column is extracted automatically
    port = simulate_return_with_TE_ER(
        spy_closing_prices = prices,
        target_annual_te   = 0.017,    # 1.7% tracking error
        target_annual_er   = 0.020,    # +2.0% net excess return
        annual_fee         = ANNUAL_FEE,
    )

    spy = prices["SPY"]
    print()
    stats = validate(port, spy, annual_fee=ANNUAL_FEE)

    print()
    plot_simulation(port, spy, annual_fee=ANNUAL_FEE)
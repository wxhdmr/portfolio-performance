"""
Portfolio Statistics Analysis
==============================
Computes and compares tracking error, return, and risk metrics for:
  - NPORT portfolio  (SPY quarterly weights, no tilt)
  - Frec portfolio   (NVDA / GOOG / GOOGL / AVGO +1.5 pp tilt)
  - SPY benchmark

Inputs (from data/)
  sp500_weights_daily.csv   -- daily NPORT weights
  frec_weights_daily.csv    -- daily Frec weights
  sp500_closing_prices.csv  -- adjusted close prices (includes SPY)

Outputs (printed + saved to data/)
  portfolio_stats_summary.csv   -- one row per portfolio, all metrics
  portfolio_stats_quarterly.csv -- quarterly breakdown for both portfolios

Run from project root:
    py src/portfolio_stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_metrics import compute_returns, compute_stats, quarterly_breakdown

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR  = Path(__file__).parent.parent / "data"
RISK_FREE = 0.045   # annual risk-free rate for Sharpe


# ── Printing ──────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


def print_comparison(nport: dict, frec: dict) -> None:
    spy_label = "SPY"
    rows = [
        ("Ann. Return",      "{:>+8.4f}%",  "ann_return",  True),
        ("Active Return/yr", "{:>+8.4f}%",  "ann_active",  True),
        ("Tracking Error",   "{:>8.4f}%",   "te",          True),
        ("IR",               "{:>8.3f}",    "ir",          False),
        ("Sharpe",           "{:>8.3f}",    "sharpe",      False),
        ("Volatility",       "{:>8.4f}%",   "vol",         True),
        ("Beta vs SPY",      "{:>8.4f}",    "beta",        False),
        ("Corr vs SPY",      "{:>8.4f}",    "corr_spy",    False),
        ("Max Drawdown",     "{:>8.4f}%",   "max_dd",      True),
        ("Full-period Ret.", "{:>+8.2f}%",  "full_period", True),
    ]

    print_header("Side-by-Side Metrics")
    print(f"  Period: {nport['start']} to {nport['end']}  ({nport['n_days']} trading days)")
    print()
    print(f"  {'Metric':<22} {'NPORT':>14} {'Frec (+1.5pp)':>16} {'SPY':>10}")
    print(f"  {'-'*22} {'-'*14} {'-'*16} {'-'*10}")

    for label, fmt, key, pct_scale in rows:
        nv = nport[key]
        fv = frec[key]
        # SPY-equivalent column
        if key == "ann_return":
            sv_str = fmt.format(nport["ann_spy"] * (100 if pct_scale else 1))
        elif key == "sharpe":
            sv_str = fmt.format(nport["spy_sharpe"])
        elif key == "max_dd":
            sv_str = fmt.format(nport["max_dd_spy"] * (100 if pct_scale else 1))
        elif key == "full_period":
            sv_str = fmt.format(nport["full_spy"] * 100)
        elif key == "vol":
            sv_str = fmt.format(nport["ann_spy"] * 0 + nport["spy_sharpe"] * 0 +
                                 # recalculate SPY vol from frec dict
                                 frec.get("spy_vol", nport["ann_spy"]) * 0)
            sv_str = "      —   "
        else:
            sv_str = "      —   "

        scale = 100 if pct_scale else 1
        print(f"  {label:<22} {fmt.format(nv * scale):>14} {fmt.format(fv * scale):>16} {sv_str:>10}")


def print_quarterly(nport_q: pd.DataFrame, frec_q: pd.DataFrame) -> None:
    print_header("Quarterly Breakdown")
    print(f"  {'Quarter':<12} {'NPORT':>8} {'Frec':>8} {'SPY':>8} "
          f"{'NPORT Act':>10} {'Frec Act':>10} {'NPORT TE':>10} {'Frec TE':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    all_quarters = sorted(set(nport_q.index) | set(frec_q.index))
    for q in all_quarters:
        n = nport_q.loc[q] if q in nport_q.index else None
        f = frec_q.loc[q]  if q in frec_q.index  else None
        np_r  = n["port_return"]    * 100 if n is not None else float("nan")
        fr_r  = f["port_return"]    * 100 if f is not None else float("nan")
        spy_r = (n["spy_return"] if n is not None else
                 f["spy_return"] if f is not None else float("nan")) * 100
        np_a  = n["active_return"]  * 100 if n is not None else float("nan")
        fr_a  = f["active_return"]  * 100 if f is not None else float("nan")
        np_te = n["tracking_error"] * 100 if n is not None else float("nan")
        fr_te = f["tracking_error"] * 100 if f is not None else float("nan")

        print(f"  {q:<12} {np_r:>+7.2f}% {fr_r:>+7.2f}% {spy_r:>+7.2f}% "
              f"{np_a:>+9.3f}% {fr_a:>+9.3f}% {np_te:>9.4f}% {fr_te:>9.4f}%")


def print_full_summary(nport: dict, frec: dict) -> None:
    def row(lbl, v, pct=False):
        if isinstance(v, float) and np.isnan(v):
            print(f"  {lbl:<24}: {'—':>10}")
        elif pct:
            print(f"  {lbl:<24}: {v*100:>+9.4f}%")
        else:
            print(f"  {lbl:<24}: {v:>10.4f}")

    print_header("Full Summary")
    print(f"  {'Period':<24}: {nport['start']} to {nport['end']}")
    print(f"  {'Trading days':<24}: {nport['n_days']}")

    for name, d in [("NPORT portfolio", nport), ("Frec portfolio", frec)]:
        print(f"\n  [{name}]")
        row("Ann. Return",      d["ann_return"],  True)
        row("Active Return/yr", d["ann_active"],  True)
        row("Tracking Error",   d["te"],          True)
        row("IR",               d["ir"],          False)
        row("Sharpe",           d["sharpe"],      False)
        row("Volatility",       d["vol"],         True)
        row("Beta vs SPY",      d["beta"],        False)
        row("Corr vs SPY",      d["corr_spy"],    False)
        row("Max Drawdown",     d["max_dd"],      True)
        row("Full-period Ret.", d["full_period"], True)

    print(f"\n  [SPY benchmark]")
    row("Ann. Return",     nport["ann_spy"],    True)
    row("Sharpe",          nport["spy_sharpe"], False)
    row("Max Drawdown",    nport["max_dd_spy"], True)
    row("Full-period Ret.",nport["full_spy"],   True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data ...")
    prices       = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                               index_col=0, parse_dates=True)
    nport_w      = pd.read_csv(DATA_DIR / "sp500_weights_daily.csv",
                               index_col=0, parse_dates=True)
    frec_w       = pd.read_csv(DATA_DIR / "frec_weights_daily.csv",
                               index_col=0, parse_dates=True)

    print(f"  Prices    : {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"  NPORT wts : {nport_w.shape[0]} days x {nport_w.shape[1]} tickers")
    print(f"  Frec wts  : {frec_w.shape[0]} days x {frec_w.shape[1]} tickers")

    # ── Compute returns ───────────────────────────────────────────────────────
    print("\nComputing portfolio returns ...")
    nport_ret = compute_returns(nport_w, prices)
    frec_ret  = compute_returns(frec_w,  prices)
    print(f"  NPORT : {len(nport_ret)} return days")
    print(f"  Frec  : {len(frec_ret)}  return days")

    # ── Compute stats ─────────────────────────────────────────────────────────
    nport_stats = compute_stats(nport_ret, risk_free=RISK_FREE)
    frec_stats  = compute_stats(frec_ret,  risk_free=RISK_FREE)

    # ── Print outputs ─────────────────────────────────────────────────────────
    print_header("Portfolio Performance Analysis  —  NPORT vs Frec vs SPY")
    print(f"  NPORT: S&P 500 quarterly NPORT-P weights, no tilt")
    print(f"  Frec : NVDA / GOOG / GOOGL / AVGO  +1.5 pp overweight, rescaled")
    print(f"  SPY  : benchmark (buy-and-hold ETF)")

    print_full_summary(nport_stats, frec_stats)
    print_comparison(nport_stats, frec_stats)

    # ── Quarterly breakdown ───────────────────────────────────────────────────
    nport_q = quarterly_breakdown(nport_ret)
    frec_q  = quarterly_breakdown(frec_ret)
    print_quarterly(nport_q, frec_q)

    # ── Rolling 252-day TE ────────────────────────────────────────────────────
    print_header("Rolling 252-Day Tracking Error (annualised)")
    roll_n = nport_ret["active_return"].rolling(252).std() * np.sqrt(252) * 100
    roll_f = frec_ret["active_return"].rolling(252).std()  * np.sqrt(252) * 100
    roll_combined = pd.DataFrame({"NPORT_TE": roll_n, "Frec_TE": roll_f}).dropna()
    print(f"\n  {'Date':<12} {'NPORT TE':>12} {'Frec TE':>12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    step = max(1, len(roll_combined) // 12)
    for dt, row in roll_combined.iloc[::step].iterrows():
        print(f"  {str(dt.date()):<12} {row['NPORT_TE']:>11.4f}%  {row['Frec_TE']:>11.4f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    # Summary CSV
    summary_rows = []
    for label, d in [("NPORT", nport_stats), ("Frec", frec_stats),
                     ("SPY",   {**nport_stats,
                                "label": "SPY",
                                "ann_return": nport_stats["ann_spy"],
                                "ann_active": 0.0,
                                "te": 0.0, "ir": 0.0,
                                "sharpe": nport_stats["spy_sharpe"],
                                "vol": 0.0, "beta": 1.0, "corr_spy": 1.0,
                                "max_dd": nport_stats["max_dd_spy"],
                                "full_period": nport_stats["full_spy"]})]:
        summary_rows.append({
            "portfolio":    label,
            "start":        d["start"],
            "end":          d["end"],
            "ann_return_%": round(d["ann_return"] * 100, 6),
            "ann_active_%": round(d["ann_active"] * 100, 6),
            "tracking_error_%": round(d["te"] * 100, 6),
            "ir":           round(d["ir"], 6) if not np.isnan(d["ir"]) else "",
            "sharpe":       round(d["sharpe"], 6),
            "volatility_%": round(d["vol"] * 100, 6),
            "beta":         round(d["beta"], 6),
            "corr_spy":     round(d["corr_spy"], 6),
            "max_drawdown_%": round(d["max_dd"] * 100, 6),
            "full_period_%":  round(d["full_period"] * 100, 4),
        })

    summary_df = pd.DataFrame(summary_rows).set_index("portfolio")
    out_summary = DATA_DIR / "portfolio_stats_summary.csv"
    summary_df.to_csv(out_summary)

    # Quarterly CSV (merge NPORT and Frec)
    nport_q.columns = ["nport_" + c for c in nport_q.columns]
    frec_q.columns  = ["frec_"  + c for c in frec_q.columns]
    qdf = nport_q.join(frec_q, how="outer")
    out_quarterly = DATA_DIR / "portfolio_stats_quarterly.csv"
    qdf.to_csv(out_quarterly)

    print(f"\n\nSaved -> {out_summary.name}")
    print(f"Saved -> {out_quarterly.name}")


if __name__ == "__main__":
    main()

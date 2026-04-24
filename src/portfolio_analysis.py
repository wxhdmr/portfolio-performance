"""
Portfolio analysis: equal-weight S&P 500 baseline vs. IT-overweight portfolio.

Construction
------------
Baseline : 1/N equal weight across all 501 constituents
IT-overweight:
  1. Start from equal weight
  2. Add +10pp total to IT sector (distributed equally across IT stocks)
  3. Rescale all weights to sum to 100%

Metrics reported (both portfolios vs. SPY)
------------------------------------------
- Annual return (geometric)
- Annual volatility
- Sharpe ratio
- Realised tracking error vs SPY
- Active return vs SPY
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
ANNUALISE = np.sqrt(252)


def build_portfolios(prices: pd.DataFrame, constituents: pd.DataFrame):
    stocks    = [c for c in prices.columns if c != "SPY"]
    N         = len(stocks)
    it_stocks = set(
        constituents[constituents["sector"] == "Information Technology"]["ticker"]
    ) & set(stocks)
    n_it      = len(it_stocks)

    # ── Baseline: equal weight ────────────────────────────────────────────────
    w_base = pd.Series(1 / N, index=stocks)

    # ── IT-overweight ─────────────────────────────────────────────────────────
    # Step 1: start from equal weight
    w_tilt = w_base.copy()
    # Step 2: add 10pp distributed equally across IT stocks
    w_tilt[list(it_stocks)] += 0.10 / n_it
    # Step 3: rescale to sum = 1
    w_tilt /= w_tilt.sum()

    # Verify
    it_weight_base = w_base[list(it_stocks)].sum()
    it_weight_tilt = w_tilt[list(it_stocks)].sum()

    print(f"Universe        : {N} stocks  ({n_it} IT stocks)")
    print(f"\nBaseline  IT weight : {it_weight_base*100:.2f}%  (equal-weight share)")
    print(f"Overweight IT weight: {it_weight_tilt*100:.2f}%  (after +10pp & rescale)")
    print(f"Weight sum check    : baseline={w_base.sum():.6f}  tilt={w_tilt.sum():.6f}")

    return w_base, w_tilt, it_stocks


def compute_metrics(label: str, weights: pd.Series,
                    returns: pd.DataFrame, spy_ret: pd.Series) -> dict:
    port_ret  = returns[weights.index].multiply(weights, axis=1).sum(axis=1)
    active    = port_ret - spy_ret
    common    = port_ret.index.intersection(spy_ret.index)
    port_ret  = port_ret.loc[common]
    active    = active.loc[common]

    n_days    = len(port_ret)
    cum_ret   = (1 + port_ret).prod() - 1
    ann_ret   = (1 + cum_ret) ** (252 / n_days) - 1
    ann_vol   = port_ret.std() * ANNUALISE
    sharpe    = ann_ret / ann_vol
    te        = active.std() * ANNUALISE          # realised TE
    act_ret   = (1 + active).prod() - 1
    ann_act   = (1 + act_ret) ** (252 / n_days) - 1

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Annual return           : {ann_ret*100:+.2f}%")
    print(f"  Annual volatility       : {ann_vol*100:.2f}%")
    print(f"  Sharpe ratio            : {sharpe:.2f}")
    print(f"  Tracking error vs SPY   : {te*100:.2f}%")
    print(f"  Active return vs SPY    : {ann_act*100:+.2f}%/yr")

    return {
        "portfolio":        label,
        "ann_return_%":     round(ann_ret * 100, 2),
        "ann_vol_%":        round(ann_vol * 100, 2),
        "sharpe":           round(sharpe, 2),
        "tracking_error_%": round(te * 100, 2),
        "active_return_%":  round(ann_act * 100, 2),
    }


def main():
    print("=" * 55)
    print("Portfolio Analysis: Equal-weight vs IT-Overweight")
    print("=" * 55)

    prices       = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                               index_col=0, parse_dates=True)
    constituents = pd.read_csv(DATA_DIR / "constituents.csv")

    # Daily returns (drop first row which is NaN after pct_change)
    all_returns = prices.pct_change().dropna(how="all")
    spy_ret     = all_returns["SPY"]
    stock_ret   = all_returns.drop(columns=["SPY"])

    w_base, w_tilt, it_stocks = build_portfolios(prices, constituents)

    # SPY benchmark stats
    spy_cum    = (1 + spy_ret).prod() - 1
    spy_ann    = (1 + spy_cum) ** (252 / len(spy_ret)) - 1
    spy_vol    = spy_ret.std() * ANNUALISE
    print(f"\n--- SPY Benchmark ---")
    print(f"  Annual return : {spy_ann*100:+.2f}%")
    print(f"  Annual vol    : {spy_vol*100:.2f}%")
    print(f"  Sharpe        : {spy_ann/spy_vol:.2f}")

    r1 = compute_metrics("Equal-Weight S&P 500",    w_base, stock_ret, spy_ret)
    r2 = compute_metrics("IT-Overweight Portfolio",  w_tilt, stock_ret, spy_ret)

    # Weight difference table: show IT sector shift
    print(f"\n--- Weight changes (IT sector, top 10 by delta) ---")
    delta = (w_tilt - w_base) * 100
    it_delta = delta[list(it_stocks)].sort_values(ascending=False)
    print(f"  Each IT stock: {it_delta.mean():+.4f}pp  "
          f"(IT total: {it_delta.sum():+.2f}pp after rescaling)")

    # Save results
    results = pd.DataFrame([r1, r2])
    results.to_csv(DATA_DIR / "portfolio_comparison.csv", index=False)
    print(f"\nSaved -> data/portfolio_comparison.csv")


if __name__ == "__main__":
    main()

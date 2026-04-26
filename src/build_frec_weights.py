"""
Build Frec Portfolio: tilted quarterly weights -> daily weights -> performance.

Strategy
--------
Start from the SPY NPORT-P quarterly weights, then for each quarter:
  1. Add TILT_PP percentage points to each of the TILT_STOCKS
  2. Rescale all weights so the quarter sums to exactly 100%
  3. Back-calculate implied share counts from the tilted weights and
     quarter-end prices (same method as reconcile_weights.py)
  4. Propagate daily weights by marking those shares to market each day

The tilt is applied at the quarterly rebalance level, so price drift
between rebalances is captured naturally — just like the base portfolio.

Outputs
-------
data/frec_weights_daily.csv   — Date × Ticker matrix of daily weights (%)
data/frec_portfolio_returns.csv — daily port / SPY / active returns + metrics
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Tilt configuration ────────────────────────────────────────────────────────
TILT_STOCKS = ["NVDA", "GOOG", "GOOGL", "AVGO"]
TILT_PP     = 1.5        # percentage points added to each stock before rescaling
RISK_FREE   = 0.045      # annual risk-free rate for Sharpe / IR


# ── Helpers (shared with reconcile_weights.py) ────────────────────────────────

def load_quarterly_weights() -> pd.DataFrame:
    """
    Load raw NPORT-P per-quarter CSVs and return a wide DataFrame
    (index = quarter-end date, columns = tickers, values = weight %).
    Normalises ticker formats to match Yahoo Finance (BRK/B -> BRK-B).
    """
    weight_dir = DATA_DIR / "sp500_weights"
    frames = {}
    for f in sorted(weight_dir.glob("*.csv")):
        qdate = pd.Timestamp(f.stem).date()
        df    = pd.read_csv(f)
        df["ticker"] = (df["ticker"]
                        .str.replace("/", "-", regex=False)
                        .str.replace(".", "-", regex=False))
        s = df.set_index("ticker")["weight_%"]
        frames[qdate] = s.groupby(level=0).sum()

    wide = pd.DataFrame(frames).T
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "quarter_end"
    return wide


def apply_tilt(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    For every quarter, add TILT_PP to each TILT_STOCK then rescale
    all weights so each row sums to 100%.
    Returns a new DataFrame with the same shape.
    """
    tilted = quarterly.copy()
    for tkr in TILT_STOCKS:
        if tkr in tilted.columns:
            tilted[tkr] = tilted[tkr] + TILT_PP

    # Rescale each quarter to 100%
    row_sums = tilted.sum(axis=1)
    tilted   = tilted.div(row_sums, axis=0) * 100
    return tilted


def nearest_price_date(prices: pd.DataFrame, target: pd.Timestamp) -> pd.Timestamp:
    idx    = prices.index
    future = idx[idx >= target]
    past   = idx[idx <= target]
    if len(future):
        return future[0]
    return past[-1]


def build_daily_weights(
    prices: pd.DataFrame,
    quarterly_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Implied-share method: for each quarter compute shares ∝ weight / price,
    then value those shares daily and normalise to get daily weights.
    """
    price_tickers   = [c for c in prices.columns if c != "SPY"]
    quarter_tickers = quarterly_weights.columns.tolist()
    tickers         = [t for t in price_tickers if t in quarter_tickers]
    prices_eq       = prices[tickers]

    print(f"  Tickers in price data and Frec weights : {len(tickers)}")
    print(f"  Price range : {prices.index[0].date()} - {prices.index[-1].date()}")

    quarters       = quarterly_weights.index.sort_values()
    quarter_shares = {}

    for q in quarters:
        price_date = nearest_price_date(prices_eq, q)
        prices_q   = prices_eq.loc[price_date]
        weights_q  = quarterly_weights.loc[q].reindex(tickers).fillna(0)

        valid    = (prices_q > 0) & (weights_q > 0) & prices_q.notna()
        shares_q = pd.Series(
            np.where(valid, weights_q / prices_q, 0), index=tickers)
        quarter_shares[q] = shares_q
        print(f"  Quarter {q.date()} (priced {price_date.date()}): "
              f"{valid.sum()} stocks")

    daily_w_list = []
    for day in prices_eq.index:
        active_q = quarters[quarters <= day]
        q        = active_q[-1] if len(active_q) else quarters[0]
        mkt_val  = quarter_shares[q] * prices_eq.loc[day]
        total    = mkt_val.sum()
        weights  = mkt_val / total * 100 if total > 0 else pd.Series(0.0, index=tickers)
        daily_w_list.append(weights)

    daily = pd.DataFrame(daily_w_list, index=prices_eq.index)
    daily.index.name = "Date"
    return daily


# ── Performance ───────────────────────────────────────────────────────────────

def compute_performance(
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily portfolio returns, SPY returns, and active returns.
    Uses lagged weights so today's return reflects yesterday's positions.
    """
    spy_ret   = prices["SPY"].pct_change().dropna()
    price_ret = prices.pct_change().dropna()
    tickers   = [c for c in daily_weights.columns if c in price_ret.columns]

    common    = daily_weights.index.intersection(price_ret.index)
    w_lag     = daily_weights[tickers].shift(1).loc[common] / 100
    r_day     = price_ret[tickers].loc[common]
    spy_day   = spy_ret.reindex(common)

    port_ret   = (w_lag * r_day).sum(axis=1)
    active_ret = port_ret - spy_day

    return pd.DataFrame({
        "port_return":   port_ret,
        "spy_return":    spy_day,
        "active_return": active_ret,
    })


def print_metrics(returns: pd.DataFrame, label: str = "Frec Portfolio") -> None:
    port = returns["port_return"]
    spy  = returns["spy_return"]
    act  = returns["active_return"]
    n    = len(port)

    ann_port   = (1 + port).prod() ** (252 / n) - 1
    ann_spy    = (1 + spy).prod()  ** (252 / n) - 1
    ann_active = (1 + act).prod()  ** (252 / n) - 1
    te         = act.std() * np.sqrt(252)
    ir         = ann_active / te
    sharpe     = (ann_port - RISK_FREE) / (port.std() * np.sqrt(252))
    full_port  = (1 + port).prod() - 1
    full_spy   = (1 + spy).prod()  - 1

    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"  Tilt: {', '.join(TILT_STOCKS)}  +{TILT_PP}pp each (before rescaling)")
    print(f"{'='*58}")
    print(f"  Period          : {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"  Trading days    : {n}")
    print(f"  Ann. Return     : {ann_port*100:>+7.4f}%  (SPY: {ann_spy*100:>+7.4f}%)")
    print(f"  Active Return   : {ann_active*100:>+7.4f}%/yr")
    print(f"  Tracking Error  : {te*100:>7.4f}%  annualised")
    print(f"  IR              : {ir:>7.3f}")
    print(f"  Sharpe          : {sharpe:>7.3f}")
    print(f"  Full-period     : {full_port*100:>+7.2f}%  (SPY: {full_spy*100:>+7.2f}%)")
    print(f"  Full-period act : {(full_port-full_spy)*100:>+7.2f}%")

    print(f"\n  Quarterly breakdown:")
    print(f"  {'Quarter':<14} {'Portfolio':>10} {'SPY':>10} {'Active':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    for qname, grp in returns.resample("Q"):
        qp = (1 + grp["port_return"]).prod() - 1
        qs = (1 + grp["spy_return"]).prod() - 1
        qa = qp - qs
        print(f"  {str(qname.date()):<14} {qp*100:>+9.2f}%  {qs*100:>+9.2f}%  {qa*100:>+9.3f}%")

    print(f"\n  Net delta after rescaling (avg across all quarters):")
    print(f"  {'Ticker':<8} {'Base avg':>10} {'Tilted avg':>12} {'Net delta':>10}")


def print_tilt_summary(base: pd.DataFrame, tilted: pd.DataFrame) -> None:
    """Show average weight before/after tilt for the tilt stocks."""
    print(f"  {'Ticker':<8} {'Base avg%':>10} {'Tilted avg%':>12} {'Net delta':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10}")
    for tkr in TILT_STOCKS:
        if tkr in base.columns and tkr in tilted.columns:
            b = base[tkr].mean()
            t = tilted[tkr].mean()
            print(f"  {tkr:<8} {b:>10.3f}%  {t:>12.3f}%  {t-b:>+9.3f}pp")
    other_base   = base.drop(columns=TILT_STOCKS, errors="ignore").sum(axis=1).mean()
    other_tilted = tilted.drop(columns=TILT_STOCKS, errors="ignore").sum(axis=1).mean()
    print(f"  {'All others':<8} {other_base:>10.3f}%  {other_tilted:>12.3f}%  "
          f"{other_tilted-other_base:>+9.3f}pp")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Build Frec Portfolio weights -> daily -> performance")
    print("=" * 60)

    # 1. Load base NPORT quarterly weights
    print("\nLoading quarterly NPORT-P weights ...")
    quarterly_base = load_quarterly_weights()
    print(f"  {len(quarterly_base)} quarters: "
          f"{quarterly_base.index[0].date()} to {quarterly_base.index[-1].date()}")

    # 2. Apply tilt
    print(f"\nApplying +{TILT_PP}pp tilt to {TILT_STOCKS} and rescaling ...")
    quarterly_frec = apply_tilt(quarterly_base)

    # 3. Verify tilt
    print("\nQuarterly weight comparison (base vs Frec):")
    print_tilt_summary(quarterly_base, quarterly_frec)

    # 4. Load prices
    print("\nLoading daily prices ...")
    prices = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                         index_col=0, parse_dates=True)

    # 5. Build daily Frec weights
    print("\nBuilding Frec daily weights ...")
    frec_daily = build_daily_weights(prices, quarterly_frec)

    row_sums = frec_daily.sum(axis=1)
    print(f"\n  Weight sum check: min={row_sums.min():.4f}%  "
          f"max={row_sums.max():.4f}%  mean={row_sums.mean():.4f}%")

    out_weights = DATA_DIR / "frec_weights_daily.csv"
    frec_daily.to_csv(out_weights)
    print(f"  Saved -> {out_weights}  "
          f"({frec_daily.shape[0]} days x {frec_daily.shape[1]} tickers)")

    # 6. Compute and print performance
    print("\nComputing performance vs SPY ...")
    returns = compute_performance(frec_daily, prices)
    print_metrics(returns, label="Frec Portfolio vs SPY")

    # 7. Save returns
    out_ret = DATA_DIR / "frec_portfolio_returns.csv"
    returns.to_csv(out_ret)
    print(f"\n  Saved -> {out_ret}")

if __name__ == "__main__":
    main()

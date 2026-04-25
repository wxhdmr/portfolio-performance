"""
Reconcile quarterly NPORT-P weights into daily market-cap weights.

Problem
-------
Raw NPORT-P pctVal figures sum to ~95-97% because SPY also holds cash,
futures, and securities-lending collateral. These are not stocks we can
price, so equity weights don't sum to 100%.

Method
------
For each quarter q and stock i:
  1. Back-calculate implied share count from NPORT weight and quarter-end price:
        implied_shares_i,q  ∝  weight_i,q (%) / price_i,q

  2. For every trading day t within that quarter's period, revalue:
        market_value_i(t) = implied_shares_i,q × price_i(t)

  3. Normalise across the equity universe:
        weight_i(t) = market_value_i(t) / Σ_j  market_value_j(t)

  Weights now sum to exactly 100% for the equity-only universe and update
  daily as prices drift — exactly how S&P 500 weights actually behave
  between quarterly rebalances.

Rebalance schedule
------------------
The implied share count is reset each time a new NPORT-P quarter-end date
falls within our price history.

Outputs
-------
data/sp500_weights_daily.csv  — Date × Ticker matrix of daily weights (%)
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_quarterly_weights() -> pd.DataFrame:
    """
    Load all per-quarter weight CSVs from data/sp500_weights/.
    Returns a wide DataFrame indexed by quarter-end date.
    Normalises ticker formats to match Yahoo Finance (BRK/B → BRK-B).
    """
    weight_dir = DATA_DIR / "sp500_weights"
    frames = {}
    for f in sorted(weight_dir.glob("*.csv")):
        qdate = pd.Timestamp(f.stem).date()
        df    = pd.read_csv(f)
        df["ticker"] = df["ticker"].str.replace("/", "-", regex=False) \
                                   .str.replace(".", "-", regex=False)
        s = df.set_index("ticker")["weight_%"]
        # Sum duplicate tickers (e.g. after normalisation BRK/B and BRK-B map to same)
        frames[qdate] = s.groupby(level=0).sum()

    wide = pd.DataFrame(frames).T          # shape: quarters × tickers
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "quarter_end"
    return wide


def nearest_price_date(prices: pd.DataFrame, target: pd.Timestamp) -> pd.Timestamp:
    """Return the closest available trading date to `target`."""
    idx = prices.index
    if target in idx:
        return target
    # Find nearest future date first, then nearest past
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
    For each trading day, compute market-cap weights using the most recent
    quarterly implied share counts.
    """
    # Common tickers between our prices and the NPORT data (exclude SPY)
    price_tickers   = [c for c in prices.columns if c != "SPY"]
    quarter_tickers = quarterly_weights.columns.tolist()
    tickers         = [t for t in price_tickers if t in quarter_tickers]

    prices_eq = prices[tickers]   # equity prices only

    print(f"  Tickers in both price data and NPORT : {len(tickers)}")
    print(f"  Price data range : {prices.index[0].date()} - {prices.index[-1].date()}")

    # Sort quarters
    quarters = quarterly_weights.index.sort_values()

    # For each quarter, compute implied shares at that quarter-end price
    # implied_shares ∝ weight% / price  (NAV cancels out in normalisation)
    quarter_shares: dict[pd.Timestamp, pd.Series] = {}
    for q in quarters:
        price_date = nearest_price_date(prices_eq, q)
        prices_q   = prices_eq.loc[price_date]
        weights_q  = quarterly_weights.loc[q].reindex(tickers).fillna(0)

        # Only include stocks with valid price and positive weight
        valid    = (prices_q > 0) & (weights_q > 0) & prices_q.notna()
        shares_q = pd.Series(np.where(valid, weights_q / prices_q, 0),
                             index=tickers)
        quarter_shares[q] = shares_q
        print(f"  Quarter {q.date()} (priced at {price_date.date()}): "
              f"{valid.sum()} stocks with valid weight+price")

    # Build daily weights
    trading_days = prices_eq.index
    daily_w_list = []

    for day in trading_days:
        # Use the most recent quarter whose end-date <= today
        active_q = quarters[quarters <= day]
        if len(active_q) == 0:
            # Before any available quarter — use the earliest
            active_q = quarters[:1]
        q = active_q[-1]

        shares = quarter_shares[q]
        prices_today = prices_eq.loc[day]

        mkt_val = shares * prices_today
        total   = mkt_val.sum()
        if total > 0:
            weights = mkt_val / total * 100   # in %
        else:
            weights = pd.Series(0.0, index=tickers)

        daily_w_list.append(weights)

    daily_weights = pd.DataFrame(daily_w_list, index=trading_days)
    daily_weights.index.name = "Date"
    return daily_weights


def print_summary(daily_weights: pd.DataFrame, quarterly_raw: pd.DataFrame) -> None:
    print("\n--- Reconciliation check ---")
    print(f"  Shape            : {daily_weights.shape[0]} days x "
          f"{daily_weights.shape[1]} tickers")
    print(f"  Weight sum stats :")
    row_sums = daily_weights.sum(axis=1)
    print(f"    Min  : {row_sums.min():.4f}%")
    print(f"    Max  : {row_sums.max():.4f}%")
    print(f"    Mean : {row_sums.mean():.4f}%")

    # Compare raw NPORT vs reconciled weights for latest quarter
    latest_q   = quarterly_raw.index[-1]
    latest_day = daily_weights.index[daily_weights.index <= latest_q]
    if len(latest_day) == 0:
        return
    latest_day = latest_day[-1]

    raw_top10  = quarterly_raw.loc[latest_q].nlargest(10)
    rec_top10  = daily_weights.loc[latest_day].nlargest(10)

    print(f"\n  Top 10 comparison at {latest_q.date()} "
          f"(raw NPORT  vs  reconciled):")
    print(f"  {'Ticker':<8} {'NPORT%':>8}  {'Reconciled%':>12}")
    print(f"  {'-'*8} {'-'*8}  {'-'*12}")
    all_top = sorted(set(raw_top10.index) | set(rec_top10.index))
    for t in all_top:
        r = raw_top10.get(t, 0)
        c = rec_top10.get(t, 0)
        print(f"  {t:<8} {r:>8.3f}%  {c:>12.3f}%")

    print(f"\n  Raw NPORT total (equity only)  : "
          f"{quarterly_raw.loc[latest_q].sum():.2f}%")
    print(f"  Reconciled total               : "
          f"{daily_weights.loc[latest_day].sum():.4f}%")


def main() -> None:
    print("=" * 60)
    print("Reconcile NPORT-P weights -> daily market-cap weights")
    print("=" * 60)

    print("\nLoading quarterly NPORT-P weights ...")
    quarterly = load_quarterly_weights()
    print(f"  {len(quarterly)} quarters: "
          f"{quarterly.index[0].date()} to {quarterly.index[-1].date()}")

    print("\nLoading daily prices ...")
    prices = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                         index_col=0, parse_dates=True)

    print("\nBuilding daily weights ...")
    daily = build_daily_weights(prices, quarterly)

    print_summary(daily, quarterly)

    out_path = DATA_DIR / "sp500_weights_daily.csv"
    daily.to_csv(out_path)
    print(f"\nSaved -> {out_path}  "
          f"({daily.shape[0]} days x {daily.shape[1]} tickers)")


if __name__ == "__main__":
    main()

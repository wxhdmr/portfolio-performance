"""
Fetch 2 years of daily closing prices for SPY + all S&P 500 constituents.

Data sources
------------
- Constituent list : Wikipedia "List of S&P 500 companies"
- Prices           : Yahoo Finance via yfinance (auto-adjusted closes)

Output
------
data/sp500_closing_prices.csv   — Date × Ticker wide table
data/constituents.csv           — ticker / company / sector metadata
data/constituents_quality_report.txt
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
END_DATE   = date.today()
START_DATE = END_DATE - timedelta(days=2 * 365)
OUTPUT_DIR = Path(__file__).parent.parent / "data"
WIKI_URL   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Quality thresholds
MAX_DAILY_MOVE_PCT   = 0.20   # flag moves > ±20 % (looser for individual stocks)
MIN_PRICE            = 0.50   # below this is almost certainly corrupt
MAX_MISSING_PCT      = 0.10   # drop a ticker if >10 % of trading days are missing


# ── Step 1: constituent list ──────────────────────────────────────────────────

def fetch_constituent_list() -> pd.DataFrame:
    import io
    import requests

    print("Fetching S&P 500 constituent list from Wikipedia …")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), header=0)
    df = tables[0]  # first table = current constituents

    df = df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["ticker", "company", "sector", "sub_industry"]

    # Yahoo Finance uses "-" where S&P uses "." (e.g. BRK.B → BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    print(f"  Found {len(df)} constituents.")

    sector_counts = df["sector"].value_counts()
    print("\n  Sector breakdown:")
    for sector, count in sector_counts.items():
        print(f"    {sector:<45s} {count:>3d} stocks  ({count/len(df)*100:.1f}%)")

    return df


# ── Step 2: batch price download ──────────────────────────────────────────────

def download_prices(tickers: list[str]) -> pd.DataFrame:
    all_tickers = ["SPY"] + [t for t in tickers if t != "SPY"]
    print(f"\nDownloading prices for {len(all_tickers)} tickers "
          f"({START_DATE} → {END_DATE}) …")
    print("  This may take 1–3 minutes for the full universe.")

    raw = yf.download(
        all_tickers,
        start=str(START_DATE),
        end=str(END_DATE),
        progress=True,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    return raw


def extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull Close column for every ticker into a single wide DataFrame."""
    all_tickers = ["SPY"] + [t for t in tickers if t != "SPY"]
    frames = {}
    for ticker in all_tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                col = raw[ticker]["Close"]
            else:
                col = raw["Close"]
            frames[ticker] = col
        except KeyError:
            pass  # handled in quality check
    return pd.DataFrame(frames)


# ── Step 3: quality checks ────────────────────────────────────────────────────

def quality_check(prices: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    def log(*lines):
        for ln in lines:
            print(ln)
            report.append(ln)

    log("", "=" * 65, "DATA QUALITY REPORT — S&P 500 CONSTITUENTS", "=" * 65)
    log(f"Tickers attempted   : {prices.shape[1]}",
        f"Trading days        : {len(prices)}",
        f"Date range          : {prices.index.min().date()} → {prices.index.max().date()}")

    # 1. Flatten MultiIndex index (timezone-aware → date)
    prices.index = pd.to_datetime(prices.index).normalize()

    # 2. Duplicate dates
    dupes = prices.index[prices.index.duplicated()]
    log(f"\n[check] Duplicate dates        : {len(dupes)}")
    if len(dupes):
        prices = prices[~prices.index.duplicated(keep="last")]
        log(f"[fix]   Kept last occurrence.")

    # 3. Tickers with all-NaN (download failed entirely)
    all_nan = prices.columns[prices.isna().all()].tolist()
    log(f"\n[check] Tickers with no data   : {len(all_nan)}")
    if all_nan:
        log(f"        {all_nan}")
        log(f"[fix]   Dropped entirely.")
        prices = prices.drop(columns=all_nan)

    # 4. Tickers with too many missing days
    missing_frac = prices.isna().mean()
    too_sparse = missing_frac[missing_frac > MAX_MISSING_PCT].index.tolist()
    log(f"\n[check] Tickers > {MAX_MISSING_PCT*100:.0f}% missing  : {len(too_sparse)}")
    if too_sparse:
        log(f"        {too_sparse}")
        log(f"[fix]   Dropped (likely delisted or recently added).")
        prices = prices.drop(columns=too_sparse)

    # 5. Remaining NaNs — forward-fill within each ticker
    remaining_nans = prices.isna().sum().sum()
    log(f"\n[check] Remaining NaN cells    : {remaining_nans}")
    if remaining_nans:
        prices = prices.ffill()
        log(f"[fix]   Forward-filled (carry last known close).")

    # 6. Prices below minimum threshold
    below_min_mask = prices < MIN_PRICE
    bad_cells = below_min_mask.sum().sum()
    log(f"\n[check] Cells below ${MIN_PRICE} : {bad_cells}")
    if bad_cells:
        tickers_affected = prices.columns[below_min_mask.any()].tolist()
        log(f"        Tickers: {tickers_affected}")
        prices[below_min_mask] = np.nan
        prices = prices.ffill()
        log(f"[fix]   Replaced with NaN then forward-filled.")

    # 7. Large single-day moves per ticker
    returns = prices.pct_change()
    large_mask = returns.abs() > MAX_DAILY_MOVE_PCT
    large_counts = large_mask.sum()
    tickers_with_large = large_counts[large_counts > 0]
    log(f"\n[check] Tickers with any day > ±{MAX_DAILY_MOVE_PCT*100:.0f}% : "
        f"{len(tickers_with_large)}")
    if len(tickers_with_large):
        log(f"        Kept as-is; extreme moves are common for individual stocks "
            f"(earnings, M&A, etc.).")
        for t, cnt in tickers_with_large.items():
            worst_day = returns[t].abs().idxmax()
            worst_ret = returns[t].loc[worst_day]
            log(f"        {t:6s}  {cnt} day(s), worst: {worst_day.date()} {worst_ret:+.1%}")

    # 8. Final summary
    log(f"\n[result] Clean shape : {prices.shape[0]} rows × {prices.shape[1]} tickers")
    log("=" * 65)

    return prices


# ── Step 4: sector returns ───────────────────────────────────────────────────

def compute_sector_data(
    prices: pd.DataFrame, constituents: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute equal-weight daily returns per GICS sector and a sector summary.

    Returns
    -------
    sector_returns  : Date × Sector DataFrame of daily equal-weight returns
    sector_summary  : one row per sector with stock count, weight, and
                      cumulative return over the full period
    """
    # Map ticker → sector (exclude SPY)
    ticker_sector = constituents.set_index("ticker")["sector"].to_dict()

    daily_returns = prices.drop(columns=["SPY"], errors="ignore").pct_change()

    sector_groups: dict[str, list[str]] = {}
    for ticker in daily_returns.columns:
        sector = ticker_sector.get(ticker)
        if sector and sector != "ETF":
            sector_groups.setdefault(sector, []).append(ticker)

    # Equal-weight average return per sector per day
    sector_ret = pd.DataFrame({
        sector: daily_returns[tickers].mean(axis=1)
        for sector, tickers in sector_groups.items()
    })
    sector_ret.index.name = "Date"

    # Summary: count, index weight, cumulative return, annualised volatility
    total_stocks = sum(len(v) for v in sector_groups.values())
    summary_rows = []
    for sector, tickers in sorted(sector_groups.items()):
        s = sector_ret[sector].dropna()
        cum_ret  = (1 + s).prod() - 1
        ann_vol  = s.std() * (252 ** 0.5)        # annualised daily vol
        ann_ret  = (1 + cum_ret) ** (252 / len(s)) - 1  # annualised return
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
        summary_rows.append({
            "sector":             sector,
            "n_stocks":           len(tickers),
            "index_weight_%":     round(len(tickers) / total_stocks * 100, 2),
            "cumulative_ret_%":   round(cum_ret * 100, 2),
            "ann_return_%":       round(ann_ret * 100, 2),
            "ann_volatility_%":   round(ann_vol * 100, 2),
            "sharpe_ratio":       round(sharpe, 2),
        })
    sector_summary = pd.DataFrame(summary_rows).sort_values(
        "ann_volatility_%", ascending=True
    )

    return sector_ret, sector_summary


# ── Step 5: save ──────────────────────────────────────────────────────────────

def save(prices: pd.DataFrame, constituents: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    price_path = OUTPUT_DIR / "sp500_closing_prices.csv"
    prices.index = prices.index.date
    prices.index.name = "Date"
    prices.to_csv(price_path)
    print(f"\nSaved price matrix → {price_path}")

    const_path = OUTPUT_DIR / "constituents.csv"
    # Keep only tickers that survived quality checks
    surviving = set(prices.columns)
    constituents_out = constituents[constituents["ticker"].isin(surviving)]
    # Add SPY row
    spy_row = pd.DataFrame([{
        "ticker": "SPY", "company": "SPDR S&P 500 ETF Trust",
        "sector": "ETF", "sub_industry": "ETF",
    }])
    constituents_out = pd.concat([spy_row, constituents_out], ignore_index=True)
    constituents_out.to_csv(const_path, index=False)
    print(f"Saved constituent metadata → {const_path}")

    sector_ret, sector_summary = compute_sector_data(prices, constituents_out)

    sector_ret_path = OUTPUT_DIR / "sector_returns.csv"
    sector_ret.to_csv(sector_ret_path)
    print(f"Saved sector returns       → {sector_ret_path}")

    sector_sum_path = OUTPUT_DIR / "sector_breakdown.csv"
    sector_summary.to_csv(sector_sum_path, index=False)
    print(f"Saved sector breakdown     → {sector_sum_path}")

    print("\n  Sector risk/return (equal-weight, annualised):")
    print(f"    {'Sector':<45s} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>7}  Stocks")
    print(f"    {'-'*45} {'-'*8} {'-'*8} {'-'*7}  ------")
    for _, row in sector_summary.iterrows():
        print(f"    {row['sector']:<45s} "
              f"{row['ann_return_%']:>+7.1f}% "
              f"{row['ann_volatility_%']:>7.1f}% "
              f"{row['sharpe_ratio']:>7.2f}  "
              f"{int(row['n_stocks'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    constituents = fetch_constituent_list()
    tickers = constituents["ticker"].tolist()

    raw = download_prices(tickers)
    prices = extract_close(raw, tickers)

    report: list[str] = [
        f"Report generated : {date.today()}",
        f"Requested window : {START_DATE} → {END_DATE}",
    ]
    prices = quality_check(prices, report)
    save(prices, constituents)

    report_path = OUTPUT_DIR / "constituents_quality_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Quality report   → {report_path}")


if __name__ == "__main__":
    main()

"""
Fetch 2 years of daily closing prices for SPY from Yahoo Finance via yfinance.
Performs data quality checks and saves clean data to data/spy_closing_prices.csv.
Findings are printed to stdout and also written to data/data_quality_report.txt.
"""

import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "SPY"
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=2 * 365)
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_CSV = OUTPUT_DIR / "spy_closing_prices.csv"
REPORT_FILE = OUTPUT_DIR / "data_quality_report.txt"

# Thresholds for anomaly detection
MAX_DAILY_MOVE_PCT = 0.15   # flag single-day moves > ±15 %
MIN_PRICE = 1.0             # price below this is almost certainly corrupt


def log(lines: list[str], report_lines: list[str]) -> None:
    for line in lines:
        print(line)
        report_lines.append(line)


def fetch_raw(ticker: str, start: date, end: date) -> pd.DataFrame:
    print(f"Downloading {ticker} from {start} to {end} via yfinance …")
    df = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        progress=False,
        auto_adjust=True,   # adjust for splits & dividends
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned an empty DataFrame for {ticker}.")
    return df


def quality_check(df: pd.DataFrame, report_lines: list[str]) -> pd.DataFrame:
    log(["", "=" * 60, "DATA QUALITY REPORT", "=" * 60], report_lines)

    # ── 1. Shape ──────────────────────────────────────────────────
    log([f"Raw rows downloaded : {len(df)}",
         f"Date range          : {df.index.min().date()} → {df.index.max().date()}",
         f"Columns             : {list(df.columns)}"], report_lines)

    # ── 2. Flatten MultiIndex columns (yfinance ≥ 0.2.38) ─────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        log(["[fix] Flattened MultiIndex columns."], report_lines)

    close = df["Close"].copy()

    # ── 3. Missing dates (trading-day gaps) ───────────────────────
    # Build full US-trading-day calendar via pandas business-day frequency
    bday_range = pd.bdate_range(start=close.index.min(), end=close.index.max())
    missing_bdays = bday_range.difference(close.index)
    # Filter out known US market holidays (NYSE closes ~9 per year)
    # yfinance already skips holidays, so missing business days = holidays
    log([f"\n[check] Missing business days (expected = holidays): {len(missing_bdays)}"],
        report_lines)
    if len(missing_bdays) > 0:
        sample = missing_bdays[:10].strftime("%Y-%m-%d").tolist()
        log([f"        First 10: {sample}"], report_lines)

    # ── 4. Duplicate dates ────────────────────────────────────────
    dupes = close.index[close.index.duplicated()]
    log([f"\n[check] Duplicate index entries: {len(dupes)}"], report_lines)
    if len(dupes) > 0:
        log([f"        Dates: {dupes.strftime('%Y-%m-%d').tolist()}",
             "[fix]   Kept last occurrence of each duplicate."], report_lines)
        close = close[~close.index.duplicated(keep="last")]

    # ── 5. NaN / null values ──────────────────────────────────────
    null_count = close.isna().sum()
    log([f"\n[check] NaN values in Close: {null_count}"], report_lines)
    if null_count > 0:
        null_dates = close[close.isna()].index.strftime("%Y-%m-%d").tolist()
        log([f"        Dates: {null_dates}",
             "[fix]   Forward-filled NaNs (carry last known close)."], report_lines)
        close = close.ffill()

    # ── 6. Prices below minimum threshold ─────────────────────────
    below_min = close[close < MIN_PRICE]
    log([f"\n[check] Prices below ${MIN_PRICE}: {len(below_min)}"], report_lines)
    if len(below_min) > 0:
        log([f"        Dates & values: {below_min.to_dict()}",
             "[fix]   Replaced with NaN then forward-filled."], report_lines)
        close[close < MIN_PRICE] = np.nan
        close = close.ffill()

    # ── 7. Large single-day moves ─────────────────────────────────
    returns = close.pct_change()
    large_moves = returns[returns.abs() > MAX_DAILY_MOVE_PCT].dropna()
    log([f"\n[check] Single-day moves > ±{MAX_DAILY_MOVE_PCT*100:.0f}%: {len(large_moves)}"],
        report_lines)
    if len(large_moves) > 0:
        for dt, ret in large_moves.items():
            log([f"        {dt.date()}  {ret:+.2%}  (price={close[dt]:.2f})"],
                report_lines)
        log(["        These are kept as-is (no removal); extreme moves can be real."],
            report_lines)

    # ── 8. Summary stats ──────────────────────────────────────────
    log(["\n--- Summary statistics on clean Close ---",
         close.describe().to_string()], report_lines)

    log(["\n[result] Clean rows: " + str(len(close)),
         "=" * 60], report_lines)

    return close.rename("Close")


def save(series: pd.Series) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = series.to_frame()
    df_out.index.name = "Date"
    df_out.index = df_out.index.date   # drop timezone / time component
    df_out.to_csv(OUTPUT_CSV)
    print(f"\nSaved {len(df_out)} rows → {OUTPUT_CSV}")


def main() -> None:
    raw = fetch_raw(TICKER, START_DATE, END_DATE)

    report_lines: list[str] = [
        f"Report generated    : {date.today()}",
        f"Ticker              : {TICKER}",
        f"Requested window    : {START_DATE} → {END_DATE}",
    ]

    clean_close = quality_check(raw, report_lines)
    save(clean_close)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Quality report      → {REPORT_FILE}")


if __name__ == "__main__":
    main()

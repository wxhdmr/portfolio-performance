"""
Fetch quarterly S&P 500 weights from SEC EDGAR NPORT-P filings for SPY.

Pipeline
--------
1. Pull the list of NPORT-P filings from EDGAR for SPY (CIK 0000884394)
2. For each filing within the past 2 years, download and parse the XML
   to extract: company name, CUSIP, portfolio weight (pctVal)
3. Batch-map CUSIPs to exchange tickers via the free OpenFIGI API
4. Save per-quarter CSVs and a combined Date × Ticker weight matrix

Outputs
-------
data/sp500_weights/YYYY-MM-DD.csv   — one file per quarter-end date
data/sp500_weights_quarterly.csv    — wide matrix: Date × Ticker (weight %)
"""

import json
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
CIK          = "0000884394"          # SPDR S&P 500 ETF Trust
EDGAR_BASE   = "https://data.sec.gov"
ARCHIVES     = "https://www.sec.gov/Archives/edgar/data/884394"
FIGI_URL     = "https://api.openfigi.com/v3/mapping"
HEADERS      = {"User-Agent": "portfolio-research zl524@nyu.edu"}
FIGI_HEADERS = {"Content-Type": "application/json"}

CUTOFF        = date.today() - timedelta(days=2 * 365)
NS            = {"n": "http://www.sec.gov/edgar/nport"}
OUTPUT_DIR    = Path(__file__).parent.parent / "data" / "sp500_weights"
PRICE_OUTPUT  = Path(__file__).parent.parent / "data" / "sp500_closing_prices.csv"
PRICE_BATCH   = 100   # tickers per yfinance call

# ── Foreign PLC name → ticker map ─────────────────────────────────────────────
# These companies are incorporated outside the US so their NPORT entries carry
# CUSIP=000000000, which OpenFIGI cannot resolve.  We map them by company name.
# "TE Connectivity Ltd" (pre-2024 name) and "TE Connectivity PLC" both → TEL.
NAME_TO_TICKER: dict[str, str] = {
    "Accenture PLC":                       "ACN",
    "Allegion plc":                        "ALLE",
    "Amcor PLC":                           "AMCR",
    "Aon PLC":                             "AON",
    "Aptiv PLC":                           "APTV",
    "Arch Capital Group Ltd":              "ACGL",
    "Bunge Global SA":                     "BG",
    "CRH PLC":                             "CRH",
    "Chubb Ltd":                           "CB",
    "Eaton Corp PLC":                      "ETN",
    "Everest Group Ltd":                   "EG",
    "Garmin Ltd":                          "GRMN",
    "Invesco Ltd":                         "IVZ",
    "Johnson Controls International plc":  "JCI",
    "Linde PLC":                           "LIN",
    "LyondellBasell Industries NV":        "LYB",
    "Medtronic PLC":                       "MDT",
    "NXP Semiconductors NV":               "NXPI",
    "Norwegian Cruise Line Holdings Ltd":  "NCLH",
    "Pentair PLC":                         "PNR",
    "Royal Caribbean Cruises Ltd":         "RCL",
    "STERIS PLC":                          "STE",
    "Seagate Technology Holdings PLC":     "STX",
    "Smurfit WestRock PLC":               "SW",
    "TE Connectivity Ltd":                 "TEL",   # pre-2024 name
    "TE Connectivity PLC":                 "TEL",   # post-2024 name
    "Trane Technologies PLC":              "TT",
    "Willis Towers Watson PLC":            "WTW",
}


# ── Step 1: list NPORT-P filings ──────────────────────────────────────────────

def get_nport_filings() -> list[dict]:
    url  = f"{EDGAR_BASE}/submissions/CIK{CIK}.json"
    data = requests.get(url, headers=HEADERS, timeout=30).json()
    f    = data["filings"]["recent"]

    filings = []
    for i, form in enumerate(f["form"]):
        if form != "NPORT-P":
            continue
        filing_date = date.fromisoformat(f["filingDate"][i])
        if filing_date < CUTOFF:
            continue
        filings.append({
            "filing_date":    filing_date,
            "accession":      f["accessionNumber"][i],
        })

    filings.sort(key=lambda x: x["filing_date"])
    print(f"Found {len(filings)} NPORT-P filings since {CUTOFF}")
    return filings


# ── Step 2: parse one NPORT-P XML ─────────────────────────────────────────────

def parse_nport(accession: str) -> tuple[date, list[dict]]:
    acc_clean = accession.replace("-", "")
    base_url  = f"{ARCHIVES}/{acc_clean}"

    # Find the primary XML in the filing index page
    idx_html  = requests.get(base_url + "/", headers=HEADERS, timeout=30).text
    import re
    xml_files = re.findall(r'href="(/Archives/edgar/data/884394/[^"]+\.xml)"', idx_html)
    if not xml_files:
        raise ValueError(f"No XML found for accession {accession}")

    xml_url = "https://www.sec.gov" + xml_files[0]
    text    = requests.get(xml_url, headers=HEADERS, timeout=60).text
    root    = ET.fromstring(text)

    # Reporting period date
    rep_date_el = root.find(".//n:repPdDate", NS)
    rep_date    = date.fromisoformat(rep_date_el.text) if rep_date_el is not None else None

    # Holdings
    holdings = []
    for sec in root.findall(".//n:invstOrSec", NS):
        name_el    = sec.find("n:name", NS)
        cusip_el   = sec.find("n:cusip", NS)
        pctval_el  = sec.find("n:pctVal", NS)
        val_el     = sec.find("n:valUSD", NS)

        if cusip_el is None or pctval_el is None:
            continue

        holdings.append({
            "name":    name_el.text if name_el is not None else "",
            "cusip":   cusip_el.text.strip(),
            "pct_val": float(pctval_el.text),
            "val_usd": float(val_el.text) if val_el is not None else 0.0,
        })

    return rep_date, holdings


# ── Step 3: CUSIP → ticker via OpenFIGI ──────────────────────────────────────

def cusip_to_ticker(cusips: list[str]) -> dict[str, str]:
    """
    Batch-map CUSIPs to primary exchange tickers using the free OpenFIGI API.
    Returns {cusip: ticker}. Rate limit: 25 jobs/min without API key.
    """
    BATCH = 10    # OpenFIGI free tier: max 10 mapping jobs per request
    mapping = {}

    for i in range(0, len(cusips), BATCH):
        batch = cusips[i: i + BATCH]
        jobs  = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in batch]
        resp  = requests.post(FIGI_URL, headers=FIGI_HEADERS,
                              data=json.dumps(jobs), timeout=30)

        if resp.status_code == 429:
            print("  [rate-limit] waiting 65 s …")
            time.sleep(65)
            resp = requests.post(FIGI_URL, headers=FIGI_HEADERS,
                                 data=json.dumps(jobs), timeout=30)

        if not resp.content:
            print(f"  [warn] empty response for batch {i//BATCH + 1}, skipping")
            time.sleep(5)
            continue

        results = resp.json()
        for cusip, result in zip(batch, results):
            if result.get("data"):
                # Prefer common stock (shareClassFIGI = None means it's the main one)
                for item in result["data"]:
                    if item.get("securityType") in ("Common Stock", "ETP"):
                        mapping[cusip] = item["ticker"]
                        break
                else:
                    mapping[cusip] = result["data"][0]["ticker"]

        time.sleep(3)   # stay well under rate limit

    return mapping


# ── Step 4: build weight DataFrame ────────────────────────────────────────────

def build_weights(holdings: list[dict], cusip_map: dict[str, str]) -> pd.Series:
    rows = []
    name_hits = name_misses = 0

    for h in holdings:
        cusip  = h["cusip"]
        ticker = None

        if cusip == "000000000":
            # Foreign PLC: resolve via company name
            ticker = NAME_TO_TICKER.get(h["name"])
            if ticker:
                name_hits += 1
            else:
                name_misses += 1
        else:
            ticker = cusip_map.get(cusip)

        if ticker and h["pct_val"] > 0:
            rows.append({"ticker": ticker, "weight_%": h["pct_val"]})

    if name_hits or name_misses:
        print(f"    name-map: {name_hits} resolved, {name_misses} still missing")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.Series(dtype=float)

    # Normalise ticker format to match Yahoo Finance (e.g. BRK/B → BRK-B)
    df["ticker"] = (df["ticker"]
                    .str.replace("/", "-", regex=False)
                    .str.replace(".", "-", regex=False))

    # If duplicate tickers (e.g. share classes), sum weights
    df = df.groupby("ticker")["weight_%"].sum()
    return df


# ── Step 5: fetch 2-year closing prices ──────────────────────────────────────

def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Download 2-year daily adjusted closing prices for all S&P 500 tickers + SPY.
    Downloads in batches of PRICE_BATCH to avoid yfinance timeouts.
    Saves the result to data/sp500_closing_prices.csv.
    Returns a Date × Ticker DataFrame of adjusted close prices.
    """
    all_tickers = sorted(set(tickers) | {"SPY"})
    start = str(date.today() - timedelta(days=2 * 365))
    end   = str(date.today())
    n_batches = (len(all_tickers) + PRICE_BATCH - 1) // PRICE_BATCH

    print(f"\nFetching 2-year closing prices for {len(all_tickers)} tickers …")
    print(f"  Date range : {start} to {end}")
    print(f"  Batches    : {n_batches}  ({PRICE_BATCH} tickers each)")

    frames: list[pd.DataFrame] = []

    for i in range(0, len(all_tickers), PRICE_BATCH):
        batch   = all_tickers[i : i + PRICE_BATCH]
        batch_n = i // PRICE_BATCH + 1
        print(f"  [{batch_n}/{n_batches}]  {len(batch)} tickers … ", end="", flush=True)
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                print("empty, skipping")
                continue

            # yfinance returns MultiIndex (Price, Ticker) when len(batch) > 1
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})

            close = close.dropna(axis=1, how="all")   # drop fully-missing tickers
            frames.append(close)
            print(f"{len(close.columns)} ok")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No price data was downloaded — check network / yfinance version.")

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]   # deduplicate columns
    combined.sort_index(inplace=True)
    combined.ffill(inplace=True)                                 # fill holiday/delist gaps
    combined.dropna(axis=1, how="all", inplace=True)

    combined.index = pd.to_datetime(combined.index).date
    combined.index.name = "Date"

    PRICE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PRICE_OUTPUT)

    print(f"\n  Saved → {PRICE_OUTPUT}")
    print(f"  Shape  : {combined.shape[0]} trading days × {combined.shape[1]} tickers")
    print(f"  Range  : {combined.index[0]} to {combined.index[-1]}")

    missing = sorted(set(all_tickers) - set(combined.columns))
    if missing:
        print(f"  No data: {len(missing)} tickers (delisted / not on Yahoo Finance)")
        if len(missing) <= 20:
            print(f"           {missing}")

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EDGAR NPORT-P → SPY quarterly weights")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filings = get_nport_filings()

    # Collect all holdings across filings first (to batch CUSIP lookup)
    print("\nParsing NPORT-P filings …")
    parsed = []   # list of (rep_date, holdings)
    for fl in filings:
        print(f"  {fl['filing_date']}  accession={fl['accession']} … ", end="", flush=True)
        try:
            rep_date, holdings = parse_nport(fl["accession"])
            parsed.append((rep_date, holdings))
            print(f"period={rep_date}  holdings={len(holdings)}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.5)   # EDGAR rate limit: 10 req/s

    # Collect unique CUSIPs across all filings
    all_cusips = list({h["cusip"] for _, holdings in parsed for h in holdings})
    print(f"\nMapping {len(all_cusips)} unique CUSIPs to tickers via OpenFIGI …")
    cusip_map = cusip_to_ticker(all_cusips)
    matched = sum(1 for c in all_cusips if c in cusip_map)
    print(f"  Matched {matched}/{len(all_cusips)} CUSIPs to tickers")

    # Build per-quarter weight series
    print("\nBuilding weight tables …")
    all_weights: dict[date, pd.Series] = {}
    for rep_date, holdings in parsed:
        if rep_date is None:
            continue
        weights = build_weights(holdings, cusip_map)
        all_weights[rep_date] = weights

        path = OUTPUT_DIR / f"{rep_date}.csv"
        weights.reset_index().rename(columns={"ticker": "ticker"}).to_csv(path, index=False)
        print(f"  {rep_date}: {len(weights)} tickers, "
              f"total weight={weights.sum():.2f}%  → {path.name}")

    # Wide matrix: Date × Ticker
    wide = pd.DataFrame(all_weights).T.sort_index()
    wide.index.name = "Date"
    out_path = Path(__file__).parent.parent / "data" / "sp500_weights_quarterly.csv"
    wide.to_csv(out_path)
    print(f"\nSaved wide matrix ({wide.shape[0]} quarters × {wide.shape[1]} tickers)")
    print(f"  → {out_path}")

    # Print a quick summary
    print("\n--- Weight summary (latest quarter) ---")
    latest = wide.iloc[-1].dropna().sort_values(ascending=False)
    print(f"  Date     : {wide.index[-1]}")
    print(f"  Top 10   :")
    for tkr, w in latest.head(10).items():
        print(f"    {tkr:<8} {w:.3f}%")

    # Step 5: fetch 2-year closing prices for all tickers in the weight data
    all_tickers = sorted({t for s in all_weights.values() for t in s.index})
    fetch_prices(all_tickers)


if __name__ == "__main__":
    main()

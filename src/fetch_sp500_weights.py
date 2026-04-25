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

# ── Config ────────────────────────────────────────────────────────────────────
CIK          = "0000884394"          # SPDR S&P 500 ETF Trust
EDGAR_BASE   = "https://data.sec.gov"
ARCHIVES     = "https://www.sec.gov/Archives/edgar/data/884394"
FIGI_URL     = "https://api.openfigi.com/v3/mapping"
HEADERS      = {"User-Agent": "portfolio-research zl524@nyu.edu"}
FIGI_HEADERS = {"Content-Type": "application/json"}

CUTOFF       = date.today() - timedelta(days=2 * 365)
NS           = {"n": "http://www.sec.gov/edgar/nport"}
OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "sp500_weights"


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
    for h in holdings:
        ticker = cusip_map.get(h["cusip"])
        if ticker and h["pct_val"] > 0:
            rows.append({"ticker": ticker, "weight_%": h["pct_val"]})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.Series(dtype=float)

    # If duplicate tickers (e.g. share classes), sum weights
    df = df.groupby("ticker")["weight_%"].sum()
    return df


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


if __name__ == "__main__":
    main()

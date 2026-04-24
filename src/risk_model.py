"""
Fama-French 5-Factor risk model for tracking error decomposition.

Steps
-----
1. Load constituent daily returns from data/sp500_closing_prices.csv
2. Fetch FF5 daily factors from Ken French's data library (direct ZIP download)
3. Estimate factor loadings per stock via OLS
4. Given a set of portfolio weights, decompose tracking error vs. SPY into:
      - Factor tracking error   (systematic)
      - Idiosyncratic tracking error (stock-specific)
      - Total tracking error

Usage
-----
Run directly for a demo with an equal-weight portfolio:
    python src/risk_model.py

Or import and call build_model() / tracking_error() from another script.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data"
PRICES_CSV = DATA_DIR / "sp500_closing_prices.csv"
FF5_URL    = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF5_CACHE  = DATA_DIR / "ff5_daily_factors.csv"

ANNUALIZE  = np.sqrt(252)


# ── 1. Load price data ────────────────────────────────────────────────────────

def load_returns() -> pd.DataFrame:
    """Load constituent prices and return daily log-returns (Date × Ticker)."""
    prices = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True)
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return returns


# ── 2. Fetch Fama-French 5 factors ────────────────────────────────────────────

def fetch_ff5(use_cache: bool = True) -> pd.DataFrame:
    """
    Download FF5 daily factors from Ken French's website.
    Returns a DataFrame with columns [Mkt-RF, SMB, HML, RMW, CMA, RF]
    as decimals (divided by 100), indexed by date.
    """
    if use_cache and FF5_CACHE.exists():
        print(f"Loading FF5 factors from cache: {FF5_CACHE}")
        return pd.read_csv(FF5_CACHE, index_col="Date", parse_dates=True)

    print("Downloading FF5 daily factors from Ken French's data library …")
    resp = requests.get(FF5_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        fname = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw_text = zf.read(fname).decode("utf-8", errors="replace")

    # The CSV has a multi-line header and a footer — find the data block
    lines = raw_text.splitlines()
    # Data rows start after the header line containing "Mkt-RF"
    header_idx = next(i for i, ln in enumerate(lines) if "Mkt-RF" in ln)
    data_lines = []
    for ln in lines[header_idx + 1:]:
        stripped = ln.strip()
        if not stripped:
            break   # blank line signals end of daily data
        data_lines.append(stripped)

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").apply(pd.to_numeric, errors="coerce").dropna()
    df = df / 100  # convert from percent to decimal

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FF5_CACHE)
    print(f"  Saved to cache: {FF5_CACHE}")
    return df


# ── 3. Estimate factor loadings ───────────────────────────────────────────────

def estimate_loadings(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    OLS regression of each ticker's excess return on FF5 factors.

    Returns
    -------
    betas      : DataFrame (Ticker × Factor), factor loadings
    idio_var   : Series (Ticker), annualised idiosyncratic variance
    """
    rf = factors["RF"]
    f  = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]

    # Align on common dates
    common = returns.index.intersection(f.index)
    exc    = returns.loc[common].subtract(rf.loc[common], axis=0)  # excess returns
    F      = f.loc[common]

    # Add intercept column
    X = np.column_stack([np.ones(len(F)), F.values])
    factor_names = ["alpha"] + list(F.columns)

    betas_dict: dict = {}
    idio_var:   dict = {}

    for ticker in returns.columns:
        y = exc[ticker].values
        mask = ~np.isnan(y)
        if mask.sum() < 60:   # need at least ~3 months of data
            continue
        X_t = X[mask]
        y_t = y[mask]
        coef, _, _, _ = np.linalg.lstsq(X_t, y_t, rcond=None)
        betas_dict[ticker] = dict(zip(factor_names, coef))
        # annualised idiosyncratic variance
        e = y_t - X_t @ coef
        idio_var[ticker] = np.var(e, ddof=len(factor_names)) * 252

    betas    = pd.DataFrame(betas_dict).T          # Ticker × Factor
    idio_var = pd.Series(idio_var, name="idio_var")
    return betas, idio_var


# ── 4. Tracking error decomposition ───────────────────────────────────────────

def tracking_error(
    weights: pd.Series,
    betas: pd.DataFrame,
    idio_var: pd.Series,
    factors: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict:
    """
    Compute annualised tracking error vs SPY, decomposed into factor and
    idiosyncratic components.

    Parameters
    ----------
    weights : portfolio weights indexed by ticker (must include 'SPY').
              Values should sum to ~1. SPY weight represents the benchmark;
              pass weights={'SPY': -1, 'AAPL': 0.01, ...} to express active weights,
              or pass portfolio weights and the function treats SPY as benchmark.
    betas   : output of estimate_loadings()
    idio_var: output of estimate_loadings()
    factors : FF5 factor DataFrame
    returns : constituent returns DataFrame
    """
    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    F = factors[factor_cols]

    # ── Factor covariance matrix (annualised) ─────────────────────────────────
    common = returns.index.intersection(F.index)
    F_aligned = F.loc[common]
    factor_cov = F_aligned.cov() * 252   # annualised

    # ── Separate portfolio vs. benchmark weights ───────────────────────────────
    if "SPY" not in weights.index:
        raise ValueError("weights must include 'SPY' as the benchmark.")

    port_w = weights.drop("SPY")
    # active weights: long portfolio, short benchmark
    port_w = port_w.reindex(betas.index).fillna(0)
    spy_betas = betas.loc["SPY", factor_cols] if "SPY" in betas.index else pd.Series(0, index=factor_cols)

    # Weighted portfolio factor exposures
    common_tickers = port_w.index.intersection(betas.index)
    B_port = betas.loc[common_tickers, factor_cols].multiply(
        port_w.loc[common_tickers], axis=0
    ).sum()

    # Active factor exposure = portfolio - benchmark
    delta_beta = B_port - spy_betas

    # ── Factor tracking error ─────────────────────────────────────────────────
    db = delta_beta.values
    factor_te_var = db @ factor_cov.values @ db
    factor_te     = np.sqrt(max(factor_te_var, 0))

    # ── Idiosyncratic tracking error ──────────────────────────────────────────
    # var = Σ w_i² × σ²_idio,i
    idio_aligned = idio_var.reindex(common_tickers).fillna(idio_var.median())
    idio_te_var  = (port_w.loc[common_tickers] ** 2 * idio_aligned).sum()
    # subtract benchmark idiosyncratic (SPY as single asset has near-zero idio)
    spy_idio     = idio_var.get("SPY", 0.0)
    idio_te_var  = max(idio_te_var - spy_idio, 0)
    idio_te      = np.sqrt(idio_te_var)

    # ── Total tracking error ──────────────────────────────────────────────────
    total_te = np.sqrt(factor_te_var + idio_te_var)

    # ── Realised tracking error (simple, from returns) ────────────────────────
    if "SPY" in returns.columns:
        common_r = returns.index.intersection(returns.index)
        active_r = returns.loc[common_r, port_w.index[port_w != 0]].multiply(
            port_w[port_w != 0], axis=1
        ).sum(axis=1) - returns.loc[common_r, "SPY"]
        realised_te = active_r.std() * ANNUALIZE
    else:
        realised_te = np.nan

    # ── Factor contribution breakdown ─────────────────────────────────────────
    factor_contrib = {}
    for f_name in factor_cols:
        contrib_var = delta_beta[f_name] ** 2 * factor_cov.loc[f_name, f_name]
        factor_contrib[f_name] = np.sqrt(max(contrib_var, 0))

    return {
        "total_te":          round(total_te * 100, 4),
        "factor_te":         round(factor_te * 100, 4),
        "idiosyncratic_te":  round(idio_te * 100, 4),
        "realised_te":       round(realised_te * 100, 4) if not np.isnan(realised_te) else None,
        "factor_contrib_%":  {k: round(v * 100, 4) for k, v in factor_contrib.items()},
        "active_factor_exposures": delta_beta.round(4).to_dict(),
    }


# ── 5. Demo: equal-weight portfolio ───────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("FF5 Risk Model — Tracking Error Decomposition")
    print("=" * 60)

    returns = load_returns()
    print(f"Loaded returns: {returns.shape[0]} days × {returns.shape[1]} tickers")

    factors = fetch_ff5()
    print(f"FF5 factors:    {factors.shape[0]} days, "
          f"{factors.index.min().date()} → {factors.index.max().date()}")

    print("\nEstimating factor loadings (OLS per ticker) …")
    betas, idio_var = estimate_loadings(returns, factors)
    print(f"  Fitted {len(betas)} tickers")

    # Build an equal-weight portfolio of all non-SPY constituents
    constituent_tickers = [t for t in returns.columns if t != "SPY" and t in betas.index]
    n = len(constituent_tickers)
    port_weights = pd.Series(1.0 / n, index=constituent_tickers)
    # Add SPY as benchmark placeholder (weight = 0 in portfolio, used as benchmark)
    all_weights = pd.concat([port_weights, pd.Series({"SPY": 0.0})])

    print(f"\nPortfolio: equal-weight across {n} constituents vs. SPY benchmark")
    result = tracking_error(all_weights, betas, idio_var, factors, returns)

    print("\n--- Tracking Error Results (annualised, in %) ---")
    print(f"  Total TE            : {result['total_te']:.2f}%")
    print(f"  Factor TE           : {result['factor_te']:.2f}%")
    print(f"  Idiosyncratic TE    : {result['idiosyncratic_te']:.2f}%")
    if result["realised_te"] is not None:
        print(f"  Realised TE (hist.) : {result['realised_te']:.2f}%")

    print("\n  Factor contributions (standalone, %):")
    for f_name, contrib in result["factor_contrib_%"].items():
        print(f"    {f_name:8s}: {contrib:.4f}%")

    print("\n  Active factor exposures (portfolio - SPY):")
    for f_name, exp in result["active_factor_exposures"].items():
        print(f"    {f_name:8s}: {exp:+.4f}")

    # Save results
    out_path = DATA_DIR / "tracking_error_results.csv"
    rows = [
        {"metric": "Total TE (%)",         "value": result["total_te"]},
        {"metric": "Factor TE (%)",         "value": result["factor_te"]},
        {"metric": "Idiosyncratic TE (%)",  "value": result["idiosyncratic_te"]},
        {"metric": "Realised TE (%)",       "value": result["realised_te"]},
    ]
    for f_name, contrib in result["factor_contrib_%"].items():
        rows.append({"metric": f"Factor contrib {f_name} (%)", "value": contrib})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved results → {out_path}")


if __name__ == "__main__":
    main()

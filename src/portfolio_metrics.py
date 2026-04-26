"""
portfolio_metrics.py — shared return and performance calculation utilities.

All return-based metrics in src/ use these functions so the logic is
defined exactly once.

Functions
---------
compute_returns(weights, prices)
    Daily portfolio / SPY / active returns from a lagged-weight matrix.
    Returns DataFrame with columns: port_return, spy_return, active_return.

compute_stats(returns, risk_free)
    Full set of annualised performance metrics from a returns DataFrame.
    Returns a dict with keys: ann_return, ann_spy, ann_active, te, ir,
    vol, sharpe, spy_sharpe, beta, corr_spy, full_period, full_spy,
    max_dd, max_dd_spy, n_days, start, end.

quarterly_breakdown(returns)
    Quarter-by-quarter cumulative returns and tracking error.
    Returns a DataFrame indexed by quarter-end date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily portfolio, SPY, and active returns from a weight matrix.

    Uses lagged weights (today's return reflects yesterday's positions).
    Tickers present in weights but missing from prices are silently ignored.

    Parameters
    ----------
    weights : pd.DataFrame
        Date × ticker matrix of portfolio weights in percent (rows sum to ~100).
    prices : pd.DataFrame
        Date × ticker closing prices, must include a "SPY" column.

    Returns
    -------
    pd.DataFrame with columns port_return, spy_return, active_return.
    Rows with any NaN are dropped.
    """
    spy_ret   = prices["SPY"].pct_change()
    price_ret = prices.pct_change()
    tickers   = [c for c in weights.columns if c in price_ret.columns]
    common    = weights.index.intersection(price_ret.index)

    w_lag      = weights[tickers].shift(1).loc[common] / 100
    r_day      = price_ret[tickers].loc[common]
    spy_day    = spy_ret.reindex(common)

    port_ret   = (w_lag * r_day).sum(axis=1)
    active_ret = port_ret - spy_day

    return pd.DataFrame({
        "port_return":   port_ret,
        "spy_return":    spy_day,
        "active_return": active_ret,
    }).dropna()


def compute_stats(
    returns: pd.DataFrame,
    risk_free: float = 0.045,
) -> dict:
    """
    Compute annualised performance metrics from a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Must have columns port_return, spy_return, active_return.
    risk_free : float
        Annual risk-free rate used for Sharpe ratio (default 4.5%).

    Returns
    -------
    dict with keys:
        ann_return, ann_spy, ann_active  — annualised geometric returns
        te                               — annualised tracking error
        ir                               — information ratio (ann_active / te)
        vol                              — annualised portfolio volatility
        sharpe                           — portfolio Sharpe ratio
        spy_sharpe                       — SPY Sharpe ratio
        beta, corr_spy                   — portfolio beta and correlation vs SPY
        full_period, full_spy            — cumulative total returns
        max_dd, max_dd_spy               — maximum drawdowns
        n_days, start, end               — metadata
    """
    port = returns["port_return"]
    spy  = returns["spy_return"]
    act  = returns["active_return"]
    n    = len(port)

    ann_port = float((1 + port).prod() ** (252 / n) - 1)
    ann_spy  = float((1 + spy).prod()  ** (252 / n) - 1)
    ann_act  = ann_port - ann_spy          # arithmetic active return (consistent)

    te       = float(act.std() * np.sqrt(252))
    ir       = ann_act / te if te > 0 else np.nan
    vol      = float(port.std() * np.sqrt(252))
    sharpe   = (ann_port - risk_free) / vol if vol > 0 else np.nan
    spy_vol  = float(spy.std() * np.sqrt(252))
    spy_sharpe = (ann_spy - risk_free) / spy_vol if spy_vol > 0 else np.nan

    full_port = float((1 + port).prod() - 1)
    full_spy  = float((1 + spy).prod()  - 1)

    cum      = (1 + port).cumprod()
    max_dd   = float(((cum - cum.cummax()) / cum.cummax()).min())
    cum_spy  = (1 + spy).cumprod()
    max_dd_spy = float(((cum_spy - cum_spy.cummax()) / cum_spy.cummax()).min())

    cov_mat  = np.cov(port.values, spy.values)
    beta     = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else np.nan
    corr     = float(port.corr(spy))

    return {
        "ann_return":  ann_port,
        "ann_spy":     ann_spy,
        "ann_active":  ann_act,
        "te":          te,
        "ir":          ir,
        "vol":         vol,
        "sharpe":      sharpe,
        "spy_sharpe":  spy_sharpe,
        "beta":        beta,
        "corr_spy":    corr,
        "full_period": full_port,
        "full_spy":    full_spy,
        "max_dd":      max_dd,
        "max_dd_spy":  max_dd_spy,
        "n_days":      n,
        "start":       str(returns.index[0].date()),
        "end":         str(returns.index[-1].date()),
    }


def quarterly_breakdown(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Quarter-by-quarter cumulative returns and annualised tracking error.

    Parameters
    ----------
    returns : pd.DataFrame
        Must have columns port_return, spy_return, active_return.

    Returns
    -------
    pd.DataFrame indexed by quarter-end date with columns:
        port_return, spy_return, active_return, tracking_error.
    Quarters with fewer than 5 trading days are skipped.
    """
    rows = []
    for qname, grp in returns.resample("Q"):
        if len(grp) < 5:
            continue
        qp   = float((1 + grp["port_return"]).prod() - 1)
        qs   = float((1 + grp["spy_return"]).prod()  - 1)
        qa   = qp - qs
        te_q = float(grp["active_return"].std() * np.sqrt(252))
        rows.append({
            "quarter":        str(qname.date()),
            "port_return":    qp,
            "spy_return":     qs,
            "active_return":  qa,
            "tracking_error": te_q,
        })
    return pd.DataFrame(rows).set_index("quarter")
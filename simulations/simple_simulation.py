"""
simple_simulation.py
====================
Generates a synthetic portfolio return series that exactly hits specified
tracking error, excess return, and fee targets relative to a provided
SPY daily closing price series.

The active return on each day is:
    active[t] = alpha[t] - annual_fee / 252
where alpha[t] is drawn from N(daily_gross_alpha, daily_TE) and then
rescaled so the full sample hits the targets exactly (not just in expectation).

Function
--------
    simulate_return_with_TE_ER(
        spy_close,
        target_annual_te,
        target_annual_er,
        annual_fee,
        output_path,
        random_seed,
    ) -> pd.Series

Run demo from project root:
    py simulations/simple_simulation.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"


def simulate_return_with_TE_ER(
    spy_close: pd.Series,
    target_annual_te: float,
    target_annual_er: float,
    annual_fee: float = 0.0009,
    output_path: str | Path | None = None,
    random_seed: int = 42,
) -> pd.Series:
    """
    Generate a synthetic portfolio closing-price series with exact tracking
    error, net excess return, and management fee properties.

    Parameters
    ----------
    spy_close : pd.Series
        SPY daily closing prices (DatetimeIndex).  Starting value is shared
        with the output series so both can be compared on the same scale.
    target_annual_te : float
        Target annualised tracking error, e.g. 0.017 for 1.7%.
    target_annual_er : float
        Target annualised net excess return vs SPY after the management fee,
        e.g. 0.02 for +2.0%.
    annual_fee : float
        Annual management fee rate (default 0.09% = 0.0009).
    output_path : str | Path | None
        Destination CSV.  Defaults to data/simple_simulation.csv.
    random_seed : int
        NumPy seed for reproducibility (default 42).

    Returns
    -------
    pd.Series
        Simulated portfolio daily closing prices with the same DatetimeIndex
        and starting value as spy_close.
    """
    spy_close = spy_close.dropna()
    n = len(spy_close)
    if n < 2:
        raise ValueError("spy_close must have at least 2 observations")

    # ── Daily parameter targets ────────────────────────────────────────────────
    daily_te          = target_annual_te / np.sqrt(252)
    daily_fee         = annual_fee / 252
    # Gross alpha must cover the fee so that net ER = target_annual_er
    daily_gross_alpha = (target_annual_er + annual_fee) / 252

    # ── Alpha: white noise rescaled to exact TE and mean ──────────────────────
    rng   = np.random.default_rng(random_seed)
    noise = rng.standard_normal(n - 1)
    noise = noise - noise.mean()                    # exact zero mean
    noise = noise / noise.std(ddof=1) * daily_te   # exact daily TE (ddof=1 matches pandas)
    alpha = noise + daily_gross_alpha         # shift to target gross alpha

    # ── SPY and portfolio daily returns ───────────────────────────────────────
    spy_ret  = spy_close.pct_change().iloc[1:].values   # shape (n-1,)
    port_ret = spy_ret + alpha - daily_fee               # net of fee

    # ── Compound into a closing-price series starting at spy_close.iloc[0] ───
    nav    = np.empty(n)
    nav[0] = float(spy_close.iloc[0])
    for i, r in enumerate(port_ret):
        nav[i + 1] = nav[i] * (1.0 + r)
    port_close = pd.Series(nav, index=spy_close.index, name="port_close")

    # ── Verify achieved metrics ────────────────────────────────────────────────
    active_ret = pd.Series(port_ret - spy_ret, index=spy_close.index[1:])
    ach_te     = float(active_ret.std() * np.sqrt(252))
    ach_er_net = float(active_ret.mean() * 252)

    print("=" * 56)
    print("  Simulated Portfolio vs SPY")
    print("=" * 56)
    print("  Period         : {} → {}".format(
          spy_close.index[0].date(), spy_close.index[-1].date()))
    print("  Trading days   : {}".format(n))
    print("  Target TE      : {:>8.4f}%   achieved {:>8.4f}%".format(
          target_annual_te * 100, ach_te * 100))
    print("  Target ER(net) : {:>+8.4f}%   achieved {:>+8.4f}%".format(
          target_annual_er * 100, ach_er_net * 100))
    print("  Annual fee     : {:>8.4f}%".format(annual_fee * 100))
    print("  Starting value : ${:>12,.4f}".format(float(spy_close.iloc[0])))
    print("  Final SPY      : ${:>12,.4f}".format(float(spy_close.iloc[-1])))
    print("  Final Port     : ${:>12,.4f}".format(float(port_close.iloc[-1])))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = DATA_DIR / "simple_simulation.csv"
    output_path = Path(output_path)

    port_ret_full   = np.concatenate([[np.nan], port_ret])
    active_ret_full = np.concatenate([[np.nan], active_ret.values])

    out = pd.DataFrame({
        "spy_close":  spy_close,
        "port_close": port_close,
        "spy_ret":    spy_close.pct_change(),
        "port_ret":   pd.Series(port_ret_full,   index=spy_close.index),
        "active_ret": pd.Series(active_ret_full, index=spy_close.index),
    })
    out.index.name = "Date"
    out.to_csv(output_path)
    print("  CSV saved      : {}".format(output_path))

    return port_close


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prices = pd.read_csv(
        DATA_DIR / "sp500_closing_prices.csv",
        index_col=0, parse_dates=True)
    spy = prices["SPY"]

    port = simulate_return_with_TE_ER(
        spy_close         = spy,
        target_annual_te  = 0.017,   # 1.7% tracking error
        target_annual_er  = 0.020,   # +2.0% net excess return
        annual_fee        = 0.0009,  # 0.09% management fee
    )

    print("\nSample (first 5 rows):")
    print(pd.DataFrame({"SPY": spy, "Port": port}).head().to_string())
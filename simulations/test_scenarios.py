"""
Test Scenarios — One per Account Activity Type
================================================
Each scenario starts from the same $100 k Frec-weighted portfolio
(top-10 holdings) and exercises exactly one activity.

On every activity the portfolio is immediately rebalanced so that
all holdings (including cash) maintain their pre-activity weights,
and the total NAV shifts by the cash value of the activity:

  Scenario 1  cash_inflow   -- deposit $50 k cash on 2024-10-15
                               new NAV = old NAV + $50 k; rebalance
  Scenario 2  cash_outflow  -- withdraw $20 k cash on 2025-01-15
                               new NAV = old NAV - $20 k; rebalance
                               (guard: amount must not exceed portfolio NAV)
  Scenario 3  stock_inflow  -- transfer in 30 AAPL shares on 2024-11-01
                               new NAV = old NAV + 30 × AAPL price; rebalance
  Scenario 4  stock_outflow -- transfer out half of initial NVDA shares 2025-04-01
                               new NAV = old NAV - shares × NVDA price; rebalance
                               (guard: shares must not exceed current holding)

Outputs saved to data/:
  test_scenario_1_cash_inflow.csv
  test_scenario_2_cash_outflow.csv
  test_scenario_3_stock_inflow.csv
  test_scenario_4_stock_outflow.csv

Run from project root:
    py simulations/test_scenarios.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# Import the account class from the same simulations folder
sys.path.insert(0, str(Path(__file__).parent))
from frec_account_activities import FrecAccount

DATA_DIR = Path(__file__).parent.parent / "data"
ANNUAL_FEE = 0.0009
TOP_N      = 10
BUDGET     = 100_000


# ── Shared helpers ────────────────────────────────────────────────────────────

def load_data():
    prices  = pd.read_csv(DATA_DIR / "sp500_closing_prices.csv",
                          index_col=0, parse_dates=True)
    weights = pd.read_csv(DATA_DIR / "frec_weights_daily.csv",
                          index_col=0, parse_dates=True)
    return prices, weights


def build_positions(prices: pd.DataFrame, weights: pd.DataFrame,
                    budget: float, top_n: int) -> tuple[dict[str, float], float]:
    """
    Allocate `budget` into top-N Frec-weighted stocks on day 1.
    Returns (share_counts, residual_cash).
    """
    first_w = weights.iloc[0].dropna().sort_values(ascending=False).head(top_n)
    first_w = first_w / first_w.sum() * 100
    px_d1   = prices.iloc[0]

    share_counts: dict[str, float] = {}
    residual = budget
    for tkr, pct in first_w.items():
        px = float(px_d1.get(tkr, 0))
        if px > 0:
            sh = round((pct / 100) * budget / px, 4)
            share_counts[tkr] = sh
            residual -= sh * px
    return share_counts, max(0.0, residual)


def print_scenario_header(n: int, title: str, activity: str) -> None:
    print("\n" + "=" * 62)
    print("  Scenario {}  |  {}".format(n, title))
    print("  Activity   : {}".format(activity))
    print("=" * 62)


def snapshot(sim: pd.DataFrame, label: str, date: str) -> None:
    """Print a one-line NAV snapshot at the first available date >= `date`."""
    ts  = pd.Timestamp(date)
    idx = sim.index[sim.index >= ts]
    if not len(idx):
        return
    row = sim.loc[idx[0]]
    print("  {:<28} {}  NAV=${:>12,.2f}  cash=${:>10,.2f}  equity=${:>10,.2f}".format(
          label, idx[0].date(),
          float(row["total"]), float(row["cash"]), float(row["equity"])))


def save_and_summarise(sim: pd.DataFrame, fname: str, prices: pd.DataFrame,
                       acc: FrecAccount, scenario_n: int) -> None:
    out = DATA_DIR / fname
    sim.to_csv(out)
    print("\n  Saved -> {}  ({} rows x {} cols)".format(
          out.name, len(sim), len(sim.columns)))

    p = acc.performance(sim)
    print("  TWR: {:+.2f}%   Ann. return: {:+.2f}%   Fees paid: ${:.2f}".format(
          p["twr"] * 100, p["ann_return"] * 100, p["total_fees"]))


# ── Scenario 1 : cash_inflow ──────────────────────────────────────────────────

def scenario_1(prices: pd.DataFrame, weights: pd.DataFrame) -> None:
    print_scenario_header(
        1, "Cash Inflow",
        "Deposit $50,000 cash on 2024-10-15 → rebalance to pre-activity weights")

    positions, residual = build_positions(prices, weights, BUDGET, TOP_N)

    acc = FrecAccount(name="S1 cash_inflow", annual_fee=ANNUAL_FEE)
    acc.cash_inflow("2024-10-15", 50_000)

    sim = acc.simulate(prices, initial_cash=residual, initial_positions=positions)

    ACTIVITY_DATE = "2024-10-15"
    ts = sim.index[sim.index >= pd.Timestamp(ACTIVITY_DATE)]
    prev_date = sim.index[sim.index < pd.Timestamp(ACTIVITY_DATE)]

    print("\n  Key snapshots:")
    snapshot(sim, "Day before cash inflow",  str(prev_date[-1].date()) if len(prev_date) else ACTIVITY_DATE)
    snapshot(sim, "Day of  cash inflow",     ACTIVITY_DATE)
    snapshot(sim, "End of period",           str(prices.index[-1].date()))

    nav_before = float(sim.loc[prev_date[-1], "total"]) if len(prev_date) else BUDGET
    nav_after  = float(sim.loc[ts[0], "total"])
    print("\n  NAV jump on activity date : ${:,.2f} -> ${:,.2f}  (delta ${:,.2f}, expected ~$50,000)".format(
          nav_before, nav_after, nav_after - nav_before))

    save_and_summarise(sim, "test_scenario_1_cash_inflow.csv", prices, acc, 1)


# ── Scenario 2 : cash_outflow ─────────────────────────────────────────────────

def scenario_2(prices: pd.DataFrame, weights: pd.DataFrame) -> None:
    print_scenario_header(
        2, "Cash Outflow",
        "Withdraw $20,000 on 2025-01-15 → rebalance to pre-activity weights")

    positions, residual = build_positions(prices, weights, BUDGET, TOP_N)

    acc = FrecAccount(name="S2 cash_outflow", annual_fee=ANNUAL_FEE)
    acc.cash_outflow("2025-01-15", 20_000)

    sim = acc.simulate(prices, initial_cash=residual, initial_positions=positions)

    ACTIVITY_DATE = "2025-01-15"
    ts = sim.index[sim.index >= pd.Timestamp(ACTIVITY_DATE)]
    prev_date = sim.index[sim.index < pd.Timestamp(ACTIVITY_DATE)]

    print("\n  Key snapshots:")
    snapshot(sim, "Day before cash outflow", str(prev_date[-1].date()) if len(prev_date) else ACTIVITY_DATE)
    snapshot(sim, "Day of  cash outflow",    ACTIVITY_DATE)
    snapshot(sim, "End of period",           str(prices.index[-1].date()))

    nav_before = float(sim.loc[prev_date[-1], "total"]) if len(prev_date) else BUDGET
    nav_after  = float(sim.loc[ts[0], "total"])
    print("\n  NAV drop on activity date : ${:,.2f} -> ${:,.2f}  (delta ${:,.2f}, expected ~-$20,000)".format(
          nav_before, nav_after, nav_after - nav_before))

    # Demonstrate the guard: try to withdraw more than portfolio NAV
    print("\n  Validation check -- attempt to withdraw more than portfolio NAV:")
    acc_bad = FrecAccount(name="guard-test", annual_fee=0)
    acc_bad.cash_outflow("2024-05-01", 999_999)
    try:
        acc_bad.simulate(prices, initial_cash=1_000.0, initial_positions={})
        print("  [FAIL] No error raised -- guard did not fire")
    except ValueError as e:
        print("  [OK] ValueError caught: {}".format(e))

    save_and_summarise(sim, "test_scenario_2_cash_outflow.csv", prices, acc, 2)


# ── Scenario 3 : stock_inflow ─────────────────────────────────────────────────

def scenario_3(prices: pd.DataFrame, weights: pd.DataFrame) -> None:
    print_scenario_header(
        3, "Stock Inflow",
        "Transfer in 30 AAPL shares on 2024-11-01 → rebalance to pre-activity weights")

    positions, residual = build_positions(prices, weights, BUDGET, TOP_N)

    acc = FrecAccount(name="S3 stock_inflow", annual_fee=ANNUAL_FEE)
    acc.stock_inflow("2024-11-01", "AAPL", 30)

    sim = acc.simulate(prices, initial_cash=residual, initial_positions=positions)

    ACTIVITY_DATE = "2024-11-01"
    ts = sim.index[sim.index >= pd.Timestamp(ACTIVITY_DATE)]
    prev_date = sim.index[sim.index < pd.Timestamp(ACTIVITY_DATE)]

    print("\n  Key snapshots:")
    snapshot(sim, "Day before stock inflow", str(prev_date[-1].date()) if len(prev_date) else ACTIVITY_DATE)
    snapshot(sim, "Day of  stock inflow",    ACTIVITY_DATE)
    snapshot(sim, "End of period",           str(prices.index[-1].date()))

    aapl_px    = float(prices.loc[ts[0], "AAPL"])
    inflow_val = 30 * aapl_px
    nav_before = float(sim.loc[prev_date[-1], "total"]) if len(prev_date) else BUDGET
    nav_after  = float(sim.loc[ts[0], "total"])
    print("\n  AAPL price on settle date    : ${:,.2f}".format(aapl_px))
    print("  In-kind value added          : ${:,.2f}  (30 × ${:,.2f})".format(inflow_val, aapl_px))
    print("  NAV jump on activity date    : ${:,.2f} -> ${:,.2f}  (delta ${:,.2f}, expected ~${:,.2f})".format(
          nav_before, nav_after, nav_after - nav_before, inflow_val))

    save_and_summarise(sim, "test_scenario_3_stock_inflow.csv", prices, acc, 3)


# ── Scenario 4 : stock_outflow ────────────────────────────────────────────────

def scenario_4(prices: pd.DataFrame, weights: pd.DataFrame) -> None:
    print_scenario_header(
        4, "Stock Outflow",
        "Transfer out half of initial NVDA shares on 2025-04-01 → rebalance")

    positions, residual = build_positions(prices, weights, BUDGET, TOP_N)

    # Determine initial NVDA share count
    nvda_initial = positions.get("NVDA", 0.0)
    nvda_half    = round(nvda_initial / 2, 4)

    print("\n  Initial NVDA position : {:.4f} shares".format(nvda_initial))
    print("  Shares to transfer out: {:.4f} (50%)".format(nvda_half))

    acc = FrecAccount(name="S4 stock_outflow", annual_fee=ANNUAL_FEE)
    acc.stock_outflow("2025-04-01", "NVDA", nvda_half)

    sim = acc.simulate(prices, initial_cash=residual, initial_positions=positions)

    ACTIVITY_DATE = "2025-04-01"
    ts = sim.index[sim.index >= pd.Timestamp(ACTIVITY_DATE)]
    prev_date = sim.index[sim.index < pd.Timestamp(ACTIVITY_DATE)]

    print("\n  Key snapshots:")
    snapshot(sim, "Day before stock outflow", str(prev_date[-1].date()) if len(prev_date) else ACTIVITY_DATE)
    snapshot(sim, "Day of  stock outflow",    ACTIVITY_DATE)
    snapshot(sim, "End of period",            str(prices.index[-1].date()))

    nvda_px     = float(prices.loc[ts[0], "NVDA"])
    outflow_val = nvda_half * nvda_px
    nav_before  = float(sim.loc[prev_date[-1], "total"]) if len(prev_date) else BUDGET
    nav_after   = float(sim.loc[ts[0], "total"])
    print("\n  NVDA price on settle date    : ${:,.2f}".format(nvda_px))
    print("  In-kind value removed        : ${:,.2f}  ({:.4f} × ${:,.2f})".format(
          outflow_val, nvda_half, nvda_px))
    print("  NAV drop on activity date    : ${:,.2f} -> ${:,.2f}  (delta ${:,.2f}, expected ~-${:,.2f})".format(
          nav_before, nav_after, nav_after - nav_before, outflow_val))

    # Demonstrate the negative-shares guard
    print("\n  Validation check -- attempt to transfer out more shares than held:")
    acc_bad = FrecAccount(name="guard-test", annual_fee=0)
    acc_bad.stock_outflow("2024-05-01", "NVDA", nvda_initial * 10)   # 10x held
    try:
        acc_bad.simulate(prices, initial_cash=0, initial_positions={"NVDA": nvda_initial})
        print("  [FAIL] No error raised -- guard did not fire")
    except ValueError as e:
        print("  [OK] ValueError caught: {}".format(e))

    save_and_summarise(sim, "test_scenario_4_stock_outflow.csv", prices, acc, 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data ...")
    prices, weights = load_data()
    print("  Price range : {} to {}  ({} trading days)".format(
          prices.index[0].date(), prices.index[-1].date(), len(prices)))
    print("  Top-{} Frec-weighted stocks  |  $100 k starting value".format(TOP_N))

    scenario_1(prices, weights)
    scenario_2(prices, weights)
    scenario_3(prices, weights)
    scenario_4(prices, weights)

    print("\n" + "=" * 62)
    print("  All 4 scenario CSVs saved to data/")
    print("=" * 62)


if __name__ == "__main__":
    main()

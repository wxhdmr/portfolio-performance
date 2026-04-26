"""
Unit tests for FrecAccount — rebalancing, simulation, and performance metrics.

Covers:
  - FrecAccount._rebalance()  — proportional weight preservation
  - FrecAccount.simulate()    — cash/stock in-out activities, fee accrual
  - FrecAccount.performance() — TWR, activity-date exclusion, guards
"""
import numpy as np
import pandas as pd
import pytest

from frec_account_activities import FrecAccount


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _flat_prices(n=10):
    """Constant prices — NAV moves only from activities or fees."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "AAPL": [100.0] * n,
        "MSFT": [200.0] * n,
        "SPY":  [400.0] * n,
    }, index=dates)


def _trending_prices(n=10, aapl_daily=0.01):
    """AAPL grows at aapl_daily per day; MSFT and SPY flat."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    aapl = [100.0 * (1 + aapl_daily) ** i for i in range(n)]
    return pd.DataFrame({
        "AAPL": aapl,
        "MSFT": [200.0] * n,
        "SPY":  [400.0] * n,
    }, index=dates)


# ── _rebalance ─────────────────────────────────────────────────────────────────

class TestRebalance:

    def _prices_and_day(self):
        prices = _flat_prices(3)
        return prices, prices.index[0]

    def test_total_equals_new_total(self):
        prices, day = self._prices_and_day()
        positions = {"AAPL": 1.0, "MSFT": 2.0}
        cash = 100.0
        new_total = 1_200.0
        new_pos, new_cash = FrecAccount._rebalance(positions, cash, prices, day, new_total)
        computed = new_pos["AAPL"] * 100 + new_pos["MSFT"] * 200 + new_cash
        assert abs(computed - new_total) < 1e-6

    def test_weights_are_proportional_to_pre_rebalance(self):
        """After rebalance, each asset's share of new_total equals its share of old_total."""
        prices, day = self._prices_and_day()
        positions = {"AAPL": 1.0, "MSFT": 2.0}
        cash = 100.0
        # Pre-rebalance: AAPL=100, MSFT=400, cash=100 → total=600
        old_total = 600.0
        new_total = 1_500.0
        new_pos, new_cash = FrecAccount._rebalance(positions, cash, prices, day, new_total)

        assert abs(new_pos["AAPL"] * 100 / new_total - 100 / old_total) < 1e-9
        assert abs(new_pos["MSFT"] * 200 / new_total - 400 / old_total) < 1e-9
        assert abs(new_cash / new_total - 100 / old_total) < 1e-9

    def test_empty_positions_all_cash(self):
        """With no equity positions the entire new_total goes to cash."""
        prices, day = self._prices_and_day()
        new_pos, new_cash = FrecAccount._rebalance({}, 10_000.0, prices, day, 20_000.0)
        assert new_pos == {}
        assert abs(new_cash - 20_000.0) < 1e-6

    def test_zero_old_total_returns_all_cash(self):
        """Edge case: old_total ≤ 0 → all new_total put in cash, empty positions."""
        prices, day = self._prices_and_day()
        new_pos, new_cash = FrecAccount._rebalance({}, 0.0, prices, day, 5_000.0)
        assert new_pos == {}
        assert abs(new_cash - 5_000.0) < 1e-6


# ── Input validation ───────────────────────────────────────────────────────────

class TestInputValidation:

    def test_cash_inflow_zero_amount_raises(self):
        acc = FrecAccount()
        with pytest.raises(ValueError):
            acc.cash_inflow("2024-01-02", 0)

    def test_cash_inflow_negative_amount_raises(self):
        acc = FrecAccount()
        with pytest.raises(ValueError):
            acc.cash_inflow("2024-01-02", -100)

    def test_cash_outflow_zero_amount_raises(self):
        acc = FrecAccount()
        with pytest.raises(ValueError):
            acc.cash_outflow("2024-01-02", 0)

    def test_stock_inflow_zero_shares_raises(self):
        acc = FrecAccount()
        with pytest.raises(ValueError):
            acc.stock_inflow("2024-01-02", "AAPL", 0)

    def test_stock_outflow_zero_shares_raises(self):
        acc = FrecAccount()
        with pytest.raises(ValueError):
            acc.stock_outflow("2024-01-02", "AAPL", 0)

    def test_cash_outflow_exceeds_nav_raises_during_simulation(self):
        prices = _flat_prices()
        acc = FrecAccount()
        acc.cash_outflow("2024-01-02", 999_999)
        with pytest.raises(ValueError, match="cash_outflow"):
            acc.simulate(prices, initial_cash=10_000)

    def test_stock_outflow_exceeds_held_raises_during_simulation(self):
        prices = _flat_prices()
        acc = FrecAccount()
        acc.stock_outflow("2024-01-02", "AAPL", 1_000)  # only 10 held
        with pytest.raises(ValueError, match="stock_outflow"):
            acc.simulate(prices, initial_cash=0, initial_positions={"AAPL": 10})


# ── simulate — basic mechanics ─────────────────────────────────────────────────

class TestSimulateMechanics:

    def test_no_activity_zero_fee_flat_prices_constant_nav(self):
        """Without activities or fees, NAV is exactly constant on flat prices."""
        prices = _flat_prices()
        acc = FrecAccount(annual_fee=0)
        sim = acc.simulate(prices, initial_cash=10_000, initial_positions={"AAPL": 50})
        expected_nav = 10_000 + 50 * 100  # cash + equity
        assert (sim["total"] - expected_nav).abs().max() < 1e-6

    def test_fee_accrual_reduces_nav(self):
        """With a management fee and flat prices, NAV must decrease each day."""
        prices = _flat_prices(5)
        acc = FrecAccount(annual_fee=0.01)  # 1% annual for visible effect
        sim = acc.simulate(prices, initial_cash=10_000)
        assert sim["total"].iloc[-1] < sim["total"].iloc[0]
        assert (sim["total"].diff().iloc[1:] <= 0).all()

    def test_nav_grows_with_rising_prices(self):
        """With rising prices and no fee, NAV increases each day."""
        prices = _trending_prices(aapl_daily=0.01)
        acc = FrecAccount(annual_fee=0)
        sim = acc.simulate(prices, initial_cash=0, initial_positions={"AAPL": 100})
        assert sim["total"].iloc[-1] > sim["total"].iloc[0]

    def test_result_index_matches_prices_index(self):
        prices = _flat_prices()
        acc = FrecAccount()
        sim = acc.simulate(prices, initial_cash=5_000)
        assert list(sim.index) == list(prices.index)

    def test_cash_and_equity_sum_to_total(self):
        prices = _flat_prices()
        acc = FrecAccount(annual_fee=0)
        sim = acc.simulate(prices, initial_cash=5_000, initial_positions={"AAPL": 10})
        diff = (sim["cash"] + sim["equity"] - sim["total"]).abs()
        assert diff.max() < 1e-6


# ── simulate — activity mechanics ─────────────────────────────────────────────

class TestSimulateActivities:

    def test_cash_inflow_increases_nav_by_amount(self):
        """NAV on the activity date should jump by exactly the inflow amount."""
        prices = _flat_prices(10)
        activity_date = prices.index[4]  # day 5
        acc = FrecAccount(annual_fee=0)
        acc.cash_inflow(str(activity_date.date()), 5_000)
        sim = acc.simulate(prices, initial_cash=10_000)

        before = sim["total"].iloc[3]
        after  = sim["total"].iloc[4]
        assert abs((after - before) - 5_000) < 1e-6

    def test_cash_outflow_decreases_nav_by_amount(self):
        prices = _flat_prices(10)
        activity_date = prices.index[4]
        acc = FrecAccount(annual_fee=0)
        acc.cash_outflow(str(activity_date.date()), 3_000)
        sim = acc.simulate(prices, initial_cash=10_000)

        before = sim["total"].iloc[3]
        after  = sim["total"].iloc[4]
        assert abs((before - after) - 3_000) < 1e-6

    def test_stock_inflow_increases_nav_by_share_value(self):
        """NAV increase = shares × AAPL price ($100 flat), so 20 shares → +$2,000."""
        prices = _flat_prices(10)
        activity_date = prices.index[4]
        acc = FrecAccount(annual_fee=0)
        acc.stock_inflow(str(activity_date.date()), "AAPL", 20)
        sim = acc.simulate(prices, initial_cash=10_000)

        before = sim["total"].iloc[3]
        after  = sim["total"].iloc[4]
        assert abs((after - before) - 20 * 100) < 1e-6

    def test_stock_outflow_decreases_nav_by_share_value(self):
        """NAV drop = shares × AAPL price ($100), so 5 shares → -$500."""
        prices = _flat_prices(10)
        activity_date = prices.index[4]
        acc = FrecAccount(annual_fee=0)
        acc.stock_outflow(str(activity_date.date()), "AAPL", 5)
        sim = acc.simulate(prices, initial_cash=0, initial_positions={"AAPL": 100})

        before = sim["total"].iloc[3]
        after  = sim["total"].iloc[4]
        assert abs((before - after) - 5 * 100) < 1e-6

    def test_multiple_activities_applied_in_order(self):
        """Two activities on the same day are applied sequentially."""
        prices = _flat_prices(10)
        activity_date = prices.index[4]
        acc = FrecAccount(annual_fee=0)
        acc.cash_inflow(str(activity_date.date()), 5_000)
        acc.cash_inflow(str(activity_date.date()), 5_000)  # second inflow same day
        sim = acc.simulate(prices, initial_cash=10_000)

        before = sim["total"].iloc[3]
        after  = sim["total"].iloc[4]
        assert abs((after - before) - 10_000) < 1e-6


# ── performance — metrics and activity exclusion ───────────────────────────────

class TestPerformance:

    def test_twr_is_zero_for_flat_prices_no_fee(self):
        """0% price movement + 0% fee → TWR = 0%."""
        prices = _flat_prices()
        acc = FrecAccount(annual_fee=0)
        sim = acc.simulate(prices, initial_cash=10_000, initial_positions={"AAPL": 50})
        perf = acc.performance(sim)
        assert abs(perf["twr"]) < 1e-9

    def test_activity_dates_excluded_from_twr(self):
        """
        With flat prices the true market return is 0%.
        A cash inflow causes a large artificial NAV jump on the activity date.
        After exclusion TWR must still be ≈ 0%, not inflated by the jump.
        """
        prices = _flat_prices(10)
        activity_date = prices.index[4]
        acc = FrecAccount(annual_fee=0)
        acc.cash_inflow(str(activity_date.date()), 50_000)   # 500% of initial NAV
        sim = acc.simulate(prices, initial_cash=10_000)

        perf = acc.performance(sim)
        # If the activity date were included, pct_change would show ~500% return
        assert abs(perf["twr"]) < 1e-6, \
            f"TWR should be ~0 but got {perf['twr']:.4f} — activity date not excluded"

    def test_total_fees_accumulate_correctly(self):
        """Fees paid over 10 flat-price days with 1% annual fee."""
        prices = _flat_prices(10)
        annual_fee = 0.01
        initial_nav = 10_000.0
        acc = FrecAccount(annual_fee=annual_fee)
        sim = acc.simulate(prices, initial_cash=initial_nav)
        perf = acc.performance(sim)

        daily_fee = annual_fee / 252
        # Each day's fee is on a slightly lower NAV (compounding), but over 10 days
        # the total should be close to initial_nav × daily_fee × 10
        expected_approx = initial_nav * daily_fee * 10
        assert abs(perf["total_fees"] - expected_approx) / expected_approx < 0.01

    def test_performance_returns_required_keys(self):
        prices = _flat_prices()
        acc = FrecAccount()
        sim = acc.simulate(prices, initial_cash=10_000)
        perf = acc.performance(sim)
        for key in ("twr", "ann_return", "ann_vol", "sharpe", "max_drawdown", "total_fees"):
            assert key in perf, f"Missing key: {key}"

    def test_benchmark_adds_tracking_error_and_ir(self):
        prices = _flat_prices()
        acc = FrecAccount()
        sim = acc.simulate(prices, initial_cash=10_000)
        spy = prices["SPY"]
        perf = acc.performance(sim, benchmark=spy)
        for key in ("tracking_error", "active_return", "ir"):
            assert key in perf, f"Missing key: {key}"
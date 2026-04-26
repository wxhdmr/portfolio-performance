"""
Unit tests for return calculation logic.

Covers:
  - apply_tilt()         in src/build_frec_weights.py
  - compute_performance() in src/build_frec_weights.py
  - compute_returns()    in src/portfolio_stats.py
"""
import numpy as np
import pandas as pd
import pytest

from build_frec_weights import apply_tilt, compute_performance, TILT_STOCKS, TILT_PP
from portfolio_stats import compute_returns


# ── Helpers ────────────────────────────────────────────────────────────────────

def _quarterly_weights():
    """Two-quarter weight matrix summing to 100% — mix of TILT_STOCKS and non-tilt stocks."""
    return pd.DataFrame({
        "NVDA":  [6.0,  6.0],
        "GOOG":  [4.0,  4.0],
        "GOOGL": [2.0,  2.0],
        "AVGO":  [3.0,  3.0],
        "AAPL":  [45.0, 45.0],
        "MSFT":  [40.0, 40.0],
    }, index=pd.to_datetime(["2024-03-31", "2024-06-30"]))


# ── apply_tilt ─────────────────────────────────────────────────────────────────

class TestApplyTilt:

    def test_tilt_stocks_weight_increases(self):
        q = _quarterly_weights()
        tilted = apply_tilt(q)
        for tkr in TILT_STOCKS:
            if tkr in q.columns:
                assert tilted[tkr].iloc[0] > q[tkr].iloc[0], \
                    f"{tkr} should have a higher weight after tilt"

    def test_non_tilt_stocks_weight_decreases(self):
        q = _quarterly_weights()
        tilted = apply_tilt(q)
        for tkr in q.columns:
            if tkr not in TILT_STOCKS:
                assert tilted[tkr].iloc[0] < q[tkr].iloc[0], \
                    f"{tkr} should have a lower weight after rescaling"

    def test_each_row_sums_to_100(self):
        q = _quarterly_weights()
        tilted = apply_tilt(q)
        for i in range(len(tilted)):
            assert abs(tilted.iloc[i].sum() - 100.0) < 1e-9, \
                f"Row {i} sums to {tilted.iloc[i].sum()}, expected 100"

    def test_no_negative_weights(self):
        q = _quarterly_weights()
        tilted = apply_tilt(q)
        assert (tilted >= 0).all().all()

    def test_does_not_mutate_input(self):
        q = _quarterly_weights()
        nvda_before = q["NVDA"].iloc[0]
        apply_tilt(q)
        assert q["NVDA"].iloc[0] == nvda_before

    def test_tilt_magnitude_before_rescaling(self):
        """The raw addition to tilt stocks equals TILT_PP before rescaling."""
        q = pd.DataFrame({"NVDA": [3.0], "AAPL": [97.0]},
                         index=pd.to_datetime(["2024-03-31"]))
        tilted_raw = q.copy()
        tilted_raw["NVDA"] += TILT_PP
        # After rescaling, NVDA share should be tilted_raw/sum × 100
        expected_nvda = (3.0 + TILT_PP) / (100.0 + TILT_PP) * 100
        tilted = apply_tilt(q)
        assert abs(tilted["NVDA"].iloc[0] - expected_nvda) < 1e-9


# ── compute_performance (build_frec_weights) ───────────────────────────────────

class TestComputePerformance:
    """Tests compute_performance() which uses lagged daily weights."""

    def _inputs_100pct_aapl(self):
        """4 trading days, 100% weight on AAPL throughout."""
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0, 110.0, 121.0, 133.1],  # +10%/day
            "MSFT": [200.0, 200.0, 200.0, 200.0],
            "SPY":  [400.0, 420.0, 441.0, 463.05],  # +5%/day
        }, index=dates)
        weights = pd.DataFrame(
            {"AAPL": [100.0] * 4, "MSFT": [0.0] * 4}, index=dates)
        return weights, prices

    def test_output_has_correct_columns(self):
        w, p = self._inputs_100pct_aapl()
        result = compute_performance(w, p)
        assert set(result.columns) == {"port_return", "spy_return", "active_return"}

    def test_active_return_equals_port_minus_spy(self):
        """active_return must be exactly port_return − spy_return at every row."""
        w, p = self._inputs_100pct_aapl()
        result = compute_performance(w, p)
        diff = (result["active_return"]
                - (result["port_return"] - result["spy_return"])).abs()
        assert diff.max() < 1e-12

    def test_100pct_single_stock_port_return_matches_stock(self):
        """With 100% weight on AAPL, port_return must equal AAPL's daily return."""
        w, p = self._inputs_100pct_aapl()
        result = compute_performance(w, p)
        aapl_ret = p["AAPL"].pct_change().dropna()
        common = result.index.intersection(aapl_ret.index)
        np.testing.assert_allclose(
            result.loc[common, "port_return"].values,
            aapl_ret.loc[common].values,
            rtol=1e-9,
        )

    def test_spy_return_column_matches_spy_pct_change(self):
        w, p = self._inputs_100pct_aapl()
        result = compute_performance(w, p)
        spy_ret = p["SPY"].pct_change().dropna()
        common = result.index.intersection(spy_ret.index)
        np.testing.assert_allclose(
            result.loc[common, "spy_return"].values,
            spy_ret.loc[common].values,
            rtol=1e-9,
        )

    def test_flat_prices_produce_zero_returns(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0] * 5,
            "SPY":  [400.0] * 5,
        }, index=dates)
        weights = pd.DataFrame({"AAPL": [100.0] * 5}, index=dates)
        result = compute_performance(weights, prices)
        assert result["port_return"].abs().max() < 1e-12
        assert result["spy_return"].abs().max() < 1e-12
        assert result["active_return"].abs().max() < 1e-12

    def test_uses_lagged_weights_not_current(self):
        """
        Weights switch from 100% AAPL (day 1) to 100% MSFT (day 2+).
        Day-2 return must reflect day-1 weight (AAPL), not day-2 weight (MSFT).
        """
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0, 110.0, 115.0],  # +10% day-2, +4.55% day-3
            "MSFT": [200.0, 200.0, 210.0],  # flat day-2, +5% day-3
            "SPY":  [400.0, 402.0, 404.0],
        }, index=dates)
        weights = pd.DataFrame({
            "AAPL": [100.0, 0.0, 0.0],
            "MSFT": [  0.0, 100.0, 100.0],
        }, index=dates)
        result = compute_performance(weights, prices)
        # Day 2: lagged = 100% AAPL → port_ret = +10%
        assert abs(result.loc[dates[1], "port_return"] - 0.10) < 1e-9
        # Day 3: lagged = 100% MSFT → port_ret = +5%
        assert abs(result.loc[dates[2], "port_return"] - 0.05) < 1e-9


# ── compute_returns (portfolio_stats) ─────────────────────────────────────────

class TestComputeReturns:
    """Tests the portfolio_stats version of the return computation."""

    def _inputs_equal_weights(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0, 105.0, 110.25, 115.76],  # +5%/day
            "MSFT": [200.0, 210.0, 220.50, 231.53],  # +5%/day
            "SPY":  [400.0, 420.0, 441.00, 463.05],  # +5%/day
        }, index=dates)
        weights = pd.DataFrame(
            {"AAPL": [50.0] * 4, "MSFT": [50.0] * 4}, index=dates)
        return weights, prices

    def test_output_columns(self):
        w, p = self._inputs_equal_weights()
        result = compute_returns(w, p)
        assert set(result.columns) == {"port_return", "spy_return", "active_return"}

    def test_active_return_identity(self):
        w, p = self._inputs_equal_weights()
        result = compute_returns(w, p)
        diff = (result["active_return"]
                - (result["port_return"] - result["spy_return"])).abs()
        assert diff.max() < 1e-12

    def test_portfolio_matching_spy_has_near_zero_active_return(self):
        """50/50 AAPL/MSFT both moving exactly like SPY → active ≈ 0."""
        w, p = self._inputs_equal_weights()
        result = compute_returns(w, p)
        # All three assets move at +5%/day, so port_return ≈ spy_return
        assert result["active_return"].abs().max() < 1e-9

    def test_returns_are_not_empty(self):
        w, p = self._inputs_equal_weights()
        result = compute_returns(w, p)
        assert len(result) > 0
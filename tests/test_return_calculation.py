"""
Unit tests for return calculation logic.

Covers:
  - apply_tilt()      in src/build_frec_weights.py
  - compute_returns() in src/portfolio_metrics.py  (used by all src files)
  - compute_stats()   in src/portfolio_metrics.py
"""
import numpy as np
import pandas as pd
import pytest

from build_frec_weights import apply_tilt, TILT_STOCKS, TILT_PP
from portfolio_metrics import compute_returns, compute_stats


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


# ── compute_returns (portfolio_metrics — canonical shared function) ────────────

class TestComputeReturns:
    """Tests the shared compute_returns() from portfolio_metrics."""

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

    def test_100pct_single_stock_port_return_matches_stock(self):
        """With 100% weight on AAPL, port_return must equal AAPL's daily return."""
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0, 110.0, 121.0, 133.1],
            "SPY":  [400.0, 420.0, 441.0, 463.05],
        }, index=dates)
        weights = pd.DataFrame({"AAPL": [100.0] * 4}, index=dates)
        result  = compute_returns(weights, prices)
        aapl_ret = prices["AAPL"].pct_change().dropna()
        common   = result.index.intersection(aapl_ret.index)
        np.testing.assert_allclose(
            result.loc[common, "port_return"].values,
            aapl_ret.loc[common].values, rtol=1e-9)

    def test_uses_lagged_weights_not_current(self):
        """Weights switch from 100% AAPL → 100% MSFT on day 2; day-2 return
        must use day-1 weight (AAPL)."""
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0, 110.0, 115.0],
            "MSFT": [200.0, 200.0, 210.0],
            "SPY":  [400.0, 402.0, 404.0],
        }, index=dates)
        weights = pd.DataFrame({
            "AAPL": [100.0, 0.0, 0.0],
            "MSFT": [  0.0, 100.0, 100.0],
        }, index=dates)
        result = compute_returns(weights, prices)
        assert abs(result.loc[dates[1], "port_return"] - 0.10) < 1e-9  # AAPL +10%
        assert abs(result.loc[dates[2], "port_return"] - 0.05) < 1e-9  # MSFT +5%

    def test_flat_prices_produce_zero_returns(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0] * 5, "SPY": [400.0] * 5,
        }, index=dates)
        weights = pd.DataFrame({"AAPL": [100.0] * 5}, index=dates)
        result  = compute_returns(weights, prices)
        assert result["port_return"].abs().max() < 1e-12
        assert result["active_return"].abs().max() < 1e-12


# ── compute_stats (portfolio_metrics) ─────────────────────────────────────────

class TestComputeStats:
    """Tests the shared compute_stats() from portfolio_metrics."""

    def _returns(self):
        """Known daily returns: port +10%/day, spy +5%/day → 4 observations."""
        dates = pd.date_range("2024-01-03", periods=4, freq="B")
        port  = pd.Series([0.10] * 4, index=dates)
        spy   = pd.Series([0.05] * 4, index=dates)
        return pd.DataFrame({
            "port_return":   port,
            "spy_return":    spy,
            "active_return": port - spy,
        })

    def test_returns_expected_keys(self):
        st = compute_stats(self._returns())
        for key in ("ann_return", "ann_spy", "ann_active", "te", "ir",
                    "vol", "sharpe", "max_dd", "n_days", "start", "end"):
            assert key in st, f"Missing key: {key}"

    def test_ann_return_geometric(self):
        """Annualised return = (1.10^4)^(252/4) - 1 = 1.10^252 - 1."""
        st = compute_stats(self._returns())
        expected = (1.10 ** 252) - 1
        np.testing.assert_allclose(st["ann_return"], expected, rtol=1e-9)

    def test_active_return_arithmetic(self):
        """ann_active = ann_port - ann_spy (arithmetic, not geometric)."""
        st = compute_stats(self._returns())
        assert abs(st["ann_active"] - (st["ann_return"] - st["ann_spy"])) < 1e-9

    def test_tracking_error_formula(self):
        """TE = std(active_return) * sqrt(252). With constant active returns, std=0."""
        st = compute_stats(self._returns())
        assert abs(st["te"]) < 1e-12

    def test_max_drawdown_non_positive(self):
        st = compute_stats(self._returns())
        assert st["max_dd"] <= 0

    def test_n_days_matches_input(self):
        ret = self._returns()
        st  = compute_stats(ret)
        assert st["n_days"] == len(ret)
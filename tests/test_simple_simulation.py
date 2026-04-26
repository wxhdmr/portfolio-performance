"""
Unit tests for the synthetic simulation functions in simulations/simple_simulation.py.

Covers:
  - simulate_return_with_TE_ER() — TE/ER construction, initial value, CSV output
  - validate()                   — realized metric computation
  - run_simulation()             — end-to-end pipeline
"""
import numpy as np
import pandas as pd
import pytest

from simple_simulation import simulate_return_with_TE_ER, validate, run_simulation


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_spy(n=300, seed=999):
    """Synthetic SPY closing-price series."""
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01
    prices = np.cumprod(1 + returns) * 400.0
    return pd.Series(
        prices,
        index=pd.date_range("2023-01-03", periods=n, freq="B"),
        name="SPY",
    )


TARGET_TE   = 0.02
TARGET_ER   = 0.01
ANNUAL_FEE  = 0.0009
INIT_VALUE  = 100_000.0
TOLERANCE   = 1e-4


# ── simulate_return_with_TE_ER ─────────────────────────────────────────────────

class TestSimulateReturnWithTEER:

    def test_output_starts_at_initial_value(self):
        spy = _make_spy()
        port = simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
        )
        assert abs(port.iloc[0] - INIT_VALUE) < 1e-6

    def test_output_length_matches_spy(self):
        spy = _make_spy()
        port = simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
        )
        assert len(port) == len(spy)

    def test_output_index_matches_spy(self):
        spy = _make_spy()
        port = simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
        )
        assert list(port.index) == list(spy.index)

    def test_achieved_te_within_tolerance(self):
        spy = _make_spy()
        port = simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
        )
        spy_ret  = spy.pct_change().dropna()
        port_ret = port.pct_change().dropna()
        active   = port_ret - spy_ret.reindex(port_ret.index)
        ach_te   = float(active.std() * np.sqrt(252))
        assert abs(ach_te - TARGET_TE) <= TOLERANCE, \
            f"TE {ach_te:.6f} deviates from target {TARGET_TE} by more than {TOLERANCE}"

    def test_achieved_er_within_tolerance(self):
        spy = _make_spy()
        port = simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
        )
        spy_ret  = spy.pct_change().dropna()
        port_ret = port.pct_change().dropna()
        active   = port_ret - spy_ret.reindex(port_ret.index)
        ach_er   = float(active.mean() * 252)
        assert abs(ach_er - TARGET_ER) <= TOLERANCE, \
            f"ER {ach_er:.6f} deviates from target {TARGET_ER} by more than {TOLERANCE}"

    def test_accepts_dataframe_with_spy_column(self):
        """Passing a DataFrame (with a 'SPY' column) should work identically to a Series."""
        spy = _make_spy()
        df  = pd.DataFrame({"SPY": spy, "OTHER": spy * 1.1})
        port_from_df     = simulate_return_with_TE_ER(df, TARGET_TE, TARGET_ER, ANNUAL_FEE,
                                                      initial_value=INIT_VALUE, tolerance=TOLERANCE)
        port_from_series = simulate_return_with_TE_ER(spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
                                                      initial_value=INIT_VALUE, tolerance=TOLERANCE)
        pd.testing.assert_series_equal(port_from_df, port_from_series)

    def test_csv_saved_at_custom_output_path(self, tmp_path):
        spy = _make_spy()
        out = tmp_path / "sim_test.csv"
        simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
            output_path=out,
        )
        assert out.exists()
        df = pd.read_csv(out, index_col=0)
        assert "port_close" in df.columns
        assert "spy_close" in df.columns

    def test_csv_spy_close_starts_at_initial_value(self, tmp_path):
        """The saved spy_close series should be scaled to start at initial_value."""
        spy = _make_spy()
        out = tmp_path / "sim_spy.csv"
        simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
            output_path=out,
        )
        df = pd.read_csv(out, index_col=0)
        assert abs(df["spy_close"].iloc[0] - INIT_VALUE) < 1e-3

    def test_insufficient_data_raises(self):
        spy = pd.Series([400.0], index=pd.DatetimeIndex(["2024-01-02"]))
        with pytest.raises(ValueError, match="at least 2"):
            simulate_return_with_TE_ER(spy, TARGET_TE, TARGET_ER, ANNUAL_FEE)

    def test_tight_tolerance_raises_value_error(self):
        """tolerance=0.0 will always raise because FP arithmetic is not exact."""
        spy = _make_spy()
        with pytest.raises(ValueError):
            simulate_return_with_TE_ER(
                spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
                initial_value=INIT_VALUE, tolerance=0.0,
            )

    def test_reproducibility_with_same_seed(self):
        spy = _make_spy()
        port1 = simulate_return_with_TE_ER(spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
                                           initial_value=INIT_VALUE, tolerance=TOLERANCE,
                                           random_seed=7)
        port2 = simulate_return_with_TE_ER(spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
                                           initial_value=INIT_VALUE, tolerance=TOLERANCE,
                                           random_seed=7)
        pd.testing.assert_series_equal(port1, port2)

    def test_different_seeds_produce_different_series(self):
        spy = _make_spy()
        port1 = simulate_return_with_TE_ER(spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
                                           initial_value=INIT_VALUE, tolerance=TOLERANCE,
                                           random_seed=1)
        port2 = simulate_return_with_TE_ER(spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
                                           initial_value=INIT_VALUE, tolerance=TOLERANCE,
                                           random_seed=2)
        assert not port1.equals(port2)


# ── validate ───────────────────────────────────────────────────────────────────

class TestValidate:

    def _make_port_and_spy(self):
        spy = _make_spy()
        port = simulate_return_with_TE_ER(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
        )
        spy_scaled = spy / float(spy.iloc[0]) * INIT_VALUE
        return port, spy_scaled

    def test_returns_expected_keys(self):
        port, spy = self._make_port_and_spy()
        stats = validate(port, spy, annual_fee=ANNUAL_FEE)
        for key in ("ann_port_ret", "ann_spy_ret", "ann_er", "ann_te",
                    "ir", "sharpe", "max_dd", "cum_port_ret", "cum_spy_ret",
                    "n_days", "annual_fee"):
            assert key in stats, f"Missing key: {key}"

    def test_realized_te_matches_simulation_target(self):
        port, spy = self._make_port_and_spy()
        stats = validate(port, spy, annual_fee=ANNUAL_FEE)
        assert abs(stats["ann_te"] - TARGET_TE) <= TOLERANCE * 5

    def test_realized_er_matches_simulation_target(self):
        port, spy = self._make_port_and_spy()
        stats = validate(port, spy, annual_fee=ANNUAL_FEE)
        assert abs(stats["ann_er"] - TARGET_ER) <= TOLERANCE * 5

    def test_active_return_equals_ann_port_minus_ann_spy_approx(self):
        """ann_er ≈ ann_port_ret − ann_spy_ret (approximation via geometric returns)."""
        port, spy = self._make_port_and_spy()
        stats = validate(port, spy, annual_fee=ANNUAL_FEE)
        implied_er = stats["ann_port_ret"] - stats["ann_spy_ret"]
        # Geometric vs arithmetic difference — allow 5 bp slack
        assert abs(stats["ann_er"] - implied_er) < 0.005

    def test_max_drawdown_is_non_positive(self):
        port, spy = self._make_port_and_spy()
        stats = validate(port, spy, annual_fee=ANNUAL_FEE)
        assert stats["max_dd"] <= 0

    def test_n_days_matches_series_length(self):
        port, spy = self._make_port_and_spy()
        stats = validate(port, spy, annual_fee=ANNUAL_FEE)
        # n_days = pct_change().dropna() length = len(series) - 1
        assert stats["n_days"] == len(port) - 1

    def test_annual_fee_stored_in_output(self):
        port, spy = self._make_port_and_spy()
        stats = validate(port, spy, annual_fee=ANNUAL_FEE)
        assert stats["annual_fee"] == ANNUAL_FEE


# ── run_simulation ─────────────────────────────────────────────────────────────

class TestRunSimulation:

    def test_returns_stats_dict(self, tmp_path):
        spy = _make_spy()
        df  = pd.DataFrame({"SPY": spy})
        stats = run_simulation(
            df, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
            output_path=tmp_path / "sim.csv",
            chart_path=tmp_path / "chart.png",
        )
        assert isinstance(stats, dict)
        assert "ann_te" in stats

    def test_chart_saved(self, tmp_path):
        spy = _make_spy()
        chart = tmp_path / "chart.png"
        run_simulation(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
            output_path=tmp_path / "sim.csv",
            chart_path=chart,
        )
        assert chart.exists()

    def test_csv_saved(self, tmp_path):
        spy = _make_spy()
        out = tmp_path / "sim.csv"
        run_simulation(
            spy, TARGET_TE, TARGET_ER, ANNUAL_FEE,
            initial_value=INIT_VALUE, tolerance=TOLERANCE,
            output_path=out,
            chart_path=tmp_path / "chart.png",
        )
        assert out.exists()

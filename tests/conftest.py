"""
Shared fixtures and path setup for all test modules.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import numpy as np
import pandas as pd
import pytest

# Make src/ and simulations/ importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulations"))


# ── Price fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def flat_prices():
    """Constant prices — NAV changes come only from fees or activities."""
    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    return pd.DataFrame({
        "AAPL": [100.0] * 10,
        "MSFT": [200.0] * 10,
        "SPY":  [400.0] * 10,
    }, index=dates)


@pytest.fixture
def trending_prices():
    """
    AAPL: +10%/day  MSFT: flat  SPY: +5%/day
    Makes expected returns easy to compute analytically.
    """
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    aapl = [100.0 * (1.10 ** i) for i in range(5)]
    msft = [200.0] * 5
    spy  = [400.0 * (1.05 ** i) for i in range(5)]
    return pd.DataFrame({"AAPL": aapl, "MSFT": msft, "SPY": spy}, index=dates)


@pytest.fixture
def synthetic_spy():
    """Synthetic SPY closing-price series (300 trading days) for simulation tests."""
    rng = np.random.default_rng(999)
    spy_ret = rng.standard_normal(300) * 0.01
    prices = np.cumprod(1 + spy_ret) * 400.0
    return pd.Series(
        prices,
        index=pd.date_range("2023-01-03", periods=300, freq="B"),
        name="SPY",
    )
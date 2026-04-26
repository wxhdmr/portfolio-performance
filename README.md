# Portfolio Performance — Frec Direct Index Analysis

End-to-end toolkit for building, simulating, and analysing a Frec-style
direct-index S&P 500 portfolio. Covers data collection, weight construction,
performance attribution, account activity simulation, and synthetic return
generation.

## Repo structure

```
portfolio_performance/
├── src/
│   ├── fetch_spy_data.py              # download 2 yrs of SPY daily closes from Yahoo Finance
│   ├── fetch_sp500_weights.py         # scrape quarterly S&P 500 weights from SEC EDGAR NPORT-P
│   ├── build_frec_weights.py          # tilt quarterly weights → daily weights; compute returns
│   ├── portfolio_stats.py             # tracking error, return, and risk metrics vs SPY
│   ├── chart_portfolio_comparison.py  # dark-theme NAV chart: Frec vs SPY
│   ├── portfolio_account.py           # PortfolioAccount data class
│   └── reconcile_weights.py           # reconcile weight sources
│
├── simulations/
│   ├── frec_account_activities.py     # FrecAccount: cash/stock in-out with rebalancing
│   ├── simulate_frec_portfolio.py     # simulate_frec_portfolio(): holdings CSV + chart
│   ├── simple_simulation.py           # run_simulation(): synthetic TE/ER portfolio
│   ├── test_scenarios.py              # four scenario tests (cash/stock in-out)
│   └── account_activity_demo.py       # standalone activity demo
│
├── data/                              # generated outputs (CSVs, PNGs)
├── notebooks/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Data pipeline

Run scripts in order from the project root:

```bash
# 1. Download SPY daily closing prices
py src/fetch_spy_data.py

# 2. Scrape S&P 500 constituent weights from SEC EDGAR
py src/fetch_sp500_weights.py

# 3. Build Frec-tilted weights and compute daily returns
py src/build_frec_weights.py

# 4. Build S&P 500 daily weights
py src/reconcile_weights.py

# 5. Compute portfolio statistics (TE, IR, Sharpe, drawdown)
py src/portfolio_stats.py

# 6. Generate Frec vs SPY comparison chart
py src/chart_portfolio_comparison.py
```

Key outputs in `data/`:

| File | Description |
|---|---|
| `sp500_closing_prices.csv` | Daily closing prices for all S&P 500 constituents + SPY |
| `frec_weights_daily.csv` | Daily portfolio weights (Frec-tilted) |
| `portfolio_comparison_chart.png` | NAV chart: Frec net vs SPY |

## Simulations

### Synthetic portfolio simulation

Generate a portfolio that exactly matches a target tracking error and net
excess return vs SPY, then validate and chart it in one call:

```python
import pandas as pd
from simulations.simple_simulation import run_simulation

prices = pd.read_csv("data/sp500_closing_prices.csv", index_col=0, parse_dates=True)

stats = run_simulation(
    spy_closing_prices = prices,
    target_annual_te   = 0.02,    # 2.0% tracking error
    target_annual_er   = 0.01,    # +1.0% net excess return
    annual_fee         = 0.0009,
    initial_value      = 100_000,
    tolerance          = 1e-4,    # raises ValueError if targets not met
)
```

Individual functions are also importable: `simulate_return_with_TE_ER`,
`validate`, `plot_simulation`.

### Frec portfolio with account activities

`FrecAccount` supports four activity types — each triggers an immediate
rebalance proportional to pre-activity weights, with NAV adjusted by the
cash value of the flow:

| Activity | Effect |
|---|---|
| `cash_inflow(date, amount)` | NAV += amount; rebalance |
| `cash_outflow(date, amount)` | NAV -= amount; rebalance |
| `stock_inflow(date, ticker, shares)` | NAV += shares × price; rebalance |
| `stock_outflow(date, ticker, shares)` | NAV -= shares × price; rebalance |

Activity dates are excluded from TE, drawdown, and return calculations.
The SPY benchmark receives the same dollar flow on each activity date for
a fair comparison.

```python
from simulations.frec_account_activities import FrecAccount

acc = FrecAccount(name="my portfolio", annual_fee=0.0009)
acc.cash_inflow("2024-10-15", 50_000)
acc.stock_outflow("2025-04-01", "NVDA", 10)

sim = acc.simulate(prices, initial_cash=500, initial_positions={"AAPL": 5})
```

Or use the higher-level wrapper that generates a holdings CSV and chart:

```python
from simulations.simulate_frec_portfolio import simulate_frec_portfolio

simulate_frec_portfolio(
    initial_value=200000,
    activities={
        "2024-10-15": {"activitytype": "cash",  "value":  50000},   # deposit
        "2025-01-15": {"activitytype": "cash",  "value": -20000},   # withdrawal
        "2024-11-01": {"activitytype": "AAPL",  "value":  300},     # +300 shares in
        "2025-04-01": {"activitytype": "NVDA",  "value": -200},     # -200 shares out
    },
    output_prefix="frec_simulation",   # → data/frec_simulation.csv + _chart.png
)
```

Run the four built-in test scenarios:

```bash
py simulations/test_scenarios.py
```

## Data source

| Field | Value |
|---|---|
| Price data | Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance) |
| Weight data | SEC EDGAR NPORT-P filings (SPY) |
| Frequency | Daily closing prices |
| Price field | Adjusted Close (`auto_adjust=True` — splits & dividends applied) |
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


## Summary for the project
# 1.a 
    Download SPY price from yfinance and download components of SP500 weights and price, I met the following issues:
        Issue: SP500 components and weights are not very easy to find, tried ask claude and see we can use quarterly State Street filing to SEC to simulate the components and weights
        Issue: not all cusip from State Street filing has corresponding ticker, could be multiple reasons as 1. company been acquired 2. foreign company that does not have mapping from cusip to ticker. Build the mapping by company name, and calculate the tracking error for this portfolio with SPY and see TE is about 1% so would be okay to use this to simulate SPY

# 1.b 
    See ### Synthetic portfolio simulation above
    Assume required true excess return equals excess return plus fee. Assume the noise are normal distributed with same vol on each date. Assume alpha is equally distributed to all dates.

    Validation: compare realized return and TE with target return and target TE, with given tolerance

    See ### Frec portfolio with account activities above
    Apart from simple simulate SP500 index, I also tried to contruct a portfolio that has low tracking error and some excess return based on S&P500 components. Since our initial portfolio has 1% TE already, start from the initial portfolio, pick a few stocks by looking at return, volatility, weight in the portfolio and manual pick NVDA/GOOG/AVGO, and add 1.5% weight to the 3 ticker, rescale the portfolio weight. This create a new portfolio with 1.75% TE and 2% annual excess return.
	    Issue: no additional weight data for year 2026, will use 2025-12-31 data as weight for year 2026
        Issue: no record of corporate action so use adjusted price from yfinance and assume no corporate action in portfolio construction

# 2 
    See ### Synthetic portfolio simulation above and ### Frec portfolio with account activities above above, both will generate chart needed

# 3
    See ### Frec portfolio with account activities above
    Assumptions: no corporation action, all activities with stocks are using adjusted number of shares, no transaction cost, cash inflow/outflow, security inflow/outflow has no delay in rebalancing, no market impact from trading, all stocks traded at close price, fractional shares are allowed 
    It directly affects portfolio value by the equivalent cash value of the inflow/outflow, and for comparison with SPY, adjust SPY portfolio with same equivalent cash value of the activities, and remove these dates from return/TE calculation

# 4
    Including account activities in the chart, include drawdown, Sharpe Ratio to the chart

# Proposal if given another 8 hours:
    1. Add risk model to the project, which can be used to calculate implied tracking error, adding transaction cost
    2. Improve portfolio construction, including integer shares, better handle account activities with rebalancing, include market rebalance delays
    3. Improve data quality, including improve data completeness of stock prices, corporate actions
    4. Include intraday trading other than rely on close price


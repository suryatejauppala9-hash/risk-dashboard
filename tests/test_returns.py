import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from analytics.returns import (
    compute_daily_returns,
    compute_portfolio_returns,
    compute_cumulative_returns,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
)

def test_all():
    # Synthetic prices — no internet needed
    np.random.seed(42)
    dates = pd.bdate_range("2019-01-01", "2024-01-01")
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]
    prices = pd.DataFrame(
        {t: 100 * np.cumprod(1 + np.random.normal(0.0003, 0.018, len(dates)))
         for t in tickers},
        index=dates,
    )
    weights = {"AAPL": 0.20, "MSFT": 0.25, "GOOGL": 0.15, "NVDA": 0.20, "SPY": 0.20}

    daily  = compute_daily_returns(prices)
    port   = compute_portfolio_returns(daily, weights)
    cum    = compute_cumulative_returns(port)

    assert len(daily) == len(dates) - 1
    assert port.name == "Portfolio"
    assert cum.iloc[0] > 0.9 and cum.iloc[0] < 1.1   
    assert -0.5 < annualised_return(port) < 1.0
    assert 0 < annualised_volatility(port) < 1.0
    assert -5 < sharpe_ratio(port) < 10

    print("Annual return:    ", f"{annualised_return(port):.2%}")
    print("Annual volatility:", f"{annualised_volatility(port):.2%}")
    print("Sharpe ratio:     ", f"{sharpe_ratio(port):.2f}")
    print()
    print("All tests passed ✓")

if __name__ == "__main__":
    test_all()
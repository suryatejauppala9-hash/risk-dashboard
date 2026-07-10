import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS=252

def fetch_prices(tickers: list[str],period: str ="5y") ->pd.DataFrame:
    if not tickers:
        raise ValueError("No tickers given")
    
    raw=yf.download(tickers,period=period,auto_adjust=True, progress=False,threads=True)

    if raw.empty:
        raise ValueError(f"No data for {tickers},Check again")
    
    if isinstance(raw.columns,pd.MultiIndex):
        prices=raw["Close"]
    else:
        prices=raw[["Close"]].rename(columns={"Close":tickers[0]})

    # Drop columns that are completely NaN (failed downloads)
    prices=prices.dropna(axis=1, how="all")

    if prices.empty or len(prices.columns) == 0:
        raise ValueError(f"No valid historical data found for {tickers}")

    # Ensure index is timezone-naive DatetimeIndex to prevent alignment issues
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    missing=[t for t in tickers if t not in prices.columns]
    if missing:
        print(f"no data for {missing}-skipping")

    return prices

def compute_daily_returns(prices: pd.DataFrame)->pd.DataFrame:
    return prices.pct_change().dropna()

def compute_portfolio_returns(daily_returns: pd.DataFrame,weights: dict[str,float])->pd.Series:
    w=pd.Series(weights).reindex(daily_returns.columns).fillna(0.0)
    w=w/w.sum()
    port=(daily_returns[w.index]*w.values).sum(axis=1)
    port.name="Portfolio"
    return port

def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()

def annualised_return(returns: pd.Series)->float:
    total=(1+returns).prod()
    n_years=len(returns)/TRADING_DAYS
    return float(total**(1/n_years)-1)

def annualised_volatility(returns: pd.Series)->float:
    return float(returns.std()*np.sqrt(TRADING_DAYS))

def sharpe_ratio(returns: pd.Series,risk_free_rate: float=0.05)->float:
    ann_r=annualised_return(returns)
    ann_vol=annualised_volatility(returns)
    if ann_vol==0:
        return 0.0
    return float((ann_r-risk_free_rate)/ann_vol)

def portfolio_summary(
    port_returns: pd.Series,
    prices: pd.DataFrame,
    weights: dict[str,float],
    risk_free_rate: float=0.05,
)->dict:
    cum=compute_cumulative_returns(port_returns)
    port_value=cum*10_000
    return{
        "total_return": float(cum.iloc[-1]-1),
        "annualised_return": annualised_return(port_returns),
        "annualised_vol": annualised_volatility(port_returns),
        "sharpe": sharpe_ratio(port_returns,risk_free_rate),
        "best_day": float(port_returns.max()),
        "worst_day": float(port_returns.min()),
        "positive_days": float((port_returns>0).mean()),
        "start_date": str(port_returns.index[0].date()),
        "end_date": str(port_returns.index[-1].date()),
        "n_days": len(port_returns),
        "portfolio_value": port_value,
        "cumulative_ret": cum,
    }
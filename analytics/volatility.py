import numpy as np
import pandas as pd


TRADING_DAYS=252

def rolling_volatility(returns: pd.Series,window:int=30)->pd.Series:
    return returns.rolling(window).std()*np.sqrt(TRADING_DAYS)

def rolling_sharpe(
    returns:pd.Series,
    window: int=90,
    risk_free_rate: float=0.05,
)->pd.Series:
    daily_rf=(1+risk_free_rate)**(1/TRADING_DAYS)-1
    excess=returns-daily_rf
    roll_mean=excess.rolling(window).mean()*TRADING_DAYS
    roll_std=returns.rolling(window).std()*np.sqrt(TRADING_DAYS)

    sharpe=roll_mean/roll_std
    sharpe.name="Rolling Sharpe"
    return sharpe

def downside_deviation(returns: pd.Series,threshold: float=0.0)->float:
    diff = returns - threshold
    downside_diff = np.minimum(diff, 0.0)
    return float(np.sqrt(np.mean(downside_diff**2)) * np.sqrt(TRADING_DAYS))

def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> float:
    """Scalar Sortino ratio for the metric card."""
    from analytics.returns import annualised_return
    ann_r = annualised_return(returns)
    dd    = downside_deviation(returns, threshold=0.0)
    if dd == 0:
        return 0.0
    return float((ann_r - risk_free_rate) / dd)


def rolling_sortino(
    returns: pd.Series,
    window: int = 90,
    risk_free_rate: float = 0.05,
) -> pd.Series:
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1

    def _sortino(x):
        excess   = x - daily_rf
        ann_ret  = excess.mean() * TRADING_DAYS
        diff     = x - 0.0
        downside = np.minimum(diff, 0.0)
        dd = np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS)
        if dd == 0:
            return np.nan
        return ann_ret / dd

    result = returns.rolling(window).apply(_sortino, raw=False)
    result.name = "Rolling Sortino"
    return result
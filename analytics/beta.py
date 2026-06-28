import numpy as np
import pandas as pd

def compute_beta(
    port_returns: pd.Series,
    market_returns: pd.Series,
)->float:
    aligned=pd.concat([port_returns,market_returns],axis=1).dropna()
    if len(aligned)<30:
        return float("nan")
    p=aligned.iloc[:,0].values
    m=aligned.iloc[:,1].values

    cov_matrix=np.cov(p,m)
    beta=cov_matrix[0,1]/cov_matrix[1,1]
    return float(beta)

def compute_alpha(
    port_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float=0.05,
)->float:
    from analytics.returns import annualised_return
    beta=compute_beta(port_returns,market_returns)
    rp=annualised_return(port_returns)
    rm=annualised_return(market_returns)
    alpha=rp-(risk_free_rate+beta*(rm-risk_free_rate))
    return float(alpha)

def compute_correlation(
    port_returns:pd.Series,
    market_returns:pd.Series,
)->float:
    aligned=pd.concat([port_returns,market_returns],axis=1).dropna()
    if len(aligned)<2:
        return float("nan")
    return float(aligned.iloc[:,0].corr(aligned.iloc[:,1]))

def rolling_beta(port_returns, market_returns, window=90):
    rolling_cov = port_returns.rolling(window=window).cov(market_returns)
    rolling_var = market_returns.rolling(window=window).var()
    
    return rolling_cov / rolling_var

def r_squared(
    port_returns:pd.Series,
    market_returns:pd.Series,
)->float:
    corr=compute_correlation(port_returns,market_returns)
    return float(corr**2)
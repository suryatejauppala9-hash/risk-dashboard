import numpy as np
import pandas as pd

def correlation_matrix(returns: pd.DataFrame)->pd.DataFrame:
    return returns.corr()


def rolling_correlation(
    returns: pd.DataFrame,
    tickera:str,
    tickerb:str,
    window:int =60,
)->pd.Series:
    if tickera not in returns.columns or tickerb not in returns.columns:
        raise ValueError(f"Tickers {tickera} or {tickerb} not in returns.")
    result=returns[tickera].rolling(window).corr(returns[tickerb])
    result.name=f"{tickera} vs {tickerb}"
    return result

def average_pairwise_correlation(returns: pd.DataFrame)->float:
    corr=correlation_matrix(returns)
    n=len(corr.columns)
    if n<2:
        return 1.0
    mask=np.triu(np.ones(corr.shape,dtype=bool),k=1)
    pairs=corr.values[mask]
    return float(pairs.mean())

def diversification_ratio(
    returns: pd.DataFrame,
    weights: dict[str,float],
)->float:
    tickers=[t for t in weights if t in returns.columns]
    w=pd.Series(weights).reindex(tickers).fillna(0)
    w=w/w.sum()
    vols=returns[tickers].std()*np.sqrt(252)
    weighted_vol=float((w*vols).sum())

    port_ret=(returns[tickers]*w.values).sum(axis=1)
    port_vol=float(port_ret.std()*np.sqrt(252))

    if port_vol==0:
        return 1.0
    return weighted_vol/port_vol

def risk_contribution(
    returns: pd.DataFrame,
    weights: dict[str,float],
)->pd.Series:
    tickers=[t for t in weights if t in returns.columns]
    w=pd.Series(weights).reindex(tickers).fillna(0)
    w=w/w.sum()
    w_arr=w.values
    cov=returns[tickers].cov()*252
    port_var=float(w_arr @ cov.values @w_arr)
    port_vol=np.sqrt(port_var)

    if port_vol==0:
        return pd.Series(0,index=tickers)
    
    marginal=cov.values@w_arr/port_vol
    contribution=w_arr*marginal
    pct=contribution/port_vol

    pct_normalised=pct/pct.sum()

    return pd.Series(pct_normalised*100,index=tickers,name="Risk Contribution %")
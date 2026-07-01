import numpy as np
import pandas as pd
import cvxpy as cp

TRADING_DAYS=252

def _annualised_mean_cov(returns: pd.DataFrame)->tuple[np.ndarray,np.ndarray]:
    mu=returns.mean().values*TRADING_DAYS
    cov=returns.cov().values*TRADING_DAYS
    return mu, cov

def _solve_min_variance(
    cov:np.ndarray,
    n:int,
    long_only:bool=True,
)->np.ndarray:
    w=cp.Variable(n)
    risk=cp.quad_form(w,cov)

    constraints=[cp.sum(w)==1]
    if long_only:
        constraints.append(w>=0)
    problem=cp.Problem(cp.Minimize(risk),constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed")
    return np.clip(w.value,0,None) if long_only else w.value

def _solve_target_return(
    mu:np.ndarray,
    cov:np.ndarray,
    target_return: float,
    n: int,
    long_only: bool=True,
)->np.ndarray:
    w=cp.Variable(n)
    risk=cp.quad_form(w,cov)
    constraints = [
        cp.sum(w) == 1,
        mu @ w >= target_return,
    ]
    if long_only:
        constraints.append(w >= 0)

    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve()

    if w.value is None:
        return None
    return np.clip(w.value, 0, None) if long_only else w.value

def minimum_variance_portfolio(
    returns: pd.DataFrame,
    long_only: bool=True,
)->dict:
    tickers=list(returns.columns)
    n=len(tickers)
    mu,cov=_annualised_mean_cov(returns)

    weights=_solve_min_variance(cov,n,long_only)
    weights=weights/weights.sum()

    port_ret=float(mu@weights)
    port_vol=float(np.sqrt(weights@cov@weights))

    return{
        "weights": dict(zip(tickers,weights)),
        "return": port_ret,
        "volatility": port_vol,
        "sharpe": None,
    }

def maximum_sharpe_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float=0.05,
    long_only: bool=True,
)->dict:
    tickers=list(returns.columns)
    n=len(tickers)
    mu,cov=_annualised_mean_cov(returns)

    min_ret=float(mu.min())
    max_ret=float(mu.max())

    target_returns=np.linspace(max(min_ret,0.0001),max_ret*0.999,100)
    
    best_sharpe=-np.inf
    best_weights=None
    best_ret=None
    best_vol=None

    for target in target_returns:
        w=_solve_target_return(mu,cov,target,n,long_only)
        if w is None:
            continue
        w=w/w.sum()
        ret=float(mu@w)
        vol=float(np.sqrt(w@cov@w))
        if vol==0:
            continue
        sharpe=(ret-risk_free_rate)/vol
        if sharpe>best_sharpe:
            best_sharpe=sharpe
            best_weights=w
            best_ret=ret
            best_vol=vol
    if best_weights is None:
        raise ValueError("couldnt fine max sharpe")
    
    return{
        "weights": dict(zip(tickers,best_weights)),
        "return": best_ret,
        "volatility": best_vol,
        "sharpe": best_sharpe,
    }

def target_return_portfolio(
    returns: pd.DataFrame,
    target_return: float,
    long_only: bool=True,
)->dict:
    tickers = list(returns.columns)
    n       = len(tickers)
    mu, cov = _annualised_mean_cov(returns)

    weights = _solve_target_return(mu, cov, target_return, n, long_only)
    if weights is None:
        raise ValueError(
            f"Target return {target_return:.2%} is not achievable with this asset set."
        )
    weights = weights / weights.sum()

    port_ret = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ cov @ weights))

    return {
        "weights":    dict(zip(tickers, weights)),
        "return":     port_ret,
        "volatility": port_vol,
        "sharpe":     None,
    }


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int=40,
    long_only: bool=True,
)->pd.DataFrame:
    tickers = list(returns.columns)
    n       = len(tickers)
    mu, cov = _annualised_mean_cov(returns)

    min_ret = float(mu.min())
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret * 0.999, n_points)

    rows = []
    for target in targets:
        w = _solve_target_return(mu, cov, target, n, long_only)
        if w is None:
            continue
        w = w / w.sum()
        ret = float(mu @ w)
        vol = float(np.sqrt(w @ cov @ w))
        rows.append({"return": ret, "volatility": vol})

    return pd.DataFrame(rows).drop_duplicates().sort_values("volatility")


def random_portfolio(
    returns:pd.DataFrame,
    n_portfolios: int=2000,
    seed:int=42,
)->pd.DataFrame:
    rng=np.random.default_rng(seed)
    tickers=list(returns.columns)
    n=len(tickers)
    mu,cov=_annualised_mean_cov(returns)

    rows=[]
    for _ in range(n_portfolios):
        w=rng.random(n)
        w=w/w.sum()
        ret=float(mu@w)
        vol=float(np.sqrt(w@cov@w))
        rows.append({"return":ret,"volatility":vol})

    return pd.DataFrame(rows)

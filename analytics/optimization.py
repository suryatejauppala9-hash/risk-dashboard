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
    max_wt: float=0.40,
    min_wt: float=0.05,
)->np.ndarray:
    w=cp.Variable(n)
    risk=cp.quad_form(w,cov)

    constraints=[cp.sum(w)==1,
                 w>=min_wt,
                 w<=max_wt,]
    problem=cp.Problem(cp.Minimize(risk),constraints)
    problem.solve()

    if w.value is None:
        raise ValueError("Optimization failed")
    return np.clip(w.value,0,None) 

def _solve_target_return(
    mu:np.ndarray,
    cov:np.ndarray,
    target_return: float,
    n: int,
    max_wt: float=0.40,
    min_wt: float=0.05,
)->np.ndarray:
    w=cp.Variable(n)
    risk=cp.quad_form(w,cov)
    constraints=[cp.sum(w)==1,
                 mu@w>=target_return,
                 w>=min_wt,
                 w<=max_wt,]
    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve()

    if w.value is None:
        return None
    return np.clip(w.value, 0, None)

def minimum_variance_portfolio(
    returns: pd.DataFrame,
    max_wt: float=0.40,
    min_wt: float=0.05,
)->dict:
    tickers=list(returns.columns)
    n=len(tickers)
    mu,cov=_annualised_mean_cov(returns)

    weights=_solve_min_variance(cov,n,max_wt,min_wt)
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
    max_wt: float=0.40,
    min_wt: float=0.05,
)->dict:
    tickers=list(returns.columns)
    n=len(tickers)
    mu,cov=_annualised_mean_cov(returns)

    y = cp.Variable(n)
    k = cp.Variable()
    risk = cp.quad_form(y, cov)
    constraints = [
        (mu - risk_free_rate) @ y == 1,
        cp.sum(y) == k,
        y >= k * min_wt,
        y <= k * max_wt,
        k >= 0,
    ]
    problem = cp.Problem(cp.Minimize(risk), constraints)
    try:
        problem.solve()
    except Exception:
        pass

    if y.value is None or k.value is None or k.value <= 1e-6:
        min_var = minimum_variance_portfolio(returns, max_wt, min_wt)
        weights_arr = np.array([min_var["weights"][t] for t in tickers])
        ret = float(mu @ weights_arr)
        vol = float(np.sqrt(weights_arr @ cov @ weights_arr))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        return {
            "weights": min_var["weights"],
            "return": ret,
            "volatility": vol,
            "sharpe": sharpe,
        }

    weights = np.clip(y.value / k.value, 0, None)
    weights = weights / weights.sum()
    ret = float(mu @ weights)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0

    return {
        "weights": dict(zip(tickers, weights)),
        "return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }

def target_return_portfolio(
    returns: pd.DataFrame,
    target_return: float,
    max_wt: float=0.40,
    min_wt: float=0.05,
)->dict:
    tickers = list(returns.columns)
    n       = len(tickers)
    mu, cov = _annualised_mean_cov(returns)

    weights = _solve_target_return(mu, cov, target_return, n, max_wt,min_wt)
    if weights is None:
        raise ValueError(
            f"Target return {target_return:.2%} is not achievable with this asset set."
            f"current constraints (max {max_wt:.0%} per stock, "
            f"min {min_wt:.0%} per stock)."
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
    max_wt: float=0.40,
    min_wt: float=0.05,
)->pd.DataFrame:
    tickers = list(returns.columns)
    n       = len(tickers)
    mu, cov = _annualised_mean_cov(returns)

    eq_w=np.ones(n)/n
    min_ret = float(mu@eq_w)*0.8
    max_ret = float(np.sort(mu)[-1]) * max_wt + float(np.sort(mu)[-2]) * (1 - max_wt)
    targets = np.linspace(min_ret, max_ret * 0.98, n_points)

    rows = []
    for target in targets:
        w = _solve_target_return(mu, cov, target, n, max_wt,min_wt)
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
    max_wt:float=0.40,
    min_wt:float=0.05,
    seed:int=42,
)->pd.DataFrame:
    rng=np.random.default_rng(seed)
    tickers=list(returns.columns)
    n=len(tickers)
    mu,cov=_annualised_mean_cov(returns)

    rows=[]
    attempts=0
    while len(rows)<n_portfolios and attempts<n_portfolios*10:
        attempts+=1
        w=rng.random(n)
        w=w/w.sum()
        if w.max() > max_wt or w.min() < min_wt:
            continue
        ret=float(mu@w)
        vol=float(np.sqrt(w@cov@w))
        rows.append({"return":ret,"volatility":vol})

    return pd.DataFrame(rows)


def max_achievable_return(
    returns: pd.DataFrame,
    max_wt: float=0.40,
    min_wt: float=0.05,
) -> float:
    tickers = list(returns.columns)
    n = len(tickers)
    mu, _ = _annualised_mean_cov(returns)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w)
    constraints = [cp.sum(w) == 1, w >= min_wt, w <= max_wt]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve()
        if w.value is not None:
            return float(mu @ w.value)
    except Exception:
        pass
    return float(mu.max())

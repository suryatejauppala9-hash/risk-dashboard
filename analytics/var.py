import numpy as np
import pandas as pd
from scipy.stats import norm

def historical_var(
    returns: pd.Series,
    confidence: float=0.95,
)->float:
    return float(abs(np.percentile(returns,(1-confidence)*100)))

def historical_cvar(
    returns: pd.Series,
    confidence: float=0.95,
)->float:
    var_threshold=-historical_var(returns,confidence)
    tail_losses=returns[returns<=var_threshold]
    if tail_losses.empty:
        return historical_var(returns,confidence)
    return float(abs(tail_losses.mean()))

def parametric_var(
    returns: pd.Series,
    confidence: float=0.95,
)->float:
    mu=returns.mean()
    sigma=returns.std()
    zscore=norm.ppf(1-confidence)
    var=-(mu+zscore*sigma)
    return float(max(var,0))

def parametric_cvar(
    returns: pd.Series,
    confidence: float=0.95,
)->float:
    mu=returns.mean()
    sigma=returns.std()
    zscore=norm.ppf(1-confidence)
    var=-mu + sigma*norm.pdf(zscore)/(1-confidence)
    return float(max(var,0))

def monte_carlo_var(
    returns: pd.Series,
    confidence: float=0.95,
    simulations: int=10_000,
    horizon: int =1,
    seed: int =42,
)->float:
    rng=np.random.default_rng(seed)
    mu=returns.mean()
    sigma=returns.std()
    simulated=rng.normal(mu,sigma,(simulations,horizon))
    if horizon==1:
        final_returns=simulated[:,0]
    else:
        final_returns=np.prod(1+simulated,axis=1)-1
    return float(abs(np.percentile(final_returns,(1-confidence)*100)))


def monte_carlo_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    horizon: int = 1,
    seed: int = 42,
) -> float:
    rng       = np.random.default_rng(seed)
    mu        = returns.mean()
    sigma     = returns.std()
    simulated = rng.normal(mu, sigma, (n_simulations, horizon))

    if horizon == 1:
        final_returns = simulated[:, 0]
    else:
        final_returns = np.prod(1 + simulated, axis=1) - 1

    threshold  = np.percentile(final_returns, (1 - confidence) * 100)
    tail       = final_returns[final_returns <= threshold]
    return float(abs(tail.mean())) if len(tail) > 0 else 0.0


def monte_carlo_paths(
    returns: pd.Series,
    n_simulations: int = 200,
    horizon: int = 30,
    seed: int = 42,
)->np.ndarray:
    rng    = np.random.default_rng(seed)
    mu     = returns.mean()
    sigma  = returns.std()

    daily=rng.normal(mu,sigma,(n_simulations,horizon))
    paths=np.cumprod(1+daily,axis=1)
    ones=np.ones((n_simulations,1))
    return np.hstack([ones,paths])

def scale_var(
    daily_var:float,
    horizon:int
)->float:
    return daily_var*np.sqrt(horizon)

def var_summary(
    returns: pd.Series,
    confidence: float=0.95,
    horizon: int=1,
    simulations: int =10_000,
)->dict:
    h_var  = historical_var(returns, confidence)
    p_var  = parametric_var(returns, confidence)
    mc_var = monte_carlo_var(returns, confidence, simulations, horizon)

    h_cvar  = historical_cvar(returns, confidence)
    p_cvar  = parametric_cvar(returns, confidence)
    mc_cvar = monte_carlo_cvar(returns, confidence, simulations, horizon)

    return{
        "confidence": confidence,
        "horizon": horizon,
        "historical_var":h_var * np.sqrt(horizon),
        "parametric_var":p_var * np.sqrt(horizon),
        "montecarlo_var":mc_var,
        "historical_cvar":h_cvar * np.sqrt(horizon),
        "parametric_cvar":p_cvar * np.sqrt(horizon),
        "montecarlo_cvar":mc_cvar,
        "historical_var_scaled":  h_var * np.sqrt(horizon),
        "parametric_var_scaled":  p_var * np.sqrt(horizon),
        "montecarlo_var_scaled":  mc_var,
    }
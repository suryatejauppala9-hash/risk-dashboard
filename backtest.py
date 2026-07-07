import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from analytics.returns import fetch_prices, compute_daily_returns, compute_portfolio_returns

def pof_test(b, obs, conf):
    p = 1 - conf
    if b == 0:
        return 1.0
    
    p_obs = b / obs
    
    t1 = b * np.log(p) + (obs - b) * np.log(1 - p)
    t2 = b * np.log(p_obs) + (obs - b) * np.log(1 - p_obs)
    
    lr = -2 * (t1 - t2)
    return 1 - chi2.cdf(lr, df=1)

def run():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "NIFTYBEES.NS"]
    weights = {
        "RELIANCE.NS": 0.20,
        "TCS.NS": 0.20,
        "INFY.NS": 0.15,
        "HDFCBANK.NS": 0.25,
        "NIFTYBEES.NS": 0.20
    }
    
    prices = fetch_prices(tickers, period="5y")
    daily = compute_daily_returns(prices)
    port = compute_portfolio_returns(daily, weights)
    
    w = 252
    c = 0.95
    
    hb = 0
    pb = 0
    n = len(port) - w
    
    for i in range(w, len(port)):
        train = port.iloc[i-w:i]
        ret = port.iloc[i]
        
        hv = -np.percentile(train, (1-c)*100)
        
        mu = train.mean()
        sig = train.std()
        z = norm.ppf(1-c)
        pv = -(mu + z * sig)
        
        if ret < -hv:
            hb += 1
            
        if ret < -pv:
            pb += 1

    exp = n * (1 - c)
    hr = hb / n
    pr = pb / n
    
    hp = pof_test(hb, n, c)
    pp = pof_test(pb, n, c)

    print(f"Total days: {n}")
    print(f"Expected: {exp:.1f}")
    print(f"Hist VaR - Breaches: {hb}, Rate: {hr*100:.2f}%, POF: {hp:.4f}")
    print(f"Param VaR - Breaches: {pb}, Rate: {pr*100:.2f}%, POF: {pp:.4f}")
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run()

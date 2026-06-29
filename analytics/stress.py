import numpy as np
import pandas as pd



SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Lehman Brothers collapse. S&P 500 fell 38% in 2008.",
        "shocks": {
            "market":    -0.38,
            "tech":      -0.42,
            "financials":-0.55,
            "energy":    -0.35,
            "default":   -0.38,
        },
    },
    "COVID Crash (Mar 2020)": {
        "description": "Fastest bear market in history. S&P 500 fell 34% in 33 days.",
        "shocks": {
            "market":    -0.34,
            "tech":      -0.27,
            "financials":-0.40,
            "energy":    -0.55,
            "default":   -0.30,
        },
    },
    "Dot-com Bust (2000-02)": {
        "description": "Tech bubble burst. NASDAQ fell 78% peak to trough.",
        "shocks": {
            "market":    -0.49,
            "tech":      -0.78,
            "financials":-0.20,
            "energy":    -0.10,
            "default":   -0.40,
        },
    },
    "Rate Shock +2%": {
        "description": "Sudden 200bps rate rise. Growth stocks hit hardest.",
        "shocks": {
            "market":    -0.15,
            "tech":      -0.25,
            "financials": 0.05,
            "energy":     0.03,
            "default":   -0.12,
        },
    },
    "Flash Crash": {
        "description": "Sudden single-day liquidity event. Sharp drop and recovery.",
        "shocks": {
            "market":    -0.10,
            "tech":      -0.12,
            "financials":-0.10,
            "energy":    -0.08,
            "default":   -0.10,
        },
    },
    "Oil Doubles": {
        "description": "Oil price doubles due to supply shock.",
        "shocks": {
            "market":    -0.08,
            "tech":      -0.06,
            "financials":-0.05,
            "energy":     0.30,
            "default":   -0.05,
        },
    },
}

# we can expand
TECH_TICKERS        = {"AAPL","MSFT","GOOGL","GOOG","META","NVDA","AMD","TSLA",
                       "ORCL","CRM","ADBE","INTC","QCOM","AVGO","TXN","NFLX"}
FINANCIAL_TICKERS   = {"JPM","BAC","GS","MS","WFC","C","BRK.B","V","MA","AXP",
                       "BLK","SCHW","USB","PNC"}
ENERGY_TICKERS      = {"XOM","CVX","COP","SLB","EOG","MPC","VLO","PSX","OXY","HAL"}

BROAD_MARKET        = {"SPY","QQQ","VTI","IVV","VOO","^NSEI","^BSESN"}


def classify_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t in TECH_TICKERS:        return "tech"
    if t in FINANCIAL_TICKERS:   return "financials"
    if t in ENERGY_TICKERS:      return "energy"
    if t in BROAD_MARKET:        return "market"
    return "default"


def apply_scenario(
    weights: dict[str, float],
    scenario_name: str,
) -> dict:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    scenario = SCENARIOS[scenario_name]
    shocks   = scenario["shocks"]

    total_w  = sum(weights.values())
    norm_w   = {k: v / total_w for k, v in weights.items()}

    rows             = []
    portfolio_impact = 0.0

    for ticker, weight in norm_w.items():
        sector       = classify_ticker(ticker)
        shock        = shocks.get(sector, shocks["default"])
        contribution = weight * shock
        portfolio_impact += contribution

        rows.append({
            "Ticker":         ticker,
            "Weight":         f"{weight*100:.1f}%",
            "Sector":         sector.capitalize(),
            "Shock":          f"{shock*100:+.1f}%",
            "Contribution":   f"{contribution*100:+.2f}%",
            "Shock (raw)":    shock,
            "Contribution (raw)": contribution,
        })

    return {
        "scenario":         scenario_name,
        "description":      scenario["description"],
        "portfolio_impact": portfolio_impact,
        "rows":             pd.DataFrame(rows),
    }


def apply_custom_scenario(
    weights: dict[str, float],
    market_shock: float,
    tech_shock: float,
    financial_shock: float,
    energy_shock: float,
) -> dict:
    custom_shocks = {
        "market":     market_shock,
        "tech":       tech_shock,
        "financials": financial_shock,
        "energy":     energy_shock,
        "default":    market_shock,
    }

    total_w  = sum(weights.values())
    norm_w   = {k: v / total_w for k, v in weights.items()}

    rows             = []
    portfolio_impact = 0.0

    for ticker, weight in norm_w.items():
        sector       = classify_ticker(ticker)
        shock        = custom_shocks.get(sector, market_shock)
        contribution = weight * shock
        portfolio_impact += contribution

        rows.append({
            "Ticker":             ticker,
            "Weight":             f"{weight*100:.1f}%",
            "Sector":             sector.capitalize(),
            "Shock":              f"{shock*100:+.1f}%",
            "Contribution":       f"{contribution*100:+.2f}%",
            "Shock (raw)":        shock,
            "Contribution (raw)": contribution,
        })

    return {
        "scenario":         "Custom",
        "description":      "User-defined shock scenario.",
        "portfolio_impact": portfolio_impact,
        "rows":             pd.DataFrame(rows),
    }


def all_scenario_impacts(weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name in SCENARIOS:
        result = apply_scenario(weights, name)
        rows.append({
            "Scenario":       name,
            "Portfolio loss": result["portfolio_impact"] * 100,
        })
    return pd.DataFrame(rows).sort_values("Portfolio loss")
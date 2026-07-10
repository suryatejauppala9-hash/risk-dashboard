import os
import json
import streamlit as st


def _get_api_key() -> str| None:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")
    
def _build_prompt(metrics:dict)->str:
    return f"""You are a senior portfolio risk analyst at an Indian asset management firm.
A client has shared their equity portfolio metrics below.
Write exactly 5 sharp, specific observations about this portfolio's risk profile.

Portfolio metrics:
{json.dumps(metrics, indent=2, default=str)}

Rules:
- Every observation must reference at least one specific number from the data.
- Observations must be relevant to an Indian retail investor (mention NIFTY/SENSEX context where relevant).
- No generic advice. No disclaimers. No preamble. No "consult a financial advisor".
- Write each as a single punchy sentence starting with a bold keyword like Risk:, Concentration:, Volatility:, Drawdown:, Correlation:, Alpha:, VaR: etc.
- Be direct, like a risk desk analyst writing a morning note, not a chatbot.

Output only the 5 observations, nothing else."""

def generate_risk_commentary(metrics: dict)->dict:
    key=_get_api_key()
    if not key:
        return{
            "status":"no_key",
            "text": None,
        }
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        response=model.generate_content(
            _build_prompt(metrics),
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=500,
            )
        )
        return {
            "status":"ok",
            "text": response.text.strip(),
        }
    except Exception as e:
        return{
            "status":"error",
            "text":str(e),
        }
    

def build_metrics_payload(
    summary: dict,
    ann_vol: float,
    sharpe: float,
    sortino: float,
    mdd:float,
    avg_dd_dur: float,
    risk_contrib: "pd.Series |None",
    beta_val:"float|None",
    alpha_val:"float|None",
    corr_val:"float|None",
    var_dict: dict,
    avg_corr: float,
    weights: dict,
    bench_label: str,
)->str:
    top_risk={}
    if risk_contrib is not None and len(risk_contrib)>0:
        top_risk={
            k:round(float(v),1)
            for k,v in risk_contrib.sort_values(ascending=False).head(3).items()
        }
    return {
        "portfolio_holdings_pct": {
            k: round(v * 100, 1) for k, v in weights.items()
        },
        "annual_return_pct":          round(summary["annualised_return"] * 100, 2),
        "annual_volatility_pct":      round(ann_vol * 100, 2),
        "total_return_pct":           round(summary["total_return"] * 100, 2),
        "sharpe_ratio":               round(sharpe, 2),
        "sortino_ratio":              round(sortino, 2),
        "max_drawdown_pct":           round(mdd * 100, 2),
        "avg_drawdown_duration_days": round(avg_dd_dur, 0),
        "top_3_risk_contributors_pct": top_risk,
        "avg_pairwise_correlation":   round(avg_corr, 2),
        "benchmark":                  bench_label,
        "beta_vs_benchmark":          round(beta_val, 2) if beta_val is not None else "N/A",
        "alpha_vs_benchmark_pct":     round(alpha_val * 100, 2) if alpha_val is not None else "N/A",
        "correlation_vs_benchmark":   round(corr_val, 2) if corr_val is not None else "N/A",
        "historical_var_95_pct":      round(var_dict.get("historical_var", 0) * 100, 3),
        "historical_cvar_95_pct":     round(var_dict.get("historical_cvar", 0) * 100, 3),
        "monte_carlo_var_95_pct":     round(var_dict.get("montecarlo_var", 0) * 100, 3),
        "positive_days_pct":          round(summary["positive_days"] * 100, 1),
        "data_period":                f"{summary['start_date']} to {summary['end_date']}",
    }
from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import skew,kurtosis

from analytics.returns import (
    fetch_prices,
    compute_daily_returns,
    compute_portfolio_returns,
    compute_cumulative_returns,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    portfolio_summary,
)
from dashboard.charts import (
    chart_cumulative_returns,
    chart_portfolio_value,
    chart_daily_returns,
    chart_return_histogram,
    chart_normalised_prices,
    chart_allocation_donut,
    chart_monthly_returns_heatmap,
    chart_rolling_annual_return,
)
from analytics.volatility import (
    rolling_volatility,
    sortino_ratio,
    rolling_sharpe,
)
from analytics.drawdown import (
    compute_drawdown_series,
    max_drawdown,
    drawdown_details,
    avg_drawdown_duration,
)
from analytics.beta import (
    compute_beta,
    compute_alpha,
    compute_correlation,
    r_squared,
)
from dashboard.charts import (
    chart_drawdown,
    chart_rolling_volatility,
    chart_rolling_sharpe,
    chart_rolling_beta,
)

from analytics.var import var_summary

from dashboard.charts import(
    chart_var_comparison,
    chart_var_loss_dollar,
    chart_monte_carlo_fan,
    chart_var_over_time,
)

from analytics.correlation import (
    correlation_matrix,
    rolling_correlation,
    average_pairwise_correlation,
    diversification_ratio,
    risk_contribution,
)
from analytics.stress import (
    SCENARIOS,
    apply_scenario,
    apply_custom_scenario,
    all_scenario_impacts,
)
from dashboard.charts import (
    chart_correlation_heatmap,
    chart_rolling_correlation,
    chart_risk_contribution,
    chart_scenario_summary,
    chart_scenario_waterfall,
)
from analytics.optimization import (
    minimum_variance_portfolio,
    maximum_sharpe_portfolio,
    target_return_portfolio,
    efficient_frontier,
    random_portfolio,
)
from dashboard.charts import (
    chart_efficient_frontier,
    chart_weight_comparison,
)
st.set_page_config(
    page_title="Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

from analytics.ai_advisor import generate_risk_commentary, build_metrics_payload

#css part
st.markdown("""
<style>
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"]  { font-size: 12px; opacity: 0.6; }
[data-testid="stMetricValue"]  { font-size: 26px; font-weight: 600; }
[data-testid="stSidebar"]      { background: rgba(15,15,20,0.6); }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

#helpers

def parse_portfolio(text: str) -> dict[str, float]:
    weights = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            ticker = parts[0].upper().strip()
            try:
                weights[ticker] = float(parts[1].replace("%", "")) / 100
            except ValueError:
                continue
    return weights


def fmt_pct(v: float, d: int = 2) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{d}f}%"

def fmt_ratio(v: float) -> str:
    return f"{v:.2f}"

def fmt_inr(v: float) -> str:
    if abs(v) >= 1_00_00_000:
        return f"Rs. {v/1_00_00_000:.2f} Cr"
    elif abs(v) >= 1_00_000:
        return f"Rs. {v/1_00_000:.2f} L"
    else:
        return f"Rs. {v:,.0f}"

with st.sidebar:
    st.title("Risk Dashboard")
    st.caption("final version")
    st.divider()

    st.markdown("**Portfolio** one per line: `TICKER WEIGHT%`")
    st.caption("NSE tickers: add `.NS` suffix (e.g. `RELIANCE.NS`). BSE: add `.BO`.")
    raw_input = st.text_area(
        "portfolio_input",
        value="RELIANCE.NS 20\nTCS.NS 20\nINFY.NS 15\nHDFCBANK.NS 25\nNIFTYBEES.NS 20",
        height=155,
        label_visibility="collapsed",
        help="Weights are auto-normalised — they don't need to sum to 100.",
    )

    st.markdown("**Settings**")
    period = st.select_slider(
        "Lookback period",
        options=["1y", "2y", "3y", "5y", "10y"],
        value="5y",
    )
    risk_free = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0, max_value=15.0,
        value=5.0, step=0.25,
    ) / 100

    benchmark_options = {
        "S&P 500 (SPY)": "SPY",
        "NASDAQ (QQQ)": "QQQ",
        "Total Market (VTI)": "VTI",
        "NIFTY 50": "^NSEI",
        "Sensex": "^BSESN",
        "None": None,
    }

    benchmark = st.selectbox(
        "Benchmark",
        list(benchmark_options.keys()),
        index=list(benchmark_options.keys()).index("NIFTY 50"),
    )

    bench_ticker = benchmark_options[benchmark]

    initial_investment = st.number_input(
        "Initial investment (Rs.)",
        min_value=10_000, max_value=10_00_00_000,
        value=1_00_000, step=10_000,
        format="%d",
    )

    st.divider()
    if st.button("▶  Run Analysis", type="primary", use_container_width=True):
        st.session_state["analysis_run"] = True
    st.caption("Data: Yahoo Finance · NSE/BSE via yfinance · Streamlit + Plotly")

#landing screen

if not st.session_state.get("analysis_run", False):
    st.markdown("## Portfolio Risk Dashboard")
    st.info("Configure your portfolio in the sidebar and click **▶ Run Analysis**.")
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in zip(
        [c1, c2, c3, c4],
        ["", "", "", ""],
        ["Performance", "Risk metrics", "Monthly heatmap", "Rolling metrics"],
        ["Cumulative returns & portfolio value",
         "Volatility, Sharpe, return distribution",
         "Calendar view of monthly returns by year",
         "1-year rolling return & 30-day trend"],
    ):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)
    st.stop()


#parse and validate

weights = parse_portfolio(raw_input)
if not weights:
    st.error("Could not parse portfolio. Format: `AAPL 20` (one per line).")
    st.stop()

tickers       = list(weights.keys())
fetch_tickers = list(set(tickers + ([bench_ticker] if bench_ticker else [])))

#fetch

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers_tuple: tuple, period: str) -> pd.DataFrame:
    return fetch_prices(list(tickers_tuple), period=period)


with st.spinner("Fetching market data from Yahoo Finance…"):
    try:
        prices_all = load_data(tuple(sorted(fetch_tickers)), period)
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

# Split portfolio vs benchmark
port_tickers = [t for t in tickers if t in prices_all.columns]
prices       = prices_all[port_tickers]

bench_cum = None
if bench_ticker and bench_ticker in prices_all.columns:
    b_ret     = prices_all[[bench_ticker]].pct_change().dropna()[bench_ticker]
    bench_cum = compute_cumulative_returns(b_ret)
    bench_ann = annualised_return(b_ret)
else:
    bench_ann = None

# Core calculations
daily_ret  = compute_daily_returns(prices)
if daily_ret.empty:
    st.error("No overlapping daily returns data available. This can happen if your tickers have no overlapping trading dates, or if one of the assets has no data for the selected lookback period.")
    st.stop()

port_ret   = compute_portfolio_returns(daily_ret, weights)
cum_ret    = compute_cumulative_returns(port_ret)
port_value = cum_ret * initial_investment
summary    = portfolio_summary(port_ret, prices, weights, risk_free_rate=risk_free)

ann_ret = summary["annualised_return"]
ann_vol = summary["annualised_vol"]
sharpe  = summary["sharpe"]

downside     = port_ret[port_ret < 0].std() * np.sqrt(252)
sortino      = (ann_ret - risk_free) / downside if downside > 0 else 0.0
alpha        = ann_ret - bench_ann if bench_ann is not None else None
bench_label = benchmark if bench_ticker else "Benchmark"

#header

st.markdown("## Portfolio Analysis")
holding_str = "  ·  ".join(f"{t} {w*100:.0f}%" for t, w in weights.items())
st.caption(
    f"{holding_str}  ·  "
    f"{summary['start_date']} → {summary['end_date']}  ·  "
    f"{summary['n_days']:,} trading days"
)
st.divider()

#cards
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric(
    "Total return",
    fmt_pct(summary["total_return"]),
    delta=f"{fmt_pct(alpha)} vs {bench_label}" if alpha is not None else None,
)
m2.metric(
    "Annual return",
    fmt_pct(ann_ret),
    delta=f"{fmt_pct(bench_ann)} benchmark" if bench_ann else None,
)
m3.metric("Annual volatility", fmt_pct(ann_vol))
m4.metric("Sharpe ratio",  fmt_ratio(sharpe),
          help="(Return − Rf) / Volatility. >1 is good, >2 is excellent.")
m5.metric("Sortino ratio", fmt_ratio(sortino),
          help="Like Sharpe but only penalises downside volatility.")
m6.metric("Positive days", f"{summary['positive_days']:.1%}")

st.divider()

b1, b2, b3 = st.columns(3)
b1.metric("Best day",    fmt_pct(summary["best_day"]),
          delta=port_ret.idxmax().strftime("%b %d, %Y"), delta_color="off")
b2.metric("Worst day",   fmt_pct(summary["worst_day"]),
          delta=port_ret.idxmin().strftime("%b %d, %Y"), delta_color="off")
b3.metric("Final value", fmt_inr(port_value.iloc[-1]),
          delta=f"from {fmt_inr(initial_investment)}", delta_color="off")

st.divider()

# for ai use
from analytics.drawdown import max_drawdown, avg_drawdown_duration
from analytics.correlation import average_pairwise_correlation, risk_contribution

_mdd          = max_drawdown(port_ret)
_avg_dd_dur   = avg_drawdown_duration(port_ret)
_avg_corr     = average_pairwise_correlation(daily_ret)
_risk_contrib = risk_contribution(daily_ret, weights)

_beta_val  = None
_alpha_val = None
_corr_val  = None
if bench_ticker and bench_ticker in prices_all.columns:
    _mkt_ret   = prices_all[[bench_ticker]].pct_change().dropna()[bench_ticker]
    _beta_val  = compute_beta(port_ret, _mkt_ret)
    _alpha_val = compute_alpha(port_ret, _mkt_ret, risk_free)
    _corr_val  = compute_correlation(port_ret, _mkt_ret)

from analytics.var import var_summary as _var_summary
_vd = _var_summary(port_ret, confidence=0.95, horizon=1, simulations=5_000)

#tabs
tab1, tab2, tab3, tab4,tab5, tab6,tab7, tab8,tab9,tab10 = st.tabs(
    ["Performance", "Distribution", "Calendar", "Holdings","Risk Metrics","VaR and CVaR","Correlation","Stress Test","Optimization",
     "AI Advisor"]
)


#1 perfromance
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            chart_cumulative_returns(cum_ret, bench_cum, bench_label),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            chart_portfolio_value(port_value, bench_cum, bench_label, initial_investment),
            use_container_width=True,
        )

    st.plotly_chart(chart_daily_returns(port_ret), use_container_width=True)
    st.plotly_chart(chart_rolling_annual_return(port_ret), use_container_width=True)


#distribution
with tab2:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.plotly_chart(chart_return_histogram(port_ret), use_container_width=True)

    with col2:
        sk  = skew(port_ret.values)
        ku  = kurtosis(port_ret.values)
        var_95  = float(np.percentile(port_ret, 5))
        cvar_95 = float(port_ret[port_ret <= var_95].mean())

        st.markdown("**Distribution statistics**")
        st.dataframe(
            pd.DataFrame({
                "Metric": [
                    "Mean daily return", "Std dev (daily)",
                    "Skewness", "Excess kurtosis",
                    "5% VaR (daily)", "5% CVaR (daily)",
                    "Min return", "Max return",
                ],
                "Value": [
                    fmt_pct(port_ret.mean(), 3), fmt_pct(port_ret.std(), 3),
                    f"{sk:.3f}", f"{ku:.3f}",
                    fmt_pct(var_95, 3), fmt_pct(cvar_95, 3),
                    fmt_pct(port_ret.min(), 3), fmt_pct(port_ret.max(), 3),
                ],
            }),
            use_container_width=True,
            hide_index=True,
            height=312,
        )

    if sk < -0.5:
        st.info(f" **Negative skew ({sk:.2f})** — more extreme negative days than positive. Common in equity portfolios.")
    elif sk > 0.5:
        st.success(f" **Positive skew ({sk:.2f})** — more extreme upside days than downside. Favourable.")
    if ku > 1:
        st.warning(f" **Fat tails (kurtosis {ku:.2f})** — extreme returns happen more often than a normal distribution predicts. Standard VaR underestimates true risk.")


#calender

with tab3:
    st.plotly_chart(chart_monthly_returns_heatmap(port_ret), use_container_width=True)
    st.caption("Green = positive month · Red = negative · Intensity = magnitude")


#holdings

with tab4:
    col1, col2 = st.columns([1, 2])

    total_w = sum(weights.values())
    norm_w  = {k: v / total_w for k, v in weights.items()}

    with col1:
        st.plotly_chart(chart_allocation_donut(norm_w), use_container_width=True)
    with col2:
        rows = []
        for ticker in port_tickers:
            t_ret = daily_ret[ticker].dropna()
            t_cum = compute_cumulative_returns(t_ret)
            rows.append({
                "Ticker":       ticker,
                "Weight":       f"{norm_w.get(ticker, 0) * 100:.1f}%",
                "Total return": fmt_pct(float(t_cum.iloc[-1] - 1)),
                "Ann. return":  fmt_pct(annualised_return(t_ret)),
                "Ann. vol":     fmt_pct(annualised_volatility(t_ret)),
                "Sharpe":       fmt_ratio(sharpe_ratio(t_ret, risk_free)),
                "Best day":     fmt_pct(float(t_ret.max()), 3),
                "Worst day":    fmt_pct(float(t_ret.min()), 3),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.plotly_chart(chart_normalised_prices(prices), use_container_width=True)

# tab 5 risk metrics

with tab5:
    mdd=max_drawdown(port_ret)
    avg_dd_dur=avg_drawdown_duration(port_ret)
    dd_table=drawdown_details(port_ret)

    has_bench=bench_cum is not None and bench_ticker in prices_all.columns

    if has_bench:
        mkt_ret=prices_all[[bench_ticker]].pct_change().dropna()[bench_ticker]
        beta_val=compute_beta(port_ret,mkt_ret)
        alpha_val=compute_alpha(port_ret,mkt_ret)
        corr_val=compute_correlation(port_ret,mkt_ret)
        r2_val=r_squared(port_ret,mkt_ret)
    else:
        beta_val=alpha_val=corr_val=r2_val=None

    sortino_val=sortino_ratio(port_ret,risk_free_rate=risk_free)
    st.markdown("Drawdown Analysis")

    r1c1,r1c2,r1c3,r1c4=st.columns(4)
    r1c1.metric("Max drawdown",      f"-{mdd*100:.2f}%")
    r1c2.metric("Avg drawdown dur.", f"{avg_dd_dur:.0f} days")
    r1c3.metric("Sortino ratio", f"{sortino_val:.2f}",
            help="Like Sharpe but only penalises negative returns.")
    r1c4.metric("Calmar ratio",
                f"{(ann_ret / mdd):.2f}" if mdd > 0 else "N/A",
                help="Annual return / Max drawdown. Higher is better.")
    
    st.plotly_chart(chart_drawdown(port_ret),use_container_width=True)

    if not dd_table.empty:
        st.markdown("**Top 5 drawdown periods**")
        display_cols=["Peak","Trough","Recovery","Drawdown","Peak-to-Trough (d)","Recovery (d)"]
        st.dataframe(
            dd_table[display_cols],
            use_container_width=True,
        )
    else:
        st.info("No completed drawdown")

    st.divider()
    st.markdown("### volatility")
    col1,col2=st.columns(2)
    with col1:
        st.plotly_chart(chart_rolling_volatility(port_ret),use_container_width=True)
    with col2:
        st.plotly_chart(chart_rolling_sharpe(port_ret,risk_free),use_container_width=True)

    st.divider()

    if has_bench:
        st.markdown(f"### Market Exposure vs {bench_label}")

        r2c1,r2c2,r2c3,r2c4=st.columns(4)
        r2c1.metric("Beta",
                    f"{beta_val:.2f}",
                    help="1.0 = moves with market exactly. >1 = amplified. <1 = dampened.")
        r2c2.metric("Jensen's Alpha",
                    f"{alpha_val*100:.2f}%",
                    help="Return above what beta alone predicts. Positive = genuine outperformance.")
        r2c3.metric("Correlation",
                    f"{corr_val:.2f}",
                    help="How closely the portfolio tracks the benchmark. 1.0 = perfect tracking.")
        r2c4.metric("R-squared",
                    f"{r2_val:.2f}",
                    help="Fraction of portfolio variance explained by the market. 0.85 means 85%.")

        st.plotly_chart(
            chart_rolling_beta(port_ret, mkt_ret, bench_label),
            use_container_width=True,
        )

        # Interpretation callouts
        if beta_val > 1.2:
            st.warning(
                f"Beta of {beta_val:.2f} — this portfolio is significantly more volatile "
                f"than {bench_label}. In a 10% market drop, expect roughly a "
                f"{beta_val*10:.0f}% portfolio drop."
            )
        elif beta_val < 0.7:
            st.info(
                f"Beta of {beta_val:.2f} — low market sensitivity. "
                f"The portfolio moves less than the benchmark in both directions."
            )

        if alpha_val is not None and alpha_val > 0.02:
            st.success(
                f"Jensen's Alpha of {alpha_val*100:.2f}% — the portfolio generated "
                f"meaningful return above what its market exposure predicts."
            )
        elif alpha_val is not None and alpha_val < -0.02:
            st.warning(
                f"Negative alpha of {alpha_val*100:.2f}% — the portfolio underperformed "
                f"what its beta exposure would predict. Consider rebalancing."
            )
    else:
        st.info("Select a benchmark in the sidebar to see beta, alpha, and correlation metrics.")

# tab6
with tab6:

    st.markdown("### Value at Risk (VaR) & Expected Shortfall (CVaR)")

    # settings
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        confidence = st.select_slider(
            "Confidence level",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%",
        )
    with cfg2:
        horizon = st.select_slider(
            "Horizon (trading days)",
            options=[1, 5, 10, 21],
            value=1,
            format_func=lambda x: {1: "1d", 5: "1w", 10: "2w", 21: "1m"}[x],
        )
    with cfg3:
        n_sims = st.select_slider(
            "Monte Carlo simulations",
            options=[1_000, 5_000, 10_000],
            value=10_000,
            format_func=lambda x: f"{x:,}",
        )

    with st.spinner("Running VaR calculations..."):
        vd = var_summary(port_ret, confidence, horizon, n_sims)

    # cards
    st.markdown("**Value at Risk** — maximum expected daily loss at selected confidence")
    v1, v2, v3 = st.columns(3)
    v1.metric("Historical VaR",   f"-{vd['historical_var']*100:.3f}%",
              delta=f"-{fmt_inr(vd['historical_var']*initial_investment)}",
              delta_color="off",
              help="Percentile of actual past returns. No distributional assumption.")
    v2.metric("Parametric VaR",   f"-{vd['parametric_var']*100:.3f}%",
              delta=f"-{fmt_inr(vd['parametric_var']*initial_investment)}",
              delta_color="off",
              help="Assumes normal distribution. Fast but underestimates fat tails.")
    v3.metric("Monte Carlo VaR",  f"-{vd['montecarlo_var']*100:.3f}%",
              delta=f"-{fmt_inr(vd['montecarlo_var']*initial_investment)}",
              delta_color="off",
              help=f"Simulated from {n_sims:,} random paths using historical mean and vol.")

    st.divider()

    # cards cvar
    st.markdown("**Expected Shortfall (CVaR)** — average loss when VaR is breached")
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical CVaR",  f"-{vd['historical_cvar']*100:.3f}%",
              delta=f"-{fmt_inr(vd['historical_cvar']*initial_investment)}",
              delta_color="off",
              help="Mean of all losses worse than the historical VaR threshold.")
    c2.metric("Parametric CVaR",  f"-{vd['parametric_cvar']*100:.3f}%",
              delta=f"-{fmt_inr(vd['parametric_cvar']*initial_investment)}",
              delta_color="off",
              help="Analytical CVaR under normality assumption.")
    c3.metric("Monte Carlo CVaR", f"-{vd['montecarlo_cvar']*100:.3f}%",
              delta=f"-{fmt_inr(vd['montecarlo_cvar']*initial_investment)}",
              delta_color="off",
              help="Mean of simulated losses beyond the Monte Carlo VaR.")

    st.divider()

    # row1 charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            chart_var_comparison(vd, initial_investment),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            chart_var_loss_dollar(vd, initial_investment),
            use_container_width=True,
        )

    st.divider()

    # MC chart
    fan_horizon = st.slider(
        "Fan chart horizon (trading days)",
        min_value=5, max_value=63, value=30, step=5,
        help="How many days forward to simulate. 21 = 1 month, 63 = 1 quarter.",
    )
    st.plotly_chart(
        chart_monte_carlo_fan(port_ret, fan_horizon, 300, confidence, initial_investment),
        use_container_width=True,
    )

    st.divider()

    # rollinv var
    st.plotly_chart(
        chart_var_over_time(port_ret, confidence, window=126),
        use_container_width=True,
    )

    # method comparison
    st.divider()
    st.markdown("**Method comparison**")
    diff_hist_param = abs(vd["historical_var"] - vd["parametric_var"]) * 100

    if diff_hist_param > 0.3:
        st.warning(
            f"Historical and Parametric VaR differ by {diff_hist_param:.2f}%. "
            f"This gap is caused by fat tails in your return distribution — "
            f"actual extreme returns are more severe than a normal distribution predicts. "
            f"Parametric VaR is underestimating your true tail risk."
        )
    else:
        st.info(
            f"All three methods produce similar VaR estimates (within {diff_hist_param:.2f}%). "
            f"Your return distribution is approximately normal — parametric assumptions hold reasonably well."
        )

    cvar_vs_var = (vd["historical_cvar"] / vd["historical_var"] - 1) * 100
    st.info(
        f"Expected Shortfall is {cvar_vs_var:.1f}% worse than VaR on average. "
        f"This is the additional loss you should expect on the days when VaR is breached — "
        f"the number banks use for capital adequacy calculations."
    )

# tab 7

with tab7:
    st.markdown("### Correlation Dashboard")

    corr_mat  = correlation_matrix(daily_ret)
    avg_corr  = average_pairwise_correlation(daily_ret)
    div_ratio = diversification_ratio(daily_ret, weights)
    risk_contrib = risk_contribution(daily_ret, weights)

    # summary
    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Avg pairwise correlation",
        f"{avg_corr:.2f}",
        help="Average correlation across all ticker pairs. Lower = better diversification.",
    )
    d2.metric(
        "Diversification ratio",
        f"{div_ratio:.2f}",
        help="Weighted avg vol / portfolio vol. Above 1 means diversification is reducing risk.",
    )
    d3.metric(
        "Tickers",
        len(port_tickers),
    )

    if avg_corr > 0.75:
        st.warning(
            f"Average pairwise correlation of {avg_corr:.2f} is high. "
            f"Your holdings move together closely — diversification benefit is limited. "
            f"Consider adding uncorrelated assets."
        )
    elif avg_corr < 0.40:
        st.success(
            f"Average pairwise correlation of {avg_corr:.2f} is low. "
            f"Good diversification across holdings."
        )

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            chart_correlation_heatmap(corr_mat),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            chart_risk_contribution(risk_contrib),
            use_container_width=True,
        )

    # risk contribution
    st.markdown("**Weight vs risk contribution breakdown**")
    total_w = sum(weights.values())
    norm_w  = {k: v / total_w for k, v in weights.items()}

    rc_rows = []
    for ticker in port_tickers:
        w_pct  = norm_w.get(ticker, 0) * 100
        rc_pct = float(risk_contrib.get(ticker, 0))
        rc_rows.append({
            "Ticker":            ticker,
            "Portfolio weight":  f"{w_pct:.1f}%",
            "Risk contribution": f"{rc_pct:.1f}%",
            "Difference":        f"{rc_pct - w_pct:+.1f}%",
        })

    st.dataframe(
        pd.DataFrame(rc_rows),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Positive difference = ticker contributes more risk than its weight suggests.")

    st.divider()

    # rolling correlation
    if len(port_tickers) >= 2:
        st.markdown("**Rolling pairwise correlation**")
        rc1, rc2 = st.columns(2)
        with rc1:
            ticker_a = st.selectbox("Ticker A", port_tickers, index=0)
        with rc2:
            remaining = [t for t in port_tickers if t != ticker_a]
            ticker_b  = st.selectbox("Ticker B", remaining, index=0)

        roll_corr = rolling_correlation(daily_ret, ticker_a, ticker_b, window=60)
        st.plotly_chart(
            chart_rolling_correlation(roll_corr, f"{ticker_a} vs {ticker_b}", 60),
            use_container_width=True,
        )


# tab 8

with tab8:
    st.markdown("### Stress Testing")

    # summary
    all_impacts = all_scenario_impacts(weights)
    st.plotly_chart(
        chart_scenario_summary(all_impacts),
        use_container_width=True,
    )

    st.divider()

    # single
    st.markdown("**Scenario deep-dive**")
    selected_scenario = st.selectbox(
        "Select scenario",
        list(SCENARIOS.keys()),
    )

    result = apply_scenario(weights, selected_scenario)

    impact_pct      = result["portfolio_impact"] * 100
    impact_dollar   = result["portfolio_impact"] * initial_investment

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Portfolio impact",
        f"{impact_pct:+.2f}%",
        delta_color="off",
    )
    s2.metric(
        "Loss",
        fmt_inr(impact_dollar),
        delta_color="off",
    )
    s3.metric(
        "Remaining value",
        fmt_inr(initial_investment + impact_dollar),
        delta=f"from {fmt_inr(initial_investment)}",
        delta_color="off",
    )

    st.caption(f"_{result['description']}_")

    st.plotly_chart(
        chart_scenario_waterfall(result),
        use_container_width=True,
    )

    st.markdown("**Per-ticker breakdown**")
    display_cols = ["Ticker", "Weight", "Sector", "Shock", "Contribution"]
    st.dataframe(
        result["rows"][display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # custom
    st.markdown("**Custom scenario**")
    st.caption("Define your own shocks and see the portfolio impact instantly.")

    cs1, cs2, cs3, cs4 = st.columns(4)
    with cs1:
        custom_market = st.slider(
            "Market shock (%)", -60, 20, -20, step=1
        ) / 100
    with cs2:
        custom_tech = st.slider(
            "Tech shock (%)", -80, 30, -30, step=1
        ) / 100
    with cs3:
        custom_fin = st.slider(
            "Financials shock (%)", -60, 20, -15, step=1
        ) / 100
    with cs4:
        custom_energy = st.slider(
            "Energy shock (%)", -60, 50, -10, step=1
        ) / 100

    custom_result = apply_custom_scenario(
        weights, custom_market, custom_tech, custom_fin, custom_energy
    )

    custom_impact     = custom_result["portfolio_impact"] * 100
    custom_dollar     = custom_result["portfolio_impact"] * initial_investment

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Portfolio impact", f"{custom_impact:+.2f}%", delta_color="off")
    cc2.metric("Loss",             fmt_inr(custom_dollar), delta_color="off")
    cc3.metric("Remaining value",  fmt_inr(initial_investment + custom_dollar))

    st.plotly_chart(
        chart_scenario_waterfall(custom_result),
        use_container_width=True,
    )

#tab 9 optimization

with tab9:
    st.markdown("### Portfolio Optimization")
    st.caption(
        "Find better weight allocations for the same set of tickers, "
        "based on historical risk and return."
    )

    if len(port_tickers) < 2:
        st.warning("Optimization needs at least 2 tickers in the portfolio.")
        st.stop()

    opt_strategy = st.radio(
        "Optimization goal",
        ["Maximum Sharpe ratio", "Minimum variance", "Target return"],
        horizontal=True,
    )

    st.markdown("**Position constraints**")
    col_a, col_b = st.columns(2)
    with col_a:
        max_wt = st.slider(
            "Maximum weight per stock (%)",
            min_value=20, max_value=100,
            value=40, step=5,
            help="No single stock will exceed this. 40% is a common real-world limit.",
        ) / 100
    with col_b:
        min_wt = st.slider(
            "Minimum weight per stock (%)",
            min_value=0, max_value=20,
            value=5, step=1,
            help="Every stock must hold at least this. Set to 0 to allow full exclusion.",
        ) / 100

    target = None
    if opt_strategy == "Target return":
        target = st.slider(
            "Target annual return (%)",
            min_value=0.0,
            max_value=float(min(60.0, daily_ret[port_tickers].mean().max() * 252 * 150)),
            value=float(max(1.0, ann_ret * 100)),
            step=0.5,
            format="%.1f%%",
        ) / 100

    # Feasibility check 
    n_stocks = len(port_tickers)
    if min_wt * n_stocks > 1.0:
        st.error(
            f"Minimum weight of {min_wt:.0%} across {n_stocks} stocks "
            f"sums to {min_wt * n_stocks:.0%} which exceeds 100%. "
            f"Reduce the minimum weight."
        )
        st.stop()

    if max_wt * n_stocks < 1.0:
        st.error(
            f"Maximum weight of {max_wt:.0%} across {n_stocks} stocks "
            f"sums to {max_wt * n_stocks:.0%} which is below 100%. "
            f"Increase the maximum weight."
        )
        st.stop()

    total_w = sum(weights.values())
    norm_w  = {k: v / total_w for k, v in weights.items() if k in port_tickers}

    st.divider()

    # optimization

    try:
        with st.spinner("Solving optimization..."):
            if opt_strategy == "Maximum Sharpe ratio":
                opt_result = maximum_sharpe_portfolio(
                    daily_ret[port_tickers], risk_free, max_wt, min_wt
                )
                opt_label = "Max Sharpe"

            elif opt_strategy == "Minimum variance":
                opt_result = minimum_variance_portfolio(
                    daily_ret[port_tickers], max_wt, min_wt
                )
                opt_result["sharpe"] = (
                    (opt_result["return"] - risk_free) / opt_result["volatility"]
                    if opt_result["volatility"] > 0 else 0.0
                )
                opt_label = "Min Variance"

            else:
                opt_result = target_return_portfolio(
                    daily_ret[port_tickers], target, max_wt, min_wt
                )
                opt_result["sharpe"] = (
                    (opt_result["return"] - risk_free) / opt_result["volatility"]
                    if opt_result["volatility"] > 0 else 0.0
                )
                opt_label = "Target Return"

    except ValueError as e:
        st.error(str(e))
        st.stop()

    # comparision

    return_delta = (opt_result["return"] - ann_ret) * 100
    vol_delta    = (opt_result["volatility"] - ann_vol) * 100
    sharpe_delta = (
        opt_result["sharpe"] - sharpe
        if opt_result["sharpe"] is not None else None
    )

    o1, o2, o3 = st.columns(3)
    o1.metric(
        "Optimized annual return",
        f"{opt_result['return'] * 100:.2f}%",
        delta=f"{return_delta:+.2f}% vs current",
    )
    o2.metric(
        "Optimized annual volatility",
        f"{opt_result['volatility'] * 100:.2f}%",
        delta=f"{vol_delta:+.2f}% vs current",
        delta_color="inverse",
    )
    o3.metric(
        "Optimized Sharpe ratio",
        f"{opt_result['sharpe']:.2f}" if opt_result["sharpe"] is not None else "N/A",
        delta=f"{sharpe_delta:+.2f} vs current" if sharpe_delta is not None else None,
    )

    st.divider()

    # efficient frontier

    with st.spinner("Computing efficient frontier..."):
        frontier_df = efficient_frontier(
            daily_ret[port_tickers], n_points=40,
            max_wt=max_wt, min_wt=min_wt,
        )
        random_df = random_portfolio(
            daily_ret[port_tickers], n_portfolios=1500,
            max_wt=max_wt,
        )
        current_portfolio_pt = {"return": ann_ret, "volatility": ann_vol}
        min_var_pt  = minimum_variance_portfolio(
            daily_ret[port_tickers], max_wt, min_wt
        )
        max_shrp_pt = maximum_sharpe_portfolio(
            daily_ret[port_tickers], risk_free, max_wt, min_wt
        )

    st.plotly_chart(
        chart_efficient_frontier(
            frontier_df, random_df, current_portfolio_pt,
            min_var_pt, max_shrp_pt,
        ),
        use_container_width=True,
    )

    st.divider()

    # weight comp

    st.plotly_chart(
        chart_weight_comparison(norm_w, opt_result["weights"], opt_label),
        use_container_width=True,
    )

    # table

    st.markdown("**Suggested new weights**")
    weight_rows = []
    for ticker in port_tickers:
        cur = norm_w.get(ticker, 0) * 100
        opt = opt_result["weights"].get(ticker, 0) * 100
        weight_rows.append({
            "Ticker":           ticker,
            "Current weight":   f"{cur:.1f}%",
            "Suggested weight": f"{opt:.1f}%",
            "Change":           f"{opt - cur:+.1f}%",
        })
    st.dataframe(
        pd.DataFrame(weight_rows),
        use_container_width=True,
        hide_index=True,
    )

    # basic comments

    if vol_delta < -1:
        st.success(
            f"The {opt_label.lower()} portfolio reduces annualised volatility by "
            f"{abs(vol_delta):.2f} percentage points while "
            f"{'maintaining similar' if abs(return_delta) < 2 else 'adjusting'} "
            f"expected returns."
        )
    if sharpe_delta is not None and sharpe_delta > 0.1:
        st.success(
            f"Sharpe ratio improves by {sharpe_delta:.2f} — this allocation "
            f"delivers more return per unit of risk than the current portfolio."
        )

    biggest_change = max(
        weight_rows,
        key=lambda r: abs(float(r["Change"].replace("%", "").replace("+", ""))),
    )
    st.info(
        f"Largest suggested change: {biggest_change['Ticker']} moves from "
        f"{biggest_change['Current weight']} to {biggest_change['Suggested weight']}. "
        f"Max weight per stock is capped at {max_wt:.0%}."
    )

    st.caption(
        f"Constraints: min {min_wt:.0%} per stock, max {max_wt:.0%} per stock. "
        f"Adjust the sliders above to explore different constraint regimes."
    )

# tab 10

with tab10:
    st.markdown("### AI Risk Advisor")
    st.caption(
        "Using Google Gemini API "
        "Interprets your portfolio metrics in plain English."
    )

    # Check key exists before showing the button
    try:
        key_present = bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        key_present = False

    if not key_present:
        key_present = bool(os.environ.get("GEMINI_API_KEY"))

    # Show the payload so the user can see what's being sent
    with st.expander("View data being analysed"):
        payload = build_metrics_payload(
            summary=summary,
            ann_vol=ann_vol,
            sharpe=sharpe,
            sortino=sortino,
            mdd=_mdd,
            avg_dd_dur=_avg_dd_dur,
            risk_contrib=_risk_contrib,
            beta_val=_beta_val,
            alpha_val=_alpha_val,
            corr_val=_corr_val,
            var_dict=_vd,
            avg_corr=_avg_corr,
            weights=norm_w,
            bench_label=bench_label,
        )
        st.json(payload)

    if st.button("Generate risk insights", type="primary"):
        with st.spinner("Analysing portfolio..."):
            result = generate_risk_commentary(payload)

        if result["status"] == "ok":
            st.session_state["ai_commentary"] = result["text"]
            st.session_state["ai_payload_snapshot"] = payload.copy()

        elif result["status"] == "error":
            st.error(f"Gemini API error: {result['text']}")

    # Render cached commentary if it exists
    if "ai_commentary" in st.session_state:
        st.divider()
        st.markdown("**Risk observations**")

        # Render each bullet on its own line cleanly
        lines = [
            line.strip()
            for line in st.session_state["ai_commentary"].splitlines()
            if line.strip()
        ]
        for line in lines:
            st.markdown(line)

        st.divider()
        st.caption(
            "Generated by Gemini 3.1 Flash Lite based on historical data only. "
            "Not financial advice."
        )

        if st.button("Clear and regenerate"):
            del st.session_state["ai_commentary"]
            st.rerun()

    else:
        st.info(
            "Click **Generate risk insights** to get an AI-powered "
            "breakdown of this portfolio's risk profile."
        )
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
st.set_page_config(
    page_title="Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

with st.sidebar:
    st.title("Risk Dashboard")
    st.caption("Phase 1 — Portfolio Analysis")
    st.divider()

    st.markdown("**Portfolio** one per line: `TICKER WEIGHT%`")
    raw_input = st.text_area(
        "portfolio_input",
        value="AAPL 20\nMSFT 25\nGOOGL 15\nNVDA 20\nSPY 20",
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
        list(benchmark_options.keys())
    )

    bench_ticker = benchmark_options[benchmark]

    initial_investment = st.number_input(
        "Initial investment ($)",
        min_value=1_000, max_value=10_000_000,
        value=10_000, step=1_000,
        format="%d",
    )

    st.divider()
    if st.button("▶  Run Analysis", type="primary", use_container_width=True):
        st.session_state["analysis_run"] = True
    st.caption("Data: Yahoo Finance · Streamlit + Plotly")

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
b3.metric("Final value", f"${port_value.iloc[-1]:,.0f}",
          delta=f"from ${initial_investment:,.0f}", delta_color="off")

st.divider()


#tabs
tab1, tab2, tab3, tab4,tab5, tab6 = st.tabs(
    ["Performance", "Distribution", "Calendar", "Holdings","Risk Metrics","VaR and CVaR"]
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
    st.markdown("### Volatality")
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
              delta=f"-${vd['historical_var']*initial_investment:,.0f}",
              delta_color="off",
              help="Percentile of actual past returns. No distributional assumption.")
    v2.metric("Parametric VaR",   f"-{vd['parametric_var']*100:.3f}%",
              delta=f"-${vd['parametric_var']*initial_investment:,.0f}",
              delta_color="off",
              help="Assumes normal distribution. Fast but underestimates fat tails.")
    v3.metric("Monte Carlo VaR",  f"-{vd['montecarlo_var']*100:.3f}%",
              delta=f"-${vd['montecarlo_var']*initial_investment:,.0f}",
              delta_color="off",
              help=f"Simulated from {n_sims:,} random paths using historical mean and vol.")

    st.divider()

    # cards cvar
    st.markdown("**Expected Shortfall (CVaR)** — average loss when VaR is breached")
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical CVaR",  f"-{vd['historical_cvar']*100:.3f}%",
              delta=f"-${vd['historical_cvar']*initial_investment:,.0f}",
              delta_color="off",
              help="Mean of all losses worse than the historical VaR threshold.")
    c2.metric("Parametric CVaR",  f"-{vd['parametric_cvar']*100:.3f}%",
              delta=f"-${vd['parametric_cvar']*initial_investment:,.0f}",
              delta_color="off",
              help="Analytical CVaR under normality assumption.")
    c3.metric("Monte Carlo CVaR", f"-{vd['montecarlo_cvar']*100:.3f}%",
              delta=f"-${vd['montecarlo_cvar']*initial_investment:,.0f}",
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
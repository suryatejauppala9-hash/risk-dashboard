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
    run = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.caption("Data: Yahoo Finance · Streamlit + Plotly")

#landing screen

if not run:
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
tab1, tab2, tab3, tab4 = st.tabs(
    ["Performance", "Distribution", "Calendar", "Holdings"]
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


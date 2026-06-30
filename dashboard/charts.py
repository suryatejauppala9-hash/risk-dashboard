import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

PORTFOLIO_COLOR = "#6366f1"
BENCHMARK_COLOR = "#f59e0b"
POSITIVE_COLOR  = "#10b981"
NEGATIVE_COLOR  = "#ef4444"
GRID_COLOR      = "rgba(128,128,128,0.12)"
TICKER_COLORS   = [
    "#6366f1", "#f59e0b", "#10b981", "#ef4444",
    "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6",
]

_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=0, r=0, t=36, b=0),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)

def _layout(title:str,height:int,extra:dict=None)->dict:
    d=dict(**_BASE,title=dict(text=title,font=dict(size=14)),height=height)
    if extra:
        d.update(extra)
    return d

# cum return

def chart_cumulative_returns(
        port_cum: pd.Series,
        bench_cum: pd.Series=None,
        bench_label: str="S&P 500",
)->go.Figure:
    fig=go.Figure()

    fig.add_trace(go.Scatter(
        x=port_cum.index,
        y=(port_cum-1)*100,
        mode="lines",
        name="Portfolio",
        line=dict(color=PORTFOLIO_COLOR,width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="%{y:.2f}%<extra> Portfolio</extra>",
    ))

    if bench_cum is not None:
        fig.add_trace(go.Scatter(
            x=bench_cum.index,
            y=(bench_cum-1)*100,
            mode="lines",
            name=bench_label,
            line=dict(color=BENCHMARK_COLOR,width=1.8,dash="dot"),
            hovertemplate="%{y:.2f}%<extra>"+bench_label+"</extra>",
        ))

    fig.add_hline(y=0,line_width=0.08,line_color="rgba(128,128,128,0.3)")
    fig.update_layout(**_layout("Cumulative return",360))
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR,zeroline=False,ticksuffix="%")
    return fig

# port value

def chart_portfolio_value(
    port_value: pd.Series,
    bench_cum: pd.Series=None,
    bench_label: str="S&P 500",
    initial: float=10_000,
) -> go.Figure:
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=port_value.index,
        y=port_value.values,
        mode="lines",
        name="Portfolio",
        line=dict(color=PORTFOLIO_COLOR,width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="Rs. %{y:,.0f}<extra>Portfolio</extra>"
    ))

    if bench_cum is not None:
        bench_val=bench_cum*initial
        fig.add_trace(go.Scatter(
            x=bench_val.index,
            y=bench_val.values,
            mode="lines",
            name=bench_label,
            line=dict(color=BENCHMARK_COLOR,width=1.8,dash="dot"),
            hovertemplate="Rs. %{y:,.0f}<extra>"+bench_label+"</extra>",
        ))

    fig.add_hline(
        y=initial,
        line_width=0.8,
        line_color="rgba(128,128,128,0.3)",
        annotation_text=f"Rs. {initial:,.0f} invested",
        annotation_position="bottom right",
    )
    fig.update_layout(**_layout(f"Portfolio value(Rs. {initial:,.0f} invested)",360))
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR,zeroline=False,tickprefix="Rs. ",tickformat=",.0f")
    return fig


# daily returns bar


def chart_daily_returns(port_returns: pd.Series) -> go.Figure:
    colors  = [POSITIVE_COLOR if r >= 0 else NEGATIVE_COLOR for r in port_returns]
    rolling = port_returns.rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=port_returns.index,
        y=port_returns.values * 100,
        marker_color=colors,
        marker_line_width=0,
        name="Daily return",
        opacity=0.8,
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.3f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rolling.index,
        y=rolling.values * 100,
        mode="lines",
        name="30d avg",
        line=dict(color="white", width=1.5, dash="dot"),
        hovertemplate="%{y:.3f}%<extra>30d avg</extra>",
    ))

    fig.update_layout(**_layout("Daily returns", 300))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=True,
                     zerolinecolor=GRID_COLOR, zerolinewidth=1, ticksuffix="%")
    return fig

# histogram

def chart_return_histogram(port_returns: pd.Series) -> go.Figure:
    mu    = port_returns.mean() * 100
    sigma = port_returns.std() * 100
    var_5 = float(np.percentile(port_returns, 5)) * 100

    x_range = np.linspace(port_returns.min() * 100 - 0.5,
                           port_returns.max() * 100 + 0.5, 300)
    y_norm  = norm.pdf(x_range, mu, sigma)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=port_returns.values * 100,
        nbinsx=80,
        name="Daily returns",
        marker_color=PORTFOLIO_COLOR,
        marker_line_width=0,
        opacity=0.7,
        histnorm="probability density",
        hovertemplate="%{x:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_norm,
        mode="lines",
        name="Normal fit",
        line=dict(color="white", width=1.5),
        hoverinfo="skip",
    ))
    fig.add_vline(
        x=var_5,
        line_dash="dash",
        line_color=NEGATIVE_COLOR,
        line_width=1.5,
        annotation_text=f"5% VaR: {var_5:.2f}%",
        annotation_position="top right",
        annotation_font=dict(color=NEGATIVE_COLOR, size=11),
    )

    fig.update_layout(**_layout("Return distribution", 300, extra=dict(showlegend=True)))
    fig.update_xaxes(showgrid=False, zeroline=False,
                     ticksuffix="%", title_text="Daily return")
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, title_text="Density")
    return fig

# normalised prices

def chart_normalised_prices(prices: pd.DataFrame) -> go.Figure:
    norm = prices / prices.iloc[0] * 100
    fig  = go.Figure()

    for i, col in enumerate(norm.columns):
        color  = TICKER_COLORS[i % len(TICKER_COLORS)]
        change = norm[col].iloc[-1] - 100
        sign   = "+" if change >= 0 else ""
        fig.add_trace(go.Scatter(
            x=norm.index,
            y=norm[col],
            mode="lines",
            name=f"{col} ({sign}{change:.0f}%)",
            line=dict(color=color, width=1.8),
            hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(**_layout("Normalised prices (base = 100)", 380))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig

#donut

def chart_allocation_donut(weights: dict[str, float]) -> go.Figure:
    labels = list(weights.keys())
    values = [w * 100 for w in weights.values()]
    colors = TICKER_COLORS[:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        rotation=90,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text="Allocation", font=dict(size=14)),
        height=300,
        annotations=[dict(
            text="Weights", x=0.5, y=0.5,
            font=dict(size=13, color="rgba(150,150,150,0.9)"),
            showarrow=False,
        )],
    )
    return fig

#monthly returns

def chart_monthly_returns_heatmap(port_returns: pd.Series) -> go.Figure:
    df = port_returns.copy().to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    monthly = (
        df.groupby(["year", "month"])["ret"]
        .apply(lambda x: float((1 + x).prod() - 1))
        .reset_index()
    )
    pivot = monthly.pivot(index="year", columns="month", values="ret")
    pivot = pivot.sort_index(ascending=False)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    z_text = [
        [f"{v*100:.1f}%" if not np.isnan(v) else "" for v in row]
        for row in pivot.values
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=[month_names[m - 1] for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        text=z_text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=[
            [0.0,  "#ef4444"],
            [0.35, "#fca5a5"],
            [0.5,  "rgba(30,30,30,0.1)"],
            [0.65, "#6ee7b7"],
            [1.0,  "#10b981"],
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(ticksuffix="%", thickness=12, len=0.8),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text="Monthly returns heatmap", font=dict(size=14)),
        height=max(200, len(pivot) * 38 + 80),
    )
    fig.update_xaxes(side="top")
    return fig

#rolling 1 yr return

def chart_rolling_annual_return(port_returns: pd.Series) -> go.Figure:
    rolling = port_returns.rolling(252).apply(
        lambda x: float((1 + x).prod() - 1), raw=True
    ).dropna() * 100
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in rolling]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rolling.index,
        y=rolling.values,
        marker_color=colors,
        marker_line_width=0,
        name="Rolling 1y return",
        opacity=0.85,
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=0.8, line_color="rgba(128,128,128,0.4)")

    fig.update_layout(**_layout("Rolling 1-year return", 280))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%")
    return fig

# phase 2

def chart_drawdown(port_returns: pd.Series) -> go.Figure:
    from analytics.drawdown import compute_drawdown_series
    dd = compute_drawdown_series(port_returns) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        mode="lines",
        name="Drawdown",
        line=dict(color=NEGATIVE_COLOR, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra>Drawdown</extra>",
    ))
    fig.add_hline(y=0, line_width=0.8, line_color="rgba(128,128,128,0.3)")

    max_dd   = dd.min()
    max_date = dd.idxmin()
    fig.add_annotation(
        x=max_date, y=max_dd,
        text=f"Max: {max_dd:.1f}%",
        showarrow=True, arrowhead=2,
        arrowcolor=NEGATIVE_COLOR,
        font=dict(color=NEGATIVE_COLOR, size=11),
        bgcolor="rgba(0,0,0,0.6)",
        borderpad=4,
    )

    fig.update_layout(**_layout("Drawdown", 300))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%")
    return fig


def chart_rolling_volatility(port_returns: pd.Series) -> go.Figure:
    from analytics.volatility import rolling_volatility

    vol_30  = rolling_volatility(port_returns, 30) * 100
    vol_90  = rolling_volatility(port_returns, 90) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vol_30.index, y=vol_30.values,
        mode="lines", name="30d vol",
        line=dict(color=PORTFOLIO_COLOR, width=1.5),
        hovertemplate="%{y:.2f}%<extra>30d vol</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=vol_90.index, y=vol_90.values,
        mode="lines", name="90d vol",
        line=dict(color=BENCHMARK_COLOR, width=2, dash="dot"),
        hovertemplate="%{y:.2f}%<extra>90d vol</extra>",
    ))

    fig.update_layout(**_layout("Rolling volatility (annualised)", 280))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%")
    return fig


def chart_rolling_sharpe(
    port_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> go.Figure:
    from analytics.volatility import rolling_sharpe

    rs = rolling_sharpe(port_returns, window=90, risk_free_rate=risk_free_rate)

    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR
              for v in rs.fillna(0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rs.index, y=rs.values,
        marker_color=colors,
        marker_line_width=0,
        name="Rolling Sharpe (90d)",
        opacity=0.85,
        hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1, line_width=1, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)",
                  annotation_text="Sharpe = 1",
                  annotation_position="right",
                  annotation_font=dict(size=10, color="rgba(255,255,255,0.5)"))
    fig.add_hline(y=0, line_width=0.8, line_color="rgba(128,128,128,0.4)")

    fig.update_layout(**_layout("Rolling Sharpe ratio (90-day)", 280))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def chart_rolling_beta(
    port_returns: pd.Series,
    market_returns: pd.Series,
    bench_label: str = "Benchmark",
) -> go.Figure:
    """90-day rolling beta vs benchmark."""
    from analytics.beta import rolling_beta

    rb = rolling_beta(port_returns, market_returns, window=90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rb.index, y=rb.values,
        mode="lines", name=f"Beta vs {bench_label}",
        line=dict(color=PORTFOLIO_COLOR, width=2),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)",
        hovertemplate="%{x|%b %Y}<br>Beta: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1, line_width=1, line_dash="dot",
                  line_color="rgba(255,255,255,0.25)",
                  annotation_text="Beta = 1 (market)",
                  annotation_position="right",
                  annotation_font=dict(size=10, color="rgba(255,255,255,0.4)"))
    fig.add_hline(y=0, line_width=0.8, line_color="rgba(128,128,128,0.3)")

    fig.update_layout(**_layout(f"Rolling beta vs {bench_label} (90-day)", 280))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig

# phase 3 charts

def chart_var_comparison(var_dict: dict, initial: float = 10_000) -> go.Figure:
    methods     = ["Historical", "Parametric", "Monte Carlo"]
    var_vals    = [
        var_dict["historical_var"] * 100,
        var_dict["parametric_var"] * 100,
        var_dict["montecarlo_var"] * 100,
    ]
    cvar_vals   = [
        var_dict["historical_cvar"] * 100,
        var_dict["parametric_cvar"] * 100,
        var_dict["montecarlo_cvar"] * 100,
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="VaR",
        x=methods,
        y=var_vals,
        marker_color=NEGATIVE_COLOR,
        opacity=0.85,
        text=[f"{v:.2f}%" for v in var_vals],
        textposition="outside",
        hovertemplate="%{x}<br>VaR: %{y:.3f}%<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="CVaR (Expected Shortfall)",
        x=methods,
        y=cvar_vals,
        marker_color="#f97316",
        opacity=0.85,
        text=[f"{v:.2f}%" for v in cvar_vals],
        textposition="outside",
        hovertemplate="%{x}<br>CVaR: %{y:.3f}%<extra></extra>",
    ))

    confidence_pct = int(var_dict["confidence"] * 100)
    fig.update_layout(
        **_layout(f"VaR vs CVaR — {confidence_pct}% confidence (daily)", 380,
                  extra=dict(barmode="group", showlegend=True)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%")
    return fig


def chart_var_loss_dollar(var_dict: dict, initial: float = 10_000) -> go.Figure:
    methods = ["Historical VaR", "Parametric VaR", "Monte Carlo VaR",
               "Historical CVaR", "Parametric CVaR", "Monte Carlo CVaR"]
    values  = [
        var_dict["historical_var"]  * initial,
        var_dict["parametric_var"]  * initial,
        var_dict["montecarlo_var"]  * initial,
        var_dict["historical_cvar"] * initial,
        var_dict["parametric_cvar"] * initial,
        var_dict["montecarlo_cvar"] * initial,
    ]
    bar_colors = [NEGATIVE_COLOR] * 3 + ["#f97316"] * 3

    fig = go.Figure(go.Bar(
        x=values,
        y=methods,
        orientation="h",
        marker_color=bar_colors,
        opacity=0.85,
        text=[f"Rs. {v:,.0f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}<br>Loss at risk: Rs. %{x:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        **_layout(f"Loss at risk (Rs. {initial:,.0f} portfolio)", 340),
    )
    fig.update_xaxes(showgrid=False, zeroline=False,
                     tickprefix="Rs. ", tickformat=",.0f")
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def chart_monte_carlo_fan(
    port_returns: pd.Series,
    horizon: int = 30,
    n_simulations: int = 200,
    confidence: float = 0.95,
    initial: float = 10_000,
) -> go.Figure:
    from analytics.var import monte_carlo_paths

    paths    = monte_carlo_paths(port_returns, n_simulations, horizon) * initial
    x_axis   = list(range(horizon + 1))

    p5       = np.percentile(paths, 5,  axis=0)
    p25      = np.percentile(paths, 25, axis=0)
    p50      = np.percentile(paths, 50, axis=0)
    p75      = np.percentile(paths, 75, axis=0)
    p95      = np.percentile(paths, 95, axis=0)

    fig = go.Figure()

    # individual ones
    for i in range(min(50, n_simulations)):
        fig.add_trace(go.Scatter(
            x=x_axis, y=paths[i],
            mode="lines",
            line=dict(color="rgba(99,102,241,0.06)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 5-95 band
    fig.add_trace(go.Scatter(
        x=x_axis + x_axis[::-1],
        y=list(p95) + list(p5[::-1]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th percentile",
        hoverinfo="skip",
    ))

    # 25-75 band
    fig.add_trace(go.Scatter(
        x=x_axis + x_axis[::-1],
        y=list(p75) + list(p25[::-1]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th–75th percentile",
        hoverinfo="skip",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=x_axis, y=p50,
        mode="lines",
        name="Median path",
        line=dict(color=PORTFOLIO_COLOR, width=2.5),
        hovertemplate="Day %{x}<br>Median: Rs. %{y:,.0f}<extra></extra>",
    ))

    # 5th percentile (worst case)
    fig.add_trace(go.Scatter(
        x=x_axis, y=p5,
        mode="lines",
        name=f"{int((1-confidence)*100)}th percentile (VaR)",
        line=dict(color=NEGATIVE_COLOR, width=1.5, dash="dash"),
        hovertemplate="Day %{x}<br>5th pct: Rs. %{y:,.0f}<extra></extra>",
    ))

    # Starting value line
    fig.add_hline(
        y=initial,
        line_width=0.8,
        line_color="rgba(128,128,128,0.4)",
        annotation_text=f"Start: Rs. {initial:,.0f}",
        annotation_position="right",
        annotation_font=dict(size=10, color="rgba(200,200,200,0.6)"),
    )

    fig.update_layout(
        **_layout(f"Monte Carlo simulation — {horizon}-day horizon "
                  f"({n_simulations} paths)", 420),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title_text="Trading days")
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False,
                     tickprefix="Rs. ", tickformat=",.0f")
    return fig


def chart_var_over_time(
    port_returns: pd.Series,
    confidence: float = 0.95,
    window: int = 126,
) -> go.Figure:
    rolling_var = port_returns.rolling(window).apply(
        lambda x: abs(np.percentile(x, (1 - confidence) * 100)),
        raw=True,
    ).dropna() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_var.index,
        y=rolling_var.values,
        mode="lines",
        name=f"Rolling {window}d VaR ({int(confidence*100)}%)",
        line=dict(color=NEGATIVE_COLOR, width=2),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.08)",
        hovertemplate="%{x|%b %d, %Y}<br>VaR: %{y:.3f}%<extra></extra>",
    ))

    fig.update_layout(**_layout(
        f"Rolling {window}-day historical VaR ({int(confidence*100)}% confidence)", 280
    ))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, ticksuffix="%")
    return fig


# phase 4

def chart_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    tickers = list(corr_matrix.columns)
    z       = corr_matrix.values
    text    = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=tickers,
        y=tickers,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[
            [0.0,  "#3b82f6"],
            [0.5,  "rgba(20,20,30,0.2)"],
            [1.0,  "#ef4444"],
        ],
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(thickness=12, len=0.8,
                      tickvals=[-1, -0.5, 0, 0.5, 1],
                      ticktext=["-1", "-0.5", "0", "0.5", "1"]),
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))

    n = len(tickers)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text="Correlation matrix", font=dict(size=14)),
        height=max(300, n * 60 + 100),
    )
    return fig


def chart_rolling_correlation(
    roll_corr: pd.Series,
    label: str,
    window: int = 60,
) -> go.Figure:
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR
              for v in roll_corr.fillna(0)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_corr.index,
        y=roll_corr.values,
        mode="lines",
        name=label,
        line=dict(color=PORTFOLIO_COLOR, width=2),
        hovertemplate="%{x|%b %d, %Y}<br>Correlation: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0,  line_width=0.8, line_color="rgba(128,128,128,0.3)")
    fig.add_hline(y=0.8, line_width=1,  line_dash="dot",
                  line_color="rgba(239,68,68,0.4)",
                  annotation_text="High correlation (0.8)",
                  annotation_position="right",
                  annotation_font=dict(size=10, color="rgba(239,68,68,0.6)"))

    fig.update_layout(**_layout(f"Rolling {window}d correlation: {label}", 280))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False,
                     range=[-1.05, 1.05])
    return fig


def chart_risk_contribution(risk_contrib: pd.Series) -> go.Figure:
    tickers = list(risk_contrib.index)
    values  = list(risk_contrib.values)
    colors  = TICKER_COLORS[:len(tickers)]

    fig = go.Figure(go.Pie(
        labels=tickers,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="%{label}<br>Risk contribution: %{value:.1f}%<extra></extra>",
        rotation=90,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text="Risk contribution to portfolio volatility", font=dict(size=14)),
        height=300,
        annotations=[dict(
            text="Vol share",
            x=0.5, y=0.5,
            font=dict(size=12, color="rgba(150,150,150,0.9)"),
            showarrow=False,
        )],
    )
    return fig


def chart_scenario_summary(scenario_df: pd.DataFrame) -> go.Figure:
    colors = [NEGATIVE_COLOR if v < 0 else POSITIVE_COLOR
              for v in scenario_df["Portfolio loss"]]

    fig = go.Figure(go.Bar(
        x=scenario_df["Portfolio loss"],
        y=scenario_df["Scenario"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        opacity=0.85,
        text=[f"{v:+.1f}%" for v in scenario_df["Portfolio loss"]],
        textposition="outside",
        hovertemplate="%{y}<br>Portfolio impact: %{x:.2f}%<extra></extra>",
    ))

    fig.add_vline(x=0, line_width=0.8, line_color="rgba(128,128,128,0.4)")
    fig.update_layout(**_layout("Scenario stress test — portfolio impact", 340))
    fig.update_xaxes(showgrid=False, zeroline=False, ticksuffix="%")
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def chart_scenario_waterfall(result: dict) -> go.Figure:
    df      = result["rows"].sort_values("Contribution (raw)")
    total   = result["portfolio_impact"]

    measures = ["relative"] * len(df) + ["total"]
    x_labels = list(df["Ticker"]) + ["TOTAL"]
    y_values = list(df["Contribution (raw)"] * 100) + [total * 100]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        connector=dict(line=dict(color="rgba(128,128,128,0.2)", width=1)),
        decreasing=dict(
            marker=dict(color=NEGATIVE_COLOR)
        ),
        increasing=dict(
            marker=dict(color=POSITIVE_COLOR)
        ),
        totals=dict(
            marker=dict(color=PORTFOLIO_COLOR)
        ),
        text=[f"{v:+.2f}%" for v in y_values],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        **_layout(f"Scenario: {result['scenario']} — contribution by ticker", 360),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=True,
                     zerolinecolor=GRID_COLOR, ticksuffix="%")
    return fig
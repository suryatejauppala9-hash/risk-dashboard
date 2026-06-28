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
        hovertemplate="$%{y:,.0f}<extra>Portfolio</extra>"
    ))

    if bench_cum is not None:
        bench_val=bench_cum*initial
        fig.add_trace(go.Scatter(
            x=bench_val.index,
            y=bench_val.values,
            mode="lines",
            name=bench_label,
            line=dict(color=BENCHMARK_COLOR,width=1.8,dash="dot"),
            hovertemplate="$%{y:,.0f}<extra>"+bench_label+"</extra>",
        ))

    fig.add_hline(
        y=initial,
        line_width=0.8,
        line_color="rgba(128,128,128,0.3)",
        annotation_text=f"${initial:,.0f} invested",
        annotation_position="bottom right",
    )
    fig.update_layout(**_layout(f"Portfolio value(${initial:,.0f} invested)",360))
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR,zeroline=False,tickprefix="$",tickformat=",.0f")
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


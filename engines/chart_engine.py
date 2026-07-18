import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def generate_rotational_heatmap(metrics_df: pd.DataFrame) -> go.Figure:
    """Generates a matrix heatmap grouping assets by momentum velocity profiles."""
    if metrics_df.empty:
        return go.Figure()
        
    heatmap_data = metrics_df.set_index("Asset")[["1M Return", "3M Return", "63D Rolling Alpha"]]
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Performance Metric", y="Asset", color="Velocity Scale"),
        x=["1M Window", "3M Window", "63D Alpha"],
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0.0
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white"
    )
    return fig

def plot_intelligence_charts(df_indexed: pd.DataFrame, df_ratios: pd.DataFrame, base_ticker: str) -> go.Figure:
    """Renders a baseline chart integrated with real-time moving average lines to track structural trend breakouts."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.5])
    
    palette = ["#1abc9c", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#2ecc71"]
    
    # Top Panel: Normalized Assets Base 100
    for idx, col in enumerate(df_indexed.columns):
        if col == base_ticker:
            fig.add_trace(go.Scatter(x=df_indexed.index, y=df_indexed[col], mode='lines', name=f"{col} Baseline", line=dict(color="#2c3e50", width=2.5, dash="dash")), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df_indexed.index, y=df_indexed[col], mode='lines', name=f"{col} (Indexed)", line=dict(color=palette[idx % len(palette)], width=1.5)), row=1, col=1)
            
    # Bottom Panel: Active Ratio Lines + Structural Trailing Moving Average Signals
    for idx, col in enumerate(df_ratios.columns):
        color = palette[idx % len(palette)]
        series = df_ratios[col]
        ma50 = series.rolling(50).mean()
        
        # Primary Ratio Curve
        fig.add_trace(go.Scatter(x=df_ratios.index, y=series, mode='lines', name=f"{col} Ratio Line", line=dict(color=color, width=2)), row=2, col=1)
        # 50-Day Overlay to spot structural base crossovers
        fig.add_trace(go.Scatter(x=df_ratios.index, y=ma50, mode='lines', name=f"{col} 50D MA Filter", line=dict(color=color, width=1, dash="dot"), showlegend=False), row=2, col=1)
        
    fig.update_layout(template="plotly_white", height=650, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title_text="Base 100 Scale", row=1, col=1)
    fig.update_yaxes(title_text="Alpha Ratio Axis", row=2, col=1)
    return fig
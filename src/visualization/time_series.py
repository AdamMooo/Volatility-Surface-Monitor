"""
Time Series Visualizations

Track metric evolution over time.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_level_vs_curvature(
    history: pd.DataFrame,
    title: str = "ATM Vol vs Curvature",
    vol_col: str = "avg_atm_vol",
    curv_col: str = "avg_atm_curvature"
) -> go.Figure:
    """
    Plot ATM volatility vs curvature on dual axis.
    
    Highlights divergences where curvature rises while vol is flat.
    
    Args:
        history: DataFrame with date index and metric columns
        title: Plot title
        vol_col: Column name for ATM volatility
        curv_col: Column name for curvature
        
    Returns:
        Plotly Figure with dual y-axes
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    dates = history.index if isinstance(history.index, pd.DatetimeIndex) else history.index
    
    if vol_col in history.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=history[vol_col] * 100,
                name="ATM Vol (%)",
                line=dict(color="#1f77b4", width=2)
            ),
            secondary_y=False
        )
    
    if curv_col in history.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=history[curv_col] * 10000,
                name="Curvature (bps)",
                line=dict(color="#ff7f0e", width=2)
            ),
            secondary_y=True
        )
    
    if vol_col in history.columns and curv_col in history.columns:
        vol_norm = (history[vol_col] - history[vol_col].mean()) / history[vol_col].std()
        curv_norm = (history[curv_col] - history[curv_col].mean()) / history[curv_col].std()
        
        divergence = curv_norm - vol_norm
        divergence_dates = dates[divergence > 1.5]
        
        for d in divergence_dates:
            fig.add_vline(
                x=d,
                line_dash="dash",
                line_color="red",
                opacity=0.3,
                annotation_text="Pre-stress signal"
            )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        template="plotly_white",
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode="x unified"
    )
    
    fig.update_yaxes(title_text="ATM Volatility (%)", secondary_y=False)
    fig.update_yaxes(title_text="Curvature (bps)", secondary_y=True)
    
    return fig


def plot_regime_timeline(
    regime_history: pd.DataFrame,
    regime_col: str = "current_regime",
    title: str = "Market Regime Timeline"
) -> go.Figure:
    """
    Create color-coded regime timeline.
    
    Args:
        regime_history: DataFrame with date index and regime column
        regime_col: Column containing regime classification
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    regime_colors = {
        'calm': '#2ecc71',
        'pre_stress': '#f1c40f',
        'elevated': '#e67e22',
        'acute': '#e74c3c',
        'recovery': '#3498db',
        'unknown': '#95a5a6'
    }
    
    fig = go.Figure()
    
    dates = regime_history.index
    regimes = regime_history[regime_col] if regime_col in regime_history.columns else ['unknown'] * len(dates)
    
    colors = [regime_colors.get(r, '#95a5a6') for r in regimes]
    
    regime_numeric = {
        'calm': 1,
        'pre_stress': 2,
        'elevated': 3,
        'acute': 4,
        'recovery': 2.5
    }
    y_vals = [regime_numeric.get(r, 1) for r in regimes]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_vals,
        mode='lines+markers',
        marker=dict(
            color=colors,
            size=10,
            line=dict(width=1, color='white')
        ),
        line=dict(color='gray', width=1),
        hovertemplate=(
            "Date: %{x}<br>"
            "Regime: %{text}<br>"
            "<extra></extra>"
        ),
        text=regimes
    ))
    
    prev_regime = None
    for i, (date, regime) in enumerate(zip(dates, regimes)):
        if prev_regime is not None and regime != prev_regime:
            fig.add_vline(
                x=date,
                line_dash="dash",
                line_color="black",
                opacity=0.5
            )
            fig.add_annotation(
                x=date,
                y=4.5,
                text=f"{prev_regime} -> {regime}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-30,
                font=dict(size=10)
            )
        prev_regime = regime
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 2.5, 3, 4],
            ticktext=['Calm', 'Pre-Stress', 'Recovery', 'Elevated', 'Acute'],
            range=[0.5, 4.5]
        ),
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_zscore_dashboard(
    metrics_history: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Z-Score Dashboard"
) -> go.Figure:
    """
    Plot multiple metrics as z-scores with threshold bands.
    
    Args:
        metrics_history: DataFrame with date index and metric columns
        metrics: List of metric columns to plot
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    if metrics is None:
        metrics = [col for col in metrics_history.columns 
                   if 'zscore' in col.lower() or col in [
                       'avg_atm_vol', 'avg_atm_curvature', 'avg_25d_skew', 'roughness'
                   ]]
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    dates = metrics_history.index
    
    for idx, metric in enumerate(metrics[:5]):
        if metric not in metrics_history.columns:
            continue
        
        values = metrics_history[metric]
        
        if 'zscore' not in metric.lower():
            mean = values.mean()
            std = values.std()
            if std > 0:
                values = (values - mean) / std
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name=metric.replace('_', ' ').title(),
            line=dict(color=colors[idx], width=2)
        ))
    
    for band, color in [(1, 'green'), (2, 'orange'), (3, 'red')]:
        fig.add_hrect(
            y0=band, y1=band+1,
            fillcolor=color, opacity=0.1,
            line_width=0
        )
        fig.add_hrect(
            y0=-band-1, y1=-band,
            fillcolor=color, opacity=0.1,
            line_width=0
        )
    
    for val in [0, 1, -1, 2, -2]:
        fig.add_hline(
            y=val,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Z-Score",
        yaxis=dict(range=[-4, 4]),
        template="plotly_white",
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        hovermode="x unified"
    )
    
    return fig


def plot_metric_correlation(
    x: pd.Series,
    y: pd.Series,
    lag: int = 0,
    title: str = "Metric Correlation"
) -> go.Figure:
    """
    Plot correlation between two metrics with optional lag.
    
    Args:
        x: First metric series
        y: Second metric series
        lag: Lag in periods (positive means x leads y)
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    if lag > 0:
        x_aligned = x.iloc[:-lag]
        y_aligned = y.iloc[lag:]
    elif lag < 0:
        x_aligned = x.iloc[-lag:]
        y_aligned = y.iloc[:lag]
    else:
        x_aligned = x
        y_aligned = y
    
    x_aligned = x_aligned.reset_index(drop=True)
    y_aligned = y_aligned.reset_index(drop=True)
    
    corr = x_aligned.corr(y_aligned)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_aligned,
        y=y_aligned,
        mode='markers',
        marker=dict(
            size=8,
            color=range(len(x_aligned)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time")
        ),
        hovertemplate=(
            f"{x.name}: %{{x:.4f}}<br>"
            f"{y.name}: %{{y:.4f}}<br>"
            "<extra></extra>"
        )
    ))
    
    z = np.polyfit(x_aligned, y_aligned, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_aligned.min(), x_aligned.max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f'Trend (r={corr:.3f})'
    ))
    
    fig.update_layout(
        title=dict(text=f"{title} (lag={lag}, r={corr:.3f})", x=0.5),
        xaxis_title=x.name if hasattr(x, 'name') else "X",
        yaxis_title=y.name if hasattr(y, 'name') else "Y",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_metrics_summary(
    history: pd.DataFrame,
    metrics: List[str],
    title: str = "Metrics Summary"
) -> go.Figure:
    """
    Create summary plot with multiple metrics on subplots.
    
    Args:
        history: DataFrame with metrics history
        metrics: List of metric column names
        title: Plot title
        
    Returns:
        Plotly Figure with subplots
    """
    n_metrics = min(len(metrics), 4)
    
    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=metrics[:n_metrics]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    dates = history.index
    
    for idx, metric in enumerate(metrics[:n_metrics]):
        if metric not in history.columns:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=history[metric],
                name=metric,
                line=dict(color=colors[idx], width=2),
                showlegend=False
            ),
            row=idx+1,
            col=1
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        height=200 * n_metrics,
        hovermode="x unified"
    )
    
    return fig

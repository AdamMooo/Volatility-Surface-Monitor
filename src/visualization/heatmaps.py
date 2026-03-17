"""
Heatmap Visualizations

2D heatmaps for curvature, skew, and other surface metrics.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_curvature_heatmap(
    curvature_data: pd.DataFrame,
    title: str = "Curvature Heatmap",
    colorscale: str = "RdYlBu_r"
) -> go.Figure:
    """
    Create curvature heatmap (strike vs maturity).
    
    Args:
        curvature_data: DataFrame with strike, maturity, curvature columns
                       OR pivot table with strikes as index, maturities as columns
        title: Plot title
        colorscale: Plotly colorscale
        
    Returns:
        Plotly Figure
    """
    if isinstance(curvature_data.index, pd.RangeIndex):
        curvature_data = curvature_data.copy()
        curvature_data['strike'] = pd.to_numeric(curvature_data['strike'], errors='coerce')
        curvature_data['time_to_expiry'] = pd.to_numeric(curvature_data['time_to_expiry'], errors='coerce')
        curvature_data['curvature'] = pd.to_numeric(curvature_data['curvature'], errors='coerce')
        curvature_data = curvature_data.dropna()
        pivot = curvature_data.pivot_table(
            index='strike', 
            columns='time_to_expiry', 
            values='curvature',
            aggfunc='mean'
        )
    else:
        pivot = curvature_data
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 10000,
        x=[f"{float(t):.2f}y" for t in pivot.columns],
        y=pivot.index,
        colorscale=colorscale,
        colorbar=dict(title="Curvature (bps)"),
        hovertemplate=(
            "Strike: %{y}<br>"
            "Maturity: %{x}<br>"
            "Curvature: %{z:.2f} bps<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time to Expiry",
        yaxis_title="Strike",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_skew_heatmap(
    skew_data: pd.DataFrame,
    title: str = "Skew Heatmap",
    colorscale: str = "RdBu"
) -> go.Figure:
    """
    Create skew heatmap with diverging colorscale.
    
    Args:
        skew_data: DataFrame with strike, maturity, skew columns
        title: Plot title
        colorscale: Plotly diverging colorscale
        
    Returns:
        Plotly Figure
    """
    if isinstance(skew_data.index, pd.RangeIndex):
        skew_data = skew_data.copy()
        skew_data['strike'] = pd.to_numeric(skew_data['strike'], errors='coerce')
        skew_data['time_to_expiry'] = pd.to_numeric(skew_data['time_to_expiry'], errors='coerce')
        skew_data['skew'] = pd.to_numeric(skew_data['skew'], errors='coerce')
        skew_data = skew_data.dropna()
        pivot = skew_data.pivot_table(
            index='strike', 
            columns='time_to_expiry', 
            values='skew',
            aggfunc='mean'
        )
    else:
        pivot = skew_data
    
    max_abs = max(abs(pivot.values.min()), abs(pivot.values.max()))
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=[f"{float(t):.2f}y" for t in pivot.columns],
        y=pivot.index,
        colorscale=colorscale,
        zmid=0,
        zmin=-max_abs * 100,
        zmax=max_abs * 100,
        colorbar=dict(title="Skew (%)"),
        hovertemplate=(
            "Strike: %{y}<br>"
            "Maturity: %{x}<br>"
            "Skew: %{z:.3f}%<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time to Expiry",
        yaxis_title="Strike",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_arbitrage_heatmap(
    violations: pd.DataFrame,
    title: str = "Arbitrage Violations",
    colorscale: str = "Reds"
) -> go.Figure:
    """
    Create heatmap showing arbitrage violations.
    
    Args:
        violations: DataFrame with strike as index, maturity as columns,
                   values indicating violation count/severity
        title: Plot title
        colorscale: Plotly colorscale (reds for violations)
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=violations.values,
        x=[f"{t:.2f}y" for t in violations.columns],
        y=violations.index,
        colorscale=colorscale,
        colorbar=dict(title="Violations"),
        hovertemplate=(
            "Strike: %{y}<br>"
            "Maturity: %{x}<br>"
            "Violations: %{z}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time to Expiry",
        yaxis_title="Strike",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_delta_heatmap(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    metric: str = 'iv',
    title: str = "Change from Previous Day"
) -> go.Figure:
    """
    Create heatmap showing change in metric from previous snapshot.
    
    Args:
        current: Current surface data
        previous: Previous surface data
        metric: Column to compare
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    current_pivot = current.pivot(
        index='strike',
        columns='time_to_expiry',
        values=metric
    )
    
    previous_pivot = previous.pivot(
        index='strike',
        columns='time_to_expiry',
        values=metric
    )
    
    common_idx = current_pivot.index.intersection(previous_pivot.index)
    common_cols = current_pivot.columns.intersection(previous_pivot.columns)
    
    delta = (current_pivot.loc[common_idx, common_cols] - 
             previous_pivot.loc[common_idx, common_cols])
    
    max_abs = max(abs(delta.values.min()), abs(delta.values.max()))
    
    fig = go.Figure(data=go.Heatmap(
        z=delta.values * 100,
        x=[f"{t:.2f}y" for t in delta.columns],
        y=delta.index,
        colorscale="RdBu",
        zmid=0,
        zmin=-max_abs * 100,
        zmax=max_abs * 100,
        colorbar=dict(title=f"Delta {metric} (%)"),
        hovertemplate=(
            "Strike: %{y}<br>"
            "Maturity: %{x}<br>"
            f"Change: %{{z:.3f}}%<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time to Expiry",
        yaxis_title="Strike",
        template="plotly_white",
        height=500
    )
    
    return fig


def plot_surface_metrics_grid(
    metrics: Dict[str, pd.DataFrame],
    titles: Optional[Dict[str, str]] = None
) -> go.Figure:
    """
    Create grid of heatmaps for multiple surface metrics.
    
    Args:
        metrics: Dictionary mapping metric names to DataFrames
        titles: Optional custom titles for each metric
        
    Returns:
        Plotly Figure with subplots
    """
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + 1) // 2
    
    if titles is None:
        titles = {name: name.replace('_', ' ').title() for name in metrics}
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=list(titles.values())
    )
    
    colorscales = {
        'iv': 'Viridis',
        'curvature': 'RdYlBu_r',
        'skew': 'RdBu',
        'roughness': 'Oranges'
    }
    
    for idx, (name, data) in enumerate(metrics.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        colorscale = colorscales.get(name, 'Viridis')
        
        fig.add_trace(
            go.Heatmap(
                z=data.values,
                x=list(data.columns),
                y=list(data.index),
                colorscale=colorscale,
                showscale=True
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        title=dict(text="Surface Metrics Overview", x=0.5),
        template="plotly_white",
        height=400 * n_rows
    )
    
    return fig

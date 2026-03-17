"""
3D Surface Visualization

Interactive Plotly visualizations of the IV surface.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter


def plot_iv_surface(
    surface,
    title: str = "Implied Volatility Surface",
    colorscale: str = "Viridis",
    spot: Optional[float] = None,
    K_steps: int = 80,
    T_steps: int = 40
) -> go.Figure:
    """
    Create a classic smooth 3D implied-volatility surface.

    Bypasses the IVSurface RBF (which can blow up) and instead
    interpolates the raw IV data directly with scipy griddata for
    a well-behaved, visually clean surface.
    """
    from scipy.interpolate import griddata as _griddata

    # ---- Extract raw IV data ----
    if isinstance(surface, pd.DataFrame):
        df = surface.copy()
        if 'implied_volatility' in df.columns:
            df = df.rename(columns={'implied_volatility': 'iv'})
        if spot is None and 'spot' in df.columns:
            spot = float(df['spot'].iloc[0])
    else:
        df = surface.to_dataframe()
        if spot is None:
            spot = surface.spot

    # Ensure numeric
    for col in ('strike', 'time_to_expiry', 'iv'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['strike', 'time_to_expiry', 'iv'])
    df = df[df['iv'] > 0]

    # ---- Restrict to a sensible moneyness window ----
    if spot:
        df = df[(df['strike'] >= spot * 0.80) & (df['strike'] <= spot * 1.20)]

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No option data in range", showarrow=False)
        return fig

    # ---- Build a regular grid ----
    K_vals = np.linspace(df['strike'].min(), df['strike'].max(), K_steps)
    T_vals = np.linspace(df['time_to_expiry'].min(), df['time_to_expiry'].max(), T_steps)
    X, Y = np.meshgrid(K_vals, T_vals)

    # Cubic interpolation on the raw data – well-behaved, no RBF blowup
    Z = _griddata(
        (df['strike'].values, df['time_to_expiry'].values),
        df['iv'].values * 100,
        (X, Y),
        method='cubic'
    )

    # Fill any NaN edges with nearest-neighbor so there are no holes
    Z_nearest = _griddata(
        (df['strike'].values, df['time_to_expiry'].values),
        df['iv'].values * 100,
        (X, Y),
        method='nearest'
    )
    Z = np.where(np.isnan(Z), Z_nearest, Z)

    # Light gaussian smooth for a polished look
    Z = gaussian_filter(Z, sigma=1.0)

    # ---- Plot ----
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        colorbar=dict(title=dict(text="IV (%)", side="right")),
        hovertemplate=(
            "Strike: %{x:.0f}<br>"
            "Maturity: %{y:.3f} yrs<br>"
            "IV: %{z:.2f}%<br>"
            "<extra></extra>"
        ),
        connectgaps=True,
        lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        )
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry (years)",
            zaxis_title="Implied Volatility (%)",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
            aspectratio=dict(x=1.2, y=1.2, z=0.7),
        ),
        template="plotly_white",
        height=700,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def plot_iv_surface_animated(
    history: List[pd.DataFrame],
    dates: List[str],
    title: str = "IV Surface Evolution"
) -> go.Figure:
    """
    Create animated 3D IV surface showing evolution over time.
    
    Args:
        history: List of surface DataFrames for each date
        dates: List of date labels
        title: Plot title
        
    Returns:
        Plotly Figure with animation
    """
    if not history:
        return go.Figure()
    
    all_strikes = set()
    all_maturities = set()
    for df in history:
        all_strikes.update(df['strike'].unique())
        all_maturities.update(df['time_to_expiry'].unique())
    
    strikes = sorted(all_strikes)
    maturities = sorted(all_maturities)
    X, Y = np.meshgrid(strikes, maturities)
    
    frames = []
    for idx, (df, date) in enumerate(zip(history, dates)):
        Z = np.full((len(maturities), len(strikes)), np.nan)
        for _, row in df.iterrows():
            if row['strike'] in strikes and row['time_to_expiry'] in maturities:
                i = maturities.index(row['time_to_expiry'])
                j = strikes.index(row['strike'])
                Z[i, j] = row['iv'] * 100
        
        frames.append(go.Frame(
            data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")],
            name=str(idx)
        ))
    
    first_z = np.full((len(maturities), len(strikes)), np.nan)
    for _, row in history[0].iterrows():
        if row['strike'] in strikes and row['time_to_expiry'] in maturities:
            i = maturities.index(row['time_to_expiry'])
            j = strikes.index(row['strike'])
            first_z[i, j] = row['iv'] * 100
    
    fig = go.Figure(
        data=[go.Surface(x=X, y=Y, z=first_z, colorscale="Viridis")],
        frames=frames
    )
    
    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=16),
            prefix="Date: ",
            visible=True,
            xanchor="right"
        ),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=[
            dict(
                args=[[str(i)], dict(frame=dict(duration=500, redraw=True), mode="immediate")],
                label=dates[i],
                method="animate"
            )
            for i in range(len(dates))
        ]
    )]
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry (years)",
            zaxis_title="Implied Volatility (%)"
        ),
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1,
            x=0.1,
            xanchor="right",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True),
                                     fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=True),
                                       mode="immediate")])
            ]
        )],
        template="plotly_white",
        height=700
    )
    
    return fig


def plot_smile_comparison(
    smiles: Dict[str, pd.DataFrame],
    title: str = "Volatility Smile Comparison",
    x_axis: str = "moneyness"
) -> go.Figure:
    """
    Compare volatility smiles across maturities or dates.
    
    Args:
        smiles: Dictionary mapping labels to smile DataFrames
        title: Plot title
        x_axis: 'strike' or 'moneyness'
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    
    for idx, (label, df) in enumerate(smiles.items()):
        x_col = 'moneyness' if x_axis == 'moneyness' and 'moneyness' in df.columns else 'strike'
        
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df['iv'] * 100,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    x_title = "Moneyness (K/S)" if x_axis == "moneyness" else "Strike"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=x_title,
        yaxis_title="Implied Volatility (%)",
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


def plot_term_structure(
    term_structure: pd.DataFrame,
    title: str = "ATM Volatility Term Structure",
    show_historical: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Plot ATM volatility term structure.
    
    Args:
        term_structure: DataFrame with maturity and iv columns
        title: Plot title
        show_historical: Optional historical term structures for comparison
        
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=term_structure['maturity'],
        y=term_structure['iv'] * 100,
        mode='lines+markers',
        name='Current',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    if show_historical is not None:
        for col in show_historical.columns:
            if col != 'maturity':
                fig.add_trace(go.Scatter(
                    x=show_historical['maturity'],
                    y=show_historical[col] * 100,
                    mode='lines',
                    name=col,
                    line=dict(width=1, dash='dash'),
                    opacity=0.5
                ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Time to Expiry (years)",
        yaxis_title="ATM Implied Volatility (%)",
        template="plotly_white",
        height=400,
        hovermode="x unified"
    )
    
    return fig

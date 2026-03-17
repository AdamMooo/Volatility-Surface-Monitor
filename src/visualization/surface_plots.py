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
    show_atm: bool = True,
    spot: Optional[float] = None,
    K_steps: int = 60,
    T_steps: int = 30
) -> go.Figure:
    """
    Create 3D IV surface plot from an IVSurface object.

    Uses the interpolated surface to produce a smooth, regular grid.

    Args:
        surface: An IVSurface object (must be built) OR a DataFrame
                 with strike, time_to_expiry, iv columns (legacy).
        title: Plot title
        colorscale: Plotly colorscale name
        show_atm: Whether to highlight ATM line
        spot: Spot price for ATM reference
        K_steps: Grid resolution along strike axis
        T_steps: Grid resolution along maturity axis

    Returns:
        Plotly Figure object
    """
    # ------------------------------------------------------------------
    # Build the regular grid from the interpolated surface
    # ------------------------------------------------------------------
    if isinstance(surface, pd.DataFrame):
        # Legacy path: caller passed a DataFrame instead of IVSurface
        df = surface
        if 'implied_volatility' in df.columns:
            df = df.rename(columns={'implied_volatility': 'iv'})
        strikes = np.linspace(df['strike'].min(), df['strike'].max(), K_steps)
        maturities = np.linspace(
            df['time_to_expiry'].min(), df['time_to_expiry'].max(), T_steps
        )
        X, Y = np.meshgrid(strikes, maturities)
        from scipy.interpolate import griddata as _griddata
        Z = _griddata(
            (df['strike'].values, df['time_to_expiry'].values),
            df['iv'].values * 100,
            (X, Y),
            method='cubic'
        )
        if spot is None and 'spot' in df.columns:
            spot = float(df['spot'].iloc[0])
    else:
        # Preferred path: IVSurface object
        if spot is None:
            spot = surface.spot

        # Use a sensible moneyness range (80%-120% of spot) so the
        # surface doesn't include extreme wings that blow up.
        K_min = spot * 0.80
        K_max = spot * 1.20
        T_min = max(min(surface.maturities), 0.02) if surface.maturities else 0.02
        T_max = max(surface.maturities) if surface.maturities else 1.0

        K_grid, T_grid, IV_grid = surface.evaluate_grid(
            K_range=(K_min, K_max),
            T_range=(T_min, T_max),
            K_steps=K_steps,
            T_steps=T_steps
        )
        X = K_grid
        Y = T_grid
        Z = IV_grid * 100  # convert to percent

        # Clamp outliers and smooth for a clean classic surface
        median_iv = np.nanmedian(Z)
        Z = np.clip(Z, 0, median_iv * 3)
        Z = np.where(np.isnan(Z), median_iv, Z)
        Z = gaussian_filter(Z, sigma=1.2)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="IV (%)", side="right")
        ),
        hovertemplate=(
            "Strike: %{x:.0f}<br>"
            "Maturity: %{y:.3f} yrs<br>"
            "IV: %{z:.2f}%<br>"
            "<extra></extra>"
        ),
        connectgaps=True
    ))

    # ATM line across maturities
    if show_atm and spot is not None:
        T_vals = np.linspace(Y.min(), Y.max(), T_steps)
        atm_ivs = []
        for T in T_vals:
            try:
                if hasattr(surface, 'evaluate'):
                    iv = surface.evaluate(spot, float(T)) * 100
                else:
                    iv = np.nan
                atm_ivs.append(iv if np.isfinite(iv) else np.nan)
            except Exception:
                atm_ivs.append(np.nan)

        fig.add_trace(go.Scatter3d(
            x=[spot] * len(T_vals),
            y=T_vals.tolist(),
            z=atm_ivs,
            mode='lines',
            line=dict(color='red', width=5),
            name='ATM'
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Expiry (years)",
            zaxis_title="Implied Volatility (%)",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1)
            )
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

"""
Regime-Specific Visualizations

Visual representation of market regime status.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REGIME_COLORS = {
    'calm': '#2ecc71',
    'pre_stress': '#f1c40f',
    'elevated': '#e67e22',
    'acute': '#e74c3c',
    'recovery': '#3498db',
    'unknown': '#95a5a6'
}


def plot_regime_gauge(
    regime_data: Dict[str, Any],
    title: str = "Market Regime"
) -> go.Figure:
    """
    Create circular gauge showing current regime.
    
    Args:
        regime_data: Dictionary with current_regime and confidence
        title: Plot title
        
    Returns:
        Plotly Figure with gauge
    """
    regime = regime_data.get('current_regime', 'unknown')
    confidence = regime_data.get('confidence', 0.5)
    
    regime_values = {
        'calm': 1,
        'pre_stress': 2,
        'recovery': 2.5,
        'elevated': 3,
        'acute': 4
    }
    
    value = regime_values.get(regime, 2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title=dict(text=f"<b>{regime.replace('_', ' ').upper()}</b>"),
        gauge=dict(
            axis=dict(
                range=[0, 5],
                tickmode='array',
                tickvals=[1, 2, 3, 4],
                ticktext=['Calm', 'Pre-Stress', 'Elevated', 'Acute']
            ),
            bar=dict(color=REGIME_COLORS.get(regime, '#95a5a6')),
            steps=[
                dict(range=[0, 1.5], color='rgba(46, 204, 113, 0.3)'),
                dict(range=[1.5, 2.5], color='rgba(241, 196, 15, 0.3)'),
                dict(range=[2.5, 3.5], color='rgba(230, 126, 34, 0.3)'),
                dict(range=[3.5, 5], color='rgba(231, 76, 60, 0.3)')
            ],
            threshold=dict(
                line=dict(color="black", width=4),
                thickness=0.75,
                value=value
            )
        )
    ))
    
    fig.add_annotation(
        x=0.5,
        y=-0.1,
        text=f"Confidence: {confidence:.0%}",
        showarrow=False,
        font=dict(size=14)
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=350,
        template="plotly_white"
    )
    
    return fig


def plot_regime_probabilities(
    probabilities: Dict[str, float],
    title: str = "Regime Probabilities"
) -> go.Figure:
    """
    Create bar chart of regime probabilities.
    
    Args:
        probabilities: Dictionary mapping regime names to probabilities
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    regimes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [REGIME_COLORS.get(r, '#95a5a6') for r in regimes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=regimes,
        y=[p * 100 for p in probs],
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition='outside',
        hovertemplate=(
            "Regime: %{x}<br>"
            "Probability: %{y:.1f}%<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Regime",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_early_warning_dashboard(
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, tuple]] = None,
    title: str = "Early Warning Dashboard"
) -> go.Figure:
    """
    Create traffic light dashboard for key metrics.
    
    Args:
        metrics: Dictionary with current metric values and percentiles
        thresholds: Optional custom thresholds for each metric
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    if thresholds is None:
        thresholds = {
            'atm_vol': (40, 70, 90),
            'curvature': (40, 70, 90),
            'skew': (40, 70, 90),
            'butterfly': (40, 70, 90)
        }
    
    def get_color(pct: float, thresh: tuple) -> str:
        if pct < thresh[0]:
            return '#2ecc71'
        elif pct < thresh[1]:
            return '#f1c40f'
        elif pct < thresh[2]:
            return '#e67e22'
        else:
            return '#e74c3c'
    
    percentiles = metrics.get('percentiles', {})
    
    if not percentiles:
        return go.Figure().update_layout(title="No percentile data available")
    
    n_metrics = len(percentiles)
    
    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        specs=[[{'type': 'indicator'}] * n_metrics],
        subplot_titles=list(percentiles.keys())
    )
    
    for idx, (metric, pct) in enumerate(percentiles.items()):
        thresh = thresholds.get(metric, (40, 70, 90))
        color = get_color(pct, thresh)
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=pct,
                number=dict(suffix="th pct"),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=color, thickness=0.8),
                    steps=[
                        dict(range=[0, thresh[0]], color='rgba(46, 204, 113, 0.3)'),
                        dict(range=[thresh[0], thresh[1]], color='rgba(241, 196, 15, 0.3)'),
                        dict(range=[thresh[1], thresh[2]], color='rgba(230, 126, 34, 0.3)'),
                        dict(range=[thresh[2], 100], color='rgba(231, 76, 60, 0.3)')
                    ]
                )
            ),
            row=1,
            col=idx+1
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=300,
        template="plotly_white"
    )
    
    return fig


def plot_regime_sankey(
    transition_history: pd.DataFrame,
    title: str = "Regime Transitions"
) -> go.Figure:
    """
    Create Sankey diagram of regime transitions.
    
    Args:
        transition_history: DataFrame with from_regime, to_regime, count columns
        title: Plot title
        
    Returns:
        Plotly Figure with Sankey diagram
    """
    regimes = ['calm', 'pre_stress', 'elevated', 'acute', 'recovery']
    
    source_regimes = [f"{r}_from" for r in regimes]
    target_regimes = [f"{r}_to" for r in regimes]
    all_labels = source_regimes + target_regimes
    
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    sources = []
    targets = []
    values = []
    colors = []
    
    if 'from_regime' in transition_history.columns:
        for _, row in transition_history.iterrows():
            from_reg = row['from_regime']
            to_reg = row['to_regime']
            count = row.get('count', 1)
            
            source_idx = label_to_idx.get(f"{from_reg}_from")
            target_idx = label_to_idx.get(f"{to_reg}_to")
            
            if source_idx is not None and target_idx is not None:
                sources.append(source_idx)
                targets.append(target_idx)
                values.append(count)
                colors.append(REGIME_COLORS.get(to_reg, '#95a5a6'))
    
    if not sources:
        return go.Figure().update_layout(title="No transition data available")
    
    node_colors = [REGIME_COLORS.get(r.replace('_from', '').replace('_to', ''), '#95a5a6') 
                   for r in all_labels]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[r.replace('_from', '').replace('_to', '').replace('_', ' ').title() 
                   for r in all_labels],
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=500,
        template="plotly_white"
    )
    
    return fig


def plot_signal_strength(
    signals: Dict[str, Any],
    title: str = "Signal Strength"
) -> go.Figure:
    """
    Create bullet chart for signal strength.
    
    Args:
        signals: Dictionary with signal types and strengths
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    signal_types = ['pre_stress_signal', 'stress_signal', 'recovery_signal']
    
    fig = go.Figure()
    
    for idx, signal in enumerate(signal_types):
        is_active = signals.get(signal, False)
        strength = signals.get('signal_strength', 0) if is_active else 0
        
        color = '#e74c3c' if is_active else '#95a5a6'
        
        fig.add_trace(go.Bar(
            x=[strength * 100],
            y=[signal.replace('_', ' ').title()],
            orientation='h',
            marker_color=color,
            text=f"{'ACTIVE' if is_active else 'Inactive'}",
            textposition='inside',
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Strength (%)", range=[0, 100]),
        yaxis=dict(title=""),
        template="plotly_white",
        height=250,
        barmode='group'
    )
    
    return fig


def create_regime_summary_card(
    regime_data: Dict[str, Any]
) -> go.Figure:
    """
    Create a summary card with all regime information.
    
    Args:
        regime_data: Complete regime classification output
        
    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{'type': 'indicator'}, {'type': 'bar'}],
            [{'type': 'table', 'colspan': 2}, None]
        ],
        subplot_titles=['Current Regime', 'Regime Probabilities', 'Key Drivers'],
        row_heights=[0.5, 0.5]
    )
    
    regime = regime_data.get('current_regime', 'unknown')
    confidence = regime_data.get('confidence', 0.5)
    
    regime_values = {'calm': 1, 'pre_stress': 2, 'recovery': 2.5, 'elevated': 3, 'acute': 4}
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=regime_values.get(regime, 2),
            title=dict(text=regime.replace('_', ' ').upper()),
            gauge=dict(
                axis=dict(range=[0, 5], visible=False),
                bar=dict(color=REGIME_COLORS.get(regime, '#95a5a6'))
            )
        ),
        row=1, col=1
    )
    
    probs = regime_data.get('regime_probability', {})
    fig.add_trace(
        go.Bar(
            x=list(probs.keys()),
            y=[p * 100 for p in probs.values()],
            marker_color=[REGIME_COLORS.get(r, '#95a5a6') for r in probs.keys()]
        ),
        row=1, col=2
    )
    
    drivers = regime_data.get('key_drivers', ['No drivers identified'])
    fig.add_trace(
        go.Table(
            cells=dict(
                values=[drivers],
                align='left',
                height=30
            )
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

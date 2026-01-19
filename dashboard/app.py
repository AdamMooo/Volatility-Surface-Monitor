"""
Volatility Surface Integrity & Stress Monitor
Simplified Dashboard - Working Version
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import project modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from data.fetcher import OptionChainFetcher
from data.cleaner import clean_option_chain


# Initialize app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    title="Vol Surface Monitor",
    suppress_callback_exceptions=True
)

server = app.server


# Colors
COLORS = {
    'bg': '#0d1117',
    'card': '#161b22',
    'border': '#30363d',
    'text': '#f0f6fc',
    'muted': '#8b949e',
    'blue': '#58a6ff',
    'green': '#3fb950',
    'yellow': '#d29922',
    'orange': '#db6d28',
    'red': '#f85149',
    'purple': '#a371f7',
}

REGIME_COLORS = {
    'CALM': COLORS['green'],
    'PRE_STRESS': COLORS['yellow'],
    'ELEVATED': COLORS['orange'],
    'ACUTE': COLORS['red'],
    'UNKNOWN': COLORS['muted'],
}


# ============ HELPER FUNCTIONS ============

def compute_iv_surface(df):
    """Compute IV surface grid from option chain."""
    if 'implied_volatility_market' not in df.columns:
        return None, None, None
    
    # Filter for meaningful data
    valid = df[
        (df['implied_volatility_market'] > 0.01) & 
        (df['implied_volatility_market'] < 2.0) &
        (df['moneyness'] > 0.8) &
        (df['moneyness'] < 1.2)
    ].copy()
    
    if len(valid) < 10:
        return None, None, None
    
    # Create grid
    moneyness_bins = np.linspace(0.85, 1.15, 25)
    expiry_bins = np.linspace(valid['days_to_expiry'].min(), 
                               min(valid['days_to_expiry'].max(), 90), 15)
    
    grid = np.full((len(expiry_bins), len(moneyness_bins)), np.nan)
    
    for i, exp in enumerate(expiry_bins):
        for j, mon in enumerate(moneyness_bins):
            mask = (
                (abs(valid['moneyness'] - mon) < 0.03) &
                (abs(valid['days_to_expiry'] - exp) < 7)
            )
            subset = valid[mask]
            if len(subset) > 0:
                grid[i, j] = subset['implied_volatility_market'].mean()
    
    # Interpolate NaNs
    from scipy.ndimage import generic_filter
    
    def nanmean_filter(x):
        valid_vals = x[~np.isnan(x)]
        return np.mean(valid_vals) if len(valid_vals) > 0 else np.nan
    
    for _ in range(3):
        mask = np.isnan(grid)
        if not mask.any():
            break
        grid = np.where(mask, generic_filter(grid, nanmean_filter, size=3, mode='nearest'), grid)
    
    return grid, moneyness_bins, expiry_bins


def compute_metrics(df):
    """Compute key surface metrics."""
    if 'implied_volatility_market' not in df.columns or len(df) == 0:
        return {}
    
    valid = df[
        (df['implied_volatility_market'] > 0.01) & 
        (df['implied_volatility_market'] < 2.0)
    ].copy()
    
    if len(valid) < 10:
        return {}
    
    # ATM vol (moneyness ~ 1.0, short-term)
    atm_mask = (valid['moneyness'] > 0.97) & (valid['moneyness'] < 1.03)
    short_term = valid['days_to_expiry'] < 45
    atm_options = valid[atm_mask & short_term]
    atm_vol = atm_options['implied_volatility_market'].mean() if len(atm_options) > 0 else 0.15
    
    # 25-Delta Skew (put wing - call wing)
    put_wing = valid[(valid['moneyness'] > 0.90) & (valid['moneyness'] < 0.95)]
    call_wing = valid[(valid['moneyness'] > 1.05) & (valid['moneyness'] < 1.10)]
    
    put_vol = put_wing['implied_volatility_market'].mean() if len(put_wing) > 0 else atm_vol
    call_vol = call_wing['implied_volatility_market'].mean() if len(call_wing) > 0 else atm_vol
    skew_25d = put_vol - call_vol
    
    # Curvature (butterfly: wing average - ATM)
    wing_avg = (put_vol + call_vol) / 2
    curvature = wing_avg - atm_vol
    
    # Term structure slope
    short_atm = valid[atm_mask & (valid['days_to_expiry'] < 30)]['implied_volatility_market'].mean()
    long_atm = valid[atm_mask & (valid['days_to_expiry'] > 60)]['implied_volatility_market'].mean()
    term_slope = (long_atm - short_atm) if not np.isnan(long_atm) and not np.isnan(short_atm) else 0
    
    return {
        'atm_vol': atm_vol,
        'skew_25d': skew_25d,
        'curvature': curvature,
        'term_slope': term_slope,
        'put_vol': put_vol,
        'call_vol': call_vol,
    }


def classify_regime(metrics):
    """Simple regime classification with detailed explanations."""
    if not metrics:
        return 'UNKNOWN', 0.5, []
    
    atm_vol = metrics.get('atm_vol', 0.15)
    curvature = metrics.get('curvature', 0)
    skew = abs(metrics.get('skew_25d', 0))
    term_slope = metrics.get('term_slope', 0)
    
    # Score based on metrics
    score = 0
    drivers = []
    
    # ATM Volatility Analysis
    if atm_vol > 0.30:
        score += 30
        drivers.append({
            'title': f"High ATM Volatility: {atm_vol*100:.1f}%",
            'explanation': "At-the-money implied volatility is elevated above 30%. This indicates the market expects large price swings. Historically, ATM vol above 30% often precedes or accompanies significant market moves. Consider reducing position sizes or hedging.",
            'severity': 'high'
        })
    elif atm_vol > 0.20:
        score += 15
        drivers.append({
            'title': f"Elevated ATM Volatility: {atm_vol*100:.1f}%",
            'explanation': "ATM volatility is moderately elevated (20-30%). The market is pricing in above-average uncertainty. This is common during earnings seasons or ahead of economic events.",
            'severity': 'medium'
        })
        
    # Curvature (Smile) Analysis
    if curvature > 0.05:
        score += 25
        drivers.append({
            'title': f"High Curvature (Smile): {curvature*100:.2f}%",
            'explanation': "The volatility smile is very pronounced - both put and call wings are expensive relative to ATM options. This 'fat tails' pricing means the market expects potential for extreme moves in either direction. Often seen before binary events or during market stress.",
            'severity': 'high'
        })
    elif curvature > 0.02:
        score += 10
        drivers.append({
            'title': f"Elevated Curvature: {curvature*100:.2f}%",
            'explanation': "Moderate smile curvature detected. Out-of-the-money options are priced higher than a flat volatility surface would suggest. Tail risk hedging demand is above normal.",
            'severity': 'medium'
        })
        
    # Skew Analysis
    if skew > 0.08:
        score += 25
        drivers.append({
            'title': f"Steep Put Skew: {skew*100:.2f}%",
            'explanation': "Put options are significantly more expensive than calls (steep negative skew). This is a classic 'crash protection' signal - institutional investors are paying up for downside protection. High skew often precedes or accompanies market selloffs.",
            'severity': 'high'
        })
    elif skew > 0.04:
        score += 10
        drivers.append({
            'title': f"Elevated Skew: {skew*100:.2f}%",
            'explanation': "Put skew is moderately elevated. Downside protection is in demand but not at extreme levels. This is typical during uncertain market conditions.",
            'severity': 'medium'
        })
    
    # Term Structure Analysis
    if term_slope < -0.03:
        score += 15
        drivers.append({
            'title': f"Inverted Term Structure: {term_slope*100:.2f}%",
            'explanation': "Short-term volatility exceeds long-term volatility (backwardation). This inversion typically signals near-term stress or an imminent event. The market expects turbulence NOW rather than later.",
            'severity': 'high'
        })
    elif term_slope > 0.05:
        drivers.append({
            'title': f"Steep Contango: {term_slope*100:.2f}%",
            'explanation': "Long-term volatility is much higher than short-term. The market is calm now but pricing in future uncertainty. Could indicate complacency or anticipated future events.",
            'severity': 'low'
        })
    
    # Classify
    if score >= 60:
        regime = 'ACUTE'
    elif score >= 40:
        regime = 'ELEVATED'
    elif score >= 20:
        regime = 'PRE_STRESS'
    else:
        regime = 'CALM'
        
    confidence = min(1.0, score / 80)
    
    return regime, confidence, drivers


# ============ LAYOUT ============

def create_metric_card(metric_id, icon, title, color):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon}", style={'color': color, 'fontSize': '1.5rem'}),
            ], style={'marginBottom': '10px'}),
            html.H3(id=metric_id, children="--", 
                   style={'color': COLORS['text'], 'marginBottom': '5px'}),
            html.Small(title, style={'color': COLORS['muted']})
        ], className='text-center')
    ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}',
              'borderRadius': '12px'})


app.layout = dbc.Container([
    # Stores
    dcc.Store(id='chain-store'),
    dcc.Store(id='metrics-store'),
    dcc.Interval(id='auto-refresh', interval=120*1000, n_intervals=0),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H3([
                html.I(className="fas fa-chart-area", style={'color': COLORS['blue'], 'marginRight': '15px'}),
                "Volatility Surface Monitor"
            ], style={'color': COLORS['text'], 'margin': '0'})
        ], md=4),
        dbc.Col([
            html.Div(id='regime-badge', 
                     style={'backgroundColor': COLORS['card'], 'padding': '10px 20px',
                            'borderRadius': '20px', 'display': 'inline-block',
                            'border': f'1px solid {COLORS["border"]}'}),
        ], md=4, className='text-center'),
        dbc.Col([
            html.Span(id='update-time', style={'color': COLORS['muted'], 'marginRight': '15px'}),
            dbc.Button([html.I(className="fas fa-sync-alt me-2"), "Refresh"],
                      id='refresh-btn', color='primary', size='sm')
        ], md=4, className='text-end')
    ], className='py-3 mb-4', style={'borderBottom': f'1px solid {COLORS["border"]}'}),
    
    # Controls
    dbc.Row([
        dbc.Col([
            html.Label("Ticker", style={'color': COLORS['muted'], 'fontSize': '0.85rem'}),
            dbc.Select(id='ticker-select', value='SPY',
                      options=[{'label': 'SPY - S&P 500', 'value': 'SPY'},
                              {'label': 'QQQ - Nasdaq 100', 'value': 'QQQ'},
                              {'label': 'IWM - Russell 2000', 'value': 'IWM'},
                              {'label': 'SPMO - S&P Momentum', 'value': 'SPMO'}],
                      style={'backgroundColor': COLORS['card'], 'color': COLORS['text'],
                             'border': f'1px solid {COLORS["border"]}'})
        ], md=2),
        dbc.Col([
            html.Label("Expiry Range (days)", style={'color': COLORS['muted'], 'fontSize': '0.85rem'}),
            dcc.RangeSlider(id='expiry-range', min=7, max=120, value=[14, 60],
                           marks={7: '7', 30: '30', 60: '60', 90: '90', 120: '120'})
        ], md=5),
        dbc.Col([
            html.Label("Moneyness Range", style={'color': COLORS['muted'], 'fontSize': '0.85rem'}),
            dcc.RangeSlider(id='moneyness-range', min=0.8, max=1.2, step=0.01,
                           value=[0.9, 1.1],
                           marks={0.8: '80%', 0.9: '90%', 1.0: 'ATM', 1.1: '110%', 1.2: '120%'})
        ], md=5)
    ], className='mb-4'),
    
    # Metrics Row
    dbc.Row([
        dbc.Col([create_metric_card('atm-vol', 'fa-chart-line', 'ATM Volatility', COLORS['blue'])], md=3),
        dbc.Col([create_metric_card('skew-val', 'fa-arrows-alt-h', '25Δ Skew', COLORS['purple'])], md=3),
        dbc.Col([create_metric_card('curv-val', 'fa-wave-square', 'Curvature', COLORS['yellow'])], md=3),
        dbc.Col([create_metric_card('term-val', 'fa-chart-bar', 'Term Slope', COLORS['green'])], md=3),
    ], className='mb-4 g-3'),
    
    # Main Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span([html.I(className="fas fa-cube me-2"), "IV Surface"]),
                    dbc.RadioItems(id='view-select', inline=True, value='3d',
                                  options=[{'label': '3D', 'value': '3d'},
                                          {'label': 'Heatmap', 'value': 'heatmap'},
                                          {'label': 'Smile', 'value': 'smile'}],
                                  className='float-end',
                                  inputClassName='btn-check',
                                  labelClassName='btn btn-outline-secondary btn-sm',
                                  labelCheckedClassName='btn btn-secondary btn-sm')
                ], style={'backgroundColor': COLORS['card'], 'borderBottom': f'1px solid {COLORS["border"]}'}),
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(id='surface-chart', config={'displayModeBar': True},
                                 style={'height': '450px'})
                    ], color=COLORS['blue'])
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}',
                     'borderRadius': '12px'})
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-heartbeat me-2"), "Regime Gauge"],
                              style={'backgroundColor': COLORS['card']}),
                dbc.CardBody([
                    dcc.Graph(id='gauge-chart', config={'displayModeBar': False},
                             style={'height': '200px'})
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}',
                     'borderRadius': '12px'}, className='mb-3'),
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-bell me-2"), "Alerts & Analysis"],
                              style={'backgroundColor': COLORS['card']}),
                dbc.CardBody(id='alerts-panel', style={'maxHeight': '280px', 'overflowY': 'auto'})
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}',
                     'borderRadius': '12px'})
        ], md=4)
    ], className='mb-4'),
    
    # Term Structure
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-chart-area me-2"), "Volatility Smile by Expiry"],
                              style={'backgroundColor': COLORS['card']}),
                dbc.CardBody([
                    dcc.Graph(id='smile-chart', config={'displayModeBar': False},
                             style={'height': '280px'})
                ])
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}',
                     'borderRadius': '12px'})
        ])
    ])
    
], fluid=True, style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'padding': '20px'})


# ============ CALLBACKS ============

@callback(
    [Output('chain-store', 'data'),
     Output('update-time', 'children')],
    [Input('refresh-btn', 'n_clicks'),
     Input('auto-refresh', 'n_intervals')],
    [State('ticker-select', 'value'),
     State('expiry-range', 'value'),
     State('moneyness-range', 'value')],
    prevent_initial_call=False
)
def fetch_data(n_clicks, n_intervals, ticker, expiry_range, moneyness_range):
    try:
        fetcher = OptionChainFetcher()
        chain = fetcher.fetch_option_chain(ticker)
        
        if chain is None or chain.empty:
            return None, "No data"
        
        config = {
            'min_moneyness': moneyness_range[0],
            'max_moneyness': moneyness_range[1],
            'min_days_to_expiry': expiry_range[0],
            'max_days_to_expiry': expiry_range[1],
            'min_volume': 1,
            'max_spread_pct': 0.50
        }
        clean_chain = clean_option_chain(chain, config)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        return clean_chain.to_dict('records'), f"Updated {timestamp}"
        
    except Exception as e:
        print(f"Fetch error: {e}")
        return None, f"Error: {str(e)[:25]}"


@callback(
    [Output('metrics-store', 'data'),
     Output('atm-vol', 'children'),
     Output('skew-val', 'children'),
     Output('curv-val', 'children'),
     Output('term-val', 'children'),
     Output('regime-badge', 'children'),
     Output('regime-badge', 'style')],
    Input('chain-store', 'data'),
    prevent_initial_call=True
)
def update_metrics(chain_data):
    base_style = {'backgroundColor': COLORS['card'], 'padding': '10px 20px',
                  'borderRadius': '20px', 'display': 'inline-block'}
    
    if not chain_data:
        badge = html.Span("NO DATA", style={'color': COLORS['muted']})
        return None, "--", "--", "--", "--", badge, {**base_style, 'border': f'1px solid {COLORS["muted"]}'}
    
    try:
        df = pd.DataFrame(chain_data)
        metrics = compute_metrics(df)
        
        if not metrics:
            badge = html.Span("NO DATA", style={'color': COLORS['muted']})
            return None, "--", "--", "--", "--", badge, {**base_style, 'border': f'1px solid {COLORS["muted"]}'}
        
        regime, conf, drivers = classify_regime(metrics)
        color = REGIME_COLORS.get(regime, COLORS['muted'])
        
        badge = html.Span([
            html.Span("●", style={'color': color, 'marginRight': '8px', 'fontSize': '1.2rem'}),
            regime.replace('_', ' ')
        ], style={'color': COLORS['text'], 'fontWeight': '600'})
        
        style = {**base_style, 'border': f'2px solid {color}'}
        
        return (
            metrics,
            f"{metrics['atm_vol']*100:.1f}%",
            f"{metrics['skew_25d']*100:.2f}",
            f"{metrics['curvature']*100:.2f}",
            f"{metrics['term_slope']*100:.2f}",
            badge,
            style
        )
        
    except Exception as e:
        print(f"Metrics error: {e}")
        badge = html.Span("ERROR", style={'color': COLORS['red']})
        return None, "ERR", "ERR", "ERR", "ERR", badge, {**base_style, 'border': f'1px solid {COLORS["red"]}'}


@callback(
    Output('surface-chart', 'figure'),
    [Input('chain-store', 'data'),
     Input('view-select', 'value')],
    prevent_initial_call=True
)
def update_surface(chain_data, view_type):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text="Loading data...", x=0.5, y=0.5, 
                         xref="paper", yref="paper", showarrow=False,
                         font=dict(size=16, color=COLORS['muted']))]
    )
    
    if not chain_data:
        return empty_fig
    
    try:
        df = pd.DataFrame(chain_data)
        grid, moneyness, expiries = compute_iv_surface(df)
        
        if grid is None:
            return empty_fig
        
        if view_type == '3d':
            fig = go.Figure(data=[go.Surface(
                x=moneyness, y=expiries, z=grid * 100,
                colorscale='Viridis',
                colorbar=dict(title='IV %', ticksuffix='%')
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title='Moneyness',
                    yaxis_title='Days to Expiry',
                    zaxis_title='IV (%)',
                    bgcolor='rgba(0,0,0,0)'
                )
            )
        elif view_type == 'heatmap':
            fig = go.Figure(data=go.Heatmap(
                x=moneyness, y=expiries, z=grid * 100,
                colorscale='Viridis',
                colorbar=dict(title='IV %')
            ))
            fig.update_layout(xaxis_title='Moneyness', yaxis_title='Days to Expiry')
        else:  # smile
            fig = go.Figure()
            for i, exp in enumerate(expiries[::3]):  # Every 3rd expiry
                if i < len(expiries[::3]):
                    idx = list(expiries).index(exp) if exp in expiries else i*3
                    if idx < len(grid):
                        fig.add_trace(go.Scatter(
                            x=moneyness, y=grid[idx] * 100,
                            mode='lines', name=f'{int(exp)}d'
                        ))
            fig.update_layout(xaxis_title='Moneyness', yaxis_title='IV (%)')
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=30, b=40),
            font=dict(color=COLORS['muted'])
        )
        return fig
        
    except Exception as e:
        print(f"Surface error: {e}")
        return empty_fig


@callback(
    Output('gauge-chart', 'figure'),
    Input('metrics-store', 'data'),
    prevent_initial_call=True
)
def update_gauge(metrics):
    if not metrics:
        value = 0
    else:
        regime, conf, _ = classify_regime(metrics)
        regime_scores = {'CALM': 20, 'PRE_STRESS': 40, 'ELEVATED': 65, 'ACUTE': 90}
        value = regime_scores.get(regime, 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS['muted']),
            bar=dict(color=COLORS['blue']),
            bgcolor=COLORS['card'],
            steps=[
                dict(range=[0, 25], color='rgba(63,185,80,0.3)'),
                dict(range=[25, 50], color='rgba(210,153,34,0.3)'),
                dict(range=[50, 75], color='rgba(219,109,40,0.3)'),
                dict(range=[75, 100], color='rgba(248,81,73,0.3)')
            ]
        )
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=10),
        font=dict(color=COLORS['text'])
    )
    return fig


@callback(
    Output('alerts-panel', 'children'),
    Input('metrics-store', 'data'),
    prevent_initial_call=True
)
def update_alerts(metrics):
    if not metrics:
        return html.Div("No data available", style={'color': COLORS['muted'], 'textAlign': 'center'})
    
    regime, conf, drivers = classify_regime(metrics)
    
    if not drivers:
        return html.Div([
            html.I(className="fas fa-check-circle", style={'color': COLORS['green'], 'marginRight': '10px', 'marginTop': '3px'}),
            html.Div([
                html.Strong("All Clear", style={'display': 'block'}),
                html.Small("Volatility surface metrics are within normal ranges. No unusual stress signals detected.", 
                          style={'color': COLORS['muted']})
            ])
        ], style={'color': COLORS['text'], 'display': 'flex', 'alignItems': 'flex-start'})
    
    # Severity colors
    severity_colors = {'high': COLORS['red'], 'medium': COLORS['yellow'], 'low': COLORS['blue']}
    severity_icons = {'high': 'fa-exclamation-circle', 'medium': 'fa-exclamation-triangle', 'low': 'fa-info-circle'}
    
    alerts = []
    for driver in drivers:
        if isinstance(driver, dict):
            severity = driver.get('severity', 'medium')
            alerts.append(html.Div([
                html.Div([
                    html.I(className=f"fas {severity_icons[severity]}", 
                          style={'color': severity_colors[severity], 'marginRight': '10px', 'marginTop': '3px'}),
                    html.Div([
                        html.Strong(driver['title'], style={'display': 'block', 'marginBottom': '5px'}),
                        html.Small(driver['explanation'], style={'color': COLORS['muted'], 'lineHeight': '1.4'})
                    ])
                ], style={'display': 'flex', 'alignItems': 'flex-start'})
            ], style={'color': COLORS['text'], 'padding': '10px 0', 
                      'borderBottom': f'1px solid {COLORS["border"]}', 'marginBottom': '5px'}))
        else:
            # Fallback for old string format
            alerts.append(html.Div([
                html.I(className="fas fa-exclamation-triangle", 
                      style={'color': COLORS['yellow'], 'marginRight': '10px'}),
                driver
            ], style={'color': COLORS['text'], 'padding': '5px 0'}))
    
    return alerts


@callback(
    Output('smile-chart', 'figure'),
    Input('chain-store', 'data'),
    prevent_initial_call=True
)
def update_smile(chain_data):
    fig = go.Figure()
    
    if chain_data:
        try:
            df = pd.DataFrame(chain_data)
            
            # Tuples: (min_days, max_days, label, color)
            expiry_ranges = [
                (7, 30, '~21d', COLORS['blue']),
                (30, 60, '~45d', COLORS['purple']),
                (60, 120, '~90d', COLORS['green'])
            ]
            
            for min_exp, max_exp, name, color in expiry_ranges:
                subset = df[(df['days_to_expiry'] >= min_exp) & 
                           (df['days_to_expiry'] <= max_exp)]
                
                if len(subset) > 5:
                    grouped = subset.groupby(pd.cut(subset['moneyness'], bins=20), observed=False)['implied_volatility_market'].mean()
                    x_vals = [interval.mid for interval in grouped.index]
                    y_vals = grouped.values * 100
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode='lines+markers', name=name,
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ))
        except Exception as e:
            print(f"Smile error: {e}")
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Moneyness',
        yaxis_title='IV (%)',
        margin=dict(l=40, r=40, t=20, b=40),
        font=dict(color=COLORS['muted']),
        legend=dict(orientation='h', y=1.1)
    )
    return fig


if __name__ == '__main__':
    print("Starting Volatility Surface Monitor...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, port=8050)
git add .
git commit -m "Add deployment files"
git push

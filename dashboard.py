"""
Volatility Surface Mathematical Analysis Dashboard

A mathematical exploration of market volatility patterns and surface dynamics.
Focuses on quantitative measures of market stress and complexity.
"""

# ============ IMPORTS ============
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
import plotly.graph_objects as go

# Project imports
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from data.fetcher import OptionChainFetcher, FREDDataFetcher
from data.cleaner import clean_option_chain
from analytics.regime_classifier import GMMRegimeClassifier

# ============ CONFIGURATION ============
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    title="Mathematical Volatility Analysis",
    suppress_callback_exceptions=True
)

server = app.server

# Color scheme optimized for mathematical visualization
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
    'cyan': '#79c0ff',
    'pink': '#f85149'
}

REGIME_COLORS = {
    'CALM': COLORS['green'],
    'PRE_STRESS': COLORS['yellow'],
    'ELEVATED': COLORS['orange'],
    'ACUTE': COLORS['red'],
    'RECOVERY': COLORS['blue'],
    'UNPRECEDENTED': COLORS['purple'],
    'UNKNOWN': COLORS['muted'],
}

# Initialize GMM regime classifier
# Try to load saved model, otherwise use default
try:
    import pickle
    model_path = project_root / "src" / "data" / "gmm_regime_classifier.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            gmm_classifier = pickle.load(f)
        print(f"Loaded trained GMM model from {model_path}")
    else:
        gmm_classifier = GMMRegimeClassifier(
            n_components=5, 
            random_state=42,
            feature_cols=['avg_atm_vol', 'avg_25d_skew', 'roughness']
        )
        print("Using default GMM classifier")
except Exception as e:
    print(f"Error loading saved model: {e}")
    gmm_classifier = GMMRegimeClassifier(
        n_components=5, 
        random_state=42,
        feature_cols=['avg_atm_vol', 'avg_25d_skew', 'roughness']
    )


# ============ MATHEMATICAL ANALYSIS FUNCTIONS ============

def load_historical_data():
    """Load historical market data for statistical analysis."""
    try:
        # Use synthetic data for reliability
        return load_synthetic_historical_data()
    except Exception as e:
        logger.warning(f"Failed to load historical data: {e}. Using synthetic data.")
        return load_synthetic_historical_data()


def load_synthetic_historical_data():
    """Load synthetic historical data representing different market regimes."""
    # Generate synthetic historical data representing different market regimes
    # This simulates what would come from a historical database
    
    np.random.seed(42)  # For reproducible synthetic data
    n_days = 8000  # 8000 trading days (~32 years) of history
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    
    # Generate synthetic metrics that cluster into regimes
    historical_data = []
    
    for i in range(n_days):
        # Create regime-like clusters in the data
        regime_type = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.2, 0.2, 0.1, 0.1])  # More calm periods
        
        if regime_type == 0:  # CALM
            atm_vol = np.random.normal(0.12, 0.015)
            curvature = np.random.normal(0.03, 0.01)
            butterfly = np.random.normal(0.015, 0.005)
            skew = np.random.normal(-0.03, 0.01)
            roughness = np.random.normal(0.04, 0.01)
        elif regime_type == 1:  # PRE_STRESS
            atm_vol = np.random.normal(0.15, 0.02)
            curvature = np.random.normal(0.05, 0.015)
            butterfly = np.random.normal(0.025, 0.008)
            skew = np.random.normal(-0.06, 0.02)
            roughness = np.random.normal(0.06, 0.015)
        elif regime_type == 2:  # ELEVATED
            atm_vol = np.random.normal(0.20, 0.03)
            curvature = np.random.normal(0.08, 0.02)
            butterfly = np.random.normal(0.04, 0.01)
            skew = np.random.normal(-0.10, 0.03)
            roughness = np.random.normal(0.08, 0.02)
        elif regime_type == 3:  # ACUTE
            atm_vol = np.random.normal(0.30, 0.05)
            curvature = np.random.normal(0.12, 0.03)
            butterfly = np.random.normal(0.07, 0.02)
            skew = np.random.normal(-0.15, 0.04)
            roughness = np.random.normal(0.12, 0.03)
        else:  # RECOVERY
            atm_vol = np.random.normal(0.18, 0.025)
            curvature = np.random.normal(0.04, 0.015)
            butterfly = np.random.normal(0.02, 0.006)
            skew = np.random.normal(-0.04, 0.015)
            roughness = np.random.normal(0.05, 0.012)
        
        # Ensure non-negative values
        atm_vol = max(atm_vol, 0.05)
        curvature = max(curvature, 0)
        butterfly = max(butterfly, 0)
        roughness = max(roughness, 0)
        
        # Generate synthetic FRED economic indicators correlated with regimes
        if regime_type == 0:  # CALM
            unemployment_rate = np.random.normal(3.5, 0.3)
            fed_funds_rate = np.random.normal(2.5, 0.2)
            treasury_10y = np.random.normal(3.5, 0.2)
            yield_curve_spread = np.random.normal(1.0, 0.1)
            cpi_inflation = np.random.normal(2.5, 0.2)
        elif regime_type == 1:  # PRE_STRESS
            unemployment_rate = np.random.normal(4.5, 0.4)
            fed_funds_rate = np.random.normal(2.5, 0.2)
            treasury_10y = np.random.normal(3.5, 0.2)
            yield_curve_spread = np.random.normal(0.8, 0.1)
            cpi_inflation = np.random.normal(2.5, 0.2)
        elif regime_type == 2:  # ELEVATED
            unemployment_rate = np.random.normal(5.5, 0.5)
            fed_funds_rate = np.random.normal(2.0, 0.3)
            treasury_10y = np.random.normal(3.0, 0.3)
            yield_curve_spread = np.random.normal(0.5, 0.15)
            cpi_inflation = np.random.normal(2.8, 0.3)
        elif regime_type == 3:  # ACUTE
            unemployment_rate = np.random.normal(8.0, 1.0)
            fed_funds_rate = np.random.normal(0.5, 0.2)
            treasury_10y = np.random.normal(2.0, 0.3)
            yield_curve_spread = np.random.normal(-0.5, 0.2)
            cpi_inflation = np.random.normal(2.0, 0.4)
        else:  # RECOVERY
            unemployment_rate = np.random.normal(4.0, 0.4)
            fed_funds_rate = np.random.normal(3.0, 0.3)
            treasury_10y = np.random.normal(4.0, 0.3)
            yield_curve_spread = np.random.normal(1.5, 0.2)
            cpi_inflation = np.random.normal(3.0, 0.3)
        
        historical_data.append({
            'date': dates[i],
            'avg_atm_vol': atm_vol,
            'avg_atm_curvature': curvature,
            'avg_butterfly': butterfly,
            'avg_25d_skew': skew,
            'roughness': roughness,
            'unemployment_rate': unemployment_rate,
            'fed_funds_rate': fed_funds_rate,
            'treasury_10y': treasury_10y,
            'yield_curve_spread': yield_curve_spread,
            'cpi_inflation': cpi_inflation,
            'spy_return': np.random.normal(0.0005, 0.015)  # Synthetic daily return
        })
    
    df = pd.DataFrame(historical_data).set_index('date')
    return df


def compute_iv_surface(df):
    """
    Construct volatility surface using spatial interpolation.

    Mathematical process:
    1. Filter valid option data
    2. Create moneyness × time grid
    3. Local averaging within bins
    4. Spatial interpolation of missing values
    """
    if 'implied_volatility_market' not in df.columns:
        return None, None, None

    # Data validation and filtering
    valid = df[
        (df['implied_volatility_market'] > 0.01) &
        (df['implied_volatility_market'] < 2.0) &
        (df['moneyness'] > 0.8) &
        (df['moneyness'] < 1.2)
    ].copy()

    if len(valid) < 10:
        return None, None, None

    # Grid discretization
    moneyness_bins = np.linspace(0.85, 1.15, 25)
    expiry_bins = np.linspace(valid['days_to_expiry'].min(),
                               min(valid['days_to_expiry'].max(), 90), 15)

    # Initialize grid
    grid = np.full((len(expiry_bins), len(moneyness_bins)), np.nan)

    # Local averaging (binning)
    for i, exp in enumerate(expiry_bins):
        for j, mon in enumerate(moneyness_bins):
            mask = (
                (abs(valid['moneyness'] - mon) < 0.03) &
                (abs(valid['days_to_expiry'] - exp) < 7)
            )
            subset = valid[mask]
            if len(subset) > 0:
                grid[i, j] = subset['implied_volatility_market'].mean()

    # Spatial interpolation for missing values
    from scipy.ndimage import generic_filter

    def nanmean_filter(values):
        """3x3 kernel average, ignoring NaN values."""
        valid_vals = values[~np.isnan(values)]
        return np.mean(valid_vals) if len(valid_vals) > 0 else np.nan

    # Iterative smoothing (3 passes for convergence)
    for _ in range(3):
        mask = np.isnan(grid)
        if not mask.any():
            break
        grid = np.where(mask,
                       generic_filter(grid, nanmean_filter, size=3, mode='nearest'),
                       grid)

    return grid, moneyness_bins, expiry_bins


def compute_metrics(df, historical_data=None):
    """
    Compute comprehensive mathematical metrics of volatility surface.

    Returns dictionary with:
    - Basic volatility measures (ATM, skew, curvature, term structure)
    - Statistical measures (z-scores, percentiles, entropy, roughness)
    - Comparative measures (realized vs implied, correlations)
    """
    try:
        if df is None or df.empty or 'implied_volatility_market' not in df.columns:
            return {}

        # Data validation and filtering
        valid = df[
            (df['implied_volatility_market'] > 0.01) &
            (df['implied_volatility_market'] < 2.0) &
            (df['implied_volatility_market'].notna())
        ].copy()

        if len(valid) < 10:
            return {}

        # ============ BASIC VOLATILITY MEASURES ============

        # ATM Volatility (geometric center of surface)
        atm_mask = (valid['moneyness'] > 0.97) & (valid['moneyness'] < 1.03)
        short_term = valid['days_to_expiry'] < 45
        atm_options = valid[atm_mask & short_term]
        atm_vol = atm_options['implied_volatility_market'].mean() if len(atm_options) > 0 else 0.15

        # Ensure all metrics are finite and reasonable
        atm_vol = max(0.01, min(2.0, atm_vol))  # 1% to 200% range

        # 25Δ Skew (asymmetry measure)
        put_wing = valid[(valid['moneyness'] > 0.90) & (valid['moneyness'] < 0.95)]
        call_wing = valid[(valid['moneyness'] > 1.05) & (valid['moneyness'] < 1.10)]
        put_vol = put_wing['implied_volatility_market'].mean() if len(put_wing) > 0 else atm_vol
        call_vol = call_wing['implied_volatility_market'].mean() if len(call_wing) > 0 else atm_vol
        
        # Ensure valid values
        put_vol = put_vol if np.isfinite(put_vol) and put_vol > 0 else atm_vol
        call_vol = call_vol if np.isfinite(call_vol) and call_vol > 0 else atm_vol
        
        skew_25d = put_vol - call_vol
        skew_25d = max(-0.5, min(0.5, skew_25d))  # -50% to 50% range

        # Curvature (second derivative approximation)
        wing_avg = (put_vol + call_vol) / 2
        curvature = wing_avg - atm_vol
        curvature = max(-0.2, min(0.2, curvature))  # -20% to 20% range

        # Term Structure (volatility term premium)
        short_atm = valid[atm_mask & (valid['days_to_expiry'] < 30)]['implied_volatility_market'].mean()
        long_atm = valid[atm_mask & (valid['days_to_expiry'] > 60)]['implied_volatility_market'].mean()
        
        # Ensure both values are valid before computing difference
        if (np.isfinite(short_atm) and np.isfinite(long_atm) and 
            not np.isnan(short_atm) and not np.isnan(long_atm)):
            term_slope = long_atm - short_atm
        else:
            term_slope = 0.0
        term_slope = max(-0.1, min(0.1, term_slope))  # -10% to 10% range

        # ============ INFORMATION-THEORETIC MEASURES ============

        vol_values = valid['implied_volatility_market'].values
        vol_values = vol_values[np.isfinite(vol_values)]  # Remove NaN/inf
        
        if len(vol_values) == 0:
            return {}

        # Surface Entropy (complexity/diversity measure)
        vol_norm = vol_values / vol_values.sum()
        entropy = -np.sum(vol_norm * np.log(vol_norm + 1e-10)) if len(vol_norm) > 0 else 0
        
        # Normalized entropy (0-1 scale, where 1 is maximum disorder)
        max_entropy = np.log(len(vol_norm)) if len(vol_norm) > 0 else 0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Surface Roughness (variability measure)
        roughness = np.std(vol_values) if len(vol_values) > 1 else 0

        # Ensure finite values
        entropy = entropy if np.isfinite(entropy) else 0
        normalized_entropy = normalized_entropy if np.isfinite(normalized_entropy) else 0
        roughness = roughness if np.isfinite(roughness) else 0

        # ============ STATISTICAL COMPARISONS ============

        z_scores = {}
        percentiles = {}

        if historical_data is not None:
            from analytics.regime_classifier import RegimeClassifier
            classifier = RegimeClassifier()
            classifier.set_history(historical_data)

            # Z-scores relative to historical distribution
            hist_atm = historical_data['avg_atm_vol']
            hist_curv = historical_data['avg_atm_curvature'] 
            hist_skew = historical_data['avg_25d_skew']
            
            z_scores['atm_vol'] = (atm_vol - hist_atm.mean()) / hist_atm.std() if hist_atm.std() > 0 else 0
            z_scores['curvature'] = (curvature - hist_curv.mean()) / hist_curv.std() if hist_curv.std() > 0 else 0
            z_scores['skew'] = (abs(skew_25d) - hist_skew.mean()) / hist_skew.std() if hist_skew.std() > 0 else 0

            # Percentile rankings
            percentiles['atm_vol'] = classifier._compute_percentile('avg_atm_vol', atm_vol)
            percentiles['curvature'] = classifier._compute_percentile('avg_atm_curvature', curvature)
            percentiles['skew'] = classifier._compute_percentile('avg_25d_skew', abs(skew_25d))

        # ============ MARKET COMPARISONS ============

        # Realized vs Implied Volatility Ratio
        realized_vol = None
        if 'realized_volatility' in df.columns:
            realized_vol = df['realized_volatility'].mean()

        # Rolling Correlation (volatility persistence)
        vol_returns_corr = None
        if 'returns' in df.columns and len(df) > 30:
            vol_returns_corr = df['implied_volatility_market'].rolling(30).corr(df['returns']).iloc[-1]

        return {
            # Basic measures
            'atm_vol': float(atm_vol),
            'skew_25d': float(skew_25d),
            'curvature': float(curvature),
            'term_slope': float(term_slope),
            'put_vol': float(put_vol),
            'call_vol': float(call_vol),

            # Information theory
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'roughness': float(roughness),

            # Statistical measures
            'z_scores': z_scores,
            'percentiles': percentiles,

            # Market comparisons
            'realized_vol': realized_vol,
            'vol_returns_corr': vol_returns_corr,
        }
        
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {}


def classify_regime(metrics):
    """
    Classify market regime using GMM-based distribution modeling.
    
    Uses Gaussian Mixture Model to identify which historical distribution
    cluster the current market state belongs to.
    """
    try:
        if not metrics or not isinstance(metrics, dict):
            return 'UNKNOWN', 0.0, []
        
        # Convert metrics to the format expected by GMM classifier
        gmm_metrics = {
            'summary': {
                'avg_atm_vol': metrics.get('atm_vol', 0.15),
                'roughness': metrics.get('roughness', 0)
            }
        }
        
        # Get GMM prediction
        prediction = gmm_classifier.predict(gmm_metrics)
        
        regime = prediction['regime'].upper()
        confidence = prediction['confidence']
        
        # Create mathematical drivers based on GMM results
        drivers = []
        
        # Distribution fit driver
        fit_score = prediction['distribution_fit']
        fit_quality = prediction.get('fit_quality', 'unknown')
        
        if fit_quality == "extreme_outlier":
            drivers.append({
                'title': f"Extreme Distribution Outlier: {fit_score:.3f}",
                'explanation': f"Current market state shows no meaningful similarity to any historical regime cluster. This represents truly unprecedented market conditions that fall outside all known distribution patterns.",
                'severity': 'high',
                'math': f"Relative likelihood = {fit_score:.3f} (no historical precedent)"
            })
        elif fit_quality == "poor_fit":
            drivers.append({
                'title': f"Poor Distribution Fit: {fit_score:.3f}",
                'explanation': f"Current market state shows poor fit with historical distributions. This may indicate an unusual or emerging regime.",
                'severity': 'medium',
                'math': f"Relative likelihood = {fit_score:.3f} (< 0.5 threshold)"
            })
        elif fit_quality == "strong_fit":
            drivers.append({
                'title': f"Strong Distribution Fit: {fit_score:.3f}",
                'explanation': f"Current market state fits extremely well with historical patterns in this regime cluster.",
                'severity': 'low',
                'math': f"Relative likelihood = {fit_score:.3f} (> 1.5)"
            })
        
        # Regime probabilities
        probabilities = prediction['probabilities']
        top_regimes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for reg, prob in top_regimes:
            if prob > 0.1:  # Only show significant probabilities
                reg_upper = reg.upper()
                if reg_upper != regime:
                    drivers.append({
                        'title': f"Alternative Regime: {reg_upper} ({prob:.1f})",
                        'explanation': f"Market also shows {prob:.1f} probability of being in {reg_upper} regime.",
                        'severity': 'low',
                        'math': f"P({reg_upper}) = {prob:.2f}"
                    })
        
        # Key metrics driver
        atm_vol = metrics.get('atm_vol', 0.15)
        curvature = metrics.get('curvature', 0)
        roughness = metrics.get('roughness', 0)
        
        drivers.append({
            'title': f"Surface Characteristics: σ={atm_vol:.2f}, κ={curvature:.4f}, ρ={roughness:.4f}",
            'explanation': f"ATM volatility, curvature, and roughness define the current surface position in distribution space.",
            'severity': 'low',
            'math': f"Surface vector: ({atm_vol:.2f}, {curvature:.4f}, {roughness:.4f})"
        })
        
        return regime, confidence, drivers
        
    except Exception as e:
        print(f"Error in classify_regime: {e}")
        return 'UNKNOWN', 0.0, []


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
    
    # Advanced Controls
    dbc.Row([
        dbc.Col([
            html.Label("GMM Features", style={'color': COLORS['muted'], 'fontSize': '0.85rem'}),
            dcc.Dropdown(
                id='feature-select',
                options=[
                    {'label': 'ATM Volatility', 'value': 'avg_atm_vol'},
                    {'label': 'Roughness', 'value': 'roughness'}
                ],
                value=['avg_atm_vol', 'roughness'],
                multi=True,
                style={'backgroundColor': COLORS['card'], 'color': COLORS['text'],
                       'border': f'1px solid {COLORS["border"]}'}
            )
        ], md=12)
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
    ]),
    
    # Mathematical Insights
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-brain me-2"), "Mathematical Pattern Analysis"],
                              style={'backgroundColor': COLORS['card']}),
                dbc.CardBody(id='math-insights-panel', style={'maxHeight': '300px', 'overflowY': 'auto'})
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}',
                     'borderRadius': '12px'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([html.I(className="fas fa-chart-line me-2"), "Surface Statistics"],
                              style={'backgroundColor': COLORS['card']}),
                dbc.CardBody(id='stats-panel')
            ], style={'backgroundColor': COLORS['card'], 'border': f'1px solid {COLORS["border"]}',
                     'borderRadius': '12px'})
        ], md=6)
    ], className='mb-4')
    
], fluid=True, style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'padding': '20px'})


# ============ CALLBACKS ============

@app.callback(
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


@app.callback(
    [Output('metrics-store', 'data'),
     Output('atm-vol', 'children'),
     Output('skew-val', 'children'),
     Output('curv-val', 'children'),
     Output('term-val', 'children'),
     Output('regime-badge', 'children'),
     Output('regime-badge', 'style')],
    [Input('chain-store', 'data'),
     Input('feature-select', 'value')],
    prevent_initial_call=False
)
def update_metrics(chain_data, selected_features):
    base_style = {'backgroundColor': COLORS['card'], 'padding': '10px 20px',
                  'borderRadius': '20px', 'display': 'inline-block'}
    
    # Update GMM features and refit if needed
    if selected_features and len(selected_features) > 0 and selected_features != gmm_classifier.feature_cols:
        gmm_classifier.feature_cols = selected_features
        gmm_classifier.is_fitted = False
        historical_data = load_historical_data()
        gmm_classifier.fit(historical_data)
    
    if not chain_data:
        badge = html.Span("NO DATA", style={'color': COLORS['muted']})
        return None, "--", "--", "--", "--", badge, {**base_style, 'border': f'1px solid {COLORS["muted"]}'}
    
    try:
        df = pd.DataFrame(chain_data)
        historical_data = load_historical_data()
        metrics = compute_metrics(df, historical_data=historical_data)
        
        # Validate metrics
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Invalid metrics returned from compute_metrics")
        
        # Get recent SPY return
        try:
            spy = yf.Ticker("SPY")
            recent_data = spy.history(period="2d")
            if len(recent_data) >= 2:
                spy_return = recent_data['Close'].iloc[-1] / recent_data['Open'].iloc[-1] - 1
            else:
                spy_return = 0.0
        except Exception as e:
            spy_return = 0.0
        
        metrics['spy_return'] = spy_return
        
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
            f"{max(0, min(100, metrics['atm_vol']*100)):.1f}%",
            f"{max(-50, min(50, metrics['skew_25d']*100)):.2f}",
            f"{max(-20, min(20, metrics['curvature']*100)):.2f}",
            f"{max(-10, min(10, metrics['term_slope']*100)):.2f}",
            badge,
            style
        )
        
    except Exception as e:
        print(f"Metrics error: {e}")
        badge = html.Span("ERROR", style={'color': COLORS['red']})
        return None, "ERR", "ERR", "ERR", "ERR", badge, {**base_style, 'border': f'1px solid {COLORS["red"]}'}


@app.callback(
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


@app.callback(
    Output('gauge-chart', 'figure'),
    Input('metrics-store', 'data'),
    prevent_initial_call=True
)
def update_gauge(metrics):
    if not metrics:
        value = 0
        regime = 'UNKNOWN'
    else:
        # Use GMM classifier for regime determination
        gmm_metrics = {
            'summary': {
                'avg_atm_vol': metrics.get('atm_vol', 0.15),
                'avg_atm_curvature': metrics.get('curvature', 0),
                'avg_butterfly': metrics.get('roughness', 0) * 10,
                'avg_25d_skew': abs(metrics.get('skew_25d', 0)),
                'roughness': metrics.get('roughness', 0)
            }
        }
        prediction = gmm_classifier.predict(gmm_metrics)
        regime = prediction['regime'].upper()
        
        # Map regimes to gauge values
        regime_scores = {
            'CALM': 20, 
            'PRE_STRESS': 40, 
            'ELEVATED': 65, 
            'ACUTE': 90,
            'RECOVERY': 30,
            'UNPRECEDENTED': 100,  # Extreme value for unprecedented
            'UNKNOWN': 0
        }
        value = regime_scores.get(regime, 50)
    
    # Choose color based on regime
    gauge_color = REGIME_COLORS.get(regime, COLORS['muted'])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS['muted']),
            bar=dict(color=gauge_color),
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


@app.callback(
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
                html.Small("All volatility surface metrics are within normal mathematical ranges. No stress signals detected.", 
                          style={'color': COLORS['muted']})
            ])
        ], style={'color': COLORS['text'], 'display': 'flex', 'alignItems': 'flex-start'})

    # Only show detailed, math-based regime explanations and their impact
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
                        html.Small(driver['explanation'], style={'color': COLORS['muted'], 'lineHeight': '1.4', 'display': 'block', 'marginBottom': '3px'})
                    ])
                ], style={'display': 'flex', 'alignItems': 'flex-start'})
            ], style={'color': COLORS['text'], 'padding': '10px 0', 
                      'borderBottom': f'1px solid {COLORS["border"]}', 'marginBottom': '5px'}))
        # Ignore any non-dict (string) drivers to enforce only math-based explanations

    return alerts


@app.callback(
    Output('math-insights-panel', 'children'),
    Input('metrics-store', 'data'),
    prevent_initial_call=True
)
def update_math_insights(metrics):
    """Display advanced mathematical pattern analysis."""
    if not metrics:
        return html.Div("No data for mathematical analysis", style={'color': COLORS['muted'], 'textAlign': 'center'})
    
    insights = []
    
    # GMM Distribution Analysis
    if gmm_classifier.is_fitted:
        gmm_metrics = {
            'summary': {
                'avg_atm_vol': metrics.get('atm_vol', 0.15),
                'avg_atm_curvature': metrics.get('curvature', 0),
                'avg_butterfly': metrics.get('roughness', 0) * 10,
                'avg_25d_skew': abs(metrics.get('skew_25d', 0)),
                'roughness': metrics.get('roughness', 0)
            }
        }
        
        prediction = gmm_classifier.predict(gmm_metrics)
        evolution = gmm_classifier.analyze_distribution_evolution(load_historical_data())
        
        # Interpret fit score
        fit_score = prediction['distribution_fit']
        fit_quality = prediction.get('fit_quality', 'unknown')
        
        if fit_quality == "extreme_outlier":
            fit_desc = "Extreme outlier - unprecedented market state"
            fit_color = COLORS['red']
        elif fit_quality == "poor_fit":
            fit_desc = "Poor fit - unusual market state"
            fit_color = COLORS['orange']
        elif fit_quality == "moderate_fit":
            fit_desc = "Moderate fit - typical market state"
            fit_color = COLORS['yellow']
        else:
            fit_desc = "Strong fit - common market state"
            fit_color = COLORS['green']
        
        insights.append(html.Div([
            html.H6("Distribution Modeling", style={'color': COLORS['purple'], 'marginBottom': '5px'}),
            html.P(f"Regime: {prediction['regime'].upper()} {'(P = N/A)' if prediction['regime'].lower() == 'unprecedented' else f'(P = {prediction['confidence']:.2f})'}", 
                  style={'color': COLORS['text'], 'marginBottom': '3px'}),
            html.P(f"Fit Quality: {fit_score:.2f} - {fit_desc}", 
                  style={'color': fit_color, 'marginBottom': '3px'}),
            html.Small("GMM clusters market states into 5 distribution regimes", 
                      style={'color': COLORS['muted']})
        ], style={'marginBottom': '15px'}))
        
        # Show regime distribution over time
        if evolution:
            regime_dist = evolution.get('regime_distribution', {})
            total_periods = evolution.get('total_periods', 0)
            if regime_dist:
                insights.append(html.Div([
                    html.H6("Historical Distribution", style={'color': COLORS['blue'], 'marginBottom': '5px'}),
                    *[html.P(f"{regime.upper()}: {count/total_periods*100:.1f}%", 
                            style={'color': COLORS['text'], 'marginBottom': '2px'}) 
                      for regime, count in sorted(regime_dist.items(), key=lambda x: x[1], reverse=True)],
                    html.Small("Percentage of time spent in each regime cluster (including unprecedented)", 
                              style={'color': COLORS['muted']})
                ], style={'marginBottom': '15px'}))
    
    # Surface Complexity Analysis
    entropy = metrics.get('entropy', 0)
    normalized_entropy = metrics.get('normalized_entropy', 0)
    roughness = metrics.get('roughness', 0)
    
    if entropy > 0:
        complexity_level = "High" if normalized_entropy > 0.8 else "Moderate" if normalized_entropy > 0.6 else "Low"
        insights.append(html.Div([
            html.H6("Surface Complexity", style={'color': COLORS['cyan'], 'marginBottom': '5px'}),
            html.P(f"Entropy: {entropy:.3f} (Normalized: {normalized_entropy:.2f}) ({complexity_level} complexity)", 
                  style={'color': COLORS['text'], 'marginBottom': '3px'}),
            html.Small("H(X) = -Σ pᵢ log(pᵢ) measures information content", 
                      style={'color': COLORS['muted']})
        ], style={'marginBottom': '15px'}))
    
    # Statistical Significance
    z_scores = metrics.get('z_scores', {})
    if z_scores:
        insights.append(html.Div([
            html.H6("Statistical Significance", style={'color': COLORS['yellow'], 'marginBottom': '5px'}),
            *[html.P(f"{k.replace('_', ' ').title()}: z = {v:.2f}", 
                    style={'color': COLORS['text'], 'marginBottom': '2px'}) for k, v in z_scores.items()],
            html.Small("z > 2 indicates statistical significance (95% confidence)", 
                      style={'color': COLORS['muted']})
        ], style={'marginBottom': '15px'}))
    
    return insights


@app.callback(
    Output('stats-panel', 'children'),
    Input('chain-store', 'data'),
    prevent_initial_call=True
)
def update_stats_panel(chain_data):
    """Display comprehensive surface statistics."""
    if not chain_data:
        return html.Div("No data for statistical analysis", style={'color': COLORS['muted'], 'textAlign': 'center'})
    
    try:
        df = pd.DataFrame(chain_data)
        metrics = compute_metrics(df)
        
        stats = []
        
        # Basic Statistics
        vol_values = df['implied_volatility_market'].values
        stats.append(html.Div([
            html.H6("Volatility Distribution", style={'color': COLORS['blue'], 'marginBottom': '8px'}),
            html.Div([
                html.Span(f"Mean: {vol_values.mean():.1%}", style={'marginRight': '15px'}),
                html.Span(f"Std: {vol_values.std():.1%}", style={'marginRight': '15px'}),
                html.Span(f"Min: {vol_values.min():.1%}", style={'marginRight': '15px'}),
                html.Span(f"Max: {vol_values.max():.1%}")
            ], style={'fontSize': '0.9em', 'color': COLORS['text']})
        ], style={'marginBottom': '15px'}))
        
        # Surface Metrics
        if metrics:
            stats.append(html.Div([
                html.H6("Surface Characteristics", style={'color': COLORS['purple'], 'marginBottom': '8px'}),
                html.Div([
                    html.Span(f"ATM Vol: {metrics.get('atm_vol', 0):.1%}", style={'marginRight': '15px'}),
                    html.Span(f"Skew: {metrics.get('skew_25d', 0):.2%}", style={'marginRight': '15px'}),
                    html.Span(f"Curvature: {metrics.get('curvature', 0):.2%}")
                ], style={'fontSize': '0.9em', 'color': COLORS['text'], 'marginBottom': '5px'}),
                html.Div([
                    html.Span(f"Entropy: {metrics.get('entropy', 0):.3f}", style={'marginRight': '15px'}),
                    html.Span(f"Roughness: {metrics.get('roughness', 0):.1%}")
                ], style={'fontSize': '0.9em', 'color': COLORS['text']})
            ], style={'marginBottom': '15px'}))
        
        # Data Quality
        stats.append(html.Div([
            html.H6("Data Quality", style={'color': COLORS['green'], 'marginBottom': '8px'}),
            html.Div([
                html.Span(f"Total: {len(df)}", style={'marginRight': '10px'}),
                html.Span(f"Valid: {len(df[df['implied_volatility_market'] > 0])}", style={'marginRight': '10px'}),
                html.Span(f"Moneyness: {df['moneyness'].min():.2f}-{df['moneyness'].max():.2f}")
            ], style={'fontSize': '0.85em', 'color': COLORS['text'], 'lineHeight': '1.4'})
        ]))
        
        return stats
        
    except Exception as e:
        return html.Div(f"Statistics error: {str(e)}", style={'color': COLORS['red']})


@app.callback(
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
    try:
        print("Starting Volatility Surface Monitor Dashboard...")
        print(f"Working directory: {Path.cwd()}")
        print(f"Python path includes: {Path(__file__).parent}")
        
        # Fit GMM classifier on historical data
        historical_data = load_historical_data()
        if historical_data is not None:
            gmm_classifier.fit(historical_data)
        
        print("Dashboard initialized successfully!")
        print("Open your browser to http://127.0.0.1:8050 to view the dashboard")
        print("Press Ctrl+C to stop the server")
        
        # Run the app
        app.run(debug=True, host='0.0.0.0', port=8050)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()


"""
Volatility Surface Mathematical Analysis Dashboard - Streamlit Version

A mathematical exploration of market volatility patterns and surface dynamics.
Focuses on quantitative measures of market stress and complexity.
"""

# ============ IMPORTS ============
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from data.fetcher import OptionChainFetcher, FREDDataFetcher
from data.cleaner import clean_option_chain
from analytics.regime_classifier import GMMRegimeClassifier

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="Volatility Surface Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ STYLING ============
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
    }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 15px;
    }
    .stMetric label {
        color: #8b949e !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #f0f6fc !important;
    }
    div[data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
    }
    .regime-badge {
        background-color: #161b22;
        padding: 10px 20px;
        border-radius: 20px;
        display: inline-block;
        border: 2px solid;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme
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

# ============ INITIALIZE SESSION STATE ============
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chain_data = None
    st.session_state.metrics = None
    st.session_state.last_update = None

# Initialize GMM regime classifier
@st.cache_resource
def load_gmm_classifier():
    """Load or create GMM classifier."""
    try:
        import pickle
        model_path = project_root / "src" / "data" / "gmm_regime_classifier.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            print(f"Loaded trained GMM model from {model_path}")
            return classifier
        else:
            classifier = GMMRegimeClassifier(
                n_components=5, 
                random_state=42,
                feature_cols=['avg_atm_vol', 'avg_25d_skew', 'roughness']
            )
            print("Using default GMM classifier")
            return classifier
    except Exception as e:
        print(f"Error loading saved model: {e}")
        return GMMRegimeClassifier(
            n_components=5, 
            random_state=42,
            feature_cols=['avg_atm_vol', 'avg_25d_skew', 'roughness']
        )

gmm_classifier = load_gmm_classifier()


# ============ DATA FUNCTIONS ============

def load_synthetic_historical_data():
    """Load synthetic historical data representing different market regimes."""
    np.random.seed(42)
    n_days = 8000  # 8000 trading days (~32 years) of history
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    
    historical_data = []
    
    for i in range(n_days):
        regime_type = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.2, 0.2, 0.1, 0.1])
        
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
            butterfly = np.random.normal(0.02, 0.008)
            skew = np.random.normal(-0.05, 0.015)
            roughness = np.random.normal(0.05, 0.015)
        
        unemployment_rate = np.random.normal(5.5, 1.5)
        fed_funds_rate = np.random.normal(2.5, 1.2)
        treasury_10y = np.random.normal(3.0, 1.0)
        yield_curve_spread = treasury_10y - fed_funds_rate
        cpi_inflation = np.random.normal(2.5, 1.0)
        
        historical_data.append({
            'date': dates[i],
            'avg_atm_vol': max(0.01, atm_vol),
            'avg_curvature': curvature,
            'avg_butterfly': butterfly,
            'avg_25d_skew': skew,
            'roughness': max(0.01, roughness),
            'unemployment_rate': unemployment_rate,
            'fed_funds_rate': fed_funds_rate,
            'treasury_10y': treasury_10y,
            'yield_curve_spread': yield_curve_spread,
            'cpi_inflation': cpi_inflation,
            'spy_return': np.random.normal(0.0005, 0.015)
        })
    
    df = pd.DataFrame(historical_data).set_index('date')
    return df


@st.cache_data(ttl=120)
def load_historical_data():
    """Load historical market data for statistical analysis."""
    try:
        return load_synthetic_historical_data()
    except Exception as e:
        logger.warning(f"Failed to load historical data: {e}. Using synthetic data.")
        return load_synthetic_historical_data()


def compute_iv_surface(df):
    """Construct volatility surface using spatial interpolation."""
    if 'implied_volatility_market' not in df.columns:
        return None, None, None

    valid = df[
        (df['implied_volatility_market'] > 0.01) &
        (df['implied_volatility_market'] < 2.0) &
        (df['moneyness'] > 0.8) &
        (df['moneyness'] < 1.2)
    ].copy()

    if len(valid) < 10:
        return None, None, None

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

    from scipy.ndimage import generic_filter

    def nanmean_filter(values):
        valid_vals = values[~np.isnan(values)]
        return np.mean(valid_vals) if len(valid_vals) > 0 else np.nan

    for _ in range(3):
        mask = np.isnan(grid)
        if not mask.any():
            break
        grid = np.where(mask,
                       generic_filter(grid, nanmean_filter, size=3, mode='nearest'),
                       grid)

    return grid, moneyness_bins, expiry_bins


def compute_metrics(df, historical_data=None):
    """Compute comprehensive mathematical metrics of volatility surface."""
    try:
        if df is None or df.empty or 'implied_volatility_market' not in df.columns:
            return {}

        valid = df[
            (df['implied_volatility_market'] > 0.01) &
            (df['implied_volatility_market'] < 2.0) &
            (df['implied_volatility_market'].notna())
        ].copy()

        if len(valid) < 10:
            return {}

        # ATM Volatility
        atm_mask = (valid['moneyness'] > 0.97) & (valid['moneyness'] < 1.03)
        short_term = valid['days_to_expiry'] < 45
        atm_options = valid[atm_mask & short_term]
        atm_vol = atm_options['implied_volatility_market'].mean() if len(atm_options) > 0 else 0.15
        atm_vol = max(0.01, min(2.0, atm_vol))

        # 25Δ Skew
        put_wing = valid[(valid['moneyness'] > 0.90) & (valid['moneyness'] < 0.95)]
        call_wing = valid[(valid['moneyness'] > 1.05) & (valid['moneyness'] < 1.10)]
        put_vol = put_wing['implied_volatility_market'].mean() if len(put_wing) > 0 else atm_vol
        call_vol = call_wing['implied_volatility_market'].mean() if len(call_wing) > 0 else atm_vol
        
        put_vol = put_vol if np.isfinite(put_vol) and put_vol > 0 else atm_vol
        call_vol = call_vol if np.isfinite(call_vol) and call_vol > 0 else atm_vol
        
        skew_25d = put_vol - call_vol
        skew_25d = max(-0.5, min(0.5, skew_25d))

        # Curvature
        wing_avg = (put_vol + call_vol) / 2
        curvature = wing_avg - atm_vol
        curvature = max(-0.2, min(0.2, curvature))

        # Term Structure
        short_atm = valid[atm_mask & (valid['days_to_expiry'] < 30)]['implied_volatility_market'].mean()
        long_atm = valid[atm_mask & (valid['days_to_expiry'] > 60)]['implied_volatility_market'].mean()
        
        if (np.isfinite(short_atm) and np.isfinite(long_atm) and 
            not np.isnan(short_atm) and not np.isnan(long_atm)):
            term_slope = long_atm - short_atm
        else:
            term_slope = 0.0
        term_slope = max(-0.1, min(0.1, term_slope))

        # Surface Roughness
        vol_values = valid['implied_volatility_market'].values
        vol_values = vol_values[np.isfinite(vol_values)]
        
        if len(vol_values) == 0:
            return {}

        vol_norm = vol_values / vol_values.sum()
        entropy = -np.sum(vol_norm * np.log(vol_norm + 1e-10)) if len(vol_norm) > 0 else 0
        entropy = max(0, min(10, entropy))

        roughness = np.std(vol_values) if len(vol_values) > 1 else 0
        roughness = max(0, min(1, roughness))

        # Z-scores relative to historical data
        if historical_data is not None and not historical_data.empty:
            atm_z = (atm_vol - historical_data['avg_atm_vol'].mean()) / (historical_data['avg_atm_vol'].std() + 1e-10)
            skew_z = (skew_25d - historical_data['avg_25d_skew'].mean()) / (historical_data['avg_25d_skew'].std() + 1e-10)
            rough_z = (roughness - historical_data['roughness'].mean()) / (historical_data['roughness'].std() + 1e-10)
        else:
            atm_z = 0
            skew_z = 0
            rough_z = 0

        metrics = {
            'atm_vol': atm_vol,
            'skew_25d': skew_25d,
            'curvature': curvature,
            'term_slope': term_slope,
            'entropy': entropy,
            'roughness': roughness,
            'avg_atm_vol': atm_vol,
            'avg_25d_skew': skew_25d,
            'avg_curvature': curvature,
            'atm_vol_z': atm_z,
            'skew_z': skew_z,
            'roughness_z': rough_z,
        }

        return metrics

    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}


def classify_regime(metrics):
    """Classify regime using GMM classifier."""
    try:
        if not metrics or not gmm_classifier.is_fitted:
            historical_data = load_historical_data()
            gmm_classifier.fit(historical_data)
        
        prediction = gmm_classifier.classify(metrics)
        regime = prediction['current_regime']
        confidence = prediction['confidence']
        
        drivers = []
        
        # Add key drivers
        if 'atm_vol_z' in metrics:
            atm_z = metrics['atm_vol_z']
            if abs(atm_z) > 2:
                severity = 'high' if abs(atm_z) > 3 else 'medium'
                direction = 'elevated' if atm_z > 0 else 'depressed'
                drivers.append({
                    'title': f"ATM Volatility {direction.upper()}: {atm_z:.1f}σ",
                    'explanation': f"ATM volatility is {abs(atm_z):.1f} standard deviations {'above' if atm_z > 0 else 'below'} historical mean.",
                    'severity': severity,
                    'math': f"Z-score = {atm_z:.2f}"
                })
        
        return regime, confidence, drivers
        
    except Exception as e:
        print(f"Error in classify_regime: {e}")
        return 'UNKNOWN', 0.0, []


def fetch_option_data(ticker, expiry_range, moneyness_range):
    """Fetch and clean option chain data."""
    try:
        fetcher = OptionChainFetcher()
        chain = fetcher.fetch_option_chain(ticker)
        
        if chain is None or chain.empty:
            return None
        
        config = {
            'min_moneyness': moneyness_range[0],
            'max_moneyness': moneyness_range[1],
            'min_days_to_expiry': expiry_range[0],
            'max_days_to_expiry': expiry_range[1],
            'min_volume': 1,
            'max_spread_pct': 0.50
        }
        clean_chain = clean_option_chain(chain, config)
        
        return clean_chain
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# ============ UI COMPONENTS ============

def create_surface_chart(df, view_type='3d'):
    """Create surface visualization chart."""
    grid, moneyness, expiries = compute_iv_surface(df)
    
    if grid is None:
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text="No data available", x=0.5, y=0.5, 
                            xref="paper", yref="paper", showarrow=False,
                            font=dict(size=16, color=COLORS['muted']))]
        )
        return fig
    
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
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
    elif view_type == 'heatmap':
        fig = go.Figure(data=go.Heatmap(
            x=moneyness, y=expiries, z=grid * 100,
            colorscale='Viridis',
            colorbar=dict(title='IV %', ticksuffix='%')
        ))
        fig.update_layout(
            xaxis_title='Moneyness',
            yaxis_title='Days to Expiry',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
    else:  # smile
        fig = go.Figure()
        for i, exp in enumerate(expiries[::3]):
            fig.add_trace(go.Scatter(
                x=moneyness,
                y=grid[i*3] * 100,
                mode='lines+markers',
                name=f"{int(exp)}d",
                line=dict(width=2)
            ))
        fig.update_layout(
            xaxis_title='Moneyness',
            yaxis_title='Implied Volatility (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
    
    return fig


def create_gauge_chart(regime, confidence):
    """Create regime gauge chart."""
    regime_order = ['CALM', 'PRE_STRESS', 'ELEVATED', 'ACUTE', 'RECOVERY']
    regime_idx = regime_order.index(regime) if regime in regime_order else 2
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=regime_idx,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {confidence:.1%}"},
        gauge={
            'axis': {'range': [0, 4], 'tickvals': list(range(5)), 
                     'ticktext': regime_order},
            'bar': {'color': REGIME_COLORS.get(regime, COLORS['muted'])},
            'steps': [
                {'range': [0, 1], 'color': COLORS['green']},
                {'range': [1, 2], 'color': COLORS['yellow']},
                {'range': [2, 3], 'color': COLORS['orange']},
                {'range': [3, 4], 'color': COLORS['red']},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': regime_idx
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_smile_chart(df):
    """Create volatility smile chart by expiry."""
    if df is None or df.empty:
        return go.Figure()
    
    valid = df[
        (df['implied_volatility_market'] > 0.01) &
        (df['implied_volatility_market'] < 2.0)
    ].copy()
    
    if len(valid) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    # Group by expiry buckets
    expiry_buckets = [(7, 21), (21, 45), (45, 90)]
    colors = [COLORS['blue'], COLORS['green'], COLORS['yellow']]
    
    for i, (min_exp, max_exp) in enumerate(expiry_buckets):
        bucket_data = valid[
            (valid['days_to_expiry'] >= min_exp) & 
            (valid['days_to_expiry'] < max_exp)
        ]
        
        if len(bucket_data) > 0:
            # Group by moneyness bins
            moneyness_bins = np.linspace(0.85, 1.15, 20)
            iv_by_moneyness = []
            
            for mon in moneyness_bins:
                subset = bucket_data[abs(bucket_data['moneyness'] - mon) < 0.03]
                if len(subset) > 0:
                    iv_by_moneyness.append(subset['implied_volatility_market'].mean())
                else:
                    iv_by_moneyness.append(None)
            
            fig.add_trace(go.Scatter(
                x=moneyness_bins,
                y=np.array(iv_by_moneyness) * 100,
                mode='lines+markers',
                name=f"{min_exp}-{max_exp} days",
                line=dict(color=colors[i], width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        xaxis_title='Moneyness',
        yaxis_title='Implied Volatility (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


# ============ MAIN APP ============

def main():
    # Header
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.title("📊 Volatility Surface Monitor")
    
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.chain_data = None
            st.rerun()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        ticker = st.selectbox(
            "Ticker",
            options=['SPY', 'QQQ', 'IWM', 'SPMO'],
            format_func=lambda x: {
                'SPY': 'SPY - S&P 500',
                'QQQ': 'QQQ - Nasdaq 100',
                'IWM': 'IWM - Russell 2000',
                'SPMO': 'SPMO - S&P Momentum'
            }[x]
        )
        
        expiry_range = st.slider(
            "Expiry Range (days)",
            min_value=7,
            max_value=120,
            value=(14, 60)
        )
        
        moneyness_range = st.slider(
            "Moneyness Range",
            min_value=0.8,
            max_value=1.2,
            value=(0.9, 1.1),
            step=0.01,
            format="%.2f"
        )
        
        st.divider()
        
        st.subheader("Advanced Settings")
        
        feature_select = st.multiselect(
            "GMM Features",
            options=['avg_atm_vol', 'roughness', 'avg_25d_skew'],
            default=['avg_atm_vol', 'roughness']
        )
        
        view_type = st.radio(
            "Surface View",
            options=['3d', 'heatmap', 'smile'],
            format_func=lambda x: {'3d': '3D Surface', 'heatmap': 'Heatmap', 'smile': 'Smile Curves'}[x]
        )
    
    # Fetch data if not in session state
    if st.session_state.chain_data is None:
        with st.spinner("Fetching option data..."):
            chain_data = fetch_option_data(ticker, expiry_range, moneyness_range)
            st.session_state.chain_data = chain_data
            st.session_state.last_update = datetime.now()
    
    chain_data = st.session_state.chain_data
    
    # Update GMM features if changed
    if feature_select and len(feature_select) > 0:
        if feature_select != gmm_classifier.feature_cols:
            gmm_classifier.feature_cols = feature_select
            gmm_classifier.is_fitted = False
            historical_data = load_historical_data()
            gmm_classifier.fit(historical_data)
    
    # Compute metrics
    if chain_data is not None and not chain_data.empty:
        historical_data = load_historical_data()
        metrics = compute_metrics(chain_data, historical_data=historical_data)
        
        # Get SPY return
        try:
            spy = yf.Ticker("SPY")
            recent_data = spy.history(period="2d")
            if len(recent_data) >= 2:
                spy_return = recent_data['Close'].iloc[-1] / recent_data['Open'].iloc[-1] - 1
            else:
                spy_return = 0.0
        except:
            spy_return = 0.0
        
        metrics['spy_return'] = spy_return
        
        # Classify regime
        regime, confidence, drivers = classify_regime(metrics)
        
        # Display regime badge
        with col2:
            regime_color = REGIME_COLORS.get(regime, COLORS['muted'])
            st.markdown(
                f'<div class="regime-badge" style="border-color: {regime_color}; text-align: center;">'
                f'<span style="color: {regime_color}; font-size: 1.2rem;">●</span> '
                f'<span style="color: {COLORS["text"]};">{regime.replace("_", " ")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Metrics row
        st.divider()
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(
                label="📈 ATM Volatility",
                value=f"{metrics.get('atm_vol', 0)*100:.1f}%",
                delta=f"{metrics.get('atm_vol_z', 0):.1f}σ" if 'atm_vol_z' in metrics else None
            )
        
        with metric_cols[1]:
            st.metric(
                label="↔️ 25Δ Skew",
                value=f"{metrics.get('skew_25d', 0)*100:.2f}",
                delta=f"{metrics.get('skew_z', 0):.1f}σ" if 'skew_z' in metrics else None
            )
        
        with metric_cols[2]:
            st.metric(
                label="🌊 Curvature",
                value=f"{metrics.get('curvature', 0)*100:.2f}",
            )
        
        with metric_cols[3]:
            st.metric(
                label="📊 Term Slope",
                value=f"{metrics.get('term_slope', 0)*100:.2f}",
            )
        
        st.divider()
        
        # Main content area
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.subheader("🧊 IV Surface")
            fig = create_surface_chart(chain_data, view_type)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_side:
            st.subheader("💓 Regime Gauge")
            gauge_fig = create_gauge_chart(regime, confidence)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.subheader("🔔 Alerts & Analysis")
            if drivers:
                for driver in drivers[:3]:
                    severity_icons = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                    icon = severity_icons.get(driver.get('severity', 'low'), '🔵')
                    st.info(f"{icon} **{driver['title']}**\n\n{driver['explanation']}")
            else:
                st.info("No significant alerts")
        
        # Volatility smile chart
        st.divider()
        st.subheader("📉 Volatility Smile by Expiry")
        smile_fig = create_smile_chart(chain_data)
        st.plotly_chart(smile_fig, use_container_width=True)
        
        # Bottom row
        st.divider()
        bottom_col1, bottom_col2 = st.columns(2)
        
        with bottom_col1:
            st.subheader("🧠 Mathematical Pattern Analysis")
            with st.expander("View Details", expanded=True):
                st.write(f"**ATM Volatility**: {metrics.get('atm_vol', 0):.4f}")
                st.write(f"**Skew (25Δ)**: {metrics.get('skew_25d', 0):.4f}")
                st.write(f"**Curvature**: {metrics.get('curvature', 0):.4f}")
                st.write(f"**Term Slope**: {metrics.get('term_slope', 0):.4f}")
                st.write(f"**Surface Roughness**: {metrics.get('roughness', 0):.4f}")
                st.write(f"**Entropy**: {metrics.get('entropy', 0):.4f}")
        
        with bottom_col2:
            st.subheader("📊 Surface Statistics")
            with st.expander("View Details", expanded=True):
                if 'atm_vol_z' in metrics:
                    st.write(f"**ATM Vol Z-Score**: {metrics['atm_vol_z']:.2f}σ")
                if 'skew_z' in metrics:
                    st.write(f"**Skew Z-Score**: {metrics['skew_z']:.2f}σ")
                if 'roughness_z' in metrics:
                    st.write(f"**Roughness Z-Score**: {metrics['roughness_z']:.2f}σ")
                st.write(f"**Regime Confidence**: {confidence:.1%}")
                st.write(f"**Current Regime**: {regime}")
        
        # Update timestamp
        if st.session_state.last_update:
            st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    else:
        st.warning("No data available. Please check the ticker and try refreshing.")


if __name__ == "__main__":
    main()

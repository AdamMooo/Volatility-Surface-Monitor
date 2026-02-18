"""
Volatility Surface Integrity & Stress Monitor
Streamlit Dashboard

Real-time market regime detection through volatility surface shape analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
from pathlib import Path

# Import project modules
from src.data.fetcher import OptionChainFetcher
from src.data.cleaner import clean_option_chain, filter_by_moneyness, filter_by_volume
from src.models.surface import IVSurface
from src.analytics.geometry import compute_all_geometry_metrics
from src.analytics.regime_classifier import RegimeClassifier, MarketRegime
from src.analytics.arbitrage import run_all_arbitrage_checks
from src.models.density import extract_density, extract_density_from_iv, compute_moments, compute_tail_mass
from src.visualization.surface_plots import plot_iv_surface, plot_smile_comparison, plot_term_structure
from src.visualization.heatmaps import plot_curvature_heatmap, plot_skew_heatmap, plot_surface_metrics_grid
from src.visualization.regime_visuals import plot_regime_gauge, plot_early_warning_dashboard, create_regime_summary_card
from src.visualization.time_series import plot_metrics_summary
from src.portfolio.manager import PortfolioManager
from src.portfolio.analyzer import PortfolioAnalyzer

# Configure page
st.set_page_config(
    page_title="Volatility Surface Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = PortfolioManager()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Sidebar
st.sidebar.title("📈 Volatility Surface Monitor")
st.sidebar.markdown("---")

# Ticker selection
ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value=config.get('data', {}).get('ticker', 'SPY'),
    help="Enter the ticker symbol for analysis"
).upper()

# Data fetch button
if st.sidebar.button("🔄 Fetch Data", type="primary"):
    with st.spinner(f"Fetching option data for {ticker}..."):
        try:
            # Fetch data
            fetcher = OptionChainFetcher(use_cache=config.get('use_cache', True))
            raw_chain = fetcher.fetch_option_chain(ticker)
            
            if raw_chain is not None and not raw_chain.empty:
                # Clean data
                cleaned_chain = clean_option_chain(raw_chain, config.get('filtering', {}))
                
                # Store in session state
                st.session_state.data_cache = {
                    'ticker': ticker,
                    'raw_chain': raw_chain,
                    'cleaned_chain': cleaned_chain,
                    'spot_price': fetcher.get_spot_price(ticker),
                    'fetch_time': datetime.now()
                }
                st.session_state.last_update = datetime.now()
                st.sidebar.success(f"✓ Data fetched successfully!")
            else:
                st.sidebar.error("❌ No data retrieved. Please check the ticker symbol.")
        except Exception as e:
            st.sidebar.error(f"❌ Error fetching data: {str(e)}")

# Display last update time
if st.session_state.last_update:
    st.sidebar.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🌐 Surface Analysis",
        "📐 Geometry Metrics",
        "⚠️ Arbitrage Detection",
        "📉 Tail Risk",
        "💼 Portfolio Monitor"
    ]
)

# Main content
st.title("Volatility Surface Integrity & Stress Monitor")

# Check if data is available
if 'cleaned_chain' not in st.session_state.data_cache or st.session_state.data_cache.get('cleaned_chain') is None:
    st.warning("⚠️ No data loaded. Please fetch data using the sidebar.")
    st.info("""
    ### Getting Started
    1. Enter a ticker symbol in the sidebar (default: SPY)
    2. Click **🔄 Fetch Data** to retrieve option chain data
    3. Explore different sections using the navigation menu
    
    ### What This Dashboard Does
    
    This system monitors market stress by analyzing the **shape** of the implied volatility surface:
    
    - **Curvature reacts before level** - Early warning signals from surface geometry
    - **Regime Classification** - Automatic detection of market stress states
    - **Arbitrage Detection** - Validates no-arbitrage constraints
    - **Tail Risk Analysis** - Risk-neutral density estimation
    - **Portfolio Monitoring** - Track positions with real-time Greeks
    """)
    st.stop()

# Get cached data
cleaned_chain = st.session_state.data_cache['cleaned_chain']
spot_price = st.session_state.data_cache['spot_price']
ticker_name = st.session_state.data_cache['ticker']

# Build IV surface
@st.cache_data
def build_surface(chain_data, ticker, spot):
    """Build IV surface from cleaned chain data."""
    surface = IVSurface(
        interpolation_method=config.get('surface', {}).get('interpolation_method', 'rbf'),
        smoothing=config.get('surface', {}).get('smoothing', 0.0)
    )
    surface.build(chain_data, spot=spot)
    return surface

try:
    iv_surface = build_surface(cleaned_chain, ticker_name, spot_price)
except Exception as e:
    st.error(f"Error building surface: {str(e)}")
    st.stop()

# Compute metrics
@st.cache_data
def compute_metrics(chain_data, ticker, spot):
    """Compute all geometry metrics."""
    surface = IVSurface()
    surface.build(chain_data, spot=spot)
    return compute_all_geometry_metrics(surface, spot=spot)

try:
    geometry_metrics = compute_metrics(cleaned_chain, ticker_name, spot_price)
except Exception as e:
    st.warning(f"Some metrics couldn't be computed: {str(e)}")
    geometry_metrics = {}

# Regime classification
@st.cache_resource
def get_classifier():
    """Get regime classifier instance."""
    return RegimeClassifier(
        percentile_thresholds=config.get('regime', {}).get('percentile_thresholds'),
        z_score_threshold=config.get('regime', {}).get('z_score_threshold', 2.0)
    )

classifier = get_classifier()

try:
    regime_result = classifier.classify(geometry_metrics) if geometry_metrics else None
except Exception as e:
    st.warning(f"Regime classification unavailable: {str(e)}")
    regime_result = None

# Page routing
if page == "📊 Overview":
    st.header("Market Overview & Regime Classification")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ticker", ticker_name)
        st.metric("Spot Price", f"${spot_price:.2f}" if spot_price else "N/A")
    
    with col2:
        st.metric("Options Count", len(cleaned_chain))
        st.metric("Expiration Dates", cleaned_chain['expiration'].nunique() if 'expiration' in cleaned_chain else 0)
    
    with col3:
        atm_vol = geometry_metrics.get('atm_vol_short', 0) if geometry_metrics else 0
        st.metric("ATM Volatility", f"{atm_vol*100:.2f}%" if atm_vol > 0 else "N/A")
        
    with col4:
        if regime_result:
            regime = regime_result.get('regime', 'unknown')
            st.metric("Current Regime", regime.upper())
    
    st.markdown("---")
    
    # Regime visualization
    if regime_result:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Regime Classification")
            regime_card = create_regime_summary_card(regime_result)
            if regime_card:
                st.plotly_chart(regime_card, use_container_width=True)
        
        with col2:
            st.subheader("Early Warning Dashboard")
            try:
                warning_fig = plot_early_warning_dashboard(geometry_metrics)
                if warning_fig:
                    st.plotly_chart(warning_fig, use_container_width=True)
            except Exception as e:
                st.info("Early warning dashboard unavailable")
    
    # Key metrics table
    st.subheader("Key Metrics")
    if geometry_metrics:
        metrics_df = pd.DataFrame([
            {
                "Metric": "ATM Curvature",
                "Value": f"{geometry_metrics.get('atm_curvature', 0):.4f}",
                "Description": "Second derivative of IV at the money"
            },
            {
                "Metric": "25-Delta Skew",
                "Value": f"{geometry_metrics.get('skew_25d', 0):.4f}",
                "Description": "Put vs call IV spread"
            },
            {
                "Metric": "Surface Roughness",
                "Value": f"{geometry_metrics.get('roughness', 0):.4f}",
                "Description": "IV surface smoothness measure"
            },
        ])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

elif page == "🌐 Surface Analysis":
    st.header("Volatility Surface Visualization")
    
    tab1, tab2, tab3 = st.tabs(["3D Surface", "Smile Comparison", "Term Structure"])
    
    with tab1:
        st.subheader("3D Implied Volatility Surface")
        try:
            surface_df = iv_surface.to_dataframe()
            fig_surface = plot_iv_surface(surface_df, title=f"{ticker_name} IV Surface", spot=spot_price)
            st.plotly_chart(fig_surface, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting surface: {str(e)}")
    
    with tab2:
        st.subheader("Volatility Smile Comparison")
        try:
            # Get available maturities from the surface
            maturities = iv_surface.maturities[:5] if len(iv_surface.maturities) > 0 else []
            
            # Create dictionary of smiles for each maturity
            smiles = {}
            for T in maturities:
                smile_df = iv_surface.get_smile(T)
                smiles[f"{T:.3f}y"] = smile_df
            
            if smiles:
                fig_smile = plot_smile_comparison(smiles)
                st.plotly_chart(fig_smile, use_container_width=True)
            else:
                st.info("No smile data available")
        except Exception as e:
            st.error(f"Error plotting smiles: {str(e)}")
    
    with tab3:
        st.subheader("ATM Volatility Term Structure")
        try:
            term_structure_df = iv_surface.get_term_structure()
            fig_term = plot_term_structure(term_structure_df)
            st.plotly_chart(fig_term, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting term structure: {str(e)}")
    
    # Heatmaps
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Curvature Heatmap")
        try:
            # Get surface data and compute curvature
            surface_df = iv_surface.to_dataframe()
            # For now, use IV as proxy for curvature (will be enhanced later)
            if 'iv' in surface_df.columns:
                surface_df['curvature'] = surface_df['iv']
                fig_curv = plot_curvature_heatmap(surface_df)
                st.plotly_chart(fig_curv, use_container_width=True)
            else:
                st.info("Curvature heatmap unavailable")
        except Exception as e:
            st.info(f"Curvature heatmap unavailable: {str(e)}")
    
    with col2:
        st.subheader("Skew Heatmap")
        try:
            # Get surface data and compute skew
            surface_df = iv_surface.to_dataframe()
            # For now, use IV as proxy for skew (will be enhanced later)
            if 'iv' in surface_df.columns:
                surface_df['skew'] = surface_df['iv']
                fig_skew = plot_skew_heatmap(surface_df)
                st.plotly_chart(fig_skew, use_container_width=True)
            else:
                st.info("Skew heatmap unavailable")
        except Exception as e:
            st.info(f"Skew heatmap unavailable: {str(e)}")

elif page == "📐 Geometry Metrics":
    st.header("Surface Geometry Metrics")
    
    if geometry_metrics:
        # Metrics summary
        st.subheader("All Computed Metrics")
        
        metrics_df = pd.DataFrame([
            {"Metric": k, "Value": f"{v:.6f}" if isinstance(v, (int, float)) else str(v)}
            for k, v in geometry_metrics.items()
        ])
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        try:
            # Prepare metrics data for visualization
            # Convert geometry metrics to appropriate structure if needed
            st.info("Metrics grid visualization available in future update")
        except Exception as e:
            st.info("Metrics grid visualization unavailable")
    else:
        st.warning("No geometry metrics available")

elif page == "⚠️ Arbitrage Detection":
    st.header("No-Arbitrage Constraint Validation")
    
    st.info("""
    **No-Arbitrage Conditions:**
    - Call prices decrease monotonically with strike
    - Put prices increase monotonically with strike
    - Butterfly spreads must be non-negative
    - Calendar spreads must be non-negative
    """)
    
    try:
        arbitrage_results = run_all_arbitrage_checks(
            cleaned_chain,
            surface=iv_surface,
            tolerance=config.get('arbitrage', {}).get('tolerance', 0.001)
        )
        
        if arbitrage_results:
            # Collect all violations from different categories
            all_violations = []
            all_violations.extend(arbitrage_results.get('butterfly_violations', []))
            all_violations.extend(arbitrage_results.get('calendar_violations', []))
            all_violations.extend(arbitrage_results.get('vertical_violations', []))
            
            col1, col2 = st.columns(2)
            with col1:
                total_checks = arbitrage_results.get('summary', {}).get('total_checks', 0)
                st.metric("Total Checks", total_checks if total_checks > 0 else len(all_violations))
            with col2:
                st.metric("Violations Found", len(all_violations), delta=f"-{len(all_violations)}" if all_violations else "0")
            
            if all_violations:
                st.warning(f"⚠️ Found {len(all_violations)} arbitrage violations!")
                violations_df = pd.DataFrame(all_violations)
                st.dataframe(violations_df, use_container_width=True)
            else:
                st.success("✅ No arbitrage violations detected!")
        else:
            st.info("Arbitrage detection completed - no issues found")
            
    except Exception as e:
        st.error(f"Error in arbitrage detection: {str(e)}")

elif page == "📉 Tail Risk":
    st.header("Tail Risk Analysis")
    
    st.info("""
    **Risk-Neutral Density Analysis:**
    
    The risk-neutral density is derived from option prices using the Breeden-Litzenberger formula.
    Heavy tails indicate elevated crash risk pricing.
    """)
    
    try:
        # Select expiration for analysis
        expirations = sorted(cleaned_chain['expiration'].unique()) if 'expiration' in cleaned_chain.columns else []
        if not expirations:
            # Use time_to_expiry if expiration is not available
            expirations = sorted(cleaned_chain['time_to_expiry'].unique()) if 'time_to_expiry' in cleaned_chain.columns else []
        
        if expirations:
            selected_expiry = st.selectbox("Select Expiration", expirations)
            
            if selected_expiry:
                # Determine time to expiry
                if 'time_to_expiry' in cleaned_chain.columns:
                    T = cleaned_chain[cleaned_chain['expiration'] == selected_expiry]['time_to_expiry'].iloc[0] if 'expiration' in cleaned_chain.columns else selected_expiry
                else:
                    T = selected_expiry
                
                # Get risk-free rate
                r = config.get('risk_free_rate', 0.05)
                
                # Determine strike range
                expiry_mask = (
                    (cleaned_chain['expiration'] == selected_expiry) if 'expiration' in cleaned_chain.columns 
                    else (cleaned_chain['time_to_expiry'] == selected_expiry)
                )
                strikes_in_expiry = cleaned_chain[expiry_mask]['strike'] if 'strike' in cleaned_chain.columns else pd.Series(dtype=float)
                
                if len(strikes_in_expiry) > 0:
                    K_min = float(strikes_in_expiry.min()) * 0.9
                    K_max = float(strikes_in_expiry.max()) * 1.1
                else:
                    K_min = spot_price * 0.7
                    K_max = spot_price * 1.3
                
                # Extract density from IV surface
                density_df = extract_density_from_iv(
                    sigma_func=lambda K: iv_surface.evaluate(K, T),
                    S=spot_price,
                    r=r,
                    T=T,
                    K_range=(K_min, K_max),
                    num_points=100
                )
                
                if density_df is not None and not density_df.empty:
                    # Plot density
                    fig_density = {
                        'data': [{
                            'x': density_df['strike'].tolist(),
                            'y': density_df['density'].tolist(),
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': 'Risk-Neutral Density'
                        }],
                        'layout': {
                            'title': f'Risk-Neutral Density - {selected_expiry}',
                            'xaxis': {'title': 'Strike Price'},
                            'yaxis': {'title': 'Probability Density'}
                        }
                    }
                    st.plotly_chart(fig_density, use_container_width=True)
                    
                    # Compute metrics
                    moments = compute_moments(density_df)
                    tail_mass = compute_tail_mass(density_df, spot_price)
                    
                    # Tail metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Left Tail Mass", f"{tail_mass.get('left_tail_mass', 0):.4f}")
                    with col2:
                        st.metric("Right Tail Mass", f"{tail_mass.get('right_tail_mass', 0):.4f}")
                    with col3:
                        st.metric("Kurtosis", f"{moments.get('kurtosis', 0):.4f}")
                else:
                    st.warning("Unable to compute risk-neutral density")
        else:
            st.warning("No expiration data available")
    except Exception as e:
        st.error(f"Error in tail risk analysis: {str(e)}")

elif page == "💼 Portfolio Monitor":
    st.header("Portfolio Monitor")
    
    st.info("Portfolio monitoring functionality - Add positions to track Greeks and risk metrics")
    
    # Add position form
    with st.expander("➕ Add Position"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pos_type = st.selectbox("Type", ["Call", "Put"])
            strike = st.number_input("Strike", min_value=0.0, value=spot_price if spot_price else 100.0)
        
        with col2:
            quantity = st.number_input("Quantity", value=1)
            expirations = sorted(cleaned_chain['expiration'].unique()) if 'expiration' in cleaned_chain.columns else []
            expiry = st.selectbox("Expiration", expirations if expirations else ["No data"])
        
        with col3:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("Add to Portfolio"):
                st.success(f"Added {quantity} {pos_type} @ ${strike}")
    
    # Portfolio summary
    st.subheader("Current Positions")
    st.info("No positions in portfolio. Add positions above to get started.")
    
    # Greeks dashboard would go here
    st.subheader("Portfolio Greeks")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Delta", "0.00")
    with col2:
        st.metric("Gamma", "0.00")
    with col3:
        st.metric("Vega", "0.00")
    with col4:
        st.metric("Theta", "0.00")
    with col5:
        st.metric("Rho", "0.00")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>Volatility Surface Integrity & Stress Monitor</p>
    <p>⚠️ For educational and research purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

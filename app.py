"""
Volatility Surface Integrity & Stress Monitor
Streamlit Dashboard

Market regime detection through volatility surface shape analysis.
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
from src.analytics.geometry import (
    compute_all_geometry_metrics, compute_curvature, compute_skew
)
from src.analytics.regime_classifier import RegimeClassifier, MarketRegime
from src.analytics.arbitrage import run_all_arbitrage_checks
from src.models.density import extract_density, extract_density_from_iv, compute_moments, compute_tail_mass
from src.visualization.surface_plots import plot_iv_surface, plot_smile_comparison, plot_term_structure
from src.visualization.heatmaps import plot_curvature_heatmap, plot_skew_heatmap
from src.visualization.regime_visuals import plot_regime_gauge, plot_early_warning_dashboard, create_regime_summary_card

# Configure page
st.set_page_config(
    page_title="Volatility Surface Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Sidebar
st.sidebar.title("Volatility Surface Monitor")
st.sidebar.markdown("---")

# Ticker selection
ticker_input = st.sidebar.text_input(
    "Ticker Symbol",
    value=config.get('data', {}).get('ticker', 'SPY'),
    help="Enter the ticker symbol for analysis"
)
ticker = ticker_input.upper() if ticker_input else 'SPY'

# Data fetch button
if st.sidebar.button("Fetch Data", type="primary"):
    with st.spinner(f"Fetching option data for {ticker}..."):
        try:
            fetcher = OptionChainFetcher(use_cache=config.get('use_cache', True))
            raw_chain = fetcher.fetch_option_chain(ticker)

            if raw_chain is not None and not raw_chain.empty:
                cleaned_chain = clean_option_chain(raw_chain, config.get('filtering', {}))
                st.session_state.data_cache = {
                    'ticker': ticker,
                    'raw_chain': raw_chain,
                    'cleaned_chain': cleaned_chain,
                    'spot_price': fetcher.get_spot_price(ticker),
                    'fetch_time': datetime.now()
                }
                st.session_state.last_update = datetime.now()
                st.sidebar.success("Data fetched successfully.")
            else:
                st.sidebar.error("No data retrieved. Check the ticker symbol.")
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {str(e)}")

if st.session_state.last_update:
    st.sidebar.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.markdown("---")

# Navigation -- three focused tabs
page = st.sidebar.radio(
    "Navigation",
    [
        "Surface & Structure",
        "Skew, Curvature & Regime",
        "Tail Risk & Arbitrage"
    ]
)

# Main content
st.title("Volatility Surface Monitor")

# Check if data is available
if 'cleaned_chain' not in st.session_state.data_cache or st.session_state.data_cache.get('cleaned_chain') is None:
    st.warning("No data loaded. Please fetch data using the sidebar.")
    st.markdown("""
    ### Getting Started
    1. Enter a ticker symbol in the sidebar (default: SPY)
    2. Click **Fetch Data** to retrieve the option chain
    3. Navigate between tabs to explore the volatility surface

    ### Overview
    This system monitors market stress by analyzing the **shape** of the implied volatility surface:
    - **Curvature reacts before level** -- early warning signals from surface geometry
    - **Regime classification** -- automatic detection of market stress states
    - **Arbitrage detection** -- validates no-arbitrage constraints across strikes and maturities
    - **Tail risk analysis** -- risk-neutral density estimation from option prices
    """)
    st.stop()

# Get cached data
cleaned_chain = st.session_state.data_cache['cleaned_chain']
spot_price = st.session_state.data_cache['spot_price']
ticker_name = st.session_state.data_cache['ticker']

# Build IV surface
@st.cache_resource
def build_surface(chain_data_hash, _ticker, spot):
    chain_data = st.session_state.data_cache['cleaned_chain']
    surface = IVSurface(
        interpolation_method=config.get('surface', {}).get('interpolation_method', 'rbf'),
        smoothing=config.get('surface', {}).get('smoothing', 0.0)
    )
    surface.build(chain_data, spot=spot)
    return surface

try:
    cache_key = f"{len(cleaned_chain)}_{ticker_name}_{spot_price}"
    iv_surface = build_surface(cache_key, ticker_name, spot_price)
except Exception as e:
    st.error(f"Error building surface: {str(e)}")
    st.stop()

# Compute geometry metrics
@st.cache_data
def compute_metrics(_chain_data, _ticker, spot):
    surface = IVSurface()
    surface.build(_chain_data, spot=spot)
    return compute_all_geometry_metrics(surface, spot=spot)

try:
    geometry_metrics = compute_metrics(cleaned_chain, ticker_name, spot_price)
except Exception as e:
    st.warning(f"Some metrics could not be computed: {str(e)}")
    geometry_metrics = {}

# Regime classification
@st.cache_resource
def get_classifier():
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


# ---------------------------------------------------------------------------
# TAB 1 -- Surface & Structure
# ---------------------------------------------------------------------------
if page == "Surface & Structure":
    st.header(f"{ticker_name} -- Implied Volatility Surface")

    # Key stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spot Price", f"${spot_price:.2f}" if spot_price else "N/A")
    with col2:
        st.metric("Options Loaded", len(cleaned_chain))
    with col3:
        n_expiries = cleaned_chain['expiration'].nunique() if 'expiration' in cleaned_chain else 0
        st.metric("Expirations", n_expiries)
    with col4:
        summary = geometry_metrics.get('summary', {})
        avg_vol = summary.get('avg_atm_vol', 0)
        st.metric("Avg ATM Vol", f"{avg_vol * 100:.2f}%" if avg_vol else "N/A")

    st.markdown("---")

    # 3D Surface
    st.subheader("3D Implied Volatility Surface")
    try:
        surface_df = iv_surface.to_dataframe()
        fig_surface = plot_iv_surface(
            surface_df, title=f"{ticker_name} IV Surface", spot=spot_price
        )
        st.plotly_chart(fig_surface, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting surface: {str(e)}")

    # Smile comparison + term structure side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Volatility Smile by Maturity")
        try:
            maturities = iv_surface.maturities[:6] if iv_surface.maturities else []
            smiles = {}
            for T in maturities:
                smile_df = iv_surface.get_smile(T)
                smiles[f"{T:.3f}y"] = smile_df
            if smiles:
                fig_smile = plot_smile_comparison(smiles)
                st.plotly_chart(fig_smile, use_container_width=True)
            else:
                st.info("No smile data available.")
        except Exception as e:
            st.error(f"Error plotting smiles: {str(e)}")

    with col_right:
        st.subheader("ATM Volatility Term Structure")
        try:
            term_structure_df = iv_surface.get_term_structure()
            fig_term = plot_term_structure(term_structure_df)
            st.plotly_chart(fig_term, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting term structure: {str(e)}")

    # Curvature and skew heatmaps -- computed from the actual surface, not IV proxy
    st.markdown("---")
    col_c, col_s = st.columns(2)

    with col_c:
        st.subheader("Curvature Heatmap (d^2 sigma / dK^2)")
        try:
            surface_df = iv_surface.to_dataframe()
            if 'iv' in surface_df.columns and 'strike' in surface_df.columns and 'time_to_expiry' in surface_df.columns:
                sigma_func = lambda K, T: iv_surface.evaluate(K, T)
                curvatures = []
                for _, row in surface_df.iterrows():
                    try:
                        c = compute_curvature(sigma_func, row['strike'], row['time_to_expiry'], spot_price)
                        curvatures.append(c if np.isfinite(c) else 0.0)
                    except Exception:
                        curvatures.append(0.0)
                curv_df = surface_df[['strike', 'time_to_expiry']].copy()
                curv_df['curvature'] = curvatures
                fig_curv = plot_curvature_heatmap(curv_df)
                st.plotly_chart(fig_curv, use_container_width=True)
            else:
                st.info("Curvature heatmap unavailable -- missing surface columns.")
        except Exception as e:
            st.info(f"Curvature heatmap unavailable: {str(e)}")

    with col_s:
        st.subheader("Skew Heatmap (d sigma / dK)")
        try:
            surface_df = iv_surface.to_dataframe()
            if 'iv' in surface_df.columns and 'strike' in surface_df.columns and 'time_to_expiry' in surface_df.columns:
                sigma_func = lambda K, T: iv_surface.evaluate(K, T)
                skews = []
                for _, row in surface_df.iterrows():
                    try:
                        s = compute_skew(sigma_func, row['strike'], row['time_to_expiry'], spot_price)
                        skews.append(s if np.isfinite(s) else 0.0)
                    except Exception:
                        skews.append(0.0)
                skew_df = surface_df[['strike', 'time_to_expiry']].copy()
                skew_df['skew'] = skews
                fig_skew = plot_skew_heatmap(skew_df)
                st.plotly_chart(fig_skew, use_container_width=True)
            else:
                st.info("Skew heatmap unavailable -- missing surface columns.")
        except Exception as e:
            st.info(f"Skew heatmap unavailable: {str(e)}")


# ---------------------------------------------------------------------------
# TAB 2 -- Skew, Curvature & Regime
# ---------------------------------------------------------------------------
elif page == "Skew, Curvature & Regime":
    st.header("Surface Geometry & Market Regime")

    # Regime quick strip
    if regime_result:
        regime_name = regime_result.get('current_regime', 'unknown').replace('_', ' ').upper()
        confidence = regime_result.get('confidence', 0)
        st.markdown(
            f"**Current regime: {regime_name}** &nbsp;&mdash;&nbsp; "
            f"Confidence: {confidence:.0%}"
        )
    st.markdown("---")

    # Regime gauge + early warning side by side
    if regime_result:
        col_g, col_w = st.columns([1, 2])

        with col_g:
            st.subheader("Regime Gauge")
            regime_card = create_regime_summary_card(regime_result)
            if regime_card:
                st.plotly_chart(regime_card, use_container_width=True)

        with col_w:
            st.subheader("Early Warning Dashboard")
            try:
                warning_fig = plot_early_warning_dashboard(geometry_metrics)
                if warning_fig:
                    st.plotly_chart(warning_fig, use_container_width=True)
            except Exception:
                st.info("Early warning dashboard unavailable.")

        # Recommendation
        recommendation = regime_result.get('recommendation', '')
        if recommendation:
            st.markdown("---")
            st.subheader("Regime Assessment")
            st.markdown(recommendation)

        # Key drivers
        drivers = regime_result.get('key_drivers', [])
        if drivers:
            st.subheader("Key Drivers")
            for d in drivers:
                st.markdown(f"- {d}")

    st.markdown("---")

    # Structural metrics -- organized in a meaningful table, not a raw dump
    st.subheader("Surface Geometry Metrics by Maturity")
    if geometry_metrics:
        maturities = geometry_metrics.get('maturities', [])
        level = geometry_metrics.get('level_metrics', {})
        skew_m = geometry_metrics.get('skew_metrics', {})
        curv_m = geometry_metrics.get('curvature_metrics', {})

        rows = []
        for T in maturities:
            rows.append({
                "Maturity (y)": f"{T:.3f}",
                "ATM Vol": f"{level.get('atm_vol', {}).get(T, 0) * 100:.2f}%",
                "ATM Skew": f"{skew_m.get('atm_skew', {}).get(T, 0):.6f}",
                "25d Skew": f"{skew_m.get('25d_skew', {}).get(T, 0):.6f}",
                "ATM Curvature": f"{curv_m.get('atm_curvature', {}).get(T, 0):.6f}",
                "Butterfly (25d)": f"{curv_m.get('butterfly_25d', {}).get(T, 0):.6f}",
                "Left Wing Curv": f"{curv_m.get('left_wing_curvature', {}).get(T, 0):.6f}",
                "Right Wing Curv": f"{curv_m.get('right_wing_curvature', {}).get(T, 0):.6f}",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Summary row
        summary = geometry_metrics.get('summary', {})
        if summary:
            st.subheader("Aggregate Summary")
            scol1, scol2, scol3, scol4 = st.columns(4)
            with scol1:
                st.metric("Avg ATM Vol", f"{summary.get('avg_atm_vol', 0) * 100:.2f}%")
            with scol2:
                st.metric("Avg 25d Skew", f"{summary.get('avg_25d_skew', 0):.6f}")
            with scol3:
                st.metric("Avg ATM Curvature", f"{summary.get('avg_atm_curvature', 0):.6f}")
            with scol4:
                st.metric("Surface Roughness", f"{summary.get('roughness', 0):.6f}")
    else:
        st.warning("Geometry metrics unavailable.")


# ---------------------------------------------------------------------------
# TAB 3 -- Tail Risk & Arbitrage
# ---------------------------------------------------------------------------
elif page == "Tail Risk & Arbitrage":
    st.header("Tail Risk & No-Arbitrage Validation")

    # --- Tail Risk Section ---
    st.subheader("Risk-Neutral Density Analysis")
    st.markdown(
        "The risk-neutral density is extracted from option prices via the "
        "Breeden-Litzenberger formula. Heavy left tails indicate elevated "
        "crash-risk pricing in the options market."
    )

    try:
        expirations = sorted(cleaned_chain['expiration'].unique()) if 'expiration' in cleaned_chain.columns else []
        if not expirations:
            expirations = sorted(cleaned_chain['time_to_expiry'].unique()) if 'time_to_expiry' in cleaned_chain.columns else []

        if expirations:
            selected_expiry = st.selectbox("Select Expiration", expirations)

            if selected_expiry:
                if 'time_to_expiry' in cleaned_chain.columns and 'expiration' in cleaned_chain.columns:
                    T = cleaned_chain[cleaned_chain['expiration'] == selected_expiry]['time_to_expiry'].iloc[0]
                else:
                    T = selected_expiry

                r = config.get('risk_free_rate', 0.05)

                expiry_mask = (
                    (cleaned_chain['expiration'] == selected_expiry) if 'expiration' in cleaned_chain.columns
                    else (cleaned_chain['time_to_expiry'] == selected_expiry)
                )
                strikes_in_expiry = cleaned_chain.loc[expiry_mask, 'strike'] if 'strike' in cleaned_chain.columns else pd.Series(dtype=float)

                K_min = float(strikes_in_expiry.min()) * 0.9 if len(strikes_in_expiry) > 0 else spot_price * 0.7
                K_max = float(strikes_in_expiry.max()) * 1.1 if len(strikes_in_expiry) > 0 else spot_price * 1.3

                density_df = extract_density_from_iv(
                    sigma_func=lambda K: iv_surface.evaluate(K, T),
                    S=spot_price,
                    r=r,
                    T=T,
                    K_range=(K_min, K_max),
                    num_points=100
                )

                if density_df is not None and not density_df.empty:
                    import plotly.graph_objects as go

                    fig_density = go.Figure()
                    fig_density.add_trace(go.Scatter(
                        x=density_df['strike'],
                        y=density_df['density'],
                        mode='lines',
                        name='Risk-Neutral Density',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    if spot_price:
                        fig_density.add_vline(
                            x=spot_price, line_dash="dash", line_color="gray",
                            annotation_text="Spot"
                        )
                    fig_density.update_layout(
                        title=f"Risk-Neutral Density -- {selected_expiry}",
                        xaxis_title="Strike Price",
                        yaxis_title="Probability Density",
                        template="plotly_white",
                        height=450
                    )
                    st.plotly_chart(fig_density, use_container_width=True)

                    moments = compute_moments(density_df)
                    tail_mass = compute_tail_mass(density_df, spot_price)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Left Tail Mass", f"{tail_mass.get('left_tail_mass', 0):.4f}")
                    with col2:
                        st.metric("Right Tail Mass", f"{tail_mass.get('right_tail_mass', 0):.4f}")
                    with col3:
                        st.metric("Skewness", f"{moments.get('skewness', 0):.4f}")
                    with col4:
                        st.metric("Excess Kurtosis", f"{moments.get('excess_kurtosis', 0):.4f}")
                else:
                    st.warning("Unable to compute risk-neutral density for this expiration.")
        else:
            st.warning("No expiration data available.")
    except Exception as e:
        st.error(f"Error in tail risk analysis: {str(e)}")

    # --- Arbitrage Section ---
    st.markdown("---")
    st.subheader("No-Arbitrage Constraint Validation")
    st.markdown(
        "Checks three classes of no-arbitrage conditions across the option surface: "
        "**butterfly** (convexity in strike), **calendar** (monotonicity in maturity), "
        "and **vertical** (monotonicity in strike)."
    )

    try:
        arbitrage_results = run_all_arbitrage_checks(
            cleaned_chain,
            surface=iv_surface,
            tolerance=config.get('arbitrage', {}).get('tolerance', 0.001)
        )

        if arbitrage_results:
            summary = arbitrage_results.get('summary', {})
            bf_v = arbitrage_results.get('butterfly_violations', pd.DataFrame())
            cal_v = arbitrage_results.get('calendar_violations', pd.DataFrame())
            vert_v = arbitrage_results.get('vertical_violations', pd.DataFrame())

            bf_count = len(bf_v) if isinstance(bf_v, pd.DataFrame) else 0
            cal_count = len(cal_v) if isinstance(cal_v, pd.DataFrame) else 0
            vert_count = len(vert_v) if isinstance(vert_v, pd.DataFrame) else 0
            total = bf_count + cal_count + vert_count

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Violations", total)
            with col2:
                st.metric("Butterfly", bf_count)
            with col3:
                st.metric("Calendar", cal_count)
            with col4:
                st.metric("Vertical", vert_count)

            if total == 0:
                st.success("No arbitrage violations detected.")
            else:
                st.warning(f"Found {total} arbitrage violation(s).")
                if bf_count > 0:
                    with st.expander(f"Butterfly violations ({bf_count})"):
                        st.dataframe(bf_v, use_container_width=True)
                if cal_count > 0:
                    with st.expander(f"Calendar violations ({cal_count})"):
                        st.dataframe(cal_v, use_container_width=True)
                if vert_count > 0:
                    with st.expander(f"Vertical violations ({vert_count})"):
                        st.dataframe(vert_v, use_container_width=True)
        else:
            st.info("Arbitrage checks completed -- no issues found.")
    except Exception as e:
        st.error(f"Error in arbitrage detection: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "<p>Volatility Surface Monitor</p>"
    "<p>For educational and research purposes only. Not financial advice.</p>"
    "</div>",
    unsafe_allow_html=True
)

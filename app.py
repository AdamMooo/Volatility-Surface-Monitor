"""
Volatility Surface Monitor
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import yaml
from pathlib import Path

from src.data.fetcher import OptionChainFetcher
from src.data.cleaner import clean_option_chain
from src.models.surface import IVSurface
from src.analytics.geometry import compute_all_geometry_metrics
from src.visualization.surface_plots import plot_iv_surface, plot_smile_comparison, plot_term_structure

st.set_page_config(page_title="Volatility Surface Monitor", layout="wide")

@st.cache_resource
def load_config():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Sidebar — ticker + fetch only
st.sidebar.title("Volatility Surface Monitor")
st.sidebar.markdown("---")

ticker_input = st.sidebar.text_input(
    "Ticker Symbol",
    value=config.get('data', {}).get('ticker', 'SPY'),
)
ticker = ticker_input.upper() if ticker_input else 'SPY'

if st.sidebar.button("Fetch Data", type="primary"):
    with st.spinner(f"Fetching option data for {ticker}..."):
        try:
            fetcher = OptionChainFetcher(use_cache=config.get('use_cache', True))
            raw_chain = fetcher.fetch_option_chain(ticker)
            if raw_chain is not None and not raw_chain.empty:
                cleaned_chain = clean_option_chain(raw_chain, config.get('filtering', {}))
                st.session_state.data_cache = {
                    'ticker': ticker,
                    'cleaned_chain': cleaned_chain,
                    'spot_price': fetcher.get_spot_price(ticker),
                    'fetch_time': datetime.now()
                }
                st.session_state.last_update = datetime.now()
                st.sidebar.success("Data fetched.")
            else:
                st.sidebar.error("No data returned. Check the ticker.")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

if st.session_state.last_update:
    st.sidebar.info(f"Updated: {st.session_state.last_update.strftime('%H:%M:%S')}")

# Main
st.title("Volatility Surface Monitor")

if 'cleaned_chain' not in st.session_state.data_cache:
    st.info("Enter a ticker and click **Fetch Data** to begin.")
    st.stop()

cleaned_chain = st.session_state.data_cache['cleaned_chain']
spot_price = st.session_state.data_cache['spot_price']
ticker_name = st.session_state.data_cache['ticker']

@st.cache_resource
def build_surface(cache_key, spot):
    chain = st.session_state.data_cache['cleaned_chain']
    surface = IVSurface(
        interpolation_method=config.get('surface', {}).get('interpolation_method', 'rbf'),
        smoothing=config.get('surface', {}).get('smoothing', 0.0)
    )
    surface.build(chain, spot=spot)
    return surface

try:
    cache_key = f"{len(cleaned_chain)}_{ticker_name}_{spot_price}"
    iv_surface = build_surface(cache_key, spot_price)
except Exception as e:
    st.error(f"Error building surface: {str(e)}")
    st.stop()

@st.cache_data
def compute_metrics(_chain, _ticker, spot):
    surface = IVSurface()
    surface.build(_chain, spot=spot)
    return compute_all_geometry_metrics(surface, spot=spot)

try:
    geometry_metrics = compute_metrics(cleaned_chain, ticker_name, spot_price)
except Exception:
    geometry_metrics = {}

# Key stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spot Price", f"${spot_price:.2f}" if spot_price else "N/A")
with col2:
    st.metric("Options Loaded", len(cleaned_chain))
with col3:
    n_expiries = cleaned_chain['expiration'].nunique() if 'expiration' in cleaned_chain else 0
    st.metric("Expirations", n_expiries)
with col4:
    avg_vol = geometry_metrics.get('summary', {}).get('avg_atm_vol', 0)
    st.metric("Avg ATM Vol", f"{avg_vol * 100:.2f}%" if avg_vol else "N/A")

st.markdown("---")

# 3D Surface
st.subheader("Implied Volatility Surface")
try:
    fig_surface = plot_iv_surface(iv_surface, title=f"{ticker_name} IV Surface", spot=spot_price)
    st.plotly_chart(fig_surface, use_container_width=True)
except Exception as e:
    st.error(f"Error plotting surface: {str(e)}")

# Smile + Term Structure
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Volatility Smile by Maturity")
    try:
        maturities = iv_surface.maturities[:6] if iv_surface.maturities else []
        smiles = {f"{T:.3f}y": iv_surface.get_smile(T) for T in maturities}
        if smiles:
            st.plotly_chart(plot_smile_comparison(smiles), use_container_width=True)
        else:
            st.info("No smile data available.")
    except Exception as e:
        st.error(f"Error plotting smiles: {str(e)}")

with col_right:
    st.subheader("ATM Term Structure")
    try:
        term_df = iv_surface.get_term_structure()
        st.plotly_chart(plot_term_structure(term_df), use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting term structure: {str(e)}")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.8em;'>"
    "Volatility Surface Monitor — for research purposes only"
    "</div>",
    unsafe_allow_html=True
)

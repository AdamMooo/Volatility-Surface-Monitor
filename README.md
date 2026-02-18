# Volatility Surface Integrity & Stress Monitor
## A Market Regime Detection System Through Shape Analysis

---

## Project Vision

This project builds an **early warning system for market stress** by analyzing the *shape* of the implied volatility surface, not just its level.

> **Curvature reacts before level.** While ATM volatility tells you what the market *currently* expects, curvature and skew reveal what sophisticated traders are *quietly pricing in* at the tails—often days before headlines catch up.

## Project Structure

```
volatility_surface_monitor/
├── src/
│   ├── data/           # Data acquisition and cleaning
│   ├── models/         # IV surface construction & pricing
│   ├── analytics/      # Arbitrage, geometry, tail risk, regime detection
│   ├── visualization/  # Plotly visualizations and dashboards
│   └── reporting/      # Automated report generation
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Unit tests
├── data/               # Raw and processed data storage
├── reports/            # Generated reports
└── dashboard/          # Dash application
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Launch Streamlit Dashboard (Recommended)

The easiest way to use the Volatility Surface Monitor is through the interactive Streamlit dashboard:

```bash
# Activate your virtual environment first
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Launch the dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501` and provides:
- **Real-time regime monitoring** with visual indicators
- **3D volatility surface visualization** 
- **Geometry metrics analysis** (skew, curvature, butterflies)
- **Arbitrage detection** across the surface
- **Tail risk analysis** via risk-neutral density
- **Portfolio monitoring** with Greeks calculation

### Basic Usage (Python API)

```python
from src.data.fetcher import OptionChainFetcher
from src.models.surface import IVSurface
from src.analytics.geometry import compute_all_geometry_metrics
from src.analytics.regime_classifier import RegimeClassifier

# Fetch option data
fetcher = OptionChainFetcher("SPY")
chain = fetcher.fetch_current_chain()

# Build IV surface
surface = IVSurface(chain)

# Compute geometry metrics
metrics = compute_all_geometry_metrics(surface)

# Classify regime
classifier = RegimeClassifier()
regime_result = classifier.classify(metrics)
print(f"Current Regime: {regime_result['regime']}")
```

## Key Metrics

| Metric | Description | Early Warning Signal |
|--------|-------------|---------------------|
| **ATM Curvature** | Second derivative of IV at money | Rising while vol flat = Pre-stress |
| **25-Delta Skew** | Put vs call IV spread | Steepening = Downside fear |
| **Tail Mass** | Implied probability of extremes | Rising = Crash risk |
| **Surface Roughness** | IV surface smoothness | Spike = Market dislocation |

## Regime Classification

- **Calm**: Normal risk pricing, low curvature
- **Pre-Stress**: Early warning - curvature rising, vol still flat
- **Elevated Stress**: Active stress, high vol and curvature
- **Acute Stress**: Crisis mode, extreme readings
- **Recovery**: Stress passing, metrics normalizing

## Documentation

### Streamlit Dashboard

The interactive dashboard (`app.py`) provides a comprehensive interface with 6 main sections:

1. **Overview** - Real-time market regime classification with key metrics
2. **Surface Analysis** - Interactive 3D volatility surface and heatmaps
3. **Geometry Metrics** - Skew and curvature term structures
4. **Arbitrage Detection** - No-arbitrage constraint validation
5. **Tail Risk** - Risk-neutral density analysis
6. **Portfolio Monitor** - Position tracking with Greeks

### Jupyter Notebooks

For detailed exploration and development, see the notebooks:
- `01_data_exploration.ipynb` - Understanding option chain data
- `02_surface_construction.ipynb` - Building IV surfaces
- `03_geometry_analysis.ipynb` - Skew and curvature analysis
- `04_regime_classification.ipynb` - Market regime classification
- `05_arbitrage_detection.ipynb` - No-arbitrage checks
- `06_tail_risk.ipynb` - Risk-neutral density
- `07_backtesting.ipynb` - Historical validation

## Disclaimer

This system is for educational and research purposes. It provides market diagnostics, not trading signals. Always conduct your own analysis before making investment decisions.

## References

- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
- Breeden, D. & Litzenberger, R. (1978). "Prices of State-Contingent Claims"
- Gatheral & Jacquier (2014). SVI Parameterization

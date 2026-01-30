# Volatility Surface Integrity & Stress Monitor
## A Market Regime Detection System Through Shape Analysis

---

## Project Vision

This project builds an **early warning system for market stress** by analyzing the *shape* of the implied volatility surface, not just its level.

> **Curvature reacts before level.** While ATM volatility tells you what the market *currently* expects, curvature and skew reveal what sophisticated traders are *quietly pricing in* at the tails—often days before headlines catch up.

## Project Structure

```
volatility_surface_monitor/
├── streamlit_app.py      # Main Streamlit application
├── dashboard.py          # Legacy Dash application (deprecated)
├── train_gmm.py          # GMM model training script
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/             # Data acquisition, cleaning, and trained models
│   ├── models/           # Core pricing models (Black-Scholes, IV surface)
│   ├── analytics/        # Geometry metrics, tail risk, regime classification
│   └── __init__.py
├── tests/                # Unit tests
└── config files          # pyproject.toml, requirements.txt, etc.
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

### Basic Usage

```python
from src.data.fetcher import OptionChainFetcher
from src.models.surface import IVSurface
from src.analytics.geometry import compute_all_geometry_metrics
from src.analytics.regime_classifier import GMMRegimeClassifier

# Fetch option data
fetcher = OptionChainFetcher()
data = fetcher.fetch_option_chain("SPY")

# Build IV surface
surface = IVSurface()
surface.build(data)

# Compute geometry metrics
metrics = compute_all_geometry_metrics(surface, spot=data['underlying_price'].iloc[0])

# Classify regime using GMM
classifier = GMMRegimeClassifier()
regime = classifier.classify(metrics)
print(f"Current Regime: {regime['current_regime']}")
```

### Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run streamlit_app.py

# The dashboard will open in your browser automatically
# Default URL: http://localhost:8501
```

### Training the GMM Model

```bash
python train_gmm.py
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

- `notebooks/gmm_backtesting.ipynb` - GMM model training, validation, and backtesting with synthetic historical data

## Disclaimer

This system is for educational and research purposes. It provides market diagnostics, not trading signals. Always conduct your own analysis before making investment decisions.

## References

- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
- Breeden, D. & Litzenberger, R. (1978). "Prices of State-Contingent Claims"
- Gatheral & Jacquier (2014). SVI Parameterization

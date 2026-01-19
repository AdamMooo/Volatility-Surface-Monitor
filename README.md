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

### Basic Usage

```python
from src.data.fetcher import OptionChainFetcher
from src.models.surface import IVSurface
from src.analytics.geometry import compute_all_geometry_metrics
from src.analytics.regime_classifier import RegimeClassifier

# Fetch option data
fetcher = OptionChainFetcher()
data = fetcher.fetch_option_chain("SPY")

# Build IV surface
surface = IVSurface()
surface.build(data)

# Compute geometry metrics
metrics = compute_all_geometry_metrics(surface, spot=data['underlying_price'].iloc[0])

# Classify regime
classifier = RegimeClassifier()
regime = classifier.classify(metrics)
print(f"Current Regime: {regime['current_regime']}")
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

See the notebooks for detailed walkthroughs:
- `01_data_exploration.ipynb` - Understanding option chain data
- `02_surface_construction.ipynb` - Building IV surfaces
- `03_arbitrage_analysis.ipynb` - No-arbitrage checks
- `04_geometry_metrics.ipynb` - Skew and curvature analysis
- `05_tail_risk_analysis.ipynb` - Risk-neutral density
- `06_regime_detection.ipynb` - Market regime classification
- `07_backtesting_signals.ipynb` - Historical validation

## Disclaimer

This system is for educational and research purposes. It provides market diagnostics, not trading signals. Always conduct your own analysis before making investment decisions.

## References

- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
- Breeden, D. & Litzenberger, R. (1978). "Prices of State-Contingent Claims"
- Gatheral & Jacquier (2014). SVI Parameterization

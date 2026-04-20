# Volatility Surface Monitor

Market stress diagnostic tool that tracks the shape of the implied volatility surface — not just its level.

## Core Insight

ATM volatility tells you what the market currently expects. Curvature and skew reveal what participants are pricing at the tails — often days before the move registers in headline vol. This tool makes those shape signals readable in real time.

## Architecture

| Module | Role |
|--------|------|
| `src/data/` | Option chain fetching, caching, and cleaning |
| `src/models/surface.py` | IV surface construction via RBF interpolation |
| `src/analytics/geometry.py` | Skew, curvature, butterfly spread, and tail-mass metrics |
| `src/visualization/` | 3D surface, cross-section smiles, ATM term structure |

## Signals

| Metric | What It Detects |
|--------|-----------------|
| ATM curvature | Rising while vol flat — early stress formation |
| 25-delta skew | Steepening — increased downside protection demand |
| Tail mass | Rising — crash risk being actively priced |
| Surface roughness | Spike — market dislocation |

## Regime Ladder

Calm → Pre-Stress → Elevated Stress → Acute Stress → Recovery

Each regime is derived from geometry metrics alone, not from price level or ATM vol.

## Stack

Python · Streamlit · Plotly · yfinance · SciPy

## References

- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*
- Breeden & Litzenberger (1978). Prices of State-Contingent Claims
- Gatheral & Jacquier (2014). SVI Parameterization

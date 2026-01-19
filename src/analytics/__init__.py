"""
Analytics Module
Arbitrage detection, geometry metrics, tail risk, and regime classification.
"""

from .arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_vertical_arbitrage,
    run_all_arbitrage_checks
)
from .geometry import (
    compute_skew,
    compute_curvature,
    compute_roughness,
    compute_all_geometry_metrics
)
from .tail_risk import (
    compute_tail_metrics,
    compute_distribution_moments,
    generate_tail_risk_report
)
from .temporal import (
    compute_delta_metrics,
    compute_rolling_stats,
    detect_regime_change
)
from .regime_classifier import RegimeClassifier

__all__ = [
    'check_butterfly_arbitrage',
    'check_calendar_arbitrage',
    'check_vertical_arbitrage',
    'run_all_arbitrage_checks',
    'compute_skew',
    'compute_curvature',
    'compute_roughness',
    'compute_all_geometry_metrics',
    'compute_tail_metrics',
    'compute_distribution_moments',
    'generate_tail_risk_report',
    'compute_delta_metrics',
    'compute_rolling_stats',
    'detect_regime_change',
    'RegimeClassifier'
]

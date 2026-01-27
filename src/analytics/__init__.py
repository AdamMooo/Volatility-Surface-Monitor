"""
Analytics Module
Geometry metrics, tail risk, and regime classification.
"""

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
from .regime_classifier import RegimeClassifier, GMMRegimeClassifier

__all__ = [
    'compute_skew',
    'compute_curvature',
    'compute_roughness',
    'compute_all_geometry_metrics',
    'compute_tail_metrics',
    'compute_distribution_moments',
    'generate_tail_risk_report',
    'RegimeClassifier',
    'GMMRegimeClassifier'
]

"""
Models Module
Option pricing, IV surface construction, and density extraction.
"""

from .black_scholes import (
    bs_price, 
    bs_delta, 
    bs_gamma, 
    bs_vega, 
    implied_volatility
)
from .surface import IVSurface
from .svi import svi_total_variance, svi_implied_vol, fit_svi_slice
from .density import extract_density, compute_moments, compute_tail_mass

__all__ = [
    'bs_price',
    'bs_delta',
    'bs_gamma',
    'bs_vega',
    'implied_volatility',
    'IVSurface',
    'svi_total_variance',
    'svi_implied_vol',
    'fit_svi_slice',
    'extract_density',
    'compute_moments',
    'compute_tail_mass'
]

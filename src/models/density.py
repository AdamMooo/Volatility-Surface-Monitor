"""
Risk-Neutral Density Module

Extract the market-implied probability distribution using Breeden-Litzenberger.

Theory:
The risk-neutral density f(K) is given by:
    f(K) = e^(rT) * d^2C/dK^2
    
where C(K) is the call price as a function of strike.
"""

from typing import Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
from loguru import logger

from .black_scholes import bs_price


def extract_density(
    call_prices: pd.Series,
    strikes: pd.Series,
    r: float,
    T: float,
    method: str = 'finite_diff'
) -> pd.DataFrame:
    """
    Extract risk-neutral density from call prices.
    
    Args:
        call_prices: Series of call option prices
        strikes: Series of corresponding strikes
        r: Risk-free rate
        T: Time to expiry in years
        method: 'finite_diff' or 'spline'
        
    Returns:
        DataFrame with strike and density columns
    """
    strikes = np.array(strikes)
    prices = np.array(call_prices)
    
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    prices = prices[sort_idx]
    
    discount_factor = np.exp(r * T)
    
    if method == 'finite_diff':
        density = _finite_diff_density(strikes, prices, discount_factor)
    elif method == 'spline':
        density = _spline_density(strikes, prices, discount_factor)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    density = np.maximum(density, 0)
    
    return pd.DataFrame({
        'strike': strikes,
        'density': density
    })


def _finite_diff_density(
    strikes: np.ndarray,
    prices: np.ndarray,
    discount_factor: float
) -> np.ndarray:
    """Compute density using finite differences."""
    n = len(strikes)
    density = np.zeros(n)
    
    for i in range(1, n - 1):
        dK = (strikes[i+1] - strikes[i-1]) / 2
        d2C_dK2 = (prices[i-1] - 2*prices[i] + prices[i+1]) / (dK**2)
        density[i] = discount_factor * d2C_dK2
    
    density[0] = density[1]
    density[-1] = density[-2]
    
    return density


def _spline_density(
    strikes: np.ndarray,
    prices: np.ndarray,
    discount_factor: float
) -> np.ndarray:
    """Compute density using spline interpolation and differentiation."""
    spline = UnivariateSpline(strikes, prices, s=0.01, k=4)
    
    d2C_dK2 = spline.derivative(2)(strikes)
    
    return discount_factor * d2C_dK2


def extract_density_from_iv(
    sigma_func: Callable,
    S: float,
    r: float,
    T: float,
    K_range: Tuple[float, float],
    num_points: int = 100
) -> pd.DataFrame:
    """
    Extract density from smooth IV surface.
    
    This is more stable than using discrete prices.
    
    Args:
        sigma_func: Function sigma(K) returning IV for strike K
        S: Spot price
        r: Risk-free rate
        T: Time to expiry
        K_range: (min_strike, max_strike)
        num_points: Number of points
        
    Returns:
        DataFrame with strike and density columns
    """
    strikes = np.linspace(K_range[0], K_range[1], num_points)
    
    prices = np.array([
        bs_price(S, K, T, r, sigma_func(K), 'call')
        for K in strikes
    ])
    
    return extract_density(
        pd.Series(prices),
        pd.Series(strikes),
        r,
        T,
        method='spline'
    )


def compute_moments(density: pd.DataFrame) -> Dict[str, float]:
    """
    Compute moments of the risk-neutral distribution.
    
    Args:
        density: DataFrame with strike and density columns
        
    Returns:
        Dictionary with mean, variance, skewness, kurtosis
    """
    K = density['strike'].values
    f = density['density'].values
    
    f = f / np.maximum(simpson(f, x=K), 1e-10)
    
    mean = simpson(K * f, x=K)
    
    variance = simpson((K - mean)**2 * f, x=K)
    std = np.sqrt(max(variance, 1e-10))
    
    skewness = simpson((K - mean)**3 * f, x=K) / (std**3)
    
    kurtosis = simpson((K - mean)**4 * f, x=K) / (std**4)
    
    return {
        'mean': float(mean),
        'variance': float(variance),
        'std': float(std),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'excess_kurtosis': float(kurtosis - 3)
    }


def compute_tail_mass(
    density: pd.DataFrame,
    spot: float,
    left_threshold: float = 0.9,
    right_threshold: float = 1.1
) -> Dict[str, float]:
    """
    Compute probability mass in the tails.
    
    Args:
        density: DataFrame with strike and density columns
        spot: Current spot price
        left_threshold: Left tail defined as K < left_threshold * S
        right_threshold: Right tail defined as K > right_threshold * S
        
    Returns:
        Dictionary with tail mass metrics
    """
    K = density['strike'].values
    f = density['density'].values
    
    total_mass = simpson(f, x=K)
    if total_mass < 1e-10:
        return {
            'left_tail_mass': 0.0,
            'right_tail_mass': 0.0,
            'tail_ratio': 1.0,
            'center_mass': 0.0
        }
    
    f = f / total_mass
    
    left_cutoff = left_threshold * spot
    right_cutoff = right_threshold * spot
    
    left_mask = K < left_cutoff
    right_mask = K > right_cutoff
    center_mask = ~(left_mask | right_mask)
    
    if np.sum(left_mask) > 1:
        left_tail = simpson(f[left_mask], x=K[left_mask])
    else:
        left_tail = 0.0
    
    if np.sum(right_mask) > 1:
        right_tail = simpson(f[right_mask], x=K[right_mask])
    else:
        right_tail = 0.0
    
    if np.sum(center_mask) > 1:
        center_mass = simpson(f[center_mask], x=K[center_mask])
    else:
        center_mass = 0.0
    
    tail_ratio = left_tail / max(right_tail, 1e-10)
    
    return {
        'left_tail_mass': float(left_tail),
        'right_tail_mass': float(right_tail),
        'tail_ratio': float(tail_ratio),
        'center_mass': float(center_mass),
        'left_threshold': left_threshold,
        'right_threshold': right_threshold
    }


def compute_implied_var(
    density: pd.DataFrame,
    confidence: float = 0.01
) -> float:
    """
    Compute implied Value-at-Risk from density.
    
    Args:
        density: DataFrame with strike and density columns
        confidence: VaR confidence level (e.g., 0.01 for 1%)
        
    Returns:
        Strike level at the given confidence percentile
    """
    K = density['strike'].values
    f = density['density'].values
    
    total = simpson(f, x=K)
    if total < 1e-10:
        return K[0]
    
    f = f / total
    
    cdf = np.zeros_like(f)
    for i in range(1, len(K)):
        cdf[i] = simpson(f[:i+1], x=K[:i+1])
    
    idx = np.searchsorted(cdf, confidence)
    if idx == 0:
        return K[0]
    if idx >= len(K):
        return K[-1]
    
    weight = (confidence - cdf[idx-1]) / (cdf[idx] - cdf[idx-1] + 1e-10)
    var = K[idx-1] + weight * (K[idx] - K[idx-1])
    
    return float(var)


def generate_density_report(
    surface,
    spot: float,
    maturities: list,
    rate: float = 0.05
) -> Dict[str, any]:
    """
    Generate comprehensive density analysis for multiple maturities.
    
    Args:
        surface: IVSurface object
        spot: Spot price
        maturities: List of maturities to analyze
        rate: Risk-free rate
        
    Returns:
        Dictionary with density analysis by maturity
    """
    results = {'by_maturity': {}, 'summary': {}}
    
    all_skewness = []
    all_kurtosis = []
    
    for T in maturities:
        K_range = (spot * 0.7, spot * 1.3)
        
        try:
            density = extract_density_from_iv(
                lambda K: surface.evaluate(K, T),
                S=spot,
                r=rate,
                T=T,
                K_range=K_range
            )
            
            moments = compute_moments(density)
            tail_5pct = compute_tail_mass(density, spot, 0.95, 1.05)
            tail_10pct = compute_tail_mass(density, spot, 0.90, 1.10)
            tail_20pct = compute_tail_mass(density, spot, 0.80, 1.20)
            
            var_1pct = compute_implied_var(density, 0.01)
            var_5pct = compute_implied_var(density, 0.05)
            
            results['by_maturity'][T] = {
                'density': density,
                'moments': moments,
                'left_tail_5pct': tail_5pct['left_tail_mass'],
                'left_tail_10pct': tail_10pct['left_tail_mass'],
                'left_tail_20pct': tail_20pct['left_tail_mass'],
                'right_tail_5pct': tail_5pct['right_tail_mass'],
                'right_tail_10pct': tail_10pct['right_tail_mass'],
                'implied_var_1pct': var_1pct,
                'implied_var_5pct': var_5pct,
                'tail_ratio': tail_10pct['tail_ratio']
            }
            
            all_skewness.append(moments['skewness'])
            all_kurtosis.append(moments['excess_kurtosis'])
            
        except Exception as e:
            logger.error(f"Failed density analysis for T={T}: {e}")
            continue
    
    if all_skewness:
        results['summary'] = {
            'avg_skewness': np.mean(all_skewness),
            'avg_excess_kurtosis': np.mean(all_kurtosis),
            'max_left_tail': max(
                r['left_tail_10pct'] 
                for r in results['by_maturity'].values()
            ),
            'maturities_analyzed': len(results['by_maturity'])
        }
    
    return results

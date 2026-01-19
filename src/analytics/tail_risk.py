"""
Tail Risk Analysis Module

Quantify the market's implied probability of extreme moves.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from loguru import logger


def compute_tail_metrics(
    density: pd.DataFrame,
    spot: float,
    thresholds: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive tail risk metrics from density.
    
    Args:
        density: DataFrame with strike and density columns
        spot: Current spot price
        thresholds: List of threshold percentages (e.g., [0.9, 0.95])
        
    Returns:
        Dictionary of tail metrics
    """
    if thresholds is None:
        thresholds = [0.80, 0.90, 0.95]
    
    K = density['strike'].values
    f = density['density'].values
    
    from scipy.integrate import simpson
    total = simpson(f, x=K)
    if total < 1e-10:
        return {'error': 'Invalid density'}
    
    f = f / total
    
    results = {}
    
    for thresh in thresholds:
        left_cutoff = thresh * spot
        right_cutoff = (2 - thresh) * spot
        
        left_mask = K < left_cutoff
        right_mask = K > right_cutoff
        
        if np.sum(left_mask) > 1:
            left_mass = simpson(f[left_mask], x=K[left_mask])
        else:
            left_mass = 0.0
        
        if np.sum(right_mask) > 1:
            right_mass = simpson(f[right_mask], x=K[right_mask])
        else:
            right_mass = 0.0
        
        pct = int((1 - thresh) * 100)
        results[f'left_tail_{pct}pct'] = left_mass
        results[f'right_tail_{pct}pct'] = right_mass
    
    left_10 = results.get('left_tail_10pct', 0)
    right_10 = results.get('right_tail_10pct', 1e-10)
    results['tail_ratio'] = left_10 / max(right_10, 1e-10)
    
    return results


def compute_distribution_moments(
    density: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute moments of the risk-neutral distribution.
    
    Args:
        density: DataFrame with strike and density columns
        
    Returns:
        Dictionary with mean, variance, skewness, kurtosis
    """
    from scipy.integrate import simpson
    
    K = density['strike'].values
    f = density['density'].values
    
    total = simpson(f, x=K)
    if total < 1e-10:
        return {'error': 'Invalid density'}
    
    f = f / total
    
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


def compute_implied_var(
    density: pd.DataFrame,
    confidence_levels: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute implied Value-at-Risk from density.
    
    Args:
        density: DataFrame with strike and density columns
        confidence_levels: List of confidence levels (e.g., [0.01, 0.05])
        
    Returns:
        Dictionary mapping confidence levels to VaR values
    """
    if confidence_levels is None:
        confidence_levels = [0.01, 0.05, 0.10]
    
    from scipy.integrate import simpson
    
    K = density['strike'].values
    f = density['density'].values
    
    total = simpson(f, x=K)
    if total < 1e-10:
        return {'error': 'Invalid density'}
    
    f = f / total
    
    cdf = np.zeros_like(f)
    for i in range(1, len(K)):
        cdf[i] = simpson(f[:i+1], x=K[:i+1])
    
    results = {}
    for conf in confidence_levels:
        idx = np.searchsorted(cdf, conf)
        if idx == 0:
            var = K[0]
        elif idx >= len(K):
            var = K[-1]
        else:
            weight = (conf - cdf[idx-1]) / (cdf[idx] - cdf[idx-1] + 1e-10)
            var = K[idx-1] + weight * (K[idx] - K[idx-1])
        
        pct = int(conf * 100)
        results[f'var_{pct}pct'] = float(var)
    
    return results


def compute_tail_premium(
    implied_tail_mass: float,
    historical_data: pd.DataFrame,
    spot: float,
    threshold: float = 0.90,
    lookback: int = 252
) -> float:
    """
    Compute tail risk premium: implied vs historical.
    
    Premium = (implied tail mass) / (historical tail frequency)
    
    Args:
        implied_tail_mass: Implied probability of tail event
        historical_data: DataFrame with price history
        spot: Current spot price
        threshold: Tail threshold (e.g., 0.90 = 10% down)
        lookback: Number of days for historical calculation
        
    Returns:
        Tail premium ratio
    """
    if 'close' not in historical_data.columns and 'Close' not in historical_data.columns:
        return 0.0
    
    price_col = 'close' if 'close' in historical_data.columns else 'Close'
    
    prices = historical_data[price_col].tail(lookback + 1)
    if len(prices) < 2:
        return 0.0
    
    returns = prices.pct_change().dropna()
    
    tail_return = threshold - 1
    historical_freq = (returns < tail_return).mean()
    
    if historical_freq < 1e-10:
        return implied_tail_mass / 0.01
    
    return implied_tail_mass / historical_freq


def generate_tail_risk_report(
    surface,
    spot: float,
    maturities: List[float],
    rate: float = 0.05
) -> Dict[str, Any]:
    """
    Generate comprehensive tail risk analysis for multiple maturities.
    
    Args:
        surface: IVSurface object
        spot: Spot price
        maturities: List of maturities to analyze
        rate: Risk-free rate
        
    Returns:
        Dictionary with tail risk analysis by maturity
    """
    from ..models.density import extract_density_from_iv
    
    results = {
        'by_maturity': {},
        'summary': {}
    }
    
    all_skewness = []
    all_kurtosis = []
    all_left_tail = []
    
    for T in maturities:
        K_range = (spot * 0.7, spot * 1.3)
        
        try:
            density = extract_density_from_iv(
                lambda K, t=T: surface.evaluate(K, t),
                S=spot,
                r=rate,
                T=T,
                K_range=K_range
            )
            
            moments = compute_distribution_moments(density)
            tail_metrics = compute_tail_metrics(density, spot)
            var_metrics = compute_implied_var(density)
            
            maturity_result = {
                **moments,
                **tail_metrics,
                **var_metrics,
                'maturity': T
            }
            
            results['by_maturity'][T] = maturity_result
            
            all_skewness.append(moments.get('skewness', 0))
            all_kurtosis.append(moments.get('excess_kurtosis', 0))
            all_left_tail.append(tail_metrics.get('left_tail_10pct', 0))
            
        except Exception as e:
            logger.warning(f"Failed tail analysis for T={T}: {e}")
            continue
    
    if all_skewness:
        term_structure = 'normal'
        if len(all_left_tail) >= 2:
            if all_left_tail[0] > all_left_tail[-1]:
                term_structure = 'inverted'
            elif all_left_tail[0] < all_left_tail[-1] * 0.5:
                term_structure = 'steep'
        
        results['summary'] = {
            'avg_skewness': float(np.mean(all_skewness)),
            'avg_excess_kurtosis': float(np.mean(all_kurtosis)),
            'max_left_tail': float(max(all_left_tail)),
            'avg_left_tail': float(np.mean(all_left_tail)),
            'tail_term_structure': term_structure,
            'maturities_analyzed': len(results['by_maturity'])
        }
    
    return results


def get_tail_risk_time_series(
    history: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Convert list of tail risk snapshots to time series DataFrame.
    
    Args:
        history: List of tail risk dictionaries with 'date' key
        
    Returns:
        DataFrame with tail metrics as columns
    """
    records = []
    
    for snapshot in history:
        record = {'date': snapshot.get('date')}
        
        summary = snapshot.get('summary', {})
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                record[key] = value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    if 'date' in df.columns:
        df = df.set_index('date')
    
    return df

"""
Geometry Metrics Module (CRITICAL FOR REGIME DETECTION)

This is the heart of the early warning system.

THE KEY INSIGHT:
Curvature changes BEFORE level changes. When sophisticated traders 
fear a crash, they buy OTM puts, steepening the skew and increasing 
curvature - often while ATM vol is still asleep.
"""

from typing import Dict, Any, Callable, Tuple, Optional, List
import numpy as np
import pandas as pd
from loguru import logger


def compute_skew(
    sigma_func: Callable,
    K: float,
    T: float,
    spot: float,
    method: str = 'central',
    h: float = 0.01
) -> float:
    """
    Compute skew (first derivative of IV with respect to strike).
    
    Args:
        sigma_func: Function sigma(K, T) returning IV
        K: Strike price
        T: Time to maturity
        spot: Spot price for relative step sizing
        method: 'central', 'forward', or 'backward'
        h: Step size as fraction of spot
        
    Returns:
        Skew value (dσ/dK)
    """
    dK = h * spot
    
    if method == 'central':
        sigma_up = sigma_func(K + dK, T)
        sigma_down = sigma_func(K - dK, T)
        return (sigma_up - sigma_down) / (2 * dK)
    elif method == 'forward':
        sigma_up = sigma_func(K + dK, T)
        sigma_mid = sigma_func(K, T)
        return (sigma_up - sigma_mid) / dK
    elif method == 'backward':
        sigma_mid = sigma_func(K, T)
        sigma_down = sigma_func(K - dK, T)
        return (sigma_mid - sigma_down) / dK
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_curvature(
    sigma_func: Callable,
    K: float,
    T: float,
    spot: float,
    method: str = 'central',
    h: float = 0.01
) -> float:
    """
    Compute curvature (second derivative of IV with respect to strike).
    
    This is the KEY metric for early warning.
    
    Args:
        sigma_func: Function sigma(K, T) returning IV
        K: Strike price
        T: Time to maturity
        spot: Spot price for relative step sizing
        method: 'central' difference method
        h: Step size as fraction of spot
        
    Returns:
        Curvature value (d²σ/dK²)
    """
    dK = h * spot
    
    sigma_up = sigma_func(K + dK, T)
    sigma_mid = sigma_func(K, T)
    sigma_down = sigma_func(K - dK, T)
    
    return (sigma_up - 2 * sigma_mid + sigma_down) / (dK ** 2)


def compute_term_slope(
    sigma_func: Callable,
    K: float,
    T: float,
    dT: float = 0.01
) -> float:
    """
    Compute term structure slope (dσ/dT).
    
    Args:
        sigma_func: Function sigma(K, T) returning IV
        K: Strike price
        T: Time to maturity
        dT: Step size in years
        
    Returns:
        Term structure slope
    """
    if T - dT < 0.01:
        sigma_up = sigma_func(K, T + dT)
        sigma_mid = sigma_func(K, T)
        return (sigma_up - sigma_mid) / dT
    
    sigma_up = sigma_func(K, T + dT)
    sigma_down = sigma_func(K, T - dT)
    return (sigma_up - sigma_down) / (2 * dT)


def compute_butterfly_spread(
    sigma_func: Callable,
    K: float,
    T: float,
    spot: float,
    width: float = 0.05
) -> float:
    """
    Compute discrete butterfly spread (proxy for curvature).
    
    Butterfly = σ(K-ΔK) + σ(K+ΔK) - 2*σ(K)
    
    Args:
        sigma_func: Function sigma(K, T) returning IV
        K: Strike price
        T: Time to maturity
        spot: Spot price
        width: Wing width as fraction of spot
        
    Returns:
        Butterfly spread value
    """
    dK = width * spot
    
    sigma_up = sigma_func(K + dK, T)
    sigma_mid = sigma_func(K, T)
    sigma_down = sigma_func(K - dK, T)
    
    return sigma_down + sigma_up - 2 * sigma_mid


def compute_25delta_skew(
    surface,
    T: float,
    spot: float
) -> float:
    """
    Compute 25-delta skew: σ(25Δ put) - σ(25Δ call).
    
    Approximates 25-delta strikes using simple moneyness offsets.
    
    Args:
        surface: IVSurface object
        T: Time to maturity
        spot: Spot price
        
    Returns:
        25-delta skew value
    """
    atm_vol = surface.evaluate(spot, T)
    
    put_25d_strike = spot * np.exp(-0.5 * atm_vol * np.sqrt(T))
    call_25d_strike = spot * np.exp(0.5 * atm_vol * np.sqrt(T))
    
    put_vol = surface.evaluate(put_25d_strike, T)
    call_vol = surface.evaluate(call_25d_strike, T)
    
    return put_vol - call_vol


def compute_10delta_skew(
    surface,
    T: float,
    spot: float
) -> float:
    """
    Compute 10-delta skew: σ(10Δ put) - σ(10Δ call).
    
    Args:
        surface: IVSurface object
        T: Time to maturity
        spot: Spot price
        
    Returns:
        10-delta skew value
    """
    atm_vol = surface.evaluate(spot, T)
    
    put_10d_strike = spot * np.exp(-1.0 * atm_vol * np.sqrt(T))
    call_10d_strike = spot * np.exp(1.0 * atm_vol * np.sqrt(T))
    
    put_vol = surface.evaluate(put_10d_strike, T)
    call_vol = surface.evaluate(call_10d_strike, T)
    
    return put_vol - call_vol


def compute_roughness(
    sigma_func: Callable,
    K_range: Tuple[float, float],
    T_range: Tuple[float, float],
    spot: float,
    grid_size: int = 20
) -> float:
    """
    Compute surface roughness (mean squared second derivative).
    
    High roughness indicates noisy/dislocated surface.
    
    Args:
        sigma_func: Function sigma(K, T) returning IV
        K_range: (min_strike, max_strike)
        T_range: (min_maturity, max_maturity)
        spot: Spot price
        grid_size: Number of points in each dimension
        
    Returns:
        Roughness metric
    """
    K_vals = np.linspace(K_range[0], K_range[1], grid_size)
    T_vals = np.linspace(max(T_range[0], 0.02), T_range[1], grid_size)
    
    curvatures = []
    
    for T in T_vals:
        for K in K_vals[1:-1]:
            try:
                curv = compute_curvature(sigma_func, K, T, spot)
                if np.isfinite(curv):
                    curvatures.append(curv ** 2)
            except Exception:
                continue
    
    if not curvatures:
        return 0.0
    
    return np.sqrt(np.mean(curvatures))


def compute_wing_curvature(
    sigma_func: Callable,
    T: float,
    spot: float,
    side: str = 'left'
) -> float:
    """
    Compute average curvature in one wing of the smile.
    
    Args:
        sigma_func: Function sigma(K, T) returning IV
        T: Time to maturity
        spot: Spot price
        side: 'left' (puts) or 'right' (calls)
        
    Returns:
        Average wing curvature
    """
    if side == 'left':
        K_range = np.linspace(spot * 0.8, spot * 0.95, 10)
    else:
        K_range = np.linspace(spot * 1.05, spot * 1.2, 10)
    
    curvatures = []
    for K in K_range:
        try:
            curv = compute_curvature(sigma_func, K, T, spot)
            if np.isfinite(curv):
                curvatures.append(curv)
        except Exception:
            continue
    
    return np.mean(curvatures) if curvatures else 0.0


def compute_all_geometry_metrics(
    surface,
    spot: float,
    maturities: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute all geometry metrics for the surface.
    
    This is the main function for regime detection signals.
    
    Args:
        surface: IVSurface object
        spot: Current spot price
        maturities: List of maturities to analyze (default: use surface maturities)
        
    Returns:
        Comprehensive dictionary of geometry metrics
    """
    if maturities is None:
        maturities = surface.maturities
    
    if not maturities:
        maturities = [0.08, 0.17, 0.25, 0.5, 1.0]
    
    def sigma_func(K, T):
        return surface.evaluate(K, T)
    
    results = {
        'level_metrics': {
            'atm_vol': {},
            'vol_term_structure': []
        },
        'skew_metrics': {
            'atm_skew': {},
            '25d_skew': {},
            '10d_skew': {},
            'skew_term_structure': []
        },
        'curvature_metrics': {
            'atm_curvature': {},
            'left_wing_curvature': {},
            'right_wing_curvature': {},
            'curvature_asymmetry': {},
            'butterfly_25d': {}
        },
        'roughness': 0.0,
        'spot': spot,
        'maturities': maturities
    }
    
    for T in maturities:
        try:
            atm_vol = surface.evaluate(spot, T)
            results['level_metrics']['atm_vol'][T] = atm_vol
            results['level_metrics']['vol_term_structure'].append({
                'maturity': T, 'vol': atm_vol
            })
        except Exception as e:
            logger.debug(f"Failed to get ATM vol for T={T}: {e}")
            continue
        
        try:
            atm_skew = compute_skew(sigma_func, spot, T, spot)
            results['skew_metrics']['atm_skew'][T] = atm_skew
        except Exception:
            pass
        
        try:
            skew_25d = compute_25delta_skew(surface, T, spot)
            results['skew_metrics']['25d_skew'][T] = skew_25d
        except Exception:
            pass
        
        try:
            skew_10d = compute_10delta_skew(surface, T, spot)
            results['skew_metrics']['10d_skew'][T] = skew_10d
        except Exception:
            pass
        
        try:
            atm_curv = compute_curvature(sigma_func, spot, T, spot)
            results['curvature_metrics']['atm_curvature'][T] = atm_curv
        except Exception:
            pass
        
        try:
            left_curv = compute_wing_curvature(sigma_func, T, spot, 'left')
            right_curv = compute_wing_curvature(sigma_func, T, spot, 'right')
            results['curvature_metrics']['left_wing_curvature'][T] = left_curv
            results['curvature_metrics']['right_wing_curvature'][T] = right_curv
            
            if right_curv != 0:
                results['curvature_metrics']['curvature_asymmetry'][T] = left_curv / right_curv
        except Exception:
            pass
        
        try:
            bf = compute_butterfly_spread(sigma_func, spot, T, spot, width=0.05)
            results['curvature_metrics']['butterfly_25d'][T] = bf
        except Exception:
            pass
    
    try:
        if len(maturities) >= 2:
            T_min = min(maturities)
            T_max = max(maturities)
            K_min = spot * 0.8
            K_max = spot * 1.2
            results['roughness'] = compute_roughness(
                sigma_func, (K_min, K_max), (T_min, T_max), spot
            )
    except Exception:
        pass
    
    atm_vols = results['level_metrics']['atm_vol']
    if len(atm_vols) >= 2:
        T_vals = sorted(atm_vols.keys())
        vols = [atm_vols[t] for t in T_vals]
        results['level_metrics']['average_vol'] = np.mean(vols)
        if len(T_vals) >= 2:
            results['level_metrics']['vol_term_structure_slope'] = (
                (vols[-1] - vols[0]) / (T_vals[-1] - T_vals[0])
            )
    
    skews = results['skew_metrics']['25d_skew']
    if len(skews) >= 2:
        T_vals = sorted(skews.keys())
        skew_vals = [skews[t] for t in T_vals]
        if len(T_vals) >= 2:
            results['skew_metrics']['skew_term_structure_slope'] = (
                (skew_vals[-1] - skew_vals[0]) / (T_vals[-1] - T_vals[0])
            )
    
    results['summary'] = _compute_summary_metrics(results)
    
    return results


def _compute_summary_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Compute summary statistics from detailed metrics."""
    summary = {}
    
    atm_vols = list(metrics['level_metrics']['atm_vol'].values())
    if atm_vols:
        summary['avg_atm_vol'] = np.mean(atm_vols)
        summary['front_atm_vol'] = atm_vols[0] if atm_vols else 0
    
    skews = list(metrics['skew_metrics']['25d_skew'].values())
    if skews:
        summary['avg_25d_skew'] = np.mean(skews)
        summary['front_25d_skew'] = skews[0] if skews else 0
    
    curvs = list(metrics['curvature_metrics']['atm_curvature'].values())
    if curvs:
        summary['avg_atm_curvature'] = np.mean(curvs)
        summary['front_atm_curvature'] = curvs[0] if curvs else 0
    
    bfs = list(metrics['curvature_metrics']['butterfly_25d'].values())
    if bfs:
        summary['avg_butterfly'] = np.mean(bfs)
        summary['front_butterfly'] = bfs[0] if bfs else 0
    
    summary['roughness'] = metrics.get('roughness', 0)
    
    return summary


def get_geometry_time_series(
    history: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Convert list of geometry metrics snapshots to time series DataFrame.
    
    Args:
        history: List of geometry metrics dictionaries with 'date' key
        
    Returns:
        DataFrame with metrics as columns and dates as index
    """
    records = []
    
    for snapshot in history:
        record = {'date': snapshot.get('date')}
        
        summary = snapshot.get('summary', {})
        for key, value in summary.items():
            record[key] = value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    if 'date' in df.columns:
        df = df.set_index('date')
    
    return df

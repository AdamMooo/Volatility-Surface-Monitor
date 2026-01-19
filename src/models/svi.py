"""
SVI Parameterization Module

Stochastic Volatility Inspired (SVI) model for volatility smiles.

The SVI parameterization:
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    
where:
    k = log(K/F) = log-moneyness
    w = sigma^2 * T = total implied variance
    a, b, rho, m, sigma = SVI parameters
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.optimize import minimize, differential_evolution
from loguru import logger


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float
) -> np.ndarray:
    """
    Calculate SVI total variance w(k).
    
    Args:
        k: Log-moneyness array (log(K/F))
        a: Overall variance level
        b: Slope of wings
        rho: Correlation (controls asymmetry)
        m: Shift parameter
        sigma: Smoothness parameter
        
    Returns:
        Total variance w = sigma^2 * T
    """
    k = np.atleast_1d(k)
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def svi_implied_vol(
    k: np.ndarray,
    T: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float
) -> np.ndarray:
    """
    Calculate SVI implied volatility.
    
    Args:
        k: Log-moneyness array
        T: Time to expiry in years
        a, b, rho, m, sigma: SVI parameters
        
    Returns:
        Implied volatility array
    """
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-8)
    return np.sqrt(w / T)


def check_svi_arbitrage(params: Dict[str, float]) -> Tuple[bool, str]:
    """
    Check Gatheral's no-butterfly-arbitrage conditions.
    
    Conditions:
    1. a + b*sigma*sqrt(1-rho^2) >= 0
    2. b >= 0
    3. |rho| < 1
    4. sigma > 0
    5. b*(1+|rho|) <= 4 (practical bound for calendar arbitrage)
    
    Args:
        params: Dictionary with SVI parameters
        
    Returns:
        Tuple of (is_valid, message)
    """
    a = params['a']
    b = params['b']
    rho = params['rho']
    m = params['m']
    sigma = params['sigma']
    
    if b < 0:
        return False, "b must be non-negative"
    
    if abs(rho) >= 1:
        return False, "|rho| must be less than 1"
    
    if sigma <= 0:
        return False, "sigma must be positive"
    
    min_variance = a + b * sigma * np.sqrt(1 - rho**2)
    if min_variance < 0:
        return False, f"Minimum variance {min_variance:.4f} is negative"
    
    wing_bound = b * (1 + abs(rho))
    if wing_bound > 4:
        return False, f"Wing slope {wing_bound:.4f} exceeds practical bound of 4"
    
    return True, "All arbitrage conditions satisfied"


def fit_svi_slice(
    k_array: np.ndarray,
    iv_array: np.ndarray,
    T: float,
    method: str = 'differential_evolution'
) -> Dict[str, float]:
    """
    Fit SVI parameters to a single maturity slice.
    
    Args:
        k_array: Log-moneyness array
        iv_array: Implied volatility array
        T: Time to expiry
        method: Optimization method
        
    Returns:
        Dictionary with fitted SVI parameters
    """
    w_market = iv_array**2 * T
    
    def objective(params):
        a, b, rho, m, sigma = params
        w_model = svi_total_variance(k_array, a, b, rho, m, sigma)
        return np.sum((w_model - w_market)**2)
    
    a_init = np.mean(w_market)
    b_init = 0.1
    rho_init = -0.3
    m_init = 0.0
    sigma_init = 0.1
    
    bounds = [
        (-0.5, 1.0),
        (0.001, 2.0),
        (-0.99, 0.99),
        (-0.5, 0.5),
        (0.001, 1.0)
    ]
    
    if method == 'differential_evolution':
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=500,
            tol=1e-8
        )
    else:
        x0 = [a_init, b_init, rho_init, m_init, sigma_init]
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )
    
    a, b, rho, m, sigma = result.x
    
    params = {
        'a': a,
        'b': b,
        'rho': rho,
        'm': m,
        'sigma': sigma,
        'T': T,
        'fit_error': result.fun,
        'success': result.success
    }
    
    is_valid, message = check_svi_arbitrage(params)
    params['arbitrage_free'] = is_valid
    params['arbitrage_message'] = message
    
    return params


def fit_svi_surface(
    surface_data,
    spot: float,
    rate: float = 0.05
) -> Dict[float, Dict[str, float]]:
    """
    Fit SVI to each maturity slice of a surface.
    
    Args:
        surface_data: DataFrame with strike, time_to_expiry, iv columns
        spot: Spot price
        rate: Risk-free rate
        
    Returns:
        Dictionary mapping maturity to SVI parameters
    """
    import pandas as pd
    
    results = {}
    
    maturities = surface_data['time_to_expiry'].unique()
    
    for T in sorted(maturities):
        slice_data = surface_data[surface_data['time_to_expiry'] == T]
        
        if len(slice_data) < 5:
            logger.warning(f"Skipping maturity {T}: insufficient data points")
            continue
        
        forward = spot * np.exp(rate * T)
        k = np.log(slice_data['strike'].values / forward)
        iv = slice_data['iv'].values
        
        try:
            params = fit_svi_slice(k, iv, T)
            results[T] = params
            logger.debug(f"Fitted SVI for T={T:.3f}: a={params['a']:.4f}, "
                        f"b={params['b']:.4f}, rho={params['rho']:.4f}")
        except Exception as e:
            logger.error(f"Failed to fit SVI for T={T}: {e}")
    
    return results


def create_svi_surface(
    svi_params: Dict[float, Dict[str, float]],
    spot: float,
    rate: float = 0.05
):
    """
    Create a callable SVI surface from fitted parameters.
    
    Args:
        svi_params: Dictionary of SVI parameters by maturity
        spot: Spot price
        rate: Risk-free rate
        
    Returns:
        Callable that evaluates IV at (strike, maturity)
    """
    maturities = sorted(svi_params.keys())
    
    def evaluate(K: float, T: float) -> float:
        forward = spot * np.exp(rate * T)
        k = np.log(K / forward)
        
        if T in svi_params:
            p = svi_params[T]
            return svi_implied_vol(k, T, p['a'], p['b'], p['rho'], p['m'], p['sigma'])[0]
        
        T_below = max([t for t in maturities if t <= T], default=None)
        T_above = min([t for t in maturities if t >= T], default=None)
        
        if T_below is None:
            T_below = maturities[0]
        if T_above is None:
            T_above = maturities[-1]
        
        if T_below == T_above:
            p = svi_params[T_below]
            return svi_implied_vol(k, T, p['a'], p['b'], p['rho'], p['m'], p['sigma'])[0]
        
        weight = (T - T_below) / (T_above - T_below)
        
        p1 = svi_params[T_below]
        p2 = svi_params[T_above]
        
        iv1 = svi_implied_vol(k, T_below, p1['a'], p1['b'], p1['rho'], p1['m'], p1['sigma'])[0]
        iv2 = svi_implied_vol(k, T_above, p2['a'], p2['b'], p2['rho'], p2['m'], p2['sigma'])[0]
        
        var1 = iv1**2 * T_below
        var2 = iv2**2 * T_above
        var_interp = var1 + weight * (var2 - var1)
        
        return np.sqrt(var_interp / T)
    
    return evaluate

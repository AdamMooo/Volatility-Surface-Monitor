"""
Black-Scholes Module

Responsibilities:
1. Implement BS pricing formula for calls and puts
2. Implement robust IV inversion (Newton-Raphson with fallbacks)
3. Handle edge cases (deep ITM/OTM, near-expiry)
4. Compute Greeks if needed
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton
from typing import Optional, Literal
from loguru import logger


# Constants
MIN_VOL = 0.001    # 0.1% minimum volatility
MAX_VOL = 5.0      # 500% maximum volatility
MIN_TIME = 1e-10   # Minimum time to expiry


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in Black-Scholes formula."""
    if T <= MIN_TIME or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in Black-Scholes formula."""
    if T <= MIN_TIME or sigma <= 0:
        return 0.0
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    if T <= MIN_TIME:
        if option_type == 'call':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    
    if sigma <= 0:
        sigma = MIN_VOL
    
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(0.0, price)


def bs_delta(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate option delta (sensitivity to spot price).
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        option_type: 'call' or 'put'
        
    Returns:
        Delta value
    """
    if T <= MIN_TIME:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = _d1(S, K, T, r, sigma)
    
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option gamma (second derivative w.r.t. spot).
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        
    Returns:
        Gamma value
    """
    if T <= MIN_TIME or sigma <= 0:
        return 0.0
    
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate option vega (sensitivity to volatility).
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        
    Returns:
        Vega value (per 1 unit of volatility, not per 1%)
    """
    if T <= MIN_TIME:
        return 0.0
    
    d1 = _d1(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d1)


def bs_theta(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float,
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    Calculate option theta (time decay per year).
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        option_type: 'call' or 'put'
        
    Returns:
        Theta value (per year)
    """
    if T <= MIN_TIME:
        return 0.0
    
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    
    if option_type == 'call':
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return term1 + term2


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal['call', 'put'] = 'call',
    method: str = 'newton',
    max_iter: int = 100,
    tol: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility from option price.
    
    Uses Newton-Raphson method with Brent's method as fallback.
    
    Args:
        price: Market price of the option
        S: Spot price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        option_type: 'call' or 'put'
        method: 'newton' or 'brent'
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Implied volatility, or None if calculation fails
    """
    if T <= MIN_TIME:
        return None
    
    if price <= 0:
        return None
    
    intrinsic = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
    if price < intrinsic - tol:
        return None
    
    max_price = S if option_type == 'call' else K * np.exp(-r * T)
    if price > max_price + tol:
        return None
    
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price
    
    def objective_with_vega(sigma):
        theo_price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)
        return theo_price - price, vega
    
    sigma_init = _initial_vol_guess(price, S, K, T, r, option_type)
    
    if method == 'newton':
        try:
            def newton_func(sigma):
                theo, vega = objective_with_vega(sigma)
                if abs(vega) < 1e-10:
                    return theo, 1e-10
                return theo, vega
            
            sigma = sigma_init
            for _ in range(max_iter):
                theo, vega = newton_func(sigma)
                if abs(theo) < tol:
                    if MIN_VOL <= sigma <= MAX_VOL:
                        return sigma
                    break
                
                if abs(vega) < 1e-10:
                    break
                
                sigma_new = sigma - theo / vega
                sigma_new = max(MIN_VOL, min(MAX_VOL, sigma_new))
                
                if abs(sigma_new - sigma) < tol:
                    if MIN_VOL <= sigma_new <= MAX_VOL:
                        return sigma_new
                    break
                
                sigma = sigma_new
        except Exception:
            pass
    
    try:
        sigma = brentq(objective, MIN_VOL, MAX_VOL, xtol=tol, maxiter=max_iter)
        return sigma
    except ValueError:
        pass
    except Exception as e:
        logger.debug(f"Brent method failed: {e}")
    
    return None


def _initial_vol_guess(
    price: float, 
    S: float, 
    K: float, 
    T: float, 
    r: float,
    option_type: str
) -> float:
    """
    Generate initial volatility guess for Newton-Raphson.
    
    Uses Brenner-Subrahmanyam approximation for ATM options,
    adjusted for moneyness.
    """
    F = S * np.exp(r * T)
    moneyness = K / F
    
    atm_approx = price / (0.4 * S * np.sqrt(T)) if T > 0 else 0.2
    
    if 0.95 <= moneyness <= 1.05:
        return max(MIN_VOL, min(MAX_VOL, atm_approx))
    
    if option_type == 'call' and moneyness > 1:
        atm_approx *= (1 + 0.5 * (moneyness - 1))
    elif option_type == 'put' and moneyness < 1:
        atm_approx *= (1 + 0.5 * (1 - moneyness))
    
    return max(MIN_VOL, min(MAX_VOL, atm_approx))


def compute_iv_surface(
    option_data,
    spot: float,
    rate: float
):
    """
    Compute implied volatilities for all options in a DataFrame.
    
    Args:
        option_data: DataFrame with option chain data
        spot: Spot price
        rate: Risk-free rate
        
    Returns:
        DataFrame with 'implied_volatility' column added
    """
    import pandas as pd
    
    df = option_data.copy()
    ivs = []
    
    for _, row in df.iterrows():
        price = row.get('mid_price', row.get('lastPrice', 0))
        strike = row['strike']
        T = row.get('time_to_expiry', row.get('days_to_expiry', 30) / 365)
        opt_type = row.get('option_type', 'call')
        r = row.get('risk_free_rate', rate)
        
        iv = implied_volatility(
            price=price,
            S=spot,
            K=strike,
            T=T,
            r=r,
            option_type=opt_type
        )
        
        ivs.append(iv)
    
    df['implied_volatility'] = ivs
    
    return df


def vectorized_bs_price(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    option_type: str = 'call'
) -> np.ndarray:
    """
    Vectorized Black-Scholes pricing for arrays of inputs.
    
    All inputs should be numpy arrays of the same shape.
    
    Returns:
        Array of option prices
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)
    
    T = np.maximum(T, MIN_TIME)
    sigma = np.maximum(sigma, MIN_VOL)
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if option_type == 'call':
        prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return np.maximum(0.0, prices)

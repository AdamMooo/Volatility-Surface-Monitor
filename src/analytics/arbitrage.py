"""
Arbitrage Detection Module

Check for no-arbitrage violations in the option surface.

Three Types of Arbitrage in Options:
1. STRIKE (Butterfly) ARBITRAGE - C(K) must be convex in K
2. CALENDAR (Horizontal) ARBITRAGE - option value must increase with T
3. VERTICAL ARBITRAGE - Call spread must be positive
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


def check_butterfly_arbitrage(
    prices: pd.Series,
    strikes: pd.Series,
    tolerance: float = 0.0
) -> pd.DataFrame:
    """
    Check for butterfly (strike) arbitrage violations.
    
    For calls at fixed T: C(K) must be convex in K
    C(K-dK) - 2*C(K) + C(K+dK) >= 0
    
    Args:
        prices: Option prices
        strikes: Corresponding strikes
        tolerance: Allowed violation tolerance
        
    Returns:
        DataFrame with violations flagged
    """
    strikes = np.array(strikes)
    prices = np.array(prices)
    
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    prices = prices[sort_idx]
    
    violations = []
    
    for i in range(1, len(strikes) - 1):
        K_low = strikes[i-1]
        K_mid = strikes[i]
        K_high = strikes[i+1]
        
        C_low = prices[i-1]
        C_mid = prices[i]
        C_high = prices[i+1]
        
        dK1 = K_mid - K_low
        dK2 = K_high - K_mid
        
        if dK1 > 0 and dK2 > 0:
            weight = dK1 / (dK1 + dK2)
            interpolated = C_low * (1 - weight) + C_high * weight
            butterfly = C_low - 2*C_mid + C_high
            
            if butterfly < -tolerance:
                severity = abs(butterfly) / C_mid if C_mid > 0 else abs(butterfly)
                violations.append({
                    'strike': K_mid,
                    'strike_low': K_low,
                    'strike_high': K_high,
                    'butterfly_value': butterfly,
                    'severity': severity,
                    'type': 'butterfly'
                })
    
    return pd.DataFrame(violations)


def check_calendar_arbitrage(
    surface_data: pd.DataFrame,
    tolerance: float = 0.0
) -> pd.DataFrame:
    """
    Check for calendar (horizontal) arbitrage violations.
    
    For fixed K: option value must increase with T
    C(K, T1) <= C(K, T2) for T1 < T2
    
    Args:
        surface_data: DataFrame with strike, time_to_expiry, price columns
        tolerance: Allowed violation tolerance
        
    Returns:
        DataFrame with violations flagged
    """
    violations = []
    
    price_col = None
    for col in ['mid_price', 'price', 'iv']:
        if col in surface_data.columns:
            price_col = col
            break
    
    if price_col is None:
        logger.warning("No price column found for calendar arbitrage check")
        return pd.DataFrame()
    
    strikes = surface_data['strike'].unique()
    
    for K in strikes:
        slice_data = surface_data[surface_data['strike'] == K].sort_values('time_to_expiry')
        
        if len(slice_data) < 2:
            continue
        
        maturities = slice_data['time_to_expiry'].values
        prices = slice_data[price_col].values
        
        for i in range(len(maturities) - 1):
            T1, T2 = maturities[i], maturities[i+1]
            P1, P2 = prices[i], prices[i+1]
            
            if P1 > P2 + tolerance:
                severity = (P1 - P2) / P2 if P2 > 0 else (P1 - P2)
                violations.append({
                    'strike': K,
                    'T_short': T1,
                    'T_long': T2,
                    'price_short': P1,
                    'price_long': P2,
                    'violation_amount': P1 - P2,
                    'severity': severity,
                    'type': 'calendar'
                })
    
    return pd.DataFrame(violations)


def check_vertical_arbitrage(
    prices: pd.Series,
    strikes: pd.Series,
    option_type: str = 'call',
    tolerance: float = 0.0
) -> pd.DataFrame:
    """
    Check for vertical (monotonicity) arbitrage violations.
    
    For calls: dC/dK <= 0 (higher strike = lower price)
    For puts: dP/dK >= 0 (higher strike = higher price)
    
    Args:
        prices: Option prices
        strikes: Corresponding strikes
        option_type: 'call' or 'put'
        tolerance: Allowed violation tolerance
        
    Returns:
        DataFrame with violations flagged
    """
    strikes = np.array(strikes)
    prices = np.array(prices)
    
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    prices = prices[sort_idx]
    
    violations = []
    
    for i in range(len(strikes) - 1):
        K1, K2 = strikes[i], strikes[i+1]
        P1, P2 = prices[i], prices[i+1]
        
        if option_type == 'call':
            if P2 > P1 + tolerance:
                violations.append({
                    'strike_low': K1,
                    'strike_high': K2,
                    'price_low_strike': P1,
                    'price_high_strike': P2,
                    'violation_amount': P2 - P1,
                    'severity': (P2 - P1) / P1 if P1 > 0 else (P2 - P1),
                    'type': 'vertical_call'
                })
        else:
            if P1 > P2 + tolerance:
                violations.append({
                    'strike_low': K1,
                    'strike_high': K2,
                    'price_low_strike': P1,
                    'price_high_strike': P2,
                    'violation_amount': P1 - P2,
                    'severity': (P1 - P2) / P2 if P2 > 0 else (P1 - P2),
                    'type': 'vertical_put'
                })
    
    return pd.DataFrame(violations)


def check_iv_butterfly_arbitrage(
    ivs: pd.Series,
    strikes: pd.Series,
    T: float,
    tolerance: float = 0.0
) -> pd.DataFrame:
    """
    Check butterfly arbitrage in IV space.
    
    Total variance w = sigma^2 * T must be convex in log-moneyness.
    
    Args:
        ivs: Implied volatilities
        strikes: Corresponding strikes
        T: Time to expiry
        tolerance: Allowed tolerance
        
    Returns:
        DataFrame with violations
    """
    strikes = np.array(strikes)
    ivs = np.array(ivs)
    
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    ivs = ivs[sort_idx]
    
    total_var = ivs**2 * T
    
    violations = []
    
    for i in range(1, len(strikes) - 1):
        butterfly = total_var[i-1] - 2*total_var[i] + total_var[i+1]
        
        if butterfly < -tolerance:
            violations.append({
                'strike': strikes[i],
                'iv': ivs[i],
                'total_variance': total_var[i],
                'butterfly_value': butterfly,
                'severity': abs(butterfly),
                'type': 'iv_butterfly'
            })
    
    return pd.DataFrame(violations)


def run_all_arbitrage_checks(
    option_data: pd.DataFrame,
    surface = None,
    tolerance: float = 0.001,
    use_bid_ask: bool = True
) -> Dict[str, Any]:
    """
    Run all arbitrage checks and return comprehensive report.
    
    Args:
        option_data: DataFrame with option chain data
        surface: Optional IVSurface object
        tolerance: Violation tolerance
        use_bid_ask: If True, use bid/ask for executable arbitrage check
        
    Returns:
        Dictionary with all violations and summary
    """
    results = {
        'butterfly_violations': [],
        'calendar_violations': [],
        'vertical_violations': [],
        'summary': {}
    }
    
    maturities = option_data['time_to_expiry'].unique() if 'time_to_expiry' in option_data.columns else []
    
    for T in maturities:
        slice_data = option_data[option_data['time_to_expiry'] == T]
        
        for opt_type in ['call', 'put']:
            type_data = slice_data[slice_data['option_type'] == opt_type]
            
            if len(type_data) < 3:
                continue
            
            price_col = 'mid_price' if 'mid_price' in type_data.columns else 'price'
            if price_col not in type_data.columns:
                continue
            
            butterfly_viol = check_butterfly_arbitrage(
                type_data[price_col],
                type_data['strike'],
                tolerance
            )
            if not butterfly_viol.empty:
                butterfly_viol['maturity'] = T
                butterfly_viol['option_type'] = opt_type
                results['butterfly_violations'].append(butterfly_viol)
            
            vertical_viol = check_vertical_arbitrage(
                type_data[price_col],
                type_data['strike'],
                opt_type,
                tolerance
            )
            if not vertical_viol.empty:
                vertical_viol['maturity'] = T
                vertical_viol['option_type'] = opt_type
                results['vertical_violations'].append(vertical_viol)
    
    calendar_viol = check_calendar_arbitrage(option_data, tolerance)
    if not calendar_viol.empty:
        results['calendar_violations'].append(calendar_viol)
    
    butterfly_df = pd.concat(results['butterfly_violations']) if results['butterfly_violations'] else pd.DataFrame()
    calendar_df = pd.concat(results['calendar_violations']) if results['calendar_violations'] else pd.DataFrame()
    vertical_df = pd.concat(results['vertical_violations']) if results['vertical_violations'] else pd.DataFrame()
    
    results['butterfly_violations'] = butterfly_df
    results['calendar_violations'] = calendar_df
    results['vertical_violations'] = vertical_df
    
    total_violations = len(butterfly_df) + len(calendar_df) + len(vertical_df)
    
    max_severity = 0
    for df in [butterfly_df, calendar_df, vertical_df]:
        if not df.empty and 'severity' in df.columns:
            max_severity = max(max_severity, df['severity'].max())
    
    results['summary'] = {
        'total_violations': total_violations,
        'butterfly_count': len(butterfly_df),
        'calendar_count': len(calendar_df),
        'vertical_count': len(vertical_df),
        'max_severity': max_severity,
        'is_arbitrage_free': total_violations == 0,
        'affected_maturities': list(set(
            list(butterfly_df['maturity'].unique() if not butterfly_df.empty else []) +
            list(vertical_df['maturity'].unique() if not vertical_df.empty else [])
        ))
    }
    
    return results


def get_arbitrage_heatmap_data(
    violations: Dict[str, Any],
    option_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare data for arbitrage violation heatmap visualization.
    
    Args:
        violations: Output from run_all_arbitrage_checks
        option_data: Original option data
        
    Returns:
        DataFrame suitable for heatmap plotting
    """
    if 'strike' not in option_data.columns or 'time_to_expiry' not in option_data.columns:
        return pd.DataFrame()
    
    strikes = sorted(option_data['strike'].unique())
    maturities = sorted(option_data['time_to_expiry'].unique())
    
    heatmap_data = pd.DataFrame(
        0,
        index=strikes,
        columns=maturities
    )
    
    for viol_type in ['butterfly_violations', 'vertical_violations']:
        df = violations.get(viol_type, pd.DataFrame())
        if df.empty:
            continue
        
        strike_col = 'strike' if 'strike' in df.columns else 'strike_low'
        if strike_col in df.columns and 'maturity' in df.columns:
            for _, row in df.iterrows():
                K = row[strike_col]
                T = row['maturity']
                if K in heatmap_data.index and T in heatmap_data.columns:
                    heatmap_data.loc[K, T] += 1
    
    return heatmap_data

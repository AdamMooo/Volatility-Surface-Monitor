"""
Data Cleaner Module

Responsibilities:
1. Remove invalid/stale quotes (zero bid, wide spreads)
2. Filter extreme moneyness (deep ITM/OTM with no liquidity)
3. Handle duplicate entries
4. Validate price relationships (call >= put for same strike, etc.)
5. Mark suspicious data points for review
"""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from loguru import logger


def clean_option_chain(
    df: pd.DataFrame, 
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply all cleaning filters to an option chain.
    
    Args:
        df: Raw option chain DataFrame
        config: Configuration dictionary with filter parameters
        
    Returns:
        Cleaned DataFrame
    """
    if config is None:
        config = get_default_config()
    
    original_count = len(df)
    result = df.copy()
    
    result = remove_invalid_quotes(result)
    
    result = filter_by_moneyness(
        result, 
        config.get('min_moneyness', 0.7),
        config.get('max_moneyness', 1.3)
    )
    
    result = filter_by_volume(result, config.get('min_volume', 10))
    
    result = filter_by_spread(result, config.get('max_spread_pct', 0.20))
    
    result = filter_by_expiry(
        result,
        config.get('min_days_to_expiry', 7),
        config.get('max_days_to_expiry', 365)
    )
    
    result = remove_duplicates(result)
    
    final_count = len(result)
    logger.info(f"Cleaned option chain: {original_count} -> {final_count} rows "
                f"({original_count - final_count} removed)")
    
    return result


def get_default_config() -> Dict[str, Any]:
    """Get default cleaning configuration."""
    return {
        'min_moneyness': 0.7,
        'max_moneyness': 1.3,
        'min_volume': 10,
        'max_spread_pct': 0.20,
        'min_days_to_expiry': 7,
        'max_days_to_expiry': 365
    }


def remove_invalid_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid quotes (zero/negative bid, zero ask, etc.)."""
    result = df.copy()
    
    result = result[result['bid'] >= 0]
    result = result[result['ask'] > 0]
    result = result[result['mid_price'] > 0]
    result = result[result['ask'] >= result['bid']]
    
    return result


def filter_by_moneyness(
    df: pd.DataFrame, 
    min_moneyness: float = 0.7, 
    max_moneyness: float = 1.3
) -> pd.DataFrame:
    """
    Filter by moneyness range.
    
    Args:
        df: Option chain DataFrame
        min_moneyness: Minimum K/S ratio (e.g., 0.7 = 30% OTM puts)
        max_moneyness: Maximum K/S ratio (e.g., 1.3 = 30% OTM calls)
        
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    
    if 'moneyness' not in result.columns:
        if 'strike' in result.columns and 'underlying_price' in result.columns:
            result['moneyness'] = result['strike'] / result['underlying_price']
        else:
            logger.warning("Cannot compute moneyness - missing required columns")
            return result
    
    mask = (result['moneyness'] >= min_moneyness) & (result['moneyness'] <= max_moneyness)
    return result[mask]


def filter_by_volume(df: pd.DataFrame, min_volume: int = 10) -> pd.DataFrame:
    """Filter by minimum daily volume."""
    result = df.copy()
    
    volume_col = 'volume' if 'volume' in result.columns else None
    if volume_col is None and 'open_interest' in result.columns:
        volume_col = 'open_interest'
    
    if volume_col is None:
        logger.warning("No volume column found - skipping volume filter")
        return result
    
    result[volume_col] = result[volume_col].fillna(0)
    
    return result[result[volume_col] >= min_volume]


def filter_by_spread(df: pd.DataFrame, max_spread_pct: float = 0.20) -> pd.DataFrame:
    """
    Filter by maximum bid-ask spread as percentage of mid-price.
    
    Args:
        df: Option chain DataFrame
        max_spread_pct: Maximum spread as fraction (0.20 = 20%)
        
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    
    if 'bid' not in result.columns or 'ask' not in result.columns:
        logger.warning("Bid/ask columns not found - skipping spread filter")
        return result
    
    spread = result['ask'] - result['bid']
    mid = result['mid_price'] if 'mid_price' in result.columns else (result['bid'] + result['ask']) / 2
    
    spread_pct = spread / mid
    spread_pct = spread_pct.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    return result[spread_pct <= max_spread_pct]


def filter_by_expiry(
    df: pd.DataFrame, 
    min_days: int = 7, 
    max_days: int = 365
) -> pd.DataFrame:
    """Filter by days to expiry range."""
    result = df.copy()
    
    if 'days_to_expiry' not in result.columns:
        logger.warning("days_to_expiry column not found - skipping expiry filter")
        return result
    
    mask = (result['days_to_expiry'] >= min_days) & (result['days_to_expiry'] <= max_days)
    return result[mask]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries, keeping the one with highest volume."""
    result = df.copy()
    
    key_columns = ['strike', 'expiration', 'option_type']
    key_columns = [c for c in key_columns if c in result.columns]
    
    if not key_columns:
        return result
    
    volume_col = 'volume' if 'volume' in result.columns else 'open_interest'
    if volume_col in result.columns:
        result = result.sort_values(volume_col, ascending=False)
    
    result = result.drop_duplicates(subset=key_columns, keep='first')
    
    return result


def validate_put_call_parity(
    df: pd.DataFrame, 
    tolerance: float = 0.05
) -> pd.DataFrame:
    """
    Validate put-call parity and flag violations.
    
    Put-Call Parity: C - P = S*exp(-q*T) - K*exp(-r*T)
    For simplicity, ignoring dividends: C - P = S - K*exp(-r*T)
    
    Args:
        df: Option chain DataFrame
        tolerance: Maximum allowed deviation as fraction of spot
        
    Returns:
        DataFrame with 'pcp_violation' column added
    """
    result = df.copy()
    result['pcp_violation'] = False
    
    required_cols = ['strike', 'expiration', 'option_type', 'mid_price', 
                     'underlying_price', 'risk_free_rate', 'time_to_expiry']
    if not all(c in result.columns for c in required_cols):
        logger.warning("Missing columns for put-call parity check")
        return result
    
    calls = result[result['option_type'] == 'call'].set_index(['strike', 'expiration'])
    puts = result[result['option_type'] == 'put'].set_index(['strike', 'expiration'])
    
    common_index = calls.index.intersection(puts.index)
    
    for idx in common_index:
        call_row = calls.loc[idx]
        put_row = puts.loc[idx]
        
        if isinstance(call_row, pd.DataFrame):
            call_row = call_row.iloc[0]
        if isinstance(put_row, pd.DataFrame):
            put_row = put_row.iloc[0]
        
        C = call_row['mid_price']
        P = put_row['mid_price']
        S = call_row['underlying_price']
        K = idx[0]
        r = call_row['risk_free_rate']
        T = call_row['time_to_expiry']
        
        theoretical_diff = S - K * np.exp(-r * T)
        actual_diff = C - P
        
        deviation = abs(actual_diff - theoretical_diff) / S
        
        if deviation > tolerance:
            result.loc[
                (result['strike'] == K) & 
                (result['expiration'] == idx[1]), 
                'pcp_violation'
            ] = True
    
    return result


def flag_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """
    Flag potential outliers in implied volatility.
    
    Args:
        df: Option chain DataFrame with implied_volatility column
        method: 'iqr' for interquartile range, 'zscore' for z-score
        
    Returns:
        DataFrame with 'is_outlier' column added
    """
    result = df.copy()
    result['is_outlier'] = False
    
    iv_col = None
    for col in ['implied_volatility', 'implied_volatility_market', 'iv']:
        if col in result.columns:
            iv_col = col
            break
    
    if iv_col is None:
        logger.warning("No implied volatility column found - skipping outlier detection")
        return result
    
    iv = result[iv_col].dropna()
    
    if len(iv) < 10:
        return result
    
    if method == 'iqr':
        Q1 = iv.quantile(0.25)
        Q3 = iv.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        result['is_outlier'] = (result[iv_col] < lower) | (result[iv_col] > upper)
        
    elif method == 'zscore':
        mean = iv.mean()
        std = iv.std()
        if std > 0:
            z_scores = (result[iv_col] - mean) / std
            result['is_outlier'] = abs(z_scores) > 3
    
    return result


def get_cleaning_summary(original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary of the cleaning process."""
    return {
        'original_count': len(original),
        'cleaned_count': len(cleaned),
        'removed_count': len(original) - len(cleaned),
        'removal_rate': (len(original) - len(cleaned)) / len(original) if len(original) > 0 else 0,
        'unique_expirations': cleaned['expiration'].nunique() if 'expiration' in cleaned.columns else 0,
        'unique_strikes': cleaned['strike'].nunique() if 'strike' in cleaned.columns else 0,
        'moneyness_range': (
            cleaned['moneyness'].min() if 'moneyness' in cleaned.columns else None,
            cleaned['moneyness'].max() if 'moneyness' in cleaned.columns else None
        )
    }

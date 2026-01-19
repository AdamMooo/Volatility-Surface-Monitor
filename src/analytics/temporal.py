"""
Temporal Dynamics Module (CRITICAL FOR REGIME DETECTION)

Track how surface metrics change over time to detect regime transitions.

THE KEY INSIGHT:
It's not just the LEVEL of curvature that matters - it's the CHANGE.
A sharp increase in curvature, even from low levels, signals 
growing stress before it manifests in ATM vol.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from loguru import logger


def compute_delta_metrics(
    current: Dict[str, Any],
    previous: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute changes in metrics between two snapshots.
    
    Args:
        current: Current metrics snapshot
        previous: Previous metrics snapshot
        
    Returns:
        Dictionary of delta metrics
    """
    deltas = {}
    
    current_summary = current.get('summary', {})
    previous_summary = previous.get('summary', {})
    
    metrics_to_track = [
        'avg_atm_vol',
        'front_atm_vol',
        'avg_25d_skew',
        'front_25d_skew',
        'avg_atm_curvature',
        'front_atm_curvature',
        'avg_butterfly',
        'front_butterfly',
        'roughness'
    ]
    
    for metric in metrics_to_track:
        curr_val = current_summary.get(metric, 0)
        prev_val = previous_summary.get(metric, 0)
        
        if curr_val is not None and prev_val is not None:
            deltas[f'delta_{metric}'] = curr_val - prev_val
            
            if prev_val != 0:
                deltas[f'pct_change_{metric}'] = (curr_val - prev_val) / abs(prev_val)
    
    return deltas


def compute_rolling_stats(
    history: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Compute rolling statistics for metric time series.
    
    Args:
        history: DataFrame with metrics as columns, dates as index
        window: Rolling window size
        
    Returns:
        DataFrame with rolling mean, std, and z-scores
    """
    result = history.copy()
    
    for col in history.columns:
        if history[col].dtype in [np.float64, np.int64, float, int]:
            result[f'{col}_rolling_mean'] = history[col].rolling(window).mean()
            result[f'{col}_rolling_std'] = history[col].rolling(window).std()
            
            mean = result[f'{col}_rolling_mean']
            std = result[f'{col}_rolling_std'].replace(0, np.nan)
            result[f'{col}_zscore'] = (history[col] - mean) / std
    
    return result


def compute_percentiles(
    current_value: float,
    history: pd.Series,
    lookback: Optional[int] = None
) -> float:
    """
    Compute percentile of current value in historical distribution.
    
    Args:
        current_value: Current metric value
        history: Historical values
        lookback: Number of periods to look back (None = all)
        
    Returns:
        Percentile (0-100)
    """
    if lookback:
        history = history.tail(lookback)
    
    history = history.dropna()
    if len(history) == 0:
        return 50.0
    
    return float(np.sum(history < current_value) / len(history) * 100)


def detect_regime_change(
    delta_metrics: Dict[str, float],
    current_metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
    history_stats: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Detect regime change signals from delta metrics.
    
    Key signals:
    - Pre-stress: curvature rising while vol flat
    - Stress onset: both rising together
    - Recovery: both declining
    
    Args:
        delta_metrics: Change in metrics
        current_metrics: Current absolute metrics
        thresholds: Z-score thresholds for signals
        history_stats: Historical mean/std for z-score calculation
        
    Returns:
        Dictionary with regime change signals
    """
    if thresholds is None:
        thresholds = {
            'curvature_zscore': 2.0,
            'vol_zscore': 1.0,
            'skew_zscore': 1.5
        }
    
    signals = {
        'pre_stress_signal': False,
        'stress_signal': False,
        'recovery_signal': False,
        'signal_strength': 0.0,
        'key_drivers': []
    }
    
    delta_curv = delta_metrics.get('delta_avg_atm_curvature', 0)
    delta_vol = delta_metrics.get('delta_avg_atm_vol', 0)
    delta_skew = delta_metrics.get('delta_avg_25d_skew', 0)
    
    curv_std = 0.0001
    vol_std = 0.01
    skew_std = 0.01
    
    if history_stats:
        curv_std = history_stats.get('curvature_std', curv_std)
        vol_std = history_stats.get('vol_std', vol_std)
        skew_std = history_stats.get('skew_std', skew_std)
    
    curv_zscore = delta_curv / curv_std if curv_std > 0 else 0
    vol_zscore = delta_vol / vol_std if vol_std > 0 else 0
    skew_zscore = delta_skew / skew_std if skew_std > 0 else 0
    
    if curv_zscore > thresholds['curvature_zscore'] and abs(vol_zscore) < thresholds['vol_zscore']:
        signals['pre_stress_signal'] = True
        signals['signal_strength'] = min(1.0, curv_zscore / 3.0)
        signals['key_drivers'].append(
            f"Curvature rising ({curv_zscore:.1f}σ) while vol stable"
        )
    
    if curv_zscore > thresholds['curvature_zscore'] and vol_zscore > thresholds['vol_zscore']:
        signals['stress_signal'] = True
        signals['signal_strength'] = min(1.0, (curv_zscore + vol_zscore) / 6.0)
        signals['key_drivers'].append(
            f"Both curvature ({curv_zscore:.1f}σ) and vol ({vol_zscore:.1f}σ) rising"
        )
    
    if curv_zscore < -thresholds['curvature_zscore'] and vol_zscore < -thresholds['vol_zscore']:
        signals['recovery_signal'] = True
        signals['signal_strength'] = min(1.0, abs(curv_zscore + vol_zscore) / 6.0)
        signals['key_drivers'].append(
            f"Curvature and vol both declining"
        )
    
    if abs(skew_zscore) > thresholds['skew_zscore']:
        direction = "steepening" if skew_zscore > 0 else "flattening"
        signals['key_drivers'].append(f"Skew {direction} ({abs(skew_zscore):.1f}σ)")
    
    signals['z_scores'] = {
        'curvature': curv_zscore,
        'vol': vol_zscore,
        'skew': skew_zscore
    }
    
    return signals


def compute_metric_momentum(
    history: pd.DataFrame,
    lookback: int = 5
) -> Dict[str, float]:
    """
    Compute momentum (trend) of key metrics.
    
    Args:
        history: DataFrame with metrics history
        lookback: Number of periods for momentum calculation
        
    Returns:
        Dictionary with momentum values
    """
    if len(history) < lookback:
        return {}
    
    recent = history.tail(lookback)
    
    momentum = {}
    
    for col in recent.columns:
        if recent[col].dtype in [np.float64, np.int64, float, int]:
            values = recent[col].dropna()
            if len(values) >= 2:
                x = np.arange(len(values))
                try:
                    slope, _ = np.polyfit(x, values, 1)
                    momentum[f'{col}_momentum'] = slope
                except Exception:
                    pass
    
    return momentum


def generate_temporal_report(
    history: pd.DataFrame,
    current_metrics: Dict[str, Any],
    lookback_zscore: int = 252
) -> Dict[str, Any]:
    """
    Generate comprehensive temporal dynamics report.
    
    Args:
        history: Historical metrics DataFrame
        current_metrics: Current snapshot metrics
        lookback_zscore: Lookback for z-score calculation
        
    Returns:
        Comprehensive temporal analysis
    """
    report = {
        'current_deltas': {},
        'z_scores': {},
        'percentiles': {},
        'regime_signals': {},
        'rolling_stats': {},
        'momentum': {}
    }
    
    if len(history) < 2:
        return report
    
    if 'summary' in current_metrics:
        current_summary = current_metrics['summary']
        prev_row = history.iloc[-1] if len(history) > 0 else {}
        
        for key in current_summary:
            if key in prev_row:
                report['current_deltas'][f'delta_{key}'] = (
                    current_summary[key] - prev_row[key]
                )
    
    stats_history = history.tail(lookback_zscore)
    
    history_stats = {}
    for col in stats_history.columns:
        if stats_history[col].dtype in [np.float64, np.int64]:
            mean = stats_history[col].mean()
            std = stats_history[col].std()
            
            if 'curvature' in col.lower():
                history_stats['curvature_std'] = std
            elif 'vol' in col.lower():
                history_stats['vol_std'] = std
            elif 'skew' in col.lower():
                history_stats['skew_std'] = std
            
            if 'summary' in current_metrics and col in current_metrics['summary']:
                current_val = current_metrics['summary'][col]
                if std > 0:
                    report['z_scores'][col] = (current_val - mean) / std
                report['percentiles'][col] = compute_percentiles(
                    current_val, stats_history[col]
                )
    
    report['regime_signals'] = detect_regime_change(
        report['current_deltas'],
        current_metrics,
        history_stats=history_stats
    )
    
    rolling = compute_rolling_stats(history.tail(30), window=20)
    if not rolling.empty:
        report['rolling_stats'] = rolling.iloc[-1].to_dict()
    
    report['momentum'] = compute_metric_momentum(history)
    
    return report


def get_regime_transition_dates(
    history: pd.DataFrame,
    regime_column: str = 'regime'
) -> List[Dict[str, Any]]:
    """
    Identify dates when regime changed.
    
    Args:
        history: DataFrame with regime classification
        regime_column: Name of regime column
        
    Returns:
        List of transition events
    """
    if regime_column not in history.columns:
        return []
    
    transitions = []
    
    prev_regime = None
    for idx, row in history.iterrows():
        current_regime = row[regime_column]
        
        if prev_regime is not None and current_regime != prev_regime:
            transitions.append({
                'date': idx,
                'from_regime': prev_regime,
                'to_regime': current_regime
            })
        
        prev_regime = current_regime
    
    return transitions

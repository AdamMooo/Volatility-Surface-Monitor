"""
Market Regime Classifier

Synthesize all metrics into actionable regime classification.

Regime Definitions:
1. CALM - Normal risk pricing, low curvature
2. PRE-STRESS - Early warning: curvature rising, vol still flat
3. ELEVATED - Active stress, high vol and curvature
4. ACUTE - Crisis mode, extreme readings
5. RECOVERY - Stress passing, metrics normalizing
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class MarketRegime(Enum):
    """Market regime classifications."""
    CALM = "calm"
    PRE_STRESS = "pre_stress"
    ELEVATED = "elevated"
    ACUTE = "acute"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class RegimeClassifier:
    """
    Classify market regime based on volatility surface metrics.
    
    Uses percentile-based thresholds relative to historical data.
    """
    
    def __init__(
        self,
        percentile_thresholds: Optional[Dict[str, float]] = None,
        z_score_threshold: float = 2.0
    ):
        """
        Initialize the classifier.
        
        Args:
            percentile_thresholds: Custom percentile thresholds
            z_score_threshold: Z-score threshold for change detection
        """
        self.thresholds = percentile_thresholds or {
            'calm_vol_max': 60,
            'calm_curv_max': 60,
            'pre_stress_curv_min': 70,
            'elevated_vol_min': 70,
            'elevated_curv_min': 70,
            'acute_vol_min': 90,
            'acute_curv_min': 90
        }
        self.z_score_threshold = z_score_threshold
        self._history: Optional[pd.DataFrame] = None
    
    def set_history(self, history: pd.DataFrame) -> None:
        """
        Set historical data for percentile calculations.
        
        Args:
            history: DataFrame with historical metrics
        """
        self._history = history
    
    def classify(
        self,
        metrics: Dict[str, Any],
        history: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Classify current market regime.
        
        Args:
            metrics: Current geometry metrics
            history: Optional historical data for percentiles
            
        Returns:
            Dictionary with regime classification
        """
        if history is not None:
            self._history = history
        
        summary = metrics.get('summary', {})
        
        atm_vol = summary.get('avg_atm_vol', 0)
        curvature = summary.get('avg_atm_curvature', 0)
        butterfly = summary.get('avg_butterfly', 0)
        skew = summary.get('avg_25d_skew', 0)
        roughness = summary.get('roughness', 0)
        
        vol_pct = self._compute_percentile('avg_atm_vol', atm_vol)
        curv_pct = self._compute_percentile('avg_atm_curvature', curvature)
        bf_pct = self._compute_percentile('avg_butterfly', butterfly)
        skew_pct = self._compute_percentile('avg_25d_skew', skew)
        
        regime, probabilities = self._determine_regime(
            vol_pct, curv_pct, bf_pct, skew_pct
        )
        
        drivers = self._identify_key_drivers(
            vol_pct, curv_pct, bf_pct, skew_pct, metrics
        )
        
        confidence = self._compute_confidence(probabilities)
        
        recommendation = self._generate_recommendation(regime, drivers)
        
        return {
            'current_regime': regime.value,
            'regime_probability': {r.value: p for r, p in probabilities.items()},
            'percentiles': {
                'atm_vol': vol_pct,
                'curvature': curv_pct,
                'butterfly': bf_pct,
                'skew': skew_pct
            },
            'key_drivers': drivers,
            'confidence': confidence,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    def _compute_percentile(self, metric_name: str, current_value: float) -> float:
        """Compute percentile of current value vs history."""
        if self._history is None or metric_name not in self._history.columns:
            return 50.0
        
        hist_values = self._history[metric_name].dropna()
        if len(hist_values) == 0:
            return 50.0
        
        return float(np.sum(hist_values < current_value) / len(hist_values) * 100)
    
    def _determine_regime(
        self,
        vol_pct: float,
        curv_pct: float,
        bf_pct: float,
        skew_pct: float
    ) -> Tuple[MarketRegime, Dict[MarketRegime, float]]:
        """
        Determine regime based on percentiles.
        
        Returns tuple of (most likely regime, probability distribution)
        """
        scores = {
            MarketRegime.CALM: 0.0,
            MarketRegime.PRE_STRESS: 0.0,
            MarketRegime.ELEVATED: 0.0,
            MarketRegime.ACUTE: 0.0,
            MarketRegime.RECOVERY: 0.0
        }
        
        if vol_pct < self.thresholds['calm_vol_max']:
            scores[MarketRegime.CALM] += 0.3
        if curv_pct < self.thresholds['calm_curv_max']:
            scores[MarketRegime.CALM] += 0.3
        if bf_pct < 60:
            scores[MarketRegime.CALM] += 0.2
        if 40 < skew_pct < 60:
            scores[MarketRegime.CALM] += 0.2
        
        if curv_pct > self.thresholds['pre_stress_curv_min'] and vol_pct < self.thresholds['elevated_vol_min']:
            scores[MarketRegime.PRE_STRESS] += 0.5
        if bf_pct > 70 and vol_pct < 70:
            scores[MarketRegime.PRE_STRESS] += 0.3
        if skew_pct > 70:
            scores[MarketRegime.PRE_STRESS] += 0.2
        
        if vol_pct > self.thresholds['elevated_vol_min']:
            scores[MarketRegime.ELEVATED] += 0.3
        if curv_pct > self.thresholds['elevated_curv_min']:
            scores[MarketRegime.ELEVATED] += 0.3
        if vol_pct > 70 and curv_pct > 70:
            scores[MarketRegime.ELEVATED] += 0.2
        if skew_pct > 80:
            scores[MarketRegime.ELEVATED] += 0.2
        
        if vol_pct > self.thresholds['acute_vol_min']:
            scores[MarketRegime.ACUTE] += 0.4
        if curv_pct > self.thresholds['acute_curv_min']:
            scores[MarketRegime.ACUTE] += 0.4
        if bf_pct > 90:
            scores[MarketRegime.ACUTE] += 0.2
        
        total = sum(scores.values())
        if total > 0:
            probabilities = {r: s/total for r, s in scores.items()}
        else:
            probabilities = {r: 0.2 for r in scores}
        
        regime = max(scores, key=scores.get)
        
        return regime, probabilities
    
    def _identify_key_drivers(
        self,
        vol_pct: float,
        curv_pct: float,
        bf_pct: float,
        skew_pct: float,
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Identify key drivers of current regime."""
        drivers = []
        
        def pct_desc(pct: float) -> str:
            if pct < 20:
                return "very low"
            elif pct < 40:
                return "low"
            elif pct < 60:
                return "normal"
            elif pct < 80:
                return "elevated"
            else:
                return "very high"
        
        drivers.append(f"ATM vol at {vol_pct:.0f}th percentile ({pct_desc(vol_pct)})")
        drivers.append(f"Curvature at {curv_pct:.0f}th percentile ({pct_desc(curv_pct)})")
        
        if curv_pct > 70 and vol_pct < 70:
            drivers.append("ALERT: Curvature elevated while vol normal - potential pre-stress signal")
        
        if bf_pct > 80:
            drivers.append(f"Butterfly spread elevated ({bf_pct:.0f}th pct) - tail hedging activity")
        
        if skew_pct > 80:
            drivers.append(f"Skew very steep ({skew_pct:.0f}th pct) - downside fear elevated")
        elif skew_pct < 20:
            drivers.append(f"Skew unusually flat ({skew_pct:.0f}th pct) - potential complacency")
        
        roughness = metrics.get('summary', {}).get('roughness', 0)
        if roughness > 0.01:
            drivers.append("Surface roughness elevated - potential liquidity stress")
        
        return drivers
    
    def _compute_confidence(
        self, 
        probabilities: Dict[MarketRegime, float]
    ) -> float:
        """Compute classification confidence."""
        if not probabilities:
            return 0.0
        
        max_prob = max(probabilities.values())
        
        probs = sorted(probabilities.values(), reverse=True)
        if len(probs) >= 2:
            separation = probs[0] - probs[1]
        else:
            separation = 1.0
        
        confidence = 0.5 * max_prob + 0.5 * separation
        
        return min(1.0, max(0.0, confidence))
    
    def _generate_recommendation(
        self, 
        regime: MarketRegime,
        drivers: List[str]
    ) -> str:
        """Generate narrative recommendation."""
        recommendations = {
            MarketRegime.CALM: (
                "Market is in a calm state with normal risk pricing. "
                "Surface shape metrics are within historical norms. "
                "Continue standard monitoring."
            ),
            MarketRegime.PRE_STRESS: (
                "EARLY WARNING: Surface shape is showing stress signals while "
                "implied volatility levels remain subdued. This pattern often "
                "precedes volatility spikes. Consider reviewing tail hedges "
                "and increasing monitoring frequency."
            ),
            MarketRegime.ELEVATED: (
                "Market is in elevated stress mode. Both volatility levels and "
                "surface shape metrics are elevated. Active stress conditions "
                "are present. Monitor for escalation or de-escalation signals."
            ),
            MarketRegime.ACUTE: (
                "ACUTE STRESS: Market is in crisis mode with extreme readings "
                "across volatility metrics. Surface may show dislocations. "
                "Exercise extreme caution and monitor for capitulation signals."
            ),
            MarketRegime.RECOVERY: (
                "Market appears to be recovering from stress. Volatility and "
                "curvature are declining from elevated levels. Watch for "
                "potential secondary stress events or confirmation of recovery."
            ),
            MarketRegime.UNKNOWN: (
                "Regime classification uncertain. Review underlying metrics "
                "and data quality."
            )
        }
        
        return recommendations.get(regime, recommendations[MarketRegime.UNKNOWN])
    
    def compute_regime_probability(
        self,
        metrics: Dict[str, Any],
        history: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Compute probability of being in each regime.
        
        Args:
            metrics: Current metrics
            history: Historical data
            
        Returns:
            Dictionary of regime probabilities
        """
        result = self.classify(metrics, history)
        return result['regime_probability']
    
    def get_regime_transitions(
        self,
        history: pd.DataFrame,
        regime_column: str = 'regime'
    ) -> List[Dict[str, Any]]:
        """
        Get historical regime transitions.
        
        Args:
            history: DataFrame with regime history
            regime_column: Column containing regime values
            
        Returns:
            List of transition events
        """
        if regime_column not in history.columns:
            return []
        
        transitions = []
        prev_regime = None
        
        for idx, row in history.iterrows():
            current = row[regime_column]
            if prev_regime is not None and current != prev_regime:
                transitions.append({
                    'date': idx,
                    'from': prev_regime,
                    'to': current
                })
            prev_regime = current
        
        return transitions


def classify_regime_simple(
    atm_vol: float,
    curvature: float,
    vol_threshold_high: float = 0.25,
    vol_threshold_low: float = 0.15,
    curv_threshold_high: float = 0.001,
    curv_threshold_low: float = 0.0005
) -> str:
    """
    Simple regime classification without historical percentiles.
    
    Useful for quick classification or when history unavailable.
    
    Args:
        atm_vol: ATM implied volatility
        curvature: ATM curvature
        vol_threshold_high: High vol threshold
        vol_threshold_low: Low vol threshold  
        curv_threshold_high: High curvature threshold
        curv_threshold_low: Low curvature threshold
        
    Returns:
        Regime string
    """
    vol_high = atm_vol > vol_threshold_high
    vol_low = atm_vol < vol_threshold_low
    curv_high = curvature > curv_threshold_high
    curv_low = curvature < curv_threshold_low
    
    if vol_low and curv_low:
        return "calm"
    elif curv_high and not vol_high:
        return "pre_stress"
    elif vol_high and curv_high:
        if atm_vol > vol_threshold_high * 1.5:
            return "acute"
        return "elevated"
    elif vol_high and not curv_high:
        return "recovery"
    else:
        return "calm"


class GMMRegimeClassifier:
    """
    Gaussian Mixture Model-based regime classifier.
    
    Models the distribution of market states using GMM clustering.
    Focuses on how market distributions evolve and transition over time.
    """
    
    def __init__(
        self,
        n_components: int = 5,
        random_state: int = 42,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize GMM classifier.
        
        Args:
            n_components: Number of mixture components (regimes)
            random_state: Random seed for reproducibility
            feature_cols: Columns to use as features
        """
        self.n_components = n_components
        self.random_state = random_state
        self.feature_cols = feature_cols or [
            'avg_atm_vol', 'unemployment_rate', 'yield_curve_spread', 'spy_return'
        ]
        self.gmm = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_names = ['calm', 'pre_stress', 'elevated', 'acute', 'recovery']
        
    def fit(self, historical_data: pd.DataFrame) -> None:
        """
        Fit GMM on historical market metrics.
        
        Args:
            historical_data: DataFrame with historical metrics
        """
        if len(historical_data) < self.n_components * 10:
            logger.warning(f"Insufficient data for GMM fitting: {len(historical_data)} samples")
            return
            
        # Extract features
        features = historical_data[self.feature_cols].dropna()
        if len(features) == 0:
            logger.warning("No valid feature data for GMM fitting")
            return
            
        # Convert to numpy array for scaling
        features_array = features.values
        
        # Scale features
        self.scaler.fit(features_array)
        scaled_features = self.scaler.transform(features_array)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=min(self.n_components, len(features)),
            random_state=self.random_state,
            covariance_type='full'
        )
        self.gmm.fit(scaled_features)
        
        # Store training data for fit score calculation
        self._training_features = features_array
        training_ll = self.gmm.score_samples(scaled_features)
        self._training_avg_log_likelihood = np.mean(training_ll)
        
        self.is_fitted = True
        
        logger.info(f"GMM fitted with {self.gmm.n_components} components on {len(features)} samples")
    
    def predict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict regime for current metrics using GMM.
        
        Args:
            metrics: Current market metrics
            
        Returns:
            Classification results
        """
        if not self.is_fitted:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'distribution_fit': 'model_not_fitted'
            }
        
        # Extract current features
        current_features = []
        for col in self.feature_cols:
            if col in metrics.get('summary', {}):
                current_features.append(metrics['summary'][col])
            else:
                current_features.append(0.0)  # Default for missing features
        
        # Convert to numpy array and scale
        current_features_array = np.array([current_features])
        scaled_features = self.scaler.transform(current_features_array)
        
        # Predict
        regime_idx = self.gmm.predict(scaled_features)[0]
        probabilities = self.gmm.predict_proba(scaled_features)[0]
        log_likelihood = self.gmm.score_samples(scaled_features)[0]
        
        # Calculate fit score as normalized likelihood relative to training data
        # Use the average log-likelihood of training data as reference
        if hasattr(self, '_training_avg_log_likelihood'):
            avg_training_ll = self._training_avg_log_likelihood
        else:
            # Calculate average log-likelihood on training data
            training_features = self.scaler.transform(self._training_features)
            training_ll = self.gmm.score_samples(training_features)
            avg_training_ll = np.mean(training_ll)
            self._training_avg_log_likelihood = avg_training_ll
        
        # Fit score: relative likelihood compared to training average
        # Values > 1.0: better fit than average training point
        # Values < 1.0: worse fit than average training point
        relative_ll = log_likelihood - avg_training_ll
        fit_score = np.exp(relative_ll)
        
        # Provide more granular interpretation
        if fit_score < 0.1:
            fit_quality = "extreme_outlier"
        elif fit_score < 0.5:
            fit_quality = "poor_fit"
        elif fit_score < 1.5:
            fit_quality = "moderate_fit"
        else:
            fit_quality = "strong_fit"
        
        # For extreme outliers, don't force classification into existing regimes
        if fit_score < 0.1:
            regime_name = "unprecedented"
            confidence = 0.0
            probabilities = {name: 0.0 for name in self.regime_names}
            probabilities["unprecedented"] = 1.0
            regime_idx = -1  # Special index for unprecedented
        else:
            regime_idx = self.gmm.predict(scaled_features)[0]
            probabilities_array = self.gmm.predict_proba(scaled_features)[0]
            regime_name = self.regime_names[regime_idx] if regime_idx < len(self.regime_names) else f"regime_{regime_idx}"
            confidence = float(probabilities_array[regime_idx])
            probabilities = {self.regime_names[i] if i < len(self.regime_names) else f"regime_{i}": float(p) 
                            for i, p in enumerate(probabilities_array)}
        
        return {
            'regime': regime_name,
            'confidence': float(confidence),
            'probabilities': probabilities,
            'distribution_fit': float(fit_score),
            'fit_quality': fit_quality,
            'log_likelihood': float(log_likelihood),
            'regime_idx': int(regime_idx)
        }
    
    def predict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict regime for current metrics.
        
        Args:
            metrics: Current market metrics
            
        Returns:
            Classification results
        """
        if not self.is_fitted:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'probabilities': {},
                'distribution_fit': 'model_not_fitted'
            }
        
        # Extract current features
        current_features = []
        for col in self.feature_cols:
            if col in metrics.get('summary', {}):
                current_features.append(metrics['summary'][col])
            else:
                current_features.append(0.0)  # Default for missing features
        
        # Convert to numpy array and scale
        current_features_array = np.array([current_features])
        scaled_features = self.scaler.transform(current_features_array)
        
        # Predict
        regime_idx = self.gmm.predict(scaled_features)[0]
        probabilities = self.gmm.predict_proba(scaled_features)[0]
        log_likelihood = self.gmm.score_samples(scaled_features)[0]
        
        # Calculate fit score as normalized likelihood relative to training data
        # Use the average log-likelihood of training data as reference
        if hasattr(self, '_training_avg_log_likelihood'):
            avg_training_ll = self._training_avg_log_likelihood
        else:
            # Calculate average log-likelihood on training data
            training_features = self.scaler.transform(self._training_features)
            training_ll = self.gmm.score_samples(training_features)
            avg_training_ll = np.mean(training_ll)
            self._training_avg_log_likelihood = avg_training_ll
        
        # Fit score: relative likelihood compared to training average
        # Values > 1.0: better fit than average training point
        # Values < 1.0: worse fit than average training point
        relative_ll = log_likelihood - avg_training_ll
        fit_score = np.exp(relative_ll)
        
        # Provide more granular interpretation
        if fit_score < 0.1:
            fit_quality = "extreme_outlier"
        elif fit_score < 0.5:
            fit_quality = "poor_fit"
        elif fit_score < 1.5:
            fit_quality = "moderate_fit"
        else:
            fit_quality = "strong_fit"
        
        # For extreme outliers, don't force classification into existing regimes
        if fit_score < 0.1:
            regime_name = "unprecedented"
            confidence = 0.0
            probabilities = {name: 0.0 for name in self.regime_names}
            probabilities["unprecedented"] = 1.0
            regime_idx = -1  # Special index for unprecedented
        else:
            regime_idx = self.gmm.predict(scaled_features)[0]
            probabilities_array = self.gmm.predict_proba(scaled_features)[0]
            regime_name = self.regime_names[regime_idx] if regime_idx < len(self.regime_names) else f"regime_{regime_idx}"
            confidence = float(probabilities_array[regime_idx])
            probabilities = {self.regime_names[i] if i < len(self.regime_names) else f"regime_{i}": float(p) 
                            for i, p in enumerate(probabilities_array)}
        
        return {
            'regime': regime_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'distribution_fit': float(fit_score),
            'fit_quality': fit_quality,
            'log_likelihood': float(log_likelihood),
            'regime_idx': int(regime_idx)
        }
    
    def analyze_distribution_evolution(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how market distributions evolve over time.
        
        Args:
            historical_data: Historical metrics with dates
            
        Returns:
            Distribution evolution analysis
        """
        if not self.is_fitted or len(historical_data) == 0:
            return {}
            
        # Get predictions for historical data
        predictions = []
        for idx, row in historical_data.iterrows():
            metrics = {'summary': {col: row[col] for col in self.feature_cols if col in row}}
            pred = self.predict(metrics)
            pred['date'] = idx
            predictions.append(pred)
        
        pred_df = pd.DataFrame(predictions)
        
        # Analyze regime transitions
        transitions = []
        prev_regime = None
        for _, row in pred_df.iterrows():
            current = row['regime']
            if prev_regime and current != prev_regime:
                transitions.append({
                    'date': row['date'],
                    'from': prev_regime,
                    'to': current,
                    'confidence': row['confidence']
                })
            prev_regime = current
        
        # Distribution statistics
        regime_counts = pred_df['regime'].value_counts()
        avg_confidence = pred_df['confidence'].mean()
        
        return {
            'regime_distribution': regime_counts.to_dict(),
            'average_confidence': float(avg_confidence),
            'transitions': transitions,
            'total_periods': len(pred_df),
            'transition_count': len(transitions)
        }
    
    def analyze_distribution_evolution(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how market distributions evolve over time.
        
        Args:
            historical_data: Historical metrics with dates
            
        Returns:
            Distribution evolution analysis
        """
        if not self.is_fitted or len(historical_data) == 0:
            return {}
            
        # Get predictions for historical data
        predictions = []
        for idx, row in historical_data.iterrows():
            metrics = {'summary': {col: row[col] for col in self.feature_cols if col in row}}
            pred = self.predict(metrics)
            pred['date'] = idx
            predictions.append(pred)
        
        pred_df = pd.DataFrame(predictions)
        
        # Analyze regime transitions
        transitions = []
        prev_regime = None
        for _, row in pred_df.iterrows():
            current = row['regime']
            if prev_regime and current != prev_regime:
                transitions.append({
                    'date': row['date'],
                    'from': prev_regime,
                    'to': current,
                    'confidence': row['confidence']
                })
            prev_regime = current
        
        # Distribution statistics
        regime_counts = pred_df['regime'].value_counts()
        avg_confidence = pred_df['confidence'].mean()
        
        return {
            'regime_distribution': regime_counts.to_dict(),
            'average_confidence': float(avg_confidence),
            'transitions': transitions,
            'total_periods': len(pred_df),
            'transition_count': len(transitions)
        }

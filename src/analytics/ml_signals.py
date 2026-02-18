"""
ML Signal Detection Module

Foundation for machine learning-based market signal detection.
Includes regime classification, anomaly detection, and pattern recognition.

Future Features:
- Gaussian Mixture Models for regime clustering
- Anomaly detection for unusual market behavior
- Pattern recognition for historical similarity
- Predictive signals based on vol surface shapes
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML features disabled.")


@dataclass
class MarketSignal:
    """A detected market signal."""
    signal_type: str  # 'regime_change', 'anomaly', 'pattern', 'trend'
    name: str
    confidence: float  # 0-1
    description: str
    timestamp: datetime
    details: Dict[str, Any]
    action_suggestion: str


class RegimeDetector:
    """
    Detects market regimes using Gaussian Mixture Models.
    
    Clusters the volatility surface characteristics to identify
    distinct market states (calm, transition, stressed, crisis).
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.gmm = None
        self.is_fitted = False
        self.regime_names = ['Calm', 'Cautious', 'Stressed', 'Crisis']
        self.history: List[Dict] = []
    
    def fit(self, historical_metrics: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit the regime detector on historical data.
        
        Args:
            historical_metrics: DataFrame with columns:
                ['atm_vol', 'skew', 'curvature', 'term_slope']
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot fit: sklearn not available")
            return self
        
        if len(historical_metrics) < 50:
            logger.warning("Insufficient data for fitting GMM")
            return self
        
        features = historical_metrics[['atm_vol', 'skew', 'curvature', 'term_slope']].values
        features_scaled = self.scaler.fit_transform(features)
        
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            n_init=5
        )
        self.gmm.fit(features_scaled)
        self.is_fitted = True
        
        # Order regimes by average volatility (calm = lowest vol cluster)
        cluster_means = []
        labels = self.gmm.predict(features_scaled)
        for i in range(self.n_regimes):
            mask = labels == i
            if mask.any():
                cluster_means.append((i, features[mask, 0].mean()))  # atm_vol is column 0
        
        # Sort by vol and create mapping
        cluster_means.sort(key=lambda x: x[1])
        self.cluster_to_regime = {cm[0]: i for i, cm in enumerate(cluster_means)}
        
        logger.info(f"Fitted regime detector on {len(historical_metrics)} samples")
        return self
    
    def detect(self, current_metrics: Dict[str, float]) -> Tuple[str, float, Dict]:
        """
        Detect current regime.
        
        Args:
            current_metrics: Dict with atm_vol, skew, curvature, term_slope
            
        Returns:
            Tuple of (regime_name, confidence, probabilities)
        """
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            # Fallback to rule-based detection
            return self._rule_based_regime(current_metrics)
        
        features = np.array([[
            current_metrics.get('atm_vol', 0.15),
            current_metrics.get('skew', 0.03),
            current_metrics.get('curvature', 0.01),
            current_metrics.get('term_slope', 0)
        ]])
        
        features_scaled = self.scaler.transform(features)
        proba = self.gmm.predict_proba(features_scaled)[0]
        cluster = self.gmm.predict(features_scaled)[0]
        
        regime_idx = self.cluster_to_regime.get(cluster, 0)
        regime_name = self.regime_names[regime_idx]
        confidence = proba[cluster]
        
        # Map probabilities to regime names
        proba_dict = {}
        for cluster_id, prob in enumerate(proba):
            regime_id = self.cluster_to_regime.get(cluster_id, cluster_id)
            if regime_id < len(self.regime_names):
                proba_dict[self.regime_names[regime_id]] = float(prob)
        
        return regime_name, confidence, proba_dict
    
    def _rule_based_regime(self, metrics: Dict[str, float]) -> Tuple[str, float, Dict]:
        """Simple rule-based regime detection as fallback."""
        atm_vol = metrics.get('atm_vol', 0.15)
        skew = metrics.get('skew', 0.03)
        
        if atm_vol > 0.30:
            return 'Crisis', 0.85, {'Crisis': 0.85, 'Stressed': 0.15}
        elif atm_vol > 0.22:
            return 'Stressed', 0.75, {'Stressed': 0.75, 'Cautious': 0.20, 'Crisis': 0.05}
        elif atm_vol > 0.17 or skew > 0.05:
            return 'Cautious', 0.70, {'Cautious': 0.70, 'Calm': 0.20, 'Stressed': 0.10}
        else:
            return 'Calm', 0.80, {'Calm': 0.80, 'Cautious': 0.20}


class AnomalyDetector:
    """
    Detects anomalies in volatility surface behavior.
    
    Uses Isolation Forest to identify unusual market states
    that don't fit historical patterns.
    """
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.detector = None
        self.is_fitted = False
    
    def fit(self, historical_metrics: pd.DataFrame) -> 'AnomalyDetector':
        """Fit on historical data."""
        if not SKLEARN_AVAILABLE:
            return self
        
        if len(historical_metrics) < 100:
            logger.warning("Insufficient data for anomaly detection")
            return self
        
        features = historical_metrics[['atm_vol', 'skew', 'curvature', 'term_slope']].values
        features_scaled = self.scaler.fit_transform(features)
        
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.detector.fit(features_scaled)
        self.is_fitted = True
        
        logger.info("Fitted anomaly detector")
        return self
    
    def detect(self, current_metrics: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Check if current state is anomalous.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score, description)
        """
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            return self._rule_based_anomaly(current_metrics)
        
        features = np.array([[
            current_metrics.get('atm_vol', 0.15),
            current_metrics.get('skew', 0.03),
            current_metrics.get('curvature', 0.01),
            current_metrics.get('term_slope', 0)
        ]])
        
        features_scaled = self.scaler.transform(features)
        
        # Isolation Forest returns -1 for anomaly, 1 for normal
        prediction = self.detector.predict(features_scaled)[0]
        score = self.detector.decision_function(features_scaled)[0]
        
        is_anomaly = prediction == -1
        
        if is_anomaly:
            description = self._describe_anomaly(current_metrics)
        else:
            description = "Market behavior within normal parameters"
        
        return is_anomaly, float(score), description
    
    def _rule_based_anomaly(self, metrics: Dict[str, float]) -> Tuple[bool, float, str]:
        """Rule-based anomaly detection as fallback."""
        atm_vol = metrics.get('atm_vol', 0.15)
        skew = metrics.get('skew', 0.03)
        curvature = metrics.get('curvature', 0.01)
        term_slope = metrics.get('term_slope', 0)
        
        anomalies = []
        score = 0
        
        if atm_vol > 0.40:
            anomalies.append("Extreme volatility levels")
            score -= 0.5
        
        if skew > 0.12:
            anomalies.append("Extreme skew (crash fear)")
            score -= 0.3
        
        if abs(term_slope) > 0.08:
            anomalies.append("Extreme term structure inversion")
            score -= 0.3
        
        if curvature > 0.06:
            anomalies.append("Extreme curvature (tail risk)")
            score -= 0.2
        
        is_anomaly = len(anomalies) > 0
        description = "; ".join(anomalies) if anomalies else "Normal market behavior"
        
        return is_anomaly, score, description
    
    def _describe_anomaly(self, metrics: Dict[str, float]) -> str:
        """Generate description of why state is anomalous."""
        parts = []
        
        if metrics.get('atm_vol', 0) > 0.30:
            parts.append("unusually high volatility")
        
        if metrics.get('skew', 0) > 0.08:
            parts.append("extreme crash protection demand")
        
        if abs(metrics.get('term_slope', 0)) > 0.05:
            parts.append("abnormal term structure")
        
        if metrics.get('curvature', 0) > 0.04:
            parts.append("elevated tail risk pricing")
        
        if parts:
            return "Anomaly detected: " + ", ".join(parts)
        return "Unusual pattern in volatility surface"


class PatternMatcher:
    """
    Matches current market state to historical patterns.
    
    Identifies similar historical periods to provide context
    and potential forward-looking insights.
    """
    
    def __init__(self):
        self.historical_patterns: List[Dict] = []
        
        # Pre-defined significant market events
        self.known_events = [
            {
                'name': 'COVID Crash',
                'date': '2020-03-16',
                'atm_vol': 0.82,
                'skew': 0.15,
                'outcome': 'Rapid V-shaped recovery over 6 months',
                'lesson': 'Extreme fear often marks bottoms'
            },
            {
                'name': 'Aug 2024 VIX Spike',
                'date': '2024-08-05',
                'atm_vol': 0.38,
                'skew': 0.12,
                'outcome': 'Quick normalization within 2 weeks',
                'lesson': 'Sudden spikes often reverse quickly'
            },
            {
                'name': 'Typical Calm Market',
                'date': '2024-01-15',
                'atm_vol': 0.13,
                'skew': 0.025,
                'outcome': 'Continued stable conditions',
                'lesson': 'Low vol can persist but eventually rises'
            },
            {
                'name': 'Pre-Correction Warning',
                'date': '2022-01-03',
                'atm_vol': 0.19,
                'skew': 0.06,
                'outcome': 'Bear market followed',
                'lesson': 'Rising skew before vol spike is a warning'
            }
        ]
    
    def find_similar_periods(self, current_metrics: Dict[str, float], top_n: int = 3) -> List[Dict]:
        """
        Find historical periods most similar to current conditions.
        
        Returns list of similar periods with context.
        """
        current_vol = current_metrics.get('atm_vol', 0.15)
        current_skew = current_metrics.get('skew', 0.03)
        
        similarities = []
        
        for event in self.known_events:
            # Simple distance metric
            vol_diff = abs(event['atm_vol'] - current_vol)
            skew_diff = abs(event['skew'] - current_skew)
            
            # Weighted similarity (0-1, higher is more similar)
            similarity = 1 / (1 + vol_diff * 5 + skew_diff * 10)
            
            similarities.append({
                **event,
                'similarity': similarity,
                'vol_diff': vol_diff,
                'skew_diff': skew_diff
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_n]


class SignalAggregator:
    """
    Aggregates signals from multiple detectors into actionable insights.
    
    This is the main interface for the ML module.
    """
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.anomaly_detector = AnomalyDetector()
        self.pattern_matcher = PatternMatcher()
    
    def analyze(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Run full analysis and generate signals.
        
        Args:
            current_metrics: Current volatility surface metrics
            
        Returns:
            Analysis results with signals and recommendations
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'signals': [],
            'regime': {},
            'anomaly': {},
            'similar_periods': [],
            'overall_assessment': '',
            'confidence': 0
        }
        
        # Regime detection
        regime_name, regime_conf, regime_proba = self.regime_detector.detect(current_metrics)
        results['regime'] = {
            'current': regime_name,
            'confidence': regime_conf,
            'probabilities': regime_proba
        }
        
        if regime_name in ['Stressed', 'Crisis']:
            results['signals'].append(MarketSignal(
                signal_type='regime',
                name=f'{regime_name} Market Regime',
                confidence=regime_conf,
                description=f"Market is in {regime_name.lower()} mode",
                timestamp=datetime.now(),
                details=regime_proba,
                action_suggestion='Reduce risk exposure and avoid panic selling'
            ))
        
        # Anomaly detection
        is_anomaly, anomaly_score, anomaly_desc = self.anomaly_detector.detect(current_metrics)
        results['anomaly'] = {
            'detected': is_anomaly,
            'score': anomaly_score,
            'description': anomaly_desc
        }
        
        if is_anomaly:
            results['signals'].append(MarketSignal(
                signal_type='anomaly',
                name='Unusual Market Behavior',
                confidence=min(1.0, abs(anomaly_score)),
                description=anomaly_desc,
                timestamp=datetime.now(),
                details={'score': anomaly_score},
                action_suggestion='Exercise extra caution - market is behaving unusually'
            ))
        
        # Pattern matching
        similar = self.pattern_matcher.find_similar_periods(current_metrics)
        results['similar_periods'] = similar
        
        if similar and similar[0]['similarity'] > 0.7:
            best_match = similar[0]
            results['signals'].append(MarketSignal(
                signal_type='pattern',
                name=f"Similar to {best_match['name']}",
                confidence=best_match['similarity'],
                description=f"Current conditions resemble {best_match['name']} ({best_match['date']})",
                timestamp=datetime.now(),
                details=best_match,
                action_suggestion=f"Historical lesson: {best_match['lesson']}"
            ))
        
        # Overall assessment
        results['overall_assessment'] = self._generate_assessment(results)
        results['confidence'] = self._calculate_confidence(results)
        
        return results
    
    def _generate_assessment(self, results: Dict) -> str:
        """Generate overall market assessment."""
        regime = results['regime']['current']
        is_anomaly = results['anomaly']['detected']
        
        if regime == 'Crisis':
            if is_anomaly:
                return "⚠️ ALERT: Extreme and unusual market conditions. This is rare - stay calm and avoid panic decisions."
            return "⚠️ Markets are in crisis mode. Historically, these periods are temporary. Patience is key."
        
        elif regime == 'Stressed':
            return "📉 Elevated stress in markets. Expect larger swings. Review risk but avoid emotional decisions."
        
        elif regime == 'Cautious':
            if is_anomaly:
                return "👀 Something unusual is happening. Stay alert but don't overreact."
            return "🌥️ Some nervousness in markets. Good time to review your positions."
        
        else:  # Calm
            return "🌤️ Markets are calm. Good conditions for normal investment activities."
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence in assessment."""
        regime_conf = results['regime']['confidence']
        
        # Higher confidence when signals agree
        if results['signals']:
            signal_conf = sum(s.confidence for s in results['signals']) / len(results['signals'])
            return (regime_conf + signal_conf) / 2
        
        return regime_conf


# ============ CONVENIENCE FUNCTIONS ============

def get_market_signals(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Main entry point for ML signal detection.
    
    Args:
        metrics: Dict with atm_vol, skew, curvature, term_slope
        
    Returns:
        Full analysis results
    """
    aggregator = SignalAggregator()
    return aggregator.analyze(metrics)


def explain_signals_simply(analysis: Dict[str, Any]) -> str:
    """
    Convert ML analysis to plain English.
    
    Args:
        analysis: Output from get_market_signals()
        
    Returns:
        Human-readable summary
    """
    lines = [f"## Market Analysis\n"]
    lines.append(analysis['overall_assessment'])
    lines.append("")
    
    # Regime
    regime = analysis['regime']
    lines.append(f"**Current State:** {regime['current']} ({regime['confidence']*100:.0f}% confident)")
    
    # Signals
    if analysis['signals']:
        lines.append("\n**Detected Signals:**")
        for signal in analysis['signals']:
            lines.append(f"- {signal.name}: {signal.description}")
            if signal.action_suggestion:
                lines.append(f"  → {signal.action_suggestion}")
    
    # Similar periods
    if analysis['similar_periods']:
        best = analysis['similar_periods'][0]
        if best['similarity'] > 0.5:
            lines.append(f"\n**Historical Context:** Current conditions are {best['similarity']*100:.0f}% similar to {best['name']}.")
            lines.append(f"*What happened then:* {best['outcome']}")
    
    return "\n".join(lines)

"""
GMM-HMM Regime Detection Module

Uses Hidden Markov Models with Gaussian Mixture emissions for market regime detection.
This is fundamentally different from supervised learning:

Key Advantages:
1. Models temporal persistence - markets STAY in regimes
2. Captures transitions probabilistically 
3. Provides uncertainty quantification
4. Adapts beliefs with Bayes rule instead of retraining

The edge isn't "predict returns" - it's "know which playbook to use":
- State 0 (Calm): Long, tight stops, ride the trend
- State 1 (Cautious): Reduce size, don't get chopped
- State 2 (Stressed): Widen stops, hedge exposure
- State 3 (Crisis): Defensive, look for capitulation signals
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger
import json
from pathlib import Path

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.info("hmmlearn not installed. Using simplified HMM implementation.")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    regime: str
    regime_id: int
    confidence: float
    probabilities: Dict[str, float]
    transition_probs: Dict[str, float]  # Probability of transitioning to each state
    persistence: float  # How long we've been in this regime
    features: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'regime': self.regime,
            'regime_id': self.regime_id,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'transition_probs': self.transition_probs,
            'persistence': self.persistence,
            'features': self.features,
            'timestamp': self.timestamp.isoformat()
        }


class SimpleHMM:
    """
    Simplified HMM implementation when hmmlearn not available.
    Uses forward algorithm for state inference.
    """
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        
        # Regime characteristics (mean, std for each feature)
        # Features: [atm_vol, skew, curvature, term_slope]
        self.regime_params = {
            0: {  # Calm
                'means': [0.12, 0.02, 0.005, 0.01],
                'stds': [0.02, 0.01, 0.005, 0.01]
            },
            1: {  # Cautious
                'means': [0.17, 0.04, 0.015, 0.005],
                'stds': [0.03, 0.015, 0.008, 0.015]
            },
            2: {  # Stressed
                'means': [0.25, 0.06, 0.025, -0.01],
                'stds': [0.04, 0.02, 0.012, 0.02]
            },
            3: {  # Crisis
                'means': [0.40, 0.10, 0.04, -0.03],
                'stds': [0.10, 0.03, 0.02, 0.03]
            }
        }
        
        # Transition matrix - regimes are persistent!
        # Rows = from state, Cols = to state
        self.transition_matrix = np.array([
            [0.92, 0.06, 0.015, 0.005],  # Calm -> mostly stays calm
            [0.10, 0.80, 0.08, 0.02],    # Cautious -> can go either way
            [0.02, 0.12, 0.78, 0.08],    # Stressed -> tends to persist or escalate
            [0.01, 0.04, 0.20, 0.75],    # Crisis -> sticky but can recover
        ])
        
        # Prior (start) probabilities
        self.prior = np.array([0.60, 0.25, 0.12, 0.03])
        
        # State history for filtering
        self.state_probs = self.prior.copy()
        self.history = []
        
    def _emission_prob(self, obs: np.ndarray, state: int) -> float:
        """Calculate P(observation | state) using Gaussian."""
        params = self.regime_params[state]
        means = np.array(params['means'])
        stds = np.array(params['stds'])
        
        # Multivariate Gaussian (assuming independence for simplicity)
        log_prob = -0.5 * np.sum(((obs - means) / stds) ** 2)
        log_prob -= np.sum(np.log(stds)) + len(obs) * 0.5 * np.log(2 * np.pi)
        
        return np.exp(np.clip(log_prob, -100, 0))
    
    def update(self, observation: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Update state beliefs given new observation.
        Uses forward algorithm (filtering).
        
        Returns:
            (most_likely_state, state_probabilities)
        """
        # Predict step: P(S_t | S_{t-1}) using transition matrix
        predicted = self.transition_matrix.T @ self.state_probs
        
        # Update step: P(S_t | observation) using Bayes rule
        likelihoods = np.array([self._emission_prob(observation, s) for s in range(self.n_states)])
        
        # Combine prediction and likelihood
        posterior = predicted * likelihoods
        
        # Normalize
        if posterior.sum() > 0:
            posterior /= posterior.sum()
        else:
            posterior = self.prior.copy()
        
        self.state_probs = posterior
        self.history.append(posterior.copy())
        
        return np.argmax(posterior), posterior
    
    def get_transition_probs(self, current_state: int) -> np.ndarray:
        """Get probability of transitioning to each state from current."""
        return self.transition_matrix[current_state]


class RegimeHMM:
    """
    Hidden Markov Model for market regime detection.
    
    Uses GMM emissions within each state to capture the multimodal
    nature of returns/volatility within regimes.
    
    States:
    - 0: Calm (low vol grind, trending)
    - 1: Cautious (elevated awareness, mixed signals)
    - 2: Stressed (high vol, choppy, fear elevated)
    - 3: Crisis (extreme stress, potential capitulation)
    """
    
    REGIME_NAMES = ['Calm', 'Cautious', 'Stressed', 'Crisis']
    REGIME_COLORS = ['#00d26a', '#ffbe0b', '#ff6b35', '#e94560']
    
    # Strategy recommendations per regime
    PLAYBOOKS = {
        0: {  # Calm
            'name': 'Calm Markets',
            'description': 'Low volatility environment with normal price action',
            'strategy': 'Standard positioning, trend following works',
            'position_sizing': 'Full size acceptable',
            'stop_loss': 'Tight stops (1-2%)',
            'actions': [
                'Ride existing trends',
                'Good time for rebalancing',
                'Standard dollar-cost averaging',
                'Consider selling premium (covered calls)'
            ],
            'avoid': [
                'Overtrading',
                'Complacency - markets can shift quickly'
            ]
        },
        1: {  # Cautious
            'name': 'Early Warning',
            'description': 'Stress indicators beginning to elevate',
            'strategy': 'Reduce risk, tighten stops, raise cash',
            'position_sizing': 'Reduce to 75%',
            'stop_loss': 'Moderate stops (2-3%)',
            'actions': [
                'Review portfolio for weak positions',
                'Take partial profits on winners',
                'Hold off on new aggressive positions',
                'Consider protective puts on core holdings'
            ],
            'avoid': [
                'Large new positions',
                'Margin usage',
                'Illiquid positions'
            ]
        },
        2: {  # Stressed
            'name': 'Elevated Stress',
            'description': 'High volatility, fear clearly elevated',
            'strategy': 'Defensive, hedge, widen stops',
            'position_sizing': 'Reduce to 50%',
            'stop_loss': 'Wide stops (3-5%) or none',
            'actions': [
                'Expect 2-3% daily swings',
                'Hedge core positions',
                'Build watchlist for opportunities',
                'Stay patient - dont chase'
            ],
            'avoid': [
                'Panic selling at lows',
                'Trying to pick the bottom',
                'Short-term options (theta decay fast)'
            ]
        },
        3: {  # Crisis
            'name': 'Crisis Mode',
            'description': 'Extreme fear, potential capitulation',
            'strategy': 'Maximum defense, but watch for opportunities',
            'position_sizing': 'Minimal (25%)',
            'stop_loss': 'Very wide or no stops',
            'actions': [
                'Preserve capital',
                'Watch for capitulation signals',
                'Build cash for recovery opportunities',
                'Historical: buying fear rewarded long-term'
            ],
            'avoid': [
                'Panic selling into the hole',
                'Averaging down too early',
                'Making emotional decisions'
            ]
        }
    }
    
    def __init__(self, n_states: int = 4, model_path: Optional[Path] = None):
        self.n_states = n_states
        self.model_path = model_path or Path("data/regime_model.json")
        
        # Use hmmlearn if available, otherwise simple implementation
        if HMM_AVAILABLE:
            self.hmm = GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            self.is_fitted = False
        else:
            self.hmm = SimpleHMM(n_states)
            self.is_fitted = True  # Simple HMM has preset parameters
        
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.current_state = 0
        self.state_history: List[int] = []
        self.regime_start_time: datetime = datetime.now()
        self.history: List[RegimeState] = []
        
        # Load saved model if exists
        self._load_model()
    
    def _load_model(self):
        """Load saved model parameters."""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                    self.current_state = data.get('current_state', 0)
                    self.state_history = data.get('state_history', [])[-100:]  # Keep last 100
                    logger.info(f"Loaded regime model: current state = {self.REGIME_NAMES[self.current_state]}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def _save_model(self):
        """Save model state."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, 'w') as f:
                json.dump({
                    'current_state': self.current_state,
                    'state_history': self.state_history[-100:],
                    'last_update': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def fit(self, historical_data: pd.DataFrame) -> 'RegimeHMM':
        """
        Fit HMM on historical volatility surface metrics.
        
        Args:
            historical_data: DataFrame with columns:
                ['atm_vol', 'skew', 'curvature', 'term_slope', 'date']
        """
        if not HMM_AVAILABLE:
            logger.info("Using preset HMM parameters (hmmlearn not available)")
            return self
        
        if len(historical_data) < 100:
            logger.warning("Insufficient data for HMM fitting")
            return self
        
        features = ['atm_vol', 'skew', 'curvature', 'term_slope']
        X = historical_data[features].values
        
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Fit HMM
        self.hmm.fit(X)
        self.is_fitted = True
        
        # Order states by average volatility
        state_vols = []
        states = self.hmm.predict(X)
        for s in range(self.n_states):
            mask = states == s
            if mask.any():
                state_vols.append((s, historical_data.loc[mask, 'atm_vol'].mean()))
        
        state_vols.sort(key=lambda x: x[1])
        self.state_mapping = {old: new for new, (old, _) in enumerate(state_vols)}
        
        logger.info(f"Fitted HMM on {len(historical_data)} samples")
        self._save_model()
        return self
    
    def detect(self, metrics: Dict[str, float]) -> RegimeState:
        """
        Detect current regime from volatility surface metrics.
        
        Args:
            metrics: Dict with 'atm_vol', 'skew', 'curvature', 'term_slope'
            
        Returns:
            RegimeState with full regime information
        """
        obs = np.array([
            metrics.get('atm_vol', 0.15),
            metrics.get('skew', 0.03),
            metrics.get('curvature', 0.01),
            metrics.get('term_slope', 0.0)
        ])
        
        if HMM_AVAILABLE and self.is_fitted and hasattr(self.hmm, 'predict_proba'):
            # Use hmmlearn
            if self.scaler:
                obs_scaled = self.scaler.transform(obs.reshape(1, -1))
            else:
                obs_scaled = obs.reshape(1, -1)
            
            state = self.hmm.predict(obs_scaled)[0]
            proba = self.hmm.predict_proba(obs_scaled)[0]
            state = self.state_mapping.get(state, state)
        else:
            # Use simple HMM
            state, proba = self.hmm.update(obs)
        
        # Check for regime change
        prev_state = self.current_state
        if state != prev_state:
            self.regime_start_time = datetime.now()
            logger.info(f"Regime change: {self.REGIME_NAMES[prev_state]} -> {self.REGIME_NAMES[state]}")
        
        self.current_state = state
        self.state_history.append(state)
        
        # Calculate persistence (how long in current regime)
        persistence = (datetime.now() - self.regime_start_time).total_seconds() / 3600  # hours
        
        # Get transition probabilities
        if isinstance(self.hmm, SimpleHMM):
            trans_probs = self.hmm.get_transition_probs(state)
        else:
            trans_probs = self.hmm.transmat_[state] if hasattr(self.hmm, 'transmat_') else np.ones(4) / 4
        
        result = RegimeState(
            regime=self.REGIME_NAMES[state],
            regime_id=state,
            confidence=float(proba[state]) if len(proba) > state else 0.5,
            probabilities={self.REGIME_NAMES[i]: float(p) for i, p in enumerate(proba[:4])},
            transition_probs={self.REGIME_NAMES[i]: float(p) for i, p in enumerate(trans_probs[:4])},
            persistence=persistence,
            features=metrics,
            timestamp=datetime.now()
        )
        
        self.history.append(result)
        self._save_model()
        
        return result
    
    def get_playbook(self, regime_id: Optional[int] = None) -> Dict:
        """Get strategy playbook for current or specified regime."""
        if regime_id is None:
            regime_id = self.current_state
        return self.PLAYBOOKS.get(regime_id, self.PLAYBOOKS[0])
    
    def get_regime_summary(self) -> Dict:
        """Get summary of current regime with actionable info."""
        playbook = self.get_playbook()
        
        # Calculate stability (how often we've been in this state recently)
        recent = self.state_history[-20:] if len(self.state_history) >= 20 else self.state_history
        if recent:
            stability = sum(1 for s in recent if s == self.current_state) / len(recent)
        else:
            stability = 0.5
        
        return {
            'regime': self.REGIME_NAMES[self.current_state],
            'regime_id': self.current_state,
            'color': self.REGIME_COLORS[self.current_state],
            'stability': stability,
            'playbook': playbook,
            'recent_history': [self.REGIME_NAMES[s] for s in self.state_history[-10:]]
        }


# Convenience function
def get_regime_detector() -> RegimeHMM:
    """Get or create the global regime detector."""
    return RegimeHMM()


def detect_regime(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    One-shot regime detection from vol surface metrics.
    
    Args:
        metrics: Dict with atm_vol, skew, curvature, term_slope
        
    Returns:
        Complete regime assessment with playbook
    """
    detector = get_regime_detector()
    state = detector.detect(metrics)
    summary = detector.get_regime_summary()
    
    return {
        'state': state.to_dict(),
        'summary': summary
    }

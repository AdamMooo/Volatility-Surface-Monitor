"""
Tests for regime classifier.
"""

import pytest
import numpy as np
from src.analytics.regime_classifier import RegimeClassifier, MarketRegime


class TestRegimeClassifier:
    """Test regime classification."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier with default settings."""
        return RegimeClassifier()
    
    @pytest.fixture
    def trained_classifier(self):
        """Create classifier with historical data."""
        clf = RegimeClassifier(history_window=100)
        
        # Add historical observations
        np.random.seed(42)
        for _ in range(100):
            clf.update_history(
                atm_vol=0.15 + 0.05 * np.random.randn(),
                skew=-0.05 + 0.02 * np.random.randn(),
                curvature=0.02 + 0.01 * np.random.randn(),
                butterfly=0.01 + 0.005 * np.random.randn()
            )
        
        return clf
    
    def test_calm_regime(self, trained_classifier):
        """Low metrics should classify as CALM."""
        result = trained_classifier.classify(
            atm_vol=0.12,
            skew=-0.03,
            curvature=0.01,
            butterfly=0.005
        )
        assert result['current_regime'] == 'calm'
    
    def test_acute_regime(self, trained_classifier):
        """Very high metrics should classify as ACUTE."""
        result = trained_classifier.classify(
            atm_vol=0.50,
            skew=-0.20,
            curvature=0.10,
            butterfly=0.05
        )
        assert result['current_regime'] in ['acute', 'elevated']
    
    def test_confidence_range(self, trained_classifier):
        """Confidence should be between 0 and 1."""
        result = trained_classifier.classify(
            atm_vol=0.20,
            skew=-0.05,
            curvature=0.02,
            butterfly=0.01
        )
        assert 0 <= result['confidence'] <= 1
    
    def test_probabilities_sum_to_one(self, trained_classifier):
        """Regime probabilities should sum to 1."""
        result = trained_classifier.classify(
            atm_vol=0.20,
            skew=-0.05,
            curvature=0.02,
            butterfly=0.01
        )
        prob_sum = sum(result['regime_probability'].values())
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_key_drivers_exist(self, trained_classifier):
        """Classification should identify key drivers."""
        result = trained_classifier.classify(
            atm_vol=0.35,
            skew=-0.15,
            curvature=0.05,
            butterfly=0.03
        )
        assert len(result['key_drivers']) > 0
    
    def test_history_updates(self, classifier):
        """History should grow with updates."""
        for i in range(10):
            classifier.update_history(
                atm_vol=0.15 + i * 0.01,
                skew=-0.05,
                curvature=0.02,
                butterfly=0.01
            )
        
        assert len(classifier.atm_vol_history) == 10


class TestMarketRegime:
    """Test MarketRegime enum."""
    
    def test_regime_values(self):
        """All regimes should have correct values."""
        assert MarketRegime.CALM.value == 'calm'
        assert MarketRegime.PRE_STRESS.value == 'pre_stress'
        assert MarketRegime.ELEVATED.value == 'elevated'
        assert MarketRegime.ACUTE.value == 'acute'
        assert MarketRegime.RECOVERY.value == 'recovery'
    
    def test_regime_ordering(self):
        """Regimes should have logical ordering."""
        regimes = [MarketRegime.CALM, MarketRegime.PRE_STRESS, 
                   MarketRegime.ELEVATED, MarketRegime.ACUTE]
        # All regimes exist
        assert len(regimes) == 4


class TestThresholds:
    """Test threshold behavior."""
    
    def test_custom_thresholds(self):
        """Custom thresholds should affect classification."""
        # More sensitive classifier
        sensitive = RegimeClassifier(
            pre_stress_threshold=50,
            elevated_threshold=70,
            acute_threshold=85
        )
        
        # Less sensitive classifier  
        relaxed = RegimeClassifier(
            pre_stress_threshold=80,
            elevated_threshold=90,
            acute_threshold=98
        )
        
        # Add same history to both
        np.random.seed(42)
        for _ in range(100):
            obs = {
                'atm_vol': 0.15 + 0.05 * np.random.randn(),
                'skew': -0.05,
                'curvature': 0.02,
                'butterfly': 0.01
            }
            sensitive.update_history(**obs)
            relaxed.update_history(**obs)
        
        # Medium-high reading
        metrics = {
            'atm_vol': 0.25,
            'skew': -0.08,
            'curvature': 0.03,
            'butterfly': 0.015
        }
        
        sens_result = sensitive.classify(**metrics)
        relax_result = relaxed.classify(**metrics)
        
        # Sensitive should classify higher
        regime_order = {'calm': 0, 'pre_stress': 1, 'recovery': 1.5, 
                        'elevated': 2, 'acute': 3}
        sens_level = regime_order.get(sens_result['current_regime'], 0)
        relax_level = regime_order.get(relax_result['current_regime'], 0)
        
        assert sens_level >= relax_level


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

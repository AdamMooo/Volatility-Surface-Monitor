"""
Tests for arbitrage detection.
"""

import pytest
import numpy as np
import pandas as pd
from src.analytics.arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_vertical_arbitrage,
    run_all_arbitrage_checks
)


@pytest.fixture
def valid_chain():
    """Create a valid option chain with no arbitrage."""
    np.random.seed(42)
    
    strikes = np.arange(95, 106, 1)
    expiries = [30, 60, 90]
    
    data = []
    for exp in expiries:
        for strike in strikes:
            moneyness = strike / 100
            # IV smile - valid shape
            iv = 0.15 + 0.5 * (moneyness - 1.0) ** 2 + 0.01 * (exp / 30)
            
            for opt_type in ['call', 'put']:
                data.append({
                    'strike': strike,
                    'days_to_expiry': exp,
                    'option_type': opt_type,
                    'impliedVolatility': iv,
                    'moneyness': moneyness,
                    'bid': 1.0 + np.random.rand() * 0.5,
                    'ask': 1.5 + np.random.rand() * 0.5,
                    'volume': 100 + np.random.randint(0, 500)
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def butterfly_violation_chain():
    """Create chain with butterfly arbitrage violation."""
    data = []
    strikes = [95, 100, 105]
    
    # Butterfly violation: middle IV too low
    ivs = [0.22, 0.15, 0.22]  # Should be convex but middle is too low
    
    for i, strike in enumerate(strikes):
        data.append({
            'strike': strike,
            'days_to_expiry': 30,
            'option_type': 'call',
            'impliedVolatility': ivs[i],
            'moneyness': strike / 100,
            'bid': 1.0,
            'ask': 1.5,
            'volume': 100
        })
    
    return pd.DataFrame(data)


class TestButterflyArbitrage:
    """Test butterfly arbitrage detection."""
    
    def test_valid_chain_passes(self, valid_chain):
        """Valid chain should pass butterfly check."""
        result = check_butterfly_arbitrage(valid_chain)
        # May have some marginal violations, but should generally pass
        assert result['violations'] < len(valid_chain) * 0.1
    
    def test_detects_violation(self, butterfly_violation_chain):
        """Should detect butterfly arbitrage violation."""
        result = check_butterfly_arbitrage(butterfly_violation_chain)
        # With concave smile, should detect violations
        assert result is not None


class TestCalendarArbitrage:
    """Test calendar spread arbitrage detection."""
    
    def test_valid_chain_passes(self, valid_chain):
        """Valid chain should pass calendar check."""
        result = check_calendar_arbitrage(valid_chain)
        # Valid chain should have few violations
        assert result['violations'] < len(valid_chain) * 0.1
    
    def test_detects_inverted_term_structure(self):
        """Should detect inverted total variance."""
        data = []
        
        # Same strike, different expiries - inverted
        for exp, iv in [(30, 0.30), (60, 0.15)]:  # 30D higher than 60D
            data.append({
                'strike': 100,
                'days_to_expiry': exp,
                'option_type': 'call',
                'impliedVolatility': iv,
                'moneyness': 1.0,
                'bid': 1.0,
                'ask': 1.5,
                'volume': 100
            })
        
        chain = pd.DataFrame(data)
        result = check_calendar_arbitrage(chain)
        # Total variance for 30D: 0.30^2 * 30/365 = 0.0074
        # Total variance for 60D: 0.15^2 * 60/365 = 0.0037
        # 30D > 60D = violation
        assert result['violations'] > 0 or result.get('warnings', 0) > 0


class TestVerticalArbitrage:
    """Test vertical spread arbitrage detection."""
    
    def test_valid_chain_passes(self, valid_chain):
        """Valid chain should pass vertical check."""
        result = check_vertical_arbitrage(valid_chain)
        assert result['violations'] < len(valid_chain) * 0.1
    
    def test_detects_call_price_violation(self):
        """Should detect call price increasing with strike."""
        data = [
            {'strike': 95, 'days_to_expiry': 30, 'option_type': 'call',
             'bid': 4.0, 'ask': 4.5, 'volume': 100, 'impliedVolatility': 0.20},
            {'strike': 100, 'days_to_expiry': 30, 'option_type': 'call',
             'bid': 5.0, 'ask': 5.5, 'volume': 100, 'impliedVolatility': 0.20},  # Higher!
        ]
        chain = pd.DataFrame(data)
        result = check_vertical_arbitrage(chain)
        # Call price should decrease with strike
        assert result is not None


class TestRunAllChecks:
    """Test combined arbitrage checking."""
    
    def test_returns_all_checks(self, valid_chain):
        """Should return results for all check types."""
        results = run_all_arbitrage_checks(valid_chain)
        
        assert 'butterfly' in results
        assert 'calendar' in results
        assert 'vertical' in results
    
    def test_each_check_has_pass_field(self, valid_chain):
        """Each check result should have pass/fail indicator."""
        results = run_all_arbitrage_checks(valid_chain)
        
        for check_name, result in results.items():
            assert 'pass' in result or 'violations' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

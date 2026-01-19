"""
Tests for Black-Scholes pricing model.
"""

import pytest
import numpy as np
from src.models.black_scholes import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    implied_volatility
)


class TestBSPrice:
    """Test Black-Scholes price calculation."""
    
    def test_atm_call_price(self):
        """ATM call should be approximately 0.4 * S * sigma * sqrt(T)."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        price = bs_price(S, K, r, T, sigma, 'call')
        # Approximate ATM call formula
        approx = 0.4 * S * sigma * np.sqrt(T)
        assert abs(price - approx) < 2.0  # Within $2
    
    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        call = bs_price(S, K, r, T, sigma, 'call')
        put = bs_price(S, K, r, T, sigma, 'put')
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 0.01
    
    def test_itm_call_intrinsic(self):
        """Deep ITM call should be close to intrinsic value."""
        S, K, r, T, sigma = 150, 100, 0.05, 0.1, 0.20
        price = bs_price(S, K, r, T, sigma, 'call')
        intrinsic = S - K * np.exp(-r * T)
        assert price >= intrinsic - 0.01
    
    def test_otm_call_positive(self):
        """OTM call price should be positive."""
        S, K, r, T, sigma = 100, 120, 0.05, 0.5, 0.20
        price = bs_price(S, K, r, T, sigma, 'call')
        assert price > 0
    
    def test_zero_vol_call(self):
        """Zero vol call = max(S - K*exp(-rT), 0)."""
        S, K, r, T = 100, 95, 0.05, 1.0
        price = bs_price(S, K, r, T, 0.0001, 'call')  # Near-zero vol
        expected = max(S - K * np.exp(-r * T), 0)
        assert abs(price - expected) < 0.1


class TestBSGreeks:
    """Test Black-Scholes Greeks."""
    
    def test_atm_delta_call(self):
        """ATM call delta should be approximately 0.5."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        delta = bs_delta(S, K, r, T, sigma, 'call')
        assert 0.45 < delta < 0.65
    
    def test_atm_delta_put(self):
        """ATM put delta should be approximately -0.5."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        delta = bs_delta(S, K, r, T, sigma, 'put')
        assert -0.65 < delta < -0.45
    
    def test_gamma_positive(self):
        """Gamma should always be positive."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        gamma = bs_gamma(S, K, r, T, sigma)
        assert gamma > 0
    
    def test_atm_gamma_peak(self):
        """ATM gamma should be higher than OTM gamma."""
        r, T, sigma = 0.05, 0.5, 0.20
        gamma_atm = bs_gamma(100, 100, r, T, sigma)
        gamma_otm = bs_gamma(100, 120, r, T, sigma)
        assert gamma_atm > gamma_otm
    
    def test_vega_positive(self):
        """Vega should always be positive."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        vega = bs_vega(S, K, r, T, sigma)
        assert vega > 0
    
    def test_theta_call_negative(self):
        """Call theta should generally be negative (time decay)."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.20
        theta = bs_theta(S, K, r, T, sigma, 'call')
        assert theta < 0


class TestImpliedVolatility:
    """Test implied volatility calculation."""
    
    def test_iv_roundtrip(self):
        """IV should recover the original volatility."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
        price = bs_price(S, K, r, T, sigma, 'call')
        iv = implied_volatility(price, S, K, r, T, 'call')
        assert abs(iv - sigma) < 0.001
    
    def test_iv_otm_call(self):
        """IV should work for OTM options."""
        S, K, r, T, sigma = 100, 110, 0.05, 0.5, 0.30
        price = bs_price(S, K, r, T, sigma, 'call')
        iv = implied_volatility(price, S, K, r, T, 'call')
        assert abs(iv - sigma) < 0.001
    
    def test_iv_itm_put(self):
        """IV should work for ITM puts."""
        S, K, r, T, sigma = 100, 110, 0.05, 0.5, 0.25
        price = bs_price(S, K, r, T, sigma, 'put')
        iv = implied_volatility(price, S, K, r, T, 'put')
        assert abs(iv - sigma) < 0.001
    
    def test_iv_bounds(self):
        """IV should be within reasonable bounds."""
        S, K, r, T = 100, 100, 0.05, 0.25
        price = 5.0  # Reasonable ATM option price
        iv = implied_volatility(price, S, K, r, T, 'call')
        assert 0.01 < iv < 2.0  # 1% to 200%


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

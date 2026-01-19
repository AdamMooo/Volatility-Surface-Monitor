"""
Tests for geometry analytics.
"""

import pytest
import numpy as np
import pandas as pd
from src.analytics.geometry import (
    compute_skew,
    compute_curvature,
    compute_roughness,
    compute_25delta_skew,
    compute_wing_curvature,
    compute_all_geometry_metrics
)


class TestSkew:
    """Test skew computation."""
    
    def test_flat_smile_zero_skew(self):
        """Flat smile should have zero skew."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = np.ones_like(moneyness) * 0.20
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        skew = compute_skew(smile)
        assert abs(skew) < 0.001
    
    def test_negative_skew(self):
        """Downward sloping smile should have negative skew."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.20 - 0.2 * (moneyness - 1.0)  # Higher IV for lower strikes
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        skew = compute_skew(smile)
        assert skew < 0
    
    def test_positive_skew(self):
        """Upward sloping smile should have positive skew."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.20 + 0.2 * (moneyness - 1.0)  # Higher IV for higher strikes
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        skew = compute_skew(smile)
        assert skew > 0


class TestCurvature:
    """Test curvature computation."""
    
    def test_flat_smile_zero_curvature(self):
        """Flat smile should have zero curvature."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = np.ones_like(moneyness) * 0.20
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        curv = compute_curvature(smile)
        assert abs(curv) < 0.001
    
    def test_linear_smile_zero_curvature(self):
        """Linear smile should have zero curvature."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.20 + 0.1 * (moneyness - 1.0)
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        curv = compute_curvature(smile)
        assert abs(curv) < 0.01
    
    def test_convex_smile_positive_curvature(self):
        """Convex smile should have positive curvature."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.15 + 0.5 * (moneyness - 1.0) ** 2
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        curv = compute_curvature(smile)
        assert curv > 0
    
    def test_concave_smile_negative_curvature(self):
        """Concave smile should have negative curvature."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.25 - 0.5 * (moneyness - 1.0) ** 2
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        curv = compute_curvature(smile)
        assert curv < 0


class TestRoughness:
    """Test roughness computation."""
    
    def test_smooth_smile_low_roughness(self):
        """Smooth smile should have low roughness."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.15 + 0.5 * (moneyness - 1.0) ** 2
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        rough = compute_roughness(smile)
        assert rough < 0.01
    
    def test_noisy_smile_high_roughness(self):
        """Noisy smile should have higher roughness."""
        np.random.seed(42)
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.20 + 0.05 * np.random.randn(50)
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        rough = compute_roughness(smile)
        assert rough > 0.001


class Test25DeltaSkew:
    """Test 25-delta skew computation."""
    
    def test_symmetric_smile_zero_skew(self):
        """Symmetric smile should have near-zero 25D skew."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.15 + 0.5 * (moneyness - 1.0) ** 2
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        skew_25d = compute_25delta_skew(smile)
        # Symmetric parabola should have symmetric wings
        assert abs(skew_25d) < 0.01
    
    def test_put_wing_higher(self):
        """Higher put wing should give positive 25D skew."""
        moneyness = np.linspace(0.8, 1.2, 50)
        # Asymmetric - higher on put side
        iv = 0.20 - 0.15 * (moneyness - 1.0)
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        skew_25d = compute_25delta_skew(smile)
        assert skew_25d > 0


class TestWingCurvature:
    """Test wing curvature computation."""
    
    def test_symmetric_wings(self):
        """Symmetric smile should have similar wing curvatures."""
        moneyness = np.linspace(0.8, 1.2, 50)
        iv = 0.15 + 0.5 * (moneyness - 1.0) ** 2
        smile = pd.DataFrame({'moneyness': moneyness, 'iv': iv})
        left_curv, right_curv = compute_wing_curvature(smile)
        assert abs(left_curv - right_curv) < 0.1 * max(abs(left_curv), abs(right_curv))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

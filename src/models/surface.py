"""
IV Surface Construction Module

Responsibilities:
1. Build raw IV surface from discrete option data
2. Interpolate to create smooth, continuous surface
3. Support multiple interpolation methods
4. Ensure surface is well-behaved (no negative IVs, reasonable extrapolation)
"""

from typing import Optional, Callable, Tuple, List, Dict, Any, Literal
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf, RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter
from loguru import logger

from .black_scholes import implied_volatility


class IVSurface:
    """
    Implied Volatility Surface object.
    
    Handles construction, interpolation, and evaluation of 
    the IV surface from discrete option data.
    """
    
    def __init__(
        self,
        interpolation_method: Literal['rbf', 'spline', 'linear'] = 'rbf',
        smoothing: float = 0.0,
        use_moneyness: bool = True
    ):
        """
        Initialize the IV surface.
        
        Args:
            interpolation_method: Method for interpolation
            smoothing: Smoothing parameter for interpolation
            use_moneyness: If True, use log-moneyness; else use strikes
        """
        self.interpolation_method = interpolation_method
        self.smoothing = smoothing
        self.use_moneyness = use_moneyness
        
        self._raw_data: Optional[pd.DataFrame] = None
        self._interpolator: Optional[Callable] = None
        self._spot: Optional[float] = None
        self._maturities: List[float] = []
        self._moneyness_range: Tuple[float, float] = (-0.5, 0.5)
        self._strike_range: Tuple[float, float] = (0, 0)
    
    def build(
        self,
        option_data: pd.DataFrame,
        spot: Optional[float] = None,
        rate: Optional[float] = None
    ) -> 'IVSurface':
        """
        Build the IV surface from option data.
        
        Args:
            option_data: DataFrame with option chain data
            spot: Spot price (if not in data)
            rate: Risk-free rate (if not in data)
            
        Returns:
            self for method chaining
        """
        if spot is None:
            if 'underlying_price' in option_data.columns:
                spot = option_data['underlying_price'].iloc[0]
            else:
                raise ValueError("Spot price required")
        
        self._spot = spot
        
        raw_surface = self._build_raw_surface(option_data, spot, rate)
        
        if raw_surface.empty:
            raise ValueError("No valid IV data to build surface")
        
        self._raw_data = raw_surface
        self._interpolator = self._create_interpolator(raw_surface)
        
        self._maturities = sorted(raw_surface['time_to_expiry'].unique())
        
        if self.use_moneyness:
            self._moneyness_range = (
                raw_surface['log_moneyness'].min(),
                raw_surface['log_moneyness'].max()
            )
        else:
            self._strike_range = (
                raw_surface['strike'].min(),
                raw_surface['strike'].max()
            )
        
        logger.info(f"Built IV surface with {len(raw_surface)} points, "
                   f"{len(self._maturities)} maturities")
        
        return self
    
    def _build_raw_surface(
        self,
        option_data: pd.DataFrame,
        spot: float,
        rate: Optional[float]
    ) -> pd.DataFrame:
        """Build raw IV surface from option data."""
        df = option_data.copy()
        
        required_cols = ['strike', 'mid_price', 'option_type']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'time_to_expiry' not in df.columns:
            if 'days_to_expiry' in df.columns:
                df['time_to_expiry'] = df['days_to_expiry'] / 365.0
            else:
                raise ValueError("Missing time_to_expiry or days_to_expiry")
        
        if rate is None:
            if 'risk_free_rate' in df.columns:
                rate = df['risk_free_rate'].iloc[0]
            else:
                rate = 0.05
        
        if 'implied_volatility' not in df.columns:
            ivs = []
            for _, row in df.iterrows():
                iv = implied_volatility(
                    price=row['mid_price'],
                    S=spot,
                    K=row['strike'],
                    T=row['time_to_expiry'],
                    r=row.get('risk_free_rate', rate),
                    option_type=row['option_type']
                )
                ivs.append(iv)
            df['implied_volatility'] = ivs
        
        df = df.dropna(subset=['implied_volatility'])
        df = df[df['implied_volatility'] > 0]
        df = df[df['implied_volatility'] < 3.0]
        
        df['moneyness'] = df['strike'] / spot
        
        forward = spot * np.exp(rate * df['time_to_expiry'])
        df['log_moneyness'] = np.log(df['strike'] / forward)
        
        result = df[[
            'strike', 'time_to_expiry', 'implied_volatility',
            'moneyness', 'log_moneyness', 'option_type'
        ]].copy()
        
        result = result.rename(columns={'implied_volatility': 'iv'})
        
        return result
    
    def _create_interpolator(self, raw_surface: pd.DataFrame) -> Callable:
        """Create interpolation function from raw surface."""
        if self.use_moneyness:
            x = raw_surface['log_moneyness'].values
        else:
            x = raw_surface['strike'].values
        
        y = raw_surface['time_to_expiry'].values
        z = raw_surface['iv'].values
        
        if self.interpolation_method == 'rbf':
            return self._create_rbf_interpolator(x, y, z)
        elif self.interpolation_method == 'spline':
            return self._create_spline_interpolator(x, y, z)
        elif self.interpolation_method == 'linear':
            return self._create_linear_interpolator(x, y, z)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
    
    def _create_rbf_interpolator(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> Callable:
        """Create RBF interpolator."""
        rbf = Rbf(x, y, z, function='thin_plate_spline', smooth=self.smoothing)
        
        def interpolator(x_new, y_new):
            result = rbf(x_new, y_new)
            return np.maximum(0.01, result)
        
        return interpolator
    
    def _create_spline_interpolator(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> Callable:
        """Create 2D spline interpolator (requires gridded data)."""
        x_unique = np.sort(np.unique(x))
        y_unique = np.sort(np.unique(y))
        
        z_grid = griddata(
            (x, y), z, 
            (x_unique[None, :], y_unique[:, None]),
            method='cubic',
            fill_value=np.nan
        )
        
        mask = ~np.isnan(z_grid)
        if not mask.all():
            z_grid = np.where(mask, z_grid, np.nanmean(z))
        
        spline = RectBivariateSpline(y_unique, x_unique, z_grid, s=self.smoothing)
        
        def interpolator(x_new, y_new):
            result = spline(y_new, x_new, grid=False)
            return np.maximum(0.01, result)
        
        return interpolator
    
    def _create_linear_interpolator(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray
    ) -> Callable:
        """Create linear interpolator using scipy griddata."""
        points = np.column_stack([x, y])
        
        def interpolator(x_new, y_new):
            x_new = np.atleast_1d(x_new)
            y_new = np.atleast_1d(y_new)
            xi = np.column_stack([x_new, y_new])
            result = griddata(points, z, xi, method='linear', fill_value=np.nan)
            result = np.where(np.isnan(result), np.nanmean(z), result)
            return np.maximum(0.01, result)
        
        return interpolator
    
    def __call__(self, K: float, T: float) -> float:
        """
        Evaluate IV at a given strike and maturity.
        
        Args:
            K: Strike price (or moneyness if use_moneyness=True)
            T: Time to maturity in years
            
        Returns:
            Implied volatility at (K, T)
        """
        return self.evaluate(K, T)
    
    def evaluate(self, K: float, T: float) -> float:
        """
        Evaluate IV at a given strike and maturity.
        
        Args:
            K: Strike price
            T: Time to maturity in years
            
        Returns:
            Implied volatility at (K, T)
        """
        if self._interpolator is None:
            raise ValueError("Surface not built. Call build() first.")
        
        if self.use_moneyness:
            x = np.log(K / self._spot) if self._spot else np.log(K)
        else:
            x = K
        
        result = self._interpolator(x, T)
        
        if hasattr(result, '__len__'):
            return float(result[0]) if len(result) == 1 else float(result)
        return float(result)
    
    def evaluate_grid(
        self,
        K_range: Tuple[float, float],
        T_range: Tuple[float, float],
        K_steps: int = 50,
        T_steps: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate IV on a regular grid.
        
        Args:
            K_range: (min_strike, max_strike) or moneyness range
            T_range: (min_maturity, max_maturity) in years
            K_steps: Number of strike steps
            T_steps: Number of maturity steps
            
        Returns:
            Tuple of (K_grid, T_grid, IV_grid)
        """
        if self._interpolator is None:
            raise ValueError("Surface not built. Call build() first.")
        
        K_vals = np.linspace(K_range[0], K_range[1], K_steps)
        T_vals = np.linspace(T_range[0], T_range[1], T_steps)
        
        K_grid, T_grid = np.meshgrid(K_vals, T_vals)
        
        if self.use_moneyness:
            x_grid = np.log(K_grid / self._spot) if self._spot else np.log(K_grid)
        else:
            x_grid = K_grid
        
        IV_grid = self._interpolator(x_grid.ravel(), T_grid.ravel())
        IV_grid = IV_grid.reshape(K_grid.shape)
        
        return K_grid, T_grid, IV_grid
    
    def get_atm_vol(self, T: float) -> float:
        """Get ATM implied volatility for a given maturity."""
        if self._spot is None:
            raise ValueError("Spot price not set")
        return self.evaluate(self._spot, T)
    
    def get_smile(
        self, 
        T: float, 
        K_range: Optional[Tuple[float, float]] = None,
        num_points: int = 50
    ) -> pd.DataFrame:
        """
        Get the volatility smile for a fixed maturity.
        
        Args:
            T: Time to maturity
            K_range: Strike range (defaults to available range)
            num_points: Number of points
            
        Returns:
            DataFrame with strike and IV columns
        """
        if K_range is None:
            if self.use_moneyness and self._spot:
                K_range = (
                    self._spot * np.exp(self._moneyness_range[0]),
                    self._spot * np.exp(self._moneyness_range[1])
                )
            else:
                K_range = self._strike_range
        
        strikes = np.linspace(K_range[0], K_range[1], num_points)
        ivs = [self.evaluate(K, T) for K in strikes]
        
        return pd.DataFrame({
            'strike': strikes,
            'iv': ivs,
            'moneyness': strikes / self._spot if self._spot else strikes
        })
    
    def get_term_structure(
        self, 
        K: Optional[float] = None,
        maturities: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Get the term structure of IV for a fixed strike.
        
        Args:
            K: Strike (defaults to ATM)
            maturities: List of maturities (defaults to available)
            
        Returns:
            DataFrame with maturity and IV columns
        """
        if K is None:
            K = self._spot
        
        if maturities is None:
            maturities = self._maturities
        
        ivs = [self.evaluate(K, T) for T in maturities]
        
        return pd.DataFrame({
            'maturity': maturities,
            'iv': ivs
        })
    
    @property
    def spot(self) -> Optional[float]:
        """Current spot price."""
        return self._spot
    
    @property
    def maturities(self) -> List[float]:
        """Available maturities."""
        return self._maturities
    
    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        """Raw IV surface data."""
        return self._raw_data
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export surface as DataFrame."""
        if self._raw_data is None:
            return pd.DataFrame()
        return self._raw_data.copy()


def build_iv_surface(
    option_data: pd.DataFrame,
    spot: Optional[float] = None,
    method: str = 'rbf',
    smoothing: float = 0.0
) -> IVSurface:
    """
    Convenience function to build an IV surface.
    
    Args:
        option_data: Option chain DataFrame
        spot: Spot price
        method: Interpolation method
        smoothing: Smoothing parameter
        
    Returns:
        Constructed IVSurface object
    """
    surface = IVSurface(
        interpolation_method=method,
        smoothing=smoothing
    )
    return surface.build(option_data, spot=spot)
    
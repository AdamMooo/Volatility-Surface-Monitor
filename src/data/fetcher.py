"""
Data Fetcher Module

Responsibilities:
1. Pull option chains for SPX/SPY from yfinance
2. Handle rate limiting and retries
3. Extract all available strikes and expirations
4. Compute mid-prices from bid/ask
5. Fetch underlying spot price
6. Fetch risk-free rate (Treasury yields or static assumption)
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logger.warning("fredapi not available. FRED data fetching disabled.")


class RiskFreeRateFetcher:
    """Fetches risk-free rates from Treasury yields."""
    
    TREASURY_TICKERS = {
        30: "^IRX",    # 13-week T-bill
        90: "^IRX",    # 13-week T-bill
        180: "^FVX",   # 5-year Treasury
        365: "^TNX",   # 10-year Treasury
    }
    
    DEFAULT_RATE = 0.05  # 5% default if fetch fails
    
    def __init__(self):
        self._cache: Dict[int, float] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)
    
    def get_risk_free_rate(self, maturity_days: int) -> float:
        """
        Get risk-free rate for a given maturity.
        
        Args:
            maturity_days: Days to maturity
            
        Returns:
            Annual risk-free rate as decimal (e.g., 0.05 for 5%)
        """
        if self._is_cache_valid():
            if maturity_days in self._cache:
                return self._cache[maturity_days]
        
        try:
            ticker = self._get_treasury_ticker(maturity_days)
            data = yf.Ticker(ticker)
            hist = data.history(period="1d")
            
            if not hist.empty:
                rate = hist['Close'].iloc[-1] / 100.0
                self._cache[maturity_days] = rate
                self._cache_time = datetime.now()
                return rate
        except Exception as e:
            logger.warning(f"Failed to fetch risk-free rate: {e}. Using default.")
        
        return self.DEFAULT_RATE
    
    def _get_treasury_ticker(self, maturity_days: int) -> str:
        """Get appropriate Treasury ticker for maturity."""
        for days, ticker in sorted(self.TREASURY_TICKERS.items()):
            if maturity_days <= days:
                return ticker
        return "^TNX"
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_duration


class FREDDataFetcher:
    """
    Fetches macroeconomic data from FRED (Federal Reserve Economic Data).
    
    Provides orthogonal features for volatility regime modeling.
    """
    
    # Key FRED series for volatility modeling
    SERIES = {
        'unemployment_rate': 'UNRATE',          # Unemployment rate
        'fed_funds_rate': 'FEDFUNDS',           # Federal funds rate
        'treasury_10y': 'GS10',                 # 10-year treasury rate
        'treasury_2y': 'GS2',                   # 2-year treasury rate
        'yield_curve_spread': 'T10Y2Y',         # 10Y-2Y spread
        'cpi_inflation': 'CPIAUCSL',            # CPI (inflation)
        'industrial_production': 'INDPRO',      # Industrial production index
        'consumer_sentiment': 'UMCSENT',        # Consumer sentiment
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED fetcher.
        
        Args:
            api_key: FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
        """
        if not FRED_AVAILABLE:
            raise ImportError("fredapi not installed. Install with: pip install fredapi")
            
        self.api_key = api_key or "9284d1ec1af2e1c12114b61681304e81"  # Real FRED API key
        if self.api_key == "your_fred_api_key_here":
            logger.warning("FRED API key not set. Using demo mode with limited data.")
            self.fred = None
        else:
            self.fred = Fred(api_key=self.api_key)
        self._cache: Dict[str, pd.Series] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=6)  # FRED data updates less frequently
    
    def fetch_economic_indicators(self, start_date: str = '2020-01-01') -> Dict[str, pd.Series]:
        """
        Fetch historical economic indicators for regime modeling.
        
        Args:
            start_date: Start date for historical context
            
        Returns:
            Dictionary of pandas Series with historical data
        """
        if not self._is_cache_valid():
            self._refresh_cache(start_date)
        
        return self._cache
    
    def _refresh_cache(self, start_date: str):
        """Refresh cached FRED data."""
        self._cache = {}
        if self.fred is None:
            logger.warning("FRED API not available. Using synthetic economic data.")
            return
            
        for name, series_id in self.SERIES.items():
            try:
                series = self.fred.get_series(series_id, start_date)
                self._cache[series_id] = series
                logger.info(f"Fetched FRED series {series_id}")
            except Exception as e:
                logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
        
        self._cache_time = datetime.now()
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        return datetime.now() - self._cache_time < self._cache_duration


class OptionChainFetcher:
    """
    Main class for option chain data acquisition.
    
    Fetches option chains from yfinance with retry logic,
    computes mid-prices, and standardizes output format.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the fetcher.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (uses exponential backoff)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_fetcher = RiskFreeRateFetcher()
    
    def fetch_option_chain(
        self, 
        ticker: str, 
        expiration: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch option chain for a ticker.
        
        Args:
            ticker: Stock/ETF ticker symbol (e.g., 'SPY')
            expiration: Specific expiration date (YYYY-MM-DD). 
                       If None, fetches all available expirations.
        
        Returns:
            DataFrame with standardized option chain data
        """
        stock = self._get_ticker_with_retry(ticker)
        spot_price = self.get_spot_price(ticker)
        
        if expiration:
            expirations = [expiration]
        else:
            expirations = self.get_all_expirations(ticker)
        
        all_chains = []
        
        for exp in expirations:
            try:
                chain = self._fetch_single_expiration(stock, exp, spot_price)
                if chain is not None and not chain.empty:
                    all_chains.append(chain)
            except Exception as e:
                logger.warning(f"Failed to fetch expiration {exp}: {e}")
                continue
        
        if not all_chains:
            raise ValueError(f"No option data available for {ticker}")
        
        result = pd.concat(all_chains, ignore_index=True)
        result = self._add_derived_columns(result, spot_price)
        
        return result
    
    def get_all_expirations(self, ticker: str) -> List[str]:
        """Get all available expiration dates for a ticker."""
        stock = self._get_ticker_with_retry(ticker)
        return list(stock.options)
    
    def get_spot_price(self, ticker: str) -> float:
        """Get current spot price for a ticker."""
        stock = self._get_ticker_with_retry(ticker)
        hist = stock.history(period="1d")
        
        if hist.empty:
            raise ValueError(f"Could not fetch spot price for {ticker}")
        
        return float(hist['Close'].iloc[-1])
    
    def _get_ticker_with_retry(self, ticker: str) -> yf.Ticker:
        """Get yfinance Ticker object with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return yf.Ticker(ticker)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise
    
    def _fetch_single_expiration(
        self, 
        stock: yf.Ticker, 
        expiration: str, 
        spot_price: float
    ) -> Optional[pd.DataFrame]:
        """Fetch option chain for a single expiration."""
        try:
            opt = stock.option_chain(expiration)
        except Exception as e:
            logger.warning(f"Failed to get option chain for {expiration}: {e}")
            return None
        
        calls = opt.calls.copy()
        puts = opt.puts.copy()
        
        calls['option_type'] = 'call'
        puts['option_type'] = 'put'
        
        chain = pd.concat([calls, puts], ignore_index=True)
        
        if chain.empty:
            return None
        
        chain['expiration'] = pd.to_datetime(expiration)
        chain['underlying_price'] = spot_price
        
        return chain
    
    def _add_derived_columns(self, df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """Add derived columns to the option chain."""
        df = df.copy()
        
        now = pd.Timestamp.now()
        df['days_to_expiry'] = (df['expiration'] - now).dt.days
        df['time_to_expiry'] = df['days_to_expiry'] / 365.0
        
        df['mid_price'] = (df['bid'] + df['ask']) / 2.0
        
        df['moneyness'] = df['strike'] / spot_price
        df['log_moneyness'] = np.log(df['moneyness'])
        
        df['risk_free_rate'] = df['days_to_expiry'].apply(
            lambda d: self.rate_fetcher.get_risk_free_rate(int(d)) if d > 0 else 0.05
        )
        
        column_mapping = {
            'strike': 'strike',
            'expiration': 'expiration',
            'days_to_expiry': 'days_to_expiry',
            'time_to_expiry': 'time_to_expiry',
            'option_type': 'option_type',
            'bid': 'bid',
            'ask': 'ask',
            'mid_price': 'mid_price',
            'volume': 'volume',
            'openInterest': 'open_interest',
            'underlying_price': 'underlying_price',
            'risk_free_rate': 'risk_free_rate',
            'moneyness': 'moneyness',
            'log_moneyness': 'log_moneyness',
            'impliedVolatility': 'implied_volatility_market'
        }
        
        result_columns = []
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                result_columns.append(new_col)
        
        available_columns = [c for c in result_columns if c in df.columns]
        
        return df[available_columns].copy()
    
    def compute_mid_prices(self, chain: pd.DataFrame) -> pd.DataFrame:
        """Compute mid-prices from bid/ask."""
        chain = chain.copy()
        chain['mid_price'] = (chain['bid'] + chain['ask']) / 2.0
        return chain


def fetch_option_data(ticker: str = "SPY") -> pd.DataFrame:
    """
    Convenience function to fetch option data.
    
    Args:
        ticker: Stock/ETF ticker symbol
        
    Returns:
        Cleaned option chain DataFrame
    """
    fetcher = OptionChainFetcher()
    return fetcher.fetch_option_chain(ticker)

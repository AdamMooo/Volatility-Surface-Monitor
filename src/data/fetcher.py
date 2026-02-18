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
from threading import Lock

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.data.cache import DataCache


# Global rate limiter to prevent API abuse
class RateLimiter:
    """Simple rate limiter with exponential backoff."""
    
    def __init__(self, min_interval: float = 2.0, max_interval: float = 60.0):
        self._lock = Lock()
        self._last_call = 0.0
        self._min_interval = min_interval
        self._max_interval = max_interval
        self._current_interval = min_interval
        self._consecutive_errors = 0
    
    def wait(self):
        """Wait appropriate time before next API call."""
        with self._lock:
            elapsed = time.time() - self._last_call
            wait_time = max(0, self._current_interval - elapsed)
            if wait_time > 0:
                logger.debug(f"Rate limiter: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            self._last_call = time.time()
    
    def report_success(self):
        """Report successful API call."""
        with self._lock:
            self._consecutive_errors = 0
            self._current_interval = self._min_interval
    
    def report_error(self):
        """Report API error, increase backoff."""
        with self._lock:
            self._consecutive_errors += 1
            self._current_interval = min(
                self._max_interval,
                self._min_interval * (2 ** self._consecutive_errors)
            )
            logger.warning(f"Rate limit backoff: {self._current_interval:.1f}s")


# Global rate limiter instance
_rate_limiter = RateLimiter(min_interval=1.5, max_interval=60.0)


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


class OptionChainFetcher:
    """
    Main class for option chain data acquisition.
    
    Fetches option chains from yfinance with retry logic,
    computes mid-prices, and standardizes output format.
    """
    
    def __init__(self, max_retries: int = 5, retry_delay: float = 2.0, use_cache: bool = True, cache_max_age_hours: float = 0.5):
        """
        Initialize the fetcher.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (uses exponential backoff)
            use_cache: Whether to use local caching
            cache_max_age_hours: Maximum age of cached data in hours (default 30 min)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_fetcher = RiskFreeRateFetcher()
        self.use_cache = use_cache
        self.cache_max_age_hours = cache_max_age_hours
        self._cache = DataCache() if use_cache else None
    
    def fetch_option_chain(
        self, 
        ticker: str, 
        expiration: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch option chain for a ticker.
        
        Args:
            ticker: Stock/ETF ticker symbol (e.g., 'SPY')
            expiration: Specific expiration date (YYYY-MM-DD). 
                       If None, fetches all available expirations.
            force_refresh: If True, bypass cache and fetch fresh data
        
        Returns:
            DataFrame with standardized option chain data
        """
        # Try cache first (only for full chain fetches without specific expiration)
        if self.use_cache and self._cache and expiration is None and not force_refresh:
            cached = self._cache.get(ticker, max_age_hours=self.cache_max_age_hours)
            if cached is not None and not cached.empty:
                logger.info(f"Using cached data for {ticker} ({len(cached)} rows)")
                return cached
        
        # Apply rate limiting before API calls
        _rate_limiter.wait()
        
        try:
            stock = self._get_ticker_with_retry(ticker)
            spot_price = self.get_spot_price(ticker)
        except Exception as e:
            _rate_limiter.report_error()
            raise
        
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
            _rate_limiter.report_error()
            raise ValueError(f"No option data available for {ticker}")
        
        result = pd.concat(all_chains, ignore_index=True)
        result = self._add_derived_columns(result, spot_price)
        
        # Report success and cache the result
        _rate_limiter.report_success()
        
        if self.use_cache and self._cache and expiration is None:
            self._cache.put(ticker, result)
            logger.info(f"Cached {len(result)} rows for {ticker}")
        
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
                _rate_limiter.report_error()
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    # Check for rate limit specific errors
                    if 'Too Many Requests' in str(e) or '429' in str(e):
                        delay = max(delay, 30)  # Wait at least 30s on rate limit
                        logger.warning(f"Rate limited! Waiting {delay}s before retry {attempt + 1}/{self.max_retries}")
                    else:
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

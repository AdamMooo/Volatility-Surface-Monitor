"""
Data Cache Module

Provides local caching for historical option chain data
to avoid repeated API calls and enable historical analysis.
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from loguru import logger


class DataCache:
    """
    Local file-based cache for option chain data.
    
    Stores data as parquet files for efficiency,
    with JSON metadata for tracking.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()
    
    def get(
        self, 
        ticker: str, 
        date: Optional[datetime] = None,
        max_age_hours: float = 24.0
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and not expired.
        
        Args:
            ticker: Stock ticker symbol
            date: Specific date for the data (default: today)
            max_age_hours: Maximum age of cached data in hours
            
        Returns:
            Cached DataFrame or None if not available/expired
        """
        if date is None:
            date = datetime.now()
        
        cache_key = self._get_cache_key(ticker, date)
        
        if cache_key not in self._metadata:
            return None
        
        entry = self._metadata[cache_key]
        cached_time = datetime.fromisoformat(entry['cached_at'])
        
        if datetime.now() - cached_time > timedelta(hours=max_age_hours):
            logger.debug(f"Cache expired for {cache_key}")
            return None
        
        cache_file = self.cache_dir / entry['filename']
        
        if not cache_file.exists():
            logger.warning(f"Cache file missing: {cache_file}")
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            logger.debug(f"Cache hit for {cache_key}")
            return df
        except Exception as e:
            logger.error(f"Failed to read cache file: {e}")
            return None
    
    def put(
        self, 
        ticker: str, 
        data: pd.DataFrame, 
        date: Optional[datetime] = None
    ) -> None:
        """
        Store data in cache.
        
        Args:
            ticker: Stock ticker symbol
            data: DataFrame to cache
            date: Date for the data (default: today)
        """
        if date is None:
            date = datetime.now()
        
        cache_key = self._get_cache_key(ticker, date)
        filename = f"{cache_key}.parquet"
        cache_file = self.cache_dir / filename
        
        try:
            data.to_parquet(cache_file, index=False)
            
            self._metadata[cache_key] = {
                'ticker': ticker,
                'date': date.isoformat(),
                'cached_at': datetime.now().isoformat(),
                'filename': filename,
                'row_count': len(data)
            }
            
            self._save_metadata()
            logger.debug(f"Cached {len(data)} rows for {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
    
    def invalidate(self, ticker: str, date: Optional[datetime] = None) -> None:
        """
        Invalidate cached data.
        
        Args:
            ticker: Stock ticker symbol
            date: Specific date (if None, invalidates all for ticker)
        """
        keys_to_remove = []
        
        for key, entry in self._metadata.items():
            if entry['ticker'] == ticker:
                if date is None or key == self._get_cache_key(ticker, date):
                    keys_to_remove.append(key)
                    cache_file = self.cache_dir / entry['filename']
                    if cache_file.exists():
                        cache_file.unlink()
        
        for key in keys_to_remove:
            del self._metadata[key]
        
        self._save_metadata()
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
    
    def clear_all(self) -> None:
        """Clear all cached data."""
        for entry in self._metadata.values():
            cache_file = self.cache_dir / entry['filename']
            if cache_file.exists():
                cache_file.unlink()
        
        self._metadata = {}
        self._save_metadata()
        logger.info("Cleared all cache entries")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache contents."""
        total_size = sum(
            (self.cache_dir / entry['filename']).stat().st_size
            for entry in self._metadata.values()
            if (self.cache_dir / entry['filename']).exists()
        )
        
        return {
            'entry_count': len(self._metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'entries': list(self._metadata.keys())
        }
    
    def list_cached_dates(self, ticker: str) -> List[datetime]:
        """List all cached dates for a ticker."""
        dates = []
        for entry in self._metadata.values():
            if entry['ticker'] == ticker:
                dates.append(datetime.fromisoformat(entry['date']))
        return sorted(dates)
    
    def _get_cache_key(self, ticker: str, date: datetime) -> str:
        """Generate cache key from ticker and date."""
        date_str = date.strftime("%Y-%m-%d")
        return f"{ticker.upper()}_{date_str}"
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            return {}
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")


class HistoricalDataStore:
    """
    Store for historical surface snapshots.
    
    Stores daily snapshots of computed metrics
    for regime analysis and backtesting.
    """
    
    def __init__(self, store_dir: str = "data/historical"):
        """
        Initialize the store.
        
        Args:
            store_dir: Directory for historical data
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.surfaces_dir = self.store_dir / "surfaces"
        self.metrics_dir = self.store_dir / "metrics"
        self.surfaces_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
    
    def save_surface_snapshot(
        self, 
        ticker: str, 
        date: datetime, 
        surface_data: pd.DataFrame
    ) -> None:
        """Save a daily surface snapshot."""
        filename = f"{ticker}_{date.strftime('%Y-%m-%d')}_surface.parquet"
        filepath = self.surfaces_dir / filename
        surface_data.to_parquet(filepath, index=False)
        logger.debug(f"Saved surface snapshot: {filename}")
    
    def save_metrics_snapshot(
        self, 
        ticker: str, 
        date: datetime, 
        metrics: Dict[str, Any]
    ) -> None:
        """Save a daily metrics snapshot."""
        filename = f"{ticker}_{date.strftime('%Y-%m-%d')}_metrics.json"
        filepath = self.metrics_dir / filename
        
        serializable_metrics = self._make_serializable(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
        
        logger.debug(f"Saved metrics snapshot: {filename}")
    
    def load_surface_snapshot(
        self, 
        ticker: str, 
        date: datetime
    ) -> Optional[pd.DataFrame]:
        """Load a surface snapshot."""
        filename = f"{ticker}_{date.strftime('%Y-%m-%d')}_surface.parquet"
        filepath = self.surfaces_dir / filename
        
        if not filepath.exists():
            return None
        
        return pd.read_parquet(filepath)
    
    def load_metrics_snapshot(
        self, 
        ticker: str, 
        date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Load a metrics snapshot."""
        filename = f"{ticker}_{date.strftime('%Y-%m-%d')}_metrics.json"
        filepath = self.metrics_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_metrics_history(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Load metrics history for a date range."""
        records = []
        current = start_date
        
        while current <= end_date:
            metrics = self.load_metrics_snapshot(ticker, current)
            if metrics:
                metrics['date'] = current
                records.append(metrics)
            current += timedelta(days=1)
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records)
    
    def list_available_dates(self, ticker: str) -> List[datetime]:
        """List all dates with available data."""
        dates = []
        
        for filepath in self.surfaces_dir.glob(f"{ticker}_*_surface.parquet"):
            date_str = filepath.stem.split('_')[1]
            dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
        
        return sorted(dates)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

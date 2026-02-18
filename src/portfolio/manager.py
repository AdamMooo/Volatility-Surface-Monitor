"""
Portfolio Manager Module

Handles portfolio storage, loading, and basic operations.
Designed for simplicity - stores portfolios as JSON files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd
import yfinance as yf
from loguru import logger


@dataclass
class Position:
    """A single position in a portfolio."""
    ticker: str
    shares: float
    cost_basis: float  # Average cost per share
    date_added: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    
    @property
    def total_cost(self) -> float:
        return self.shares * self.cost_basis


@dataclass 
class Portfolio:
    """A complete portfolio with positions and metadata."""
    name: str
    owner: str
    positions: List[Position] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    
    def add_position(self, ticker: str, shares: float, cost_basis: float, notes: str = ""):
        """Add or update a position."""
        ticker = ticker.upper()
        
        # Check if position exists
        for pos in self.positions:
            if pos.ticker == ticker:
                # Update existing - average the cost basis
                total_shares = pos.shares + shares
                pos.cost_basis = (pos.shares * pos.cost_basis + shares * cost_basis) / total_shares
                pos.shares = total_shares
                if notes:
                    pos.notes = notes
                return
        
        # Add new position
        self.positions.append(Position(
            ticker=ticker,
            shares=shares,
            cost_basis=cost_basis,
            notes=notes
        ))
    
    def remove_position(self, ticker: str):
        """Remove a position entirely."""
        ticker = ticker.upper()
        self.positions = [p for p in self.positions if p.ticker != ticker]
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get a specific position."""
        ticker = ticker.upper()
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None
    
    @property
    def tickers(self) -> List[str]:
        """Get list of all tickers in portfolio."""
        return [p.ticker for p in self.positions]
    
    @property
    def total_cost_basis(self) -> float:
        """Total amount invested."""
        return sum(p.total_cost for p in self.positions)


class PortfolioManager:
    """
    Manages multiple portfolios with file-based persistence.
    
    Simple JSON storage - no database needed.
    """
    
    def __init__(self, data_dir: str = "data/portfolios"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._price_cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None
    
    def save_portfolio(self, portfolio: Portfolio) -> None:
        """Save a portfolio to disk."""
        filename = self._get_filename(portfolio.name)
        
        data = {
            'name': portfolio.name,
            'owner': portfolio.owner,
            'description': portfolio.description,
            'created_at': portfolio.created_at,
            'positions': [asdict(p) for p in portfolio.positions]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved portfolio '{portfolio.name}' with {len(portfolio.positions)} positions")
    
    def load_portfolio(self, name: str) -> Optional[Portfolio]:
        """Load a portfolio from disk."""
        filename = self._get_filename(name)
        
        if not filename.exists():
            return None
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            portfolio = Portfolio(
                name=data['name'],
                owner=data['owner'],
                description=data.get('description', ''),
                created_at=data.get('created_at', datetime.now().isoformat())
            )
            
            for pos_data in data.get('positions', []):
                portfolio.positions.append(Position(**pos_data))
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to load portfolio '{name}': {e}")
            return None
    
    def list_portfolios(self) -> List[str]:
        """List all saved portfolio names."""
        return [f.stem for f in self.data_dir.glob("*.json")]
    
    def delete_portfolio(self, name: str) -> bool:
        """Delete a portfolio."""
        filename = self._get_filename(name)
        if filename.exists():
            filename.unlink()
            return True
        return False
    
    def get_current_prices(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Get current prices for a list of tickers.
        
        Returns dict with price, change, change_pct for each ticker.
        Caches for 5 minutes.
        """
        # Check cache
        if not force_refresh and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < 300:  # 5 minutes
                cached = {t: self._price_cache[t] for t in tickers if t in self._price_cache}
                if len(cached) == len(tickers):
                    return cached
        
        prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                
                if len(hist) >= 1:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) >= 2 else current
                    
                    prices[ticker] = {
                        'price': current,
                        'prev_close': prev,
                        'change': current - prev,
                        'change_pct': ((current - prev) / prev) * 100 if prev else 0,
                        'updated_at': datetime.now().isoformat()
                    }
                else:
                    prices[ticker] = {'price': 0, 'change': 0, 'change_pct': 0, 'error': 'No data'}
                    
            except Exception as e:
                logger.warning(f"Failed to get price for {ticker}: {e}")
                prices[ticker] = {'price': 0, 'change': 0, 'change_pct': 0, 'error': str(e)}
        
        # Update cache
        self._price_cache.update(prices)
        self._cache_time = datetime.now()
        
        return prices
    
    def get_portfolio_value(self, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Calculate current portfolio value and performance.
        """
        if not portfolio.positions:
            return {
                'total_value': 0,
                'total_cost': 0,
                'total_gain': 0,
                'total_gain_pct': 0,
                'positions': []
            }
        
        prices = self.get_current_prices(portfolio.tickers)
        
        positions_data = []
        total_value = 0
        total_cost = 0
        
        for pos in portfolio.positions:
            price_info = prices.get(pos.ticker, {})
            current_price = price_info.get('price', 0)
            
            current_value = pos.shares * current_price
            cost = pos.total_cost
            gain = current_value - cost
            gain_pct = (gain / cost * 100) if cost > 0 else 0
            
            positions_data.append({
                'ticker': pos.ticker,
                'shares': pos.shares,
                'cost_basis': pos.cost_basis,
                'current_price': current_price,
                'current_value': current_value,
                'gain': gain,
                'gain_pct': gain_pct,
                'day_change': price_info.get('change', 0),
                'day_change_pct': price_info.get('change_pct', 0),
                'weight': 0,  # Will be calculated after total
                'notes': pos.notes
            })
            
            total_value += current_value
            total_cost += cost
        
        # Calculate weights
        for pos_data in positions_data:
            pos_data['weight'] = (pos_data['current_value'] / total_value * 100) if total_value > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_gain': total_value - total_cost,
            'total_gain_pct': ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0,
            'positions': sorted(positions_data, key=lambda x: x['current_value'], reverse=True)
        }
    
    def _get_filename(self, name: str) -> Path:
        """Get filename for a portfolio."""
        safe_name = "".join(c for c in name if c.isalnum() or c in "_ -").strip()
        return self.data_dir / f"{safe_name}.json"


# ============ DEMO PORTFOLIOS ============

def create_demo_portfolios(manager: PortfolioManager):
    """Create demo portfolios for testing."""
    
    # Your portfolio - growth/tech focused
    your_portfolio = Portfolio(
        name="My Portfolio",
        owner="You",
        description="Growth-focused portfolio with tech exposure"
    )
    your_portfolio.add_position("AAPL", 50, 175.00, "Core holding")
    your_portfolio.add_position("MSFT", 30, 380.00, "Cloud & AI exposure")
    your_portfolio.add_position("GOOGL", 20, 140.00, "Search & AI")
    your_portfolio.add_position("NVDA", 15, 450.00, "AI chips play")
    your_portfolio.add_position("SPY", 25, 480.00, "Broad market exposure")
    your_portfolio.add_position("QQQ", 20, 400.00, "Nasdaq exposure")
    
    # Dad's portfolio - more conservative, dividend focused
    dad_portfolio = Portfolio(
        name="Dad's Portfolio", 
        owner="Dad",
        description="Conservative income-focused portfolio"
    )
    dad_portfolio.add_position("VTI", 100, 220.00, "Total market ETF")
    dad_portfolio.add_position("VYM", 75, 115.00, "High dividend yield ETF")
    dad_portfolio.add_position("JNJ", 40, 155.00, "Healthcare dividend king")
    dad_portfolio.add_position("PG", 35, 150.00, "Consumer staples - stable")
    dad_portfolio.add_position("KO", 50, 60.00, "Coca-Cola - dividend aristocrat")
    dad_portfolio.add_position("O", 60, 55.00, "Realty Income - monthly dividends")
    dad_portfolio.add_position("SCHD", 50, 75.00, "Dividend growth ETF")
    
    manager.save_portfolio(your_portfolio)
    manager.save_portfolio(dad_portfolio)
    
    logger.info("Created demo portfolios")
    return [your_portfolio, dad_portfolio]

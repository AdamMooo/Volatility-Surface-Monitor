"""
Portfolio Analyzer Module

Provides risk metrics, sector analysis, and portfolio health checks.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


class PortfolioAnalyzer:
    """
    Analyzes portfolio risk, diversification, and performance.
    
    Designed to provide metrics that are meaningful for non-experts.
    """
    
    # ETFs that are inherently diversified - don't trigger concentration warnings
    DIVERSIFIED_ETFS = {
        # Broad market ETFs
        'SPY', 'VOO', 'VTI', 'IVV', 'ITOT', 'SCHB', 'SPTM', 'VV', "VFV.TO"
        # Total stock market
        'VTI', 'ITOT', 'SCHB', 'SWTSX', 'FZROX',
        # S&P 500
        'SPY', 'VOO', 'IVV', 'SPLG', 'SPYG', 'SPYD',
        # Nasdaq
        'QQQ', 'QQQM', 'ONEQ',
        # Small cap
        'IWM', 'VB', 'SCHA', 'IJR', 'VIOO',
        # Mid cap
        'VO', 'IJH', 'SCHM', 'MDY',
        # International
        'VEA', 'VWO', 'IEFA', 'IEMG', 'VXUS', 'IXUS', 'EFA', 'EEM',
        # Bond ETFs
        'BND', 'AGG', 'SCHZ', 'BNDX', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
        # Dividend ETFs
        'VYM', 'SCHD', 'DVY', 'VIG', 'DGRO', 'HDV', 'SPYD',
        # Balanced / Target date
        'AOR', 'AOA', 'AOM', 'AOK',
        # REIT ETFs
        'VNQ', 'SCHH', 'IYR', 'XLRE',
        # Sector ETFs (diversified within sector)
        'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    }
    
    # Sector mappings for common stocks
    SECTOR_MAP = {
        # Tech
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
        'GOOG': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'CRM': 'Technology',
        'ADBE': 'Technology', 'ORCL': 'Technology',
        
        # Healthcare
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
        'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'LLY': 'Healthcare',
        
        # Finance
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'BRK-B': 'Financials', 'V': 'Financials',
        'MA': 'Financials',
        
        # Consumer
        'AMZN': 'Consumer', 'TSLA': 'Consumer', 'HD': 'Consumer',
        'NKE': 'Consumer', 'MCD': 'Consumer', 'SBUX': 'Consumer',
        'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer',
        'WMT': 'Consumer', 'COST': 'Consumer',
        
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        
        # REITs
        'O': 'Real Estate', 'VNQ': 'Real Estate', 'AMT': 'Real Estate',
        
        # ETFs - market
        'SPY': 'Broad Market', 'QQQ': 'Technology', 'VTI': 'Broad Market',
        'VOO': 'Broad Market', 'IWM': 'Small Cap', 'DIA': 'Broad Market',
        
        # ETFs - dividend
        'VYM': 'Dividend', 'SCHD': 'Dividend', 'DVY': 'Dividend',
        'VIG': 'Dividend',
        
        # Bonds
        'BND': 'Bonds', 'AGG': 'Bonds', 'TLT': 'Bonds', 'LQD': 'Bonds',
    }
    
    def __init__(self):
        self._history_cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Optional[datetime] = None
    
    def analyze_portfolio(self, positions: List[Dict], total_value: float) -> Dict[str, Any]:
        """
        Full portfolio analysis.
        
        Args:
            positions: List of position dicts with ticker, current_value, weight
            total_value: Total portfolio value
            
        Returns:
            Comprehensive analysis dict
        """
        if not positions or total_value <= 0:
            return self._empty_analysis()
        
        tickers = [p['ticker'] for p in positions]
        weights = {p['ticker']: p['weight'] / 100 for p in positions}
        
        # Get historical data for risk calculations
        hist_data = self._get_historical_data(tickers, days=252)
        
        analysis = {
            'diversification': self._analyze_diversification(positions, total_value),
            'risk_metrics': self._calculate_risk_metrics(hist_data, weights),
            'sector_exposure': self._analyze_sectors(positions),
            'concentration': self._analyze_concentration(positions),
            'health_score': 0,  # Calculated at the end
            'health_grade': 'N/A',
            'warnings': [],
            'strengths': []
        }
        
        # Calculate overall health score
        analysis['health_score'], analysis['health_grade'] = self._calculate_health_score(analysis)
        
        # Generate warnings and strengths
        analysis['warnings'] = self._generate_warnings(analysis)
        analysis['strengths'] = self._generate_strengths(analysis)
        
        return analysis
    
    def _analyze_diversification(self, positions: List[Dict], total_value: float) -> Dict:
        """Analyze portfolio diversification."""
        n_positions = len(positions)
        
        weights = [p['weight'] for p in positions]
        
        # Herfindahl-Hirschman Index (HHI) - measure of concentration
        hhi = sum(w**2 for w in weights)
        
        # Effective number of stocks (1/HHI gives equivalent equal-weight positions)
        effective_n = 10000 / hhi if hhi > 0 else 0
        
        # Diversification score (0-100)
        # Perfect diversification = 100, single stock = 0
        max_hhi = 10000  # All in one stock
        min_hhi = 10000 / n_positions if n_positions > 0 else 10000  # Equal weight
        
        if max_hhi == min_hhi:
            div_score = 50
        else:
            div_score = 100 * (1 - (hhi - min_hhi) / (max_hhi - min_hhi))
            div_score = max(0, min(100, div_score))
        
        return {
            'n_positions': n_positions,
            'effective_positions': round(effective_n, 1),
            'hhi': round(hhi, 0),
            'score': round(div_score, 0),
            'rating': self._get_diversification_rating(div_score, n_positions)
        }
    
    def _calculate_risk_metrics(self, hist_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict:
        """Calculate portfolio risk metrics."""
        if not hist_data:
            return {
                'volatility_annual': 0,
                'volatility_rating': 'Unknown',
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'beta': 1.0,
                'var_95': 0
            }
        
        try:
            # Calculate portfolio returns
            returns_df = pd.DataFrame()
            for ticker, df in hist_data.items():
                if ticker in weights and not df.empty:
                    returns_df[ticker] = df['Close'].pct_change()
            
            if returns_df.empty:
                return self._empty_risk_metrics()
            
            returns_df = returns_df.dropna()
            
            # Portfolio returns (weighted)
            port_weights = np.array([weights.get(t, 0) for t in returns_df.columns])
            port_weights = port_weights / port_weights.sum()  # Normalize
            
            portfolio_returns = (returns_df * port_weights).sum(axis=1)
            
            # Volatility (annualized)
            vol_daily = portfolio_returns.std()
            vol_annual = vol_daily * np.sqrt(252) * 100
            
            # Max Drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() * 100
            
            # Sharpe Ratio (assuming 5% risk-free rate)
            excess_return = portfolio_returns.mean() * 252 - 0.05
            sharpe = excess_return / (vol_daily * np.sqrt(252)) if vol_daily > 0 else 0
            
            # Beta (vs SPY if in portfolio, else estimate)
            beta = 1.0
            if 'SPY' in hist_data and not hist_data['SPY'].empty:
                spy_returns = hist_data['SPY']['Close'].pct_change().dropna()
                common_idx = portfolio_returns.index.intersection(spy_returns.index)
                if len(common_idx) > 20:
                    cov = np.cov(portfolio_returns.loc[common_idx], spy_returns.loc[common_idx])
                    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
            
            # Value at Risk (95%)
            var_95 = np.percentile(portfolio_returns, 5) * 100
            
            return {
                'volatility_annual': round(vol_annual, 1),
                'volatility_rating': self._get_volatility_rating(vol_annual),
                'max_drawdown': round(abs(max_dd), 1),
                'sharpe_ratio': round(sharpe, 2),
                'beta': round(beta, 2),
                'var_95': round(abs(var_95), 2)
            }
            
        except Exception as e:
            logger.warning(f"Risk calculation error: {e}")
            return self._empty_risk_metrics()
    
    def _analyze_sectors(self, positions: List[Dict]) -> Dict[str, float]:
        """Analyze sector allocation."""
        sector_weights = {}
        
        for pos in positions:
            ticker = pos['ticker']
            weight = pos['weight']
            sector = self.SECTOR_MAP.get(ticker, 'Other')
            
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return dict(sorted(sector_weights.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_concentration(self, positions: List[Dict]) -> Dict:
        """Analyze position concentration risk."""
        if not positions:
            return {'top_1': 0, 'top_3': 0, 'top_5': 0, 'risk_level': 'Unknown'}
        
        sorted_pos = sorted(positions, key=lambda x: x['weight'], reverse=True)
        
        top_1 = sorted_pos[0]['weight'] if len(sorted_pos) >= 1 else 0
        top_3 = sum(p['weight'] for p in sorted_pos[:3])
        top_5 = sum(p['weight'] for p in sorted_pos[:5])
        
        # Risk level based on top holding
        if top_1 > 40:
            risk_level = 'High'
        elif top_1 > 25:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'
        
        return {
            'top_1': round(top_1, 1),
            'top_3': round(top_3, 1),
            'top_5': round(top_5, 1),
            'top_holding': sorted_pos[0]['ticker'] if sorted_pos else 'N/A',
            'risk_level': risk_level
        }
    
    def _calculate_health_score(self, analysis: Dict) -> Tuple[int, str]:
        """Calculate overall portfolio health score."""
        score = 50  # Start at neutral
        
        # Diversification (up to +/- 20 points)
        div_score = analysis['diversification']['score']
        score += (div_score - 50) * 0.4
        
        # Volatility (up to +/- 15 points)
        vol = analysis['risk_metrics']['volatility_annual']
        if vol < 12:
            score += 10
        elif vol < 18:
            score += 5
        elif vol > 30:
            score -= 15
        elif vol > 25:
            score -= 10
        
        # Concentration (up to +/- 15 points)
        conc = analysis['concentration']
        if conc['top_1'] < 15:
            score += 10
        elif conc['top_1'] < 25:
            score += 5
        elif conc['top_1'] > 40:
            score -= 15
        elif conc['top_1'] > 30:
            score -= 10
        
        # Number of positions bonus
        n_pos = analysis['diversification']['n_positions']
        if n_pos >= 10:
            score += 5
        elif n_pos >= 5:
            score += 2
        elif n_pos <= 2:
            score -= 10
        
        score = max(0, min(100, score))
        
        # Letter grade
        if score >= 85:
            grade = 'A'
        elif score >= 70:
            grade = 'B'
        elif score >= 55:
            grade = 'C'
        elif score >= 40:
            grade = 'D'
        else:
            grade = 'F'
        
        return int(score), grade
    
    def _generate_warnings(self, analysis: Dict) -> List[str]:
        """Generate warning messages based on analysis."""
        warnings = []
        
        # Concentration warnings - skip for diversified ETFs
        conc = analysis['concentration']
        top_holding = conc.get('top_holding', '').upper()
        if top_holding not in self.DIVERSIFIED_ETFS:
            if conc['top_1'] > 40:
                warnings.append(f"Very high concentration: {conc['top_holding']} is {conc['top_1']:.0f}% of portfolio")
            elif conc['top_1'] > 30:
                warnings.append(f"High concentration in {conc['top_holding']} ({conc['top_1']:.0f}%)")
        
        # Volatility warnings
        vol = analysis['risk_metrics']['volatility_annual']
        if vol > 30:
            warnings.append(f"High volatility ({vol:.0f}% annual) - expect large swings")
        
        # Diversification warnings
        if analysis['diversification']['n_positions'] < 5:
            warnings.append("Limited diversification - consider adding more positions")
        
        # Sector warnings
        sectors = analysis['sector_exposure']
        for sector, weight in sectors.items():
            if weight > 50 and sector != 'Broad Market':
                warnings.append(f"Heavy {sector} exposure ({weight:.0f}%) - sector risk")
        
        return warnings
    
    def _generate_strengths(self, analysis: Dict) -> List[str]:
        """Generate strength messages based on analysis."""
        strengths = []
        
        # Diversification
        if analysis['diversification']['score'] > 70:
            strengths.append("Well diversified across positions")
        
        # Low volatility
        vol = analysis['risk_metrics']['volatility_annual']
        if vol < 15:
            strengths.append("Low volatility - more stable returns")
        
        # Good Sharpe ratio
        if analysis['risk_metrics']['sharpe_ratio'] > 1:
            strengths.append("Good risk-adjusted returns (Sharpe > 1)")
        
        # Balanced sectors
        sectors = analysis['sector_exposure']
        if len(sectors) >= 4:
            max_sector = max(sectors.values())
            if max_sector < 40:
                strengths.append("Balanced sector exposure")
        
        return strengths
    
    def _get_diversification_rating(self, score: float, n_positions: int) -> str:
        """Get human-readable diversification rating."""
        if n_positions < 3:
            return "Very Concentrated"
        if score >= 80:
            return "Excellent"
        if score >= 60:
            return "Good"
        if score >= 40:
            return "Moderate"
        return "Concentrated"
    
    def _get_volatility_rating(self, vol_annual: float) -> str:
        """Get human-readable volatility rating."""
        if vol_annual < 10:
            return "Very Low"
        if vol_annual < 15:
            return "Low"
        if vol_annual < 20:
            return "Moderate"
        if vol_annual < 30:
            return "High"
        return "Very High"
    
    def _get_historical_data(self, tickers: List[str], days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch historical price data."""
        # Add SPY for beta calculation
        all_tickers = list(set(tickers + ['SPY']))
        
        result = {}
        for ticker in all_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=f"{days}d")
                if not hist.empty:
                    result[ticker] = hist
            except Exception as e:
                logger.warning(f"Failed to get history for {ticker}: {e}")
        
        return result
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure."""
        return {
            'diversification': {'n_positions': 0, 'score': 0, 'rating': 'N/A'},
            'risk_metrics': self._empty_risk_metrics(),
            'sector_exposure': {},
            'concentration': {'top_1': 0, 'risk_level': 'Unknown'},
            'health_score': 0,
            'health_grade': 'N/A',
            'warnings': [],
            'strengths': []
        }
    
    def _empty_risk_metrics(self) -> Dict:
        """Return empty risk metrics."""
        return {
            'volatility_annual': 0,
            'volatility_rating': 'Unknown',
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'beta': 1.0,
            'var_95': 0
        }

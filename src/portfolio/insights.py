"""
Insight Engine Module

Generates plain-English insights and recommendations.
Designed to be understandable by non-experts.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger


class InsightEngine:
    """
    Generates actionable insights and plain-English explanations.
    
    This is the "translator" between complex market data and
    understandable advice for non-expert investors.
    """
    
    # Market regime explanations
    REGIME_EXPLANATIONS = {
        'CALM': {
            'title': 'Calm Markets',
            'short': 'Markets are stable with normal volatility',
            'detail': (
                "The market is behaving normally with typical day-to-day movements. "
                "This is a good environment for most investment strategies. "
                "Volatility is low, meaning prices aren't swinging wildly."
            ),
            'advice': [
                "Good time for regular investing (dollar-cost averaging)",
                "Consider rebalancing if positions have drifted",
                "Don't get complacent - markets can change quickly"
            ]
        },
        'PRE_STRESS': {
            'title': 'Early Warning Signs',
            'short': 'Some stress indicators are elevated',
            'detail': (
                "The market is showing early signs of nervousness. "
                "Traders are paying more for protection (put options), which suggests "
                "some professionals are worried about potential declines. "
                "This doesn't mean a crash is coming, but caution is warranted."
            ),
            'advice': [
                "Review your risk tolerance - are you comfortable with potential drops?",
                "Consider holding off on large new purchases",
                "Ensure you have cash reserves for opportunities"
            ]
        },
        'ELEVATED': {
            'title': 'Elevated Stress',
            'short': 'Market fear is notably higher than normal',
            'detail': (
                "The market is clearly stressed. Volatility is elevated, meaning larger "
                "daily swings are expected. Options traders are paying significant premiums "
                "for protection. This often happens before or during market corrections."
            ),
            'advice': [
                "Expect larger daily swings (2-3% moves possible)",
                "Consider reducing risky positions if you're nervous",
                "If you're a long-term investor, stay the course",
                "This can be a good time to buy quality stocks at discounts"
            ]
        },
        'ACUTE': {
            'title': 'High Stress / Crisis Mode',
            'short': 'Markets are in crisis mode with extreme fear',
            'detail': (
                "The market is experiencing extreme stress - this is rare. "
                "Fear is at very high levels, and we're seeing panic-like behavior. "
                "Prices can swing 5%+ in a single day. This is when headlines get scary."
            ),
            'advice': [
                "Avoid panic selling - the worst days are often near the bottom",
                "Only make changes if absolutely necessary",
                "Historically, buying during extreme fear has been rewarded long-term",
                "Take a breath - these periods are temporary"
            ]
        },
        'UNKNOWN': {
            'title': 'Unable to Determine',
            'short': 'Market data is unavailable',
            'detail': "We couldn't get current market data to assess the regime.",
            'advice': ["Check back later when data is available"]
        }
    }
    
    # Metric explanations
    METRIC_EXPLANATIONS = {
        'atm_vol': {
            'name': 'ATM Volatility',
            'simple_name': 'Market Fear Level',
            'what_it_means': (
                "This measures how much the market expects prices to move. "
                "Higher numbers = more expected movement = more fear/uncertainty."
            ),
            'typical_values': "Normal: 12-18% | Elevated: 18-25% | High: 25%+"
        },
        'skew': {
            'name': '25-Delta Skew',
            'simple_name': 'Crash Protection Cost',
            'what_it_means': (
                "This measures how expensive 'crash insurance' is. "
                "Higher numbers mean traders are paying more to protect against big drops. "
                "It's like checking how much people are paying for flood insurance - "
                "if premiums spike, people are worried about flooding."
            ),
            'typical_values': "Normal: 2-5% | Worried: 5-8% | Fearful: 8%+"
        },
        'curvature': {
            'name': 'Volatility Curvature',
            'simple_name': 'Tail Risk Premium',
            'what_it_means': (
                "This measures fear of extreme moves in EITHER direction. "
                "High curvature means traders expect potential big swings up OR down."
            ),
            'typical_values': "Normal: 0-2% | Elevated: 2-4% | High: 4%+"
        },
        'term_slope': {
            'name': 'Term Structure',
            'simple_name': 'Future vs Present Fear',
            'what_it_means': (
                "Negative means fear is focused on the near-term (next few weeks). "
                "Positive means markets expect future volatility to be higher. "
                "Sharply negative often occurs during market panics."
            ),
            'typical_values': "Normal: slightly positive | Inverted/negative: near-term stress"
        }
    }
    
    def __init__(self):
        pass
    
    def explain_regime(self, regime: str) -> Dict[str, Any]:
        """Get full explanation for a market regime."""
        return self.REGIME_EXPLANATIONS.get(regime, self.REGIME_EXPLANATIONS['UNKNOWN'])
    
    def explain_metric(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Get explanation for a specific metric with context for its current value."""
        base = self.METRIC_EXPLANATIONS.get(metric_name, {
            'name': metric_name,
            'simple_name': metric_name,
            'what_it_means': 'Technical indicator',
            'typical_values': 'Varies'
        })
        
        # Add value interpretation
        interpretation = self._interpret_metric_value(metric_name, value)
        
        return {
            **base,
            'current_value': value,
            'interpretation': interpretation
        }
    
    def _interpret_metric_value(self, metric_name: str, value: float) -> str:
        """Generate interpretation of a metric value."""
        if metric_name == 'atm_vol':
            if value < 0.12:
                return "Very calm - markets are quiet"
            elif value < 0.18:
                return "Normal range - typical market conditions"
            elif value < 0.25:
                return "Elevated - above normal fear"
            else:
                return "High - significant market stress"
        
        elif metric_name == 'skew':
            if value < 0.02:
                return "Low - little worry about crashes"
            elif value < 0.05:
                return "Normal - typical protection buying"
            elif value < 0.08:
                return "Elevated - increased crash concern"
            else:
                return "High - significant crash fear"
        
        elif metric_name == 'curvature':
            if value < 0.01:
                return "Low - calm expectations"
            elif value < 0.03:
                return "Normal - typical tail risk pricing"
            else:
                return "Elevated - fear of big moves"
        
        elif metric_name == 'term_slope':
            if value > 0.02:
                return "Normal - future vol expected higher"
            elif value > -0.02:
                return "Flat - balanced expectations"
            else:
                return "Inverted - near-term stress elevated"
        
        return f"Value: {value:.2%}"
    
    def generate_portfolio_insights(
        self, 
        portfolio_value: Dict, 
        analysis: Dict, 
        market_regime: str
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized insights for a portfolio.
        
        Returns list of insight objects with priority, category, and content.
        """
        insights = []
        
        # Overall health insight
        health_score = analysis.get('health_score', 50)
        health_grade = analysis.get('health_grade', 'C')
        
        if health_score >= 70:
            insights.append({
                'priority': 1,
                'category': 'health',
                'emoji': '💪',
                'title': 'Portfolio in Good Shape',
                'message': f"Your portfolio scores {health_score}/100 (Grade: {health_grade}). Keep up the good work!",
                'action': None
            })
        elif health_score >= 50:
            insights.append({
                'priority': 1,
                'category': 'health',
                'emoji': '📊',
                'title': 'Portfolio Needs Some Attention',
                'message': f"Your portfolio scores {health_score}/100 (Grade: {health_grade}). There's room for improvement.",
                'action': 'Review the warnings below for specific suggestions.'
            })
        else:
            insights.append({
                'priority': 1,
                'category': 'health',
                'emoji': '⚠️',
                'title': 'Portfolio Needs Review',
                'message': f"Your portfolio scores {health_score}/100 (Grade: {health_grade}). Consider making some changes.",
                'action': 'Address the warnings below to improve your risk profile.'
            })
        
        # Market regime + portfolio combo insight
        regime_info = self.explain_regime(market_regime)
        if market_regime in ['ELEVATED', 'ACUTE']:
            vol = analysis.get('risk_metrics', {}).get('volatility_annual', 15)
            beta = analysis.get('risk_metrics', {}).get('beta', 1.0)
            
            if beta > 1.2:
                insights.append({
                    'priority': 2,
                    'category': 'market_risk',
                    'emoji': '📉',
                    'title': 'Your Portfolio Amplifies Market Moves',
                    'message': (
                        f"With beta of {beta:.1f}, your portfolio moves more than the market. "
                        f"In the current {regime_info['title'].lower()}, expect amplified swings."
                    ),
                    'action': 'Consider if you\'re comfortable with larger-than-market moves.'
                })
            elif beta < 0.8:
                insights.append({
                    'priority': 2,
                    'category': 'market_risk',
                    'emoji': '🛡️',
                    'title': 'Your Portfolio is Defensive',
                    'message': (
                        f"With beta of {beta:.1f}, your portfolio is less volatile than the market. "
                        f"You should experience smaller swings than the headlines suggest."
                    ),
                    'action': None
                })
        
        # Performance insight
        total_gain_pct = portfolio_value.get('total_gain_pct', 0)
        if total_gain_pct > 20:
            insights.append({
                'priority': 3,
                'category': 'performance',
                'emoji': '🎉',
                'title': 'Strong Gains!',
                'message': f"Your portfolio is up {total_gain_pct:.1f}% overall. Consider if it's time to take some profits.",
                'action': 'Rebalancing can lock in gains and reduce risk.'
            })
        elif total_gain_pct < -10:
            insights.append({
                'priority': 3,
                'category': 'performance',
                'emoji': '💡',
                'title': 'Portfolio Down',
                'message': (
                    f"Your portfolio is down {abs(total_gain_pct):.1f}%. "
                    "Remember: paper losses only become real if you sell."
                ),
                'action': 'Review your positions - are the investment reasons still valid?'
            })
        
        # Concentration insights
        concentration = analysis.get('concentration', {})
        if concentration.get('top_1', 0) > 30:
            insights.append({
                'priority': 2,
                'category': 'risk',
                'emoji': '⚖️',
                'title': 'High Concentration Risk',
                'message': (
                    f"{concentration['top_holding']} is {concentration['top_1']:.0f}% of your portfolio. "
                    "If this one stock drops, it will significantly impact your total."
                ),
                'action': 'Consider trimming to reduce single-stock risk.'
            })
        
        return sorted(insights, key=lambda x: x['priority'])
    
    def generate_daily_summary(
        self,
        regime: str,
        metrics: Dict,
        portfolios: List[Dict]
    ) -> str:
        """Generate a daily summary in plain English."""
        regime_info = self.explain_regime(regime)
        
        summary = f"""
## 📊 Daily Market Summary

### Market Condition: {regime_info['emoji']} {regime_info['title']}

{regime_info['detail']}

### What This Means For You:
"""
        for advice in regime_info['advice']:
            summary += f"\n{advice}"
        
        if portfolios:
            summary += "\n\n### Your Portfolios Today:\n"
            for port in portfolios:
                name = port.get('name', 'Portfolio')
                value = port.get('total_value', 0)
                change = port.get('day_change', 0)
                change_pct = port.get('day_change_pct', 0)
                
                emoji = '📈' if change >= 0 else '📉'
                summary += f"\n{emoji} **{name}**: ${value:,.0f} ({change_pct:+.1f}% today)"
        
        summary += f"\n\n---\n*Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*"
        
        return summary
    
    def get_action_recommendation(
        self,
        regime: str,
        portfolio_analysis: Dict,
        risk_tolerance: str = 'moderate'
    ) -> Dict[str, Any]:
        """
        Generate specific action recommendations.
        
        Args:
            regime: Current market regime
            portfolio_analysis: Full portfolio analysis
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
            
        Returns:
            Recommendation dict with actions
        """
        recommendations = {
            'overall_action': 'hold',
            'confidence': 'medium',
            'reasons': [],
            'specific_actions': []
        }
        
        health_score = portfolio_analysis.get('health_score', 50)
        warnings = portfolio_analysis.get('warnings', [])
        
        # Regime-based recommendations
        if regime == 'CALM':
            recommendations['overall_action'] = 'normal_operations'
            recommendations['reasons'].append("Markets are calm - normal investing conditions")
            recommendations['specific_actions'].append({
                'action': 'Continue regular contributions',
                'priority': 'normal'
            })
            if health_score < 60:
                recommendations['specific_actions'].append({
                    'action': 'Good time to rebalance and address portfolio warnings',
                    'priority': 'normal'
                })
        
        elif regime == 'PRE_STRESS':
            recommendations['overall_action'] = 'cautious'
            recommendations['reasons'].append("Early stress signals - increased caution warranted")
            recommendations['specific_actions'].append({
                'action': 'Review positions for any you\'re uncomfortable holding through volatility',
                'priority': 'medium'
            })
            recommendations['specific_actions'].append({
                'action': 'Ensure cash reserves are adequate',
                'priority': 'medium'
            })
        
        elif regime == 'ELEVATED':
            recommendations['overall_action'] = 'defensive'
            recommendations['reasons'].append("Elevated market stress - defensive posture recommended")
            if risk_tolerance == 'conservative':
                recommendations['specific_actions'].append({
                    'action': 'Consider reducing equity exposure',
                    'priority': 'high'
                })
            recommendations['specific_actions'].append({
                'action': 'Avoid large new purchases until stress subsides',
                'priority': 'high'
            })
            recommendations['specific_actions'].append({
                'action': 'Keep a buy list ready for if quality stocks become cheap',
                'priority': 'medium'
            })
        
        elif regime == 'ACUTE':
            recommendations['overall_action'] = 'stay_calm'
            recommendations['reasons'].append("Crisis conditions - patience is key")
            recommendations['specific_actions'].append({
                'action': '🛑 DO NOT panic sell',
                'priority': 'critical'
            })
            recommendations['specific_actions'].append({
                'action': 'Avoid checking portfolio constantly - it only increases stress',
                'priority': 'high'
            })
            if risk_tolerance == 'aggressive':
                recommendations['specific_actions'].append({
                    'action': 'Consider buying quality positions if you have cash and long time horizon',
                    'priority': 'medium'
                })
        
        return recommendations

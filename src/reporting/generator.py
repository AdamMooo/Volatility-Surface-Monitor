"""
Report Generator

Automated HTML and PDF report generation.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


class ReportGenerator:
    """
    Generate HTML/PDF reports for volatility surface analysis.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for report output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def _get_regime_color(self, regime: str) -> str:
        """Get color for regime."""
        colors = {
            'calm': '#2ecc71',
            'pre_stress': '#f1c40f',
            'elevated': '#e67e22',
            'acute': '#e74c3c',
            'recovery': '#3498db'
        }
        return colors.get(regime, '#95a5a6')
    
    def _create_header_html(
        self,
        title: str,
        date: datetime,
        ticker: str
    ) -> str:
        """Create report header HTML."""
        return f'''
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    color: white; padding: 30px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">{title}</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.8;">
                {ticker} | {date.strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </div>
        '''
    
    def _create_regime_summary_html(
        self,
        regime_data: Dict[str, Any]
    ) -> str:
        """Create regime summary section."""
        regime = regime_data.get('current_regime', 'unknown')
        confidence = regime_data.get('confidence', 0)
        color = self._get_regime_color(regime)
        
        drivers_html = ""
        for driver in regime_data.get('key_drivers', []):
            drivers_html += f"<li>{driver}</li>"
        
        return f'''
        <div style="background: white; border-radius: 8px; padding: 20px; 
                    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="margin-top: 0;">Current Regime</h2>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="background: {color}; color: white; padding: 20px 40px; 
                            border-radius: 8px; font-size: 24px; font-weight: bold;">
                    {regime.replace('_', ' ').upper()}
                </div>
                <div>
                    <p style="margin: 0;"><strong>Confidence:</strong> {confidence:.1%}</p>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h3>Key Drivers</h3>
                <ul>{drivers_html}</ul>
            </div>
        </div>
        '''
    
    def _create_metrics_table_html(
        self,
        metrics: Dict[str, float]
    ) -> str:
        """Create metrics table section."""
        rows_html = ""
        for metric, value in metrics.items():
            rows_html += f'''
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #eee;">
                    {metric.replace('_', ' ').title()}
                </td>
                <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right;">
                    {value:.4f}
                </td>
            </tr>
            '''
        
        return f'''
        <div style="background: white; border-radius: 8px; padding: 20px; 
                    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="margin-top: 0;">Key Metrics</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 10px; text-align: left;">Metric</th>
                        <th style="padding: 10px; text-align: right;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        '''
    
    def _create_alerts_html(
        self,
        alerts: List[Dict[str, Any]]
    ) -> str:
        """Create alerts section."""
        if not alerts:
            return ""
        
        alerts_html = ""
        for alert in alerts:
            severity = alert.get('severity', 'info')
            colors = {
                'critical': '#e74c3c',
                'warning': '#f39c12',
                'info': '#3498db'
            }
            color = colors.get(severity, '#3498db')
            
            alerts_html += f'''
            <div style="background: {color}20; border-left: 4px solid {color}; 
                        padding: 15px; margin-bottom: 10px;">
                <strong>{alert.get('title', 'Alert')}</strong>
                <p style="margin: 5px 0 0 0;">{alert.get('message', '')}</p>
            </div>
            '''
        
        return f'''
        <div style="background: white; border-radius: 8px; padding: 20px; 
                    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="margin-top: 0;">Alerts</h2>
            {alerts_html}
        </div>
        '''
    
    def _embed_plotly_figure(
        self,
        fig: go.Figure,
        title: str
    ) -> str:
        """Convert Plotly figure to embeddable HTML."""
        fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        
        return f'''
        <div style="background: white; border-radius: 8px; padding: 20px; 
                    margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h2 style="margin-top: 0;">{title}</h2>
            {fig_html}
        </div>
        '''
    
    def generate_html_report(
        self,
        data: Dict[str, Any],
        figures: Optional[List[tuple]] = None,
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate complete HTML report.
        
        Args:
            data: Report data including regime, metrics, alerts
            figures: List of (figure, title) tuples
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vol_surface_report_{timestamp}.html"
        
        date = data.get('timestamp', datetime.now())
        ticker = data.get('ticker', 'SPY')
        
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Volatility Surface Report - {ticker}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 
                                 Roboto, Oxygen, Ubuntu, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: #f5f6fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                {self._create_header_html("Volatility Surface Report", date, ticker)}
                {self._create_regime_summary_html(data.get('regime', {}))}
                {self._create_metrics_table_html(data.get('metrics', {}))}
                {self._create_alerts_html(data.get('alerts', []))}
        '''
        
        if figures:
            for fig, title in figures:
                html_content += self._embed_plotly_figure(fig, title)
        
        html_content += '''
            </div>
        </body>
        </html>
        '''
        
        output_path = self.output_dir / filename
        output_path.write_text(html_content, encoding='utf-8')
        
        return output_path
    
    def generate_json_report(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate JSON report for programmatic consumption.
        
        Args:
            data: Report data
            filename: Output filename
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vol_surface_report_{timestamp}.json"
        
        serializable = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                serializable[key] = value.isoformat()
            elif isinstance(value, pd.DataFrame):
                serializable[key] = value.to_dict(orient='records')
            else:
                serializable[key] = value
        
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(serializable, indent=2), encoding='utf-8')
        
        return output_path


def generate_daily_report(
    regime_data: Dict[str, Any],
    metrics: Dict[str, float],
    ticker: str = "SPY",
    figures: Optional[List[tuple]] = None,
    output_dir: str = "reports"
) -> Path:
    """
    Generate daily end-of-day report.
    
    Args:
        regime_data: Current regime classification
        metrics: Surface geometry metrics
        ticker: Underlying ticker
        figures: Optional list of (figure, title) tuples
        output_dir: Output directory
        
    Returns:
        Path to generated report
    """
    generator = ReportGenerator(output_dir)
    
    data = {
        'timestamp': datetime.now(),
        'ticker': ticker,
        'regime': regime_data,
        'metrics': metrics,
        'alerts': []
    }
    
    regime = regime_data.get('current_regime', 'unknown')
    if regime in ['elevated', 'acute']:
        data['alerts'].append({
            'severity': 'critical' if regime == 'acute' else 'warning',
            'title': f'Market Regime: {regime.upper()}',
            'message': f'The volatility surface indicates {regime} stress conditions. '
                      f'Key drivers: {", ".join(regime_data.get("key_drivers", []))}'
        })
    
    return generator.generate_html_report(data, figures)


def generate_alert_report(
    alert_type: str,
    alert_data: Dict[str, Any],
    ticker: str = "SPY",
    output_dir: str = "reports"
) -> Path:
    """
    Generate ad-hoc alert report.
    
    Args:
        alert_type: Type of alert (regime_change, arbitrage, spike)
        alert_data: Alert-specific data
        ticker: Underlying ticker
        output_dir: Output directory
        
    Returns:
        Path to generated report
    """
    generator = ReportGenerator(output_dir)
    
    timestamp = datetime.now()
    filename = f"alert_{alert_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    
    data = {
        'timestamp': timestamp,
        'ticker': ticker,
        'regime': alert_data.get('regime', {}),
        'metrics': alert_data.get('metrics', {}),
        'alerts': [{
            'severity': alert_data.get('severity', 'warning'),
            'title': f'{alert_type.replace("_", " ").title()} Alert',
            'message': alert_data.get('message', '')
        }]
    }
    
    return generator.generate_html_report(data, filename=filename)

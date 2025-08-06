"""
AlphaSheet Intelligence™ -- Alert System Utility
Manages real-time alerts for Premium tier customers
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AlertSystem:
    """Manages alerts and notifications for premium tier"""
    
    # Alert thresholds
    PORTFOLIO_DROP_THRESHOLD = -0.05  # -5% portfolio drop
    POSITION_MOVE_THRESHOLD = 0.15    # 15% position move (up or down)
    ML_CONFIDENCE_THRESHOLD = 0.80    # 80% ML signal confidence
    REBALANCE_DRIFT_THRESHOLD = 0.10  # 10% drift from target allocation
    VIX_SPIKE_THRESHOLD = 30          # VIX above 30
    CORRELATION_BREAK_THRESHOLD = 0.3  # 30% correlation change
    
    def __init__(self):
        """Initialize alert system"""
        self.alert_history = []
        self.last_alert_time = {}
        self.cooldown_period = timedelta(hours=1)  # Don't spam alerts
    
    def check_all_alerts(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check all alert conditions
        
        Args:
            portfolio_data: Complete portfolio data including metrics
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        # Check portfolio drop
        portfolio_alert = self._check_portfolio_drop(portfolio_data)
        if portfolio_alert:
            alerts.append(portfolio_alert)
        
        # Check position movements
        position_alerts = self._check_position_moves(portfolio_data)
        alerts.extend(position_alerts)
        
        # Check ML signals
        ml_alert = self._check_ml_signals(portfolio_data)
        if ml_alert:
            alerts.append(ml_alert)
        
        # Check rebalancing needs
        rebalance_alert = self._check_rebalancing(portfolio_data)
        if rebalance_alert:
            alerts.append(rebalance_alert)
        
        # Check market conditions
        market_alerts = self._check_market_conditions(portfolio_data)
        alerts.extend(market_alerts)
        
        # Filter out alerts that are in cooldown
        alerts = self._filter_cooldown(alerts)
        
        # Record alerts in history
        for alert in alerts:
            self._record_alert(alert)
        
        return alerts
    
    def _check_portfolio_drop(self, portfolio_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if portfolio has dropped significantly
        
        Args:
            portfolio_data: Portfolio data with daily change
            
        Returns:
            Alert if triggered, None otherwise
        """
        daily_change = portfolio_data.get('daily_change', 0)
        
        if daily_change <= self.PORTFOLIO_DROP_THRESHOLD:
            return {
                'type': 'portfolio_drop',
                'severity': 'high',
                'title': '🚨 Portfolio Alert: Significant Drop',
                'message': f'Your portfolio is down {abs(daily_change):.2%} today',
                'value': daily_change,
                'timestamp': datetime.now(),
                'action_required': True,
                'suggested_action': 'Review your positions and consider your risk tolerance'
            }
        
        return None
    
    def _check_position_moves(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for significant position movements
        
        Args:
            portfolio_data: Portfolio data with position changes
            
        Returns:
            List of position alerts
        """
        alerts = []
        positions = portfolio_data.get('positions', {})
        
        for ticker, position_data in positions.items():
            daily_change = position_data.get('daily_change', 0)
            
            if abs(daily_change) >= self.POSITION_MOVE_THRESHOLD:
                direction = 'up' if daily_change > 0 else 'down'
                severity = 'medium' if abs(daily_change) < 0.20 else 'high'
                
                alerts.append({
                    'type': 'position_move',
                    'severity': severity,
                    'title': f'📊 Position Alert: {ticker}',
                    'message': f'{ticker} is {direction} {abs(daily_change):.2%} today',
                    'ticker': ticker,
                    'value': daily_change,
                    'timestamp': datetime.now(),
                    'action_required': severity == 'high',
                    'suggested_action': f'Review your {ticker} position and consider rebalancing'
                })
        
        return alerts
    
    def _check_ml_signals(self, portfolio_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check for high-confidence ML signals
        
        Args:
            portfolio_data: Portfolio data with ML predictions
            
        Returns:
            Alert if high-confidence signal detected
        """
        ml_data = portfolio_data.get('ml_signals', {})
        confidence = ml_data.get('confidence', 0)
        signal = ml_data.get('signal', 'HOLD')
        
        if confidence >= self.ML_CONFIDENCE_THRESHOLD and signal != 'HOLD':
            return {
                'type': 'ml_signal',
                'severity': 'medium',
                'title': '🤖 ML Signal Alert',
                'message': f'High confidence {signal} signal detected ({confidence:.1%} confidence)',
                'signal': signal,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'action_required': True,
                'suggested_action': f'Consider {signal.lower()}ing based on ML analysis'
            }
        
        return None
    
    def _check_rebalancing(self, portfolio_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if portfolio needs rebalancing
        
        Args:
            portfolio_data: Portfolio data with allocation drift
            
        Returns:
            Alert if rebalancing needed
        """
        max_drift = portfolio_data.get('max_allocation_drift', 0)
        
        if max_drift >= self.REBALANCE_DRIFT_THRESHOLD:
            return {
                'type': 'rebalancing',
                'severity': 'low',
                'title': '⚖️ Rebalancing Alert',
                'message': f'Portfolio has drifted {max_drift:.1%} from target allocation',
                'value': max_drift,
                'timestamp': datetime.now(),
                'action_required': True,
                'suggested_action': 'Review the rebalancing recommendations in your report'
            }
        
        return None
    
    def _check_market_conditions(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check market conditions (VIX, correlations, regime)
        
        Args:
            portfolio_data: Portfolio data with market metrics
            
        Returns:
            List of market condition alerts
        """
        alerts = []
        market_data = portfolio_data.get('market_data', {})
        
        # Check VIX spike
        vix = market_data.get('vix', 0)
        if vix > self.VIX_SPIKE_THRESHOLD:
            alerts.append({
                'type': 'vix_spike',
                'severity': 'high',
                'title': '⚡ Market Volatility Alert',
                'message': f'VIX has spiked to {vix:.1f}, indicating high market volatility',
                'value': vix,
                'timestamp': datetime.now(),
                'action_required': False,
                'suggested_action': 'Consider reducing position sizes or hedging'
            })
        
        # Check correlation breaks
        correlation_change = market_data.get('correlation_change', 0)
        if abs(correlation_change) > self.CORRELATION_BREAK_THRESHOLD:
            alerts.append({
                'type': 'correlation_break',
                'severity': 'medium',
                'title': '🔄 Correlation Break Alert',
                'message': 'Significant change in asset correlations detected',
                'value': correlation_change,
                'timestamp': datetime.now(),
                'action_required': False,
                'suggested_action': 'Review your diversification strategy'
            })
        
        # Check regime change
        regime_change = market_data.get('regime_change', False)
        if regime_change:
            new_regime = market_data.get('current_regime', 'Unknown')
            alerts.append({
                'type': 'regime_change',
                'severity': 'medium',
                'title': '📈 Market Regime Change',
                'message': f'Market regime has shifted to {new_regime}',
                'value': new_regime,
                'timestamp': datetime.now(),
                'action_required': True,
                'suggested_action': 'Consider adjusting your strategy for the new market regime'
            })
        
        return alerts
    
    def _filter_cooldown(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out alerts that are in cooldown period
        
        Args:
            alerts: List of potential alerts
            
        Returns:
            Filtered list of alerts
        """
        filtered_alerts = []
        now = datetime.now()
        
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert.get('ticker', 'portfolio')}"
            last_sent = self.last_alert_time.get(alert_key)
            
            if not last_sent or (now - last_sent) > self.cooldown_period:
                filtered_alerts.append(alert)
                self.last_alert_time[alert_key] = now
        
        return filtered_alerts
    
    def _record_alert(self, alert: Dict[str, Any]):
        """
        Record alert in history
        
        Args:
            alert: Alert to record
        """
        self.alert_history.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def get_alert_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get summary of recent alerts
        
        Args:
            days: Number of days to look back
            
        Returns:
            Summary of alerts
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > cutoff_date]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'high_severity': len([a for a in recent_alerts if a['severity'] == 'high']),
            'medium_severity': len([a for a in recent_alerts if a['severity'] == 'medium']),
            'low_severity': len([a for a in recent_alerts if a['severity'] == 'low']),
            'by_type': {},
            'recent_alerts': recent_alerts[-5:]  # Last 5 alerts
        }
        
        # Count by type
        for alert in recent_alerts:
            alert_type = alert['type']
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
        
        return summary
    
    def format_alert_email(self, alerts: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Format alerts into email content
        
        Args:
            alerts: List of alerts to format
            
        Returns:
            Tuple of (subject, html_content)
        """
        if not alerts:
            return None, None
        
        # Determine subject based on highest severity
        severities = [a['severity'] for a in alerts]
        if 'high' in severities:
            subject = "🚨 URGENT: Portfolio Alert - Action Required"
        elif 'medium' in severities:
            subject = "📊 Portfolio Alert - Review Recommended"
        else:
            subject = "📌 Portfolio Update - For Your Information"
        
        # Build HTML content
        html_content = """
        <html>
        <head>
            <style>
                .alert-high { background: #f8d7da; border-left: 4px solid #dc3545; }
                .alert-medium { background: #fff3cd; border-left: 4px solid #ffc107; }
                .alert-low { background: #d4edda; border-left: 4px solid #28a745; }
                .alert-box { padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <h2>Portfolio Alerts</h2>
        """
        
        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            html_content += f"""
            <div class="alert-box {severity_class}">
                <h3>{alert['title']}</h3>
                <p>{alert['message']}</p>
                {f"<p><strong>Recommended Action:</strong> {alert['suggested_action']}</p>" if alert.get('action_required') else ""}
                <p style="font-size: 12px; color: #666;">
                    {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}
                </p>
            </div>
            """
        
        html_content += """
            <div style="margin-top: 30px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                <p><strong>What to do next:</strong></p>
                <ol>
                    <li>Review the alerts above</li>
                    <li>Log in to view your full portfolio report</li>
                    <li>Consider the suggested actions</li>
                    <li>Adjust your portfolio if needed</li>
                </ol>
                <a href="#" style="display: inline-block; margin-top: 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">
                    View Full Report
                </a>
            </div>
        </body>
        </html>
        """
        
        return subject, html_content
    
    def should_send_daily_brief(self, portfolio_data: Dict[str, Any]) -> bool:
        """
        Check if daily brief should be sent (Premium tier only)
        
        Args:
            portfolio_data: Portfolio data with daily change
            
        Returns:
            True if daily brief should be sent
        """
        # Send if portfolio moved more than 2%
        daily_change = abs(portfolio_data.get('daily_change', 0))
        return daily_change > 0.02
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about alert history"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'avg_per_day': 0,
                'most_common_type': None
            }
        
        # Calculate time span
        oldest_alert = min(self.alert_history, key=lambda x: x['timestamp'])
        newest_alert = max(self.alert_history, key=lambda x: x['timestamp'])
        days_span = (newest_alert['timestamp'] - oldest_alert['timestamp']).days + 1
        
        # Count by type
        type_counts = {}
        for alert in self.alert_history:
            alert_type = alert['type']
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        most_common_type = max(type_counts, key=type_counts.get) if type_counts else None
        
        return {
            'total_alerts': len(self.alert_history),
            'avg_per_day': len(self.alert_history) / days_span if days_span > 0 else 0,
            'most_common_type': most_common_type,
            'type_distribution': type_counts,
            'days_tracked': days_span
        }
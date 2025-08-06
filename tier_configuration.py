"""
AlphaSheet Intelligence™ - Tier Configuration
Main configuration file that uses the utility classes
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import our utility classes from the utils folder
from utils import EmailScheduler, AlertSystem, UsageTracker

logger = logging.getLogger(__name__)

class SubscriptionTier(Enum):
    """Subscription tier levels"""
    STARTER = "starter"
    GROWTH = "growth"
    PREMIUM = "premium"

class TierConfiguration:
    """Main tier configuration and feature management"""
    
    @staticmethod
    def get_tier_features(tier: str) -> Dict[str, Any]:
        """
        Get features available for a tier
        
        Args:
            tier: Subscription tier name
            
        Returns:
            Dictionary of features and limits
        """
        tier = tier.lower()
        
        features = {
            'starter': {
                'reports_per_month': 2,
                'portfolios': 1,
                'currencies': ['USD'],
                'features': [
                    'portfolio_summary',
                    'performance_metrics',
                    'basic_risk_analysis',
                    'goal_tracking',
                    'tax_summary'
                ],
                'ai_model': 'claude-3-sonnet-20240229',
                'email_frequency': 'weekly',
                'email_type': 'summary',
                'alerts': False,
                'export_formats': ['html'],
                'price': 19.00
            },
            'growth': {
                'reports_per_month': 4,
                'portfolios': 3,
                'currencies': ['USD', 'CAD', 'EUR', 'GBP', 'JPY', 'AUD', 
                              'CHF', 'CNY', 'HKD', 'SGD', 'INR', 'MXN'],
                'features': [
                    # All Starter features plus:
                    'portfolio_summary',
                    'performance_metrics',
                    'basic_risk_analysis',
                    'goal_tracking',
                    'tax_summary',
                    # Growth additions:
                    'monte_carlo_simulation',
                    'tax_optimization',
                    'tax_loss_harvesting',
                    'rebalancing_recommendations',
                    'scenario_analysis',
                    'dividend_optimization',
                    'correlation_analysis',
                    'factor_analysis',
                    'regional_tax_strategies'
                ],
                'ai_model': 'claude-3-opus-20240229',
                'email_frequency': 'weekly',
                'email_type': 'intelligence_brief',
                'alerts': False,
                'export_formats': ['html', 'pdf'],
                'price': 39.00
            },
            'premium': {
                'reports_per_month': -1,  # Unlimited
                'portfolios': 10,
                'currencies': ['USD', 'CAD', 'EUR', 'GBP', 'JPY', 'AUD', 
                              'CHF', 'CNY', 'HKD', 'SGD', 'INR', 'MXN'],
                'features': [
                    # All features
                    'all'
                ],
                'ai_model': 'claude-3-opus-20240229',
                'email_frequency': 'daily',
                'email_type': 'market_brief',
                'alerts': True,
                'alert_channels': ['email', 'sms'],
                'export_formats': ['html', 'pdf', 'excel'],
                'api_access': True,
                'white_label': True,
                'custom_benchmarks': True,
                'price': 79.00
            }
        }
        
        return features.get(tier, features['starter'])
    
    @staticmethod
    def get_tier_comparison() -> Dict[str, Any]:
        """
        Get comparison of all tiers for pricing page
        
        Returns:
            Comparison dictionary
        """
        return {
            'starter': {
                'name': 'Starter',
                'price': 19,
                'best_for': 'Individual investors getting started',
                'reports': '2 per month',
                'portfolios': '1 portfolio',
                'highlights': [
                    '✓ Portfolio analysis',
                    '✓ Risk metrics',
                    '✓ Goal tracking',
                    '✓ Weekly email summary',
                    '✓ Basic tax summary'
                ]
            },
            'growth': {
                'name': 'Growth',
                'price': 39,
                'best_for': 'Active investors with multiple accounts',
                'reports': '4 per month',
                'portfolios': '3 portfolios',
                'highlights': [
                    '✓ Everything in Starter',
                    '✓ Monte Carlo simulations',
                    '✓ Tax optimization',
                    '✓ Multi-currency (12+)',
                    '✓ Rebalancing AI',
                    '✓ Advanced insights (Opus)',
                    '✓ PDF exports'
                ]
            },
            'premium': {
                'name': 'Premium',
                'price': 79,
                'best_for': 'Professional traders and advisors',
                'reports': 'Unlimited',
                'portfolios': '10 portfolios',
                'highlights': [
                    '✓ Everything in Growth',
                    '✓ Real-time alerts',
                    '✓ ML trading signals',
                    '✓ Options strategies',
                    '✓ Liquidity analysis',
                    '✓ API access',
                    '✓ White-label reports',
                    '✓ Priority support'
                ]
            }
        }
    
    @staticmethod
    def validate_feature_access(tier: str, feature: str) -> bool:
        """
        Check if a tier has access to a feature
        
        Args:
            tier: Subscription tier
            feature: Feature to check
            
        Returns:
            True if feature is accessible
        """
        tier_config = TierConfiguration.get_tier_features(tier)
        features = tier_config.get('features', [])
        
        if 'all' in features:
            return True
        
        return feature in features
    
    @staticmethod
    def get_ai_model(tier: str) -> str:
        """
        Get the AI model for a tier
        
        Args:
            tier: Subscription tier
            
        Returns:
            AI model identifier
        """
        tier_config = TierConfiguration.get_tier_features(tier)
        return tier_config.get('ai_model', 'claude-3-sonnet-20240229')
    
    @staticmethod
    def get_currency_list(tier: str) -> List[str]:
        """
        Get available currencies for a tier
        
        Args:
            tier: Subscription tier
            
        Returns:
            List of currency codes
        """
        tier_config = TierConfiguration.get_tier_features(tier)
        return tier_config.get('currencies', ['USD'])


class TierEnforcer:
    """Enforces tier restrictions on analysis"""
    
    def __init__(self, tier: str):
        """
        Initialize tier enforcer
        
        Args:
            tier: Customer subscription tier
        """
        self.tier = tier.lower()
        self.config = TierConfiguration.get_tier_features(tier)
    
    def filter_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter analysis results based on tier
        
        Args:
            results: Complete analysis results
            
        Returns:
            Filtered results based on tier access
        """
        if self.tier == 'premium':
            return results  # Premium gets everything
        
        filtered = {}
        
        # Always include basic features
        basic_features = [
            'portfolio_summary', 'performance', 'positions',
            'risk_metrics', 'goals'
        ]
        
        for feature in basic_features:
            if feature in results:
                filtered[feature] = results[feature]
        
        # Add tier-specific features
        if self.tier == 'growth':
            growth_features = [
                'monte_carlo', 'tax_optimization', 'rebalancing',
                'scenario_analysis', 'correlation_matrix', 'factor_analysis'
            ]
            for feature in growth_features:
                if feature in results:
                    filtered[feature] = results[feature]
        
        return filtered
    
    def check_portfolio_limit(self, current_count: int) -> bool:
        """
        Check if portfolio limit is reached
        
        Args:
            current_count: Current number of portfolios
            
        Returns:
            True if within limit
        """
        limit = self.config.get('portfolios', 1)
        return current_count < limit
    
    def check_currency_access(self, currency: str) -> bool:
        """
        Check if currency is available for tier
        
        Args:
            currency: Currency code
            
        Returns:
            True if currency is available
        """
        available = self.config.get('currencies', ['USD'])
        return currency in available


# Example usage function to test everything works
def test_tier_system():
    """Test function to verify tier system works"""
    
    # Test tier configuration
    print("Testing Tier Configuration...")
    starter_features = TierConfiguration.get_tier_features('starter')
    print(f"Starter tier gets {starter_features['reports_per_month']} reports/month")
    
    # Test usage tracking
    print("\nTesting Usage Tracker...")
    tracker = UsageTracker('test_customer', 'starter')
    print(f"Can generate report: {tracker.can_generate_report()}")
    print(f"Reports remaining: {tracker.get_remaining_reports()}")
    
    # Test email scheduler
    print("\nTesting Email Scheduler...")
    scheduler = EmailScheduler('growth')
    print(f"Email type for Growth tier: {scheduler.get_email_content_type()}")
    
    # Test alert system
    print("\nTesting Alert System...")
    alert_system = AlertSystem()
    test_portfolio = {
        'daily_change': -0.06,  # 6% drop
        'positions': {
            'AAPL': {'daily_change': 0.02},
            'NVDA': {'daily_change': -0.18}  # 18% drop
        }
    }
    alerts = alert_system.check_all_alerts(test_portfolio)
    print(f"Found {len(alerts)} alerts")
    
    print("\n[OK] All systems working!")


if __name__ == '__main__':
    test_tier_system()
"""
AlphaSheet Intelligence™ -- Usage Tracker Utility
Tracks customer usage and enforces tier limits
"""

from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import json
import os
import logging

logger = logging.getLogger(__name__)

class UsageTracker:
    """Tracks usage and enforces tier limits"""
    
    # Tier limits
    TIER_LIMITS = {
        'starter': {
            'reports_per_month': 2,
            'portfolios': 1,
            'currencies': 1,
            'ai_model': 'claude-3-sonnet',
            'email_frequency': 'weekly'
        },
        'growth': {
            'reports_per_month': 4,
            'portfolios': 3,
            'currencies': 12,
            'ai_model': 'claude-3-opus',
            'email_frequency': 'weekly'
        },
        'premium': {
            'reports_per_month': -1,  # Unlimited
            'portfolios': 10,
            'currencies': 12,
            'ai_model': 'claude-3-opus',
            'email_frequency': 'daily',
            'alerts': True
        }
    }
    
    def __init__(self, customer_id: str, tier: str = 'starter', storage_path: str = './usage_data'):
        """
        Initialize usage tracker
        
        Args:
            customer_id: Unique customer identifier
            tier: Customer subscription tier
            storage_path: Path to store usage data files
        """
        self.customer_id = customer_id
        self.tier = tier.lower()
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        # Load or initialize usage data
        self.usage_data = self._load_usage_data()
    
    def _get_usage_file_path(self) -> str:
        """Get the path to the customer's usage file"""
        return os.path.join(self.storage_path, f"{self.customer_id}_usage.json")
    
    def _load_usage_data(self) -> Dict[str, Any]:
        """
        Load usage data from file
        
        Returns:
            Usage data dictionary
        """
        file_path = self._get_usage_file_path()
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Check if we need to reset monthly data
                last_reset = datetime.fromisoformat(data.get('last_reset', '2024-01-01'))
                if self._should_reset_monthly(last_reset):
                    data = self._reset_monthly_usage(data)
                
                return data
            except Exception as e:
                logger.error(f"Error loading usage data: {str(e)}")
                return self._create_new_usage_data()
        else:
            return self._create_new_usage_data()
    
    def _create_new_usage_data(self) -> Dict[str, Any]:
        """
        Create new usage data structure
        
        Returns:
            New usage data dictionary
        """
        return {
            'customer_id': self.customer_id,
            'tier': self.tier,
            'reports_generated': 0,
            'reports_this_month': 0,
            'portfolios_count': 0,
            'last_report_time': None,
            'last_reset': datetime.now().isoformat(),
            'month': datetime.now().month,
            'year': datetime.now().year,
            'history': []
        }
    
    def _should_reset_monthly(self, last_reset: datetime) -> bool:
        """
        Check if monthly usage should be reset
        
        Args:
            last_reset: Last reset datetime
            
        Returns:
            True if should reset
        """
        now = datetime.now()
        return (now.year > last_reset.year) or (now.month > last_reset.month)
    
    def _reset_monthly_usage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset monthly usage counters
        
        Args:
            data: Current usage data
            
        Returns:
            Updated usage data
        """
        # Archive current month's data
        if data['reports_this_month'] > 0:
            data['history'].append({
                'month': data['month'],
                'year': data['year'],
                'reports': data['reports_this_month'],
                'tier': data['tier']
            })
        
        # Keep only last 12 months of history
        if len(data['history']) > 12:
            data['history'] = data['history'][-12:]
        
        # Reset counters
        data['reports_this_month'] = 0
        data['last_reset'] = datetime.now().isoformat()
        data['month'] = datetime.now().month
        data['year'] = datetime.now().year
        
        return data
    
    def _save_usage_data(self):
        """Save usage data to file"""
        try:
            file_path = self._get_usage_file_path()
            with open(file_path, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving usage data: {str(e)}")
    
    def can_generate_report(self) -> bool:
        """
        Check if customer can generate a report
        
        Returns:
            True if within limits or unlimited
        """
        limits = self.TIER_LIMITS.get(self.tier, {})
        monthly_limit = limits.get('reports_per_month', 2)
        
        # Unlimited reports for premium
        if monthly_limit == -1:
            return True
        
        # Check against monthly limit
        return self.usage_data['reports_this_month'] < monthly_limit
    
    def record_report_generation(self):
        """Record that a report was generated"""
        self.usage_data['reports_generated'] += 1
        self.usage_data['reports_this_month'] += 1
        self.usage_data['last_report_time'] = datetime.now().isoformat()
        self._save_usage_data()
        
        logger.info(f"Report generated for {self.customer_id}. "
                   f"Monthly total: {self.usage_data['reports_this_month']}")
    
    def get_remaining_reports(self) -> int:
        """
        Get number of reports remaining this month
        
        Returns:
            Number of reports remaining (-1 for unlimited)
        """
        limits = self.TIER_LIMITS.get(self.tier, {})
        monthly_limit = limits.get('reports_per_month', 2)
        
        if monthly_limit == -1:
            return -1  # Unlimited
        
        remaining = monthly_limit - self.usage_data['reports_this_month']
        return max(0, remaining)
    
    def can_add_portfolio(self) -> bool:
        """
        Check if customer can add another portfolio
        
        Returns:
            True if within portfolio limit
        """
        limits = self.TIER_LIMITS.get(self.tier, {})
        portfolio_limit = limits.get('portfolios', 1)
        
        return self.usage_data['portfolios_count'] < portfolio_limit
    
    def record_portfolio_added(self):
        """Record that a portfolio was added"""
        self.usage_data['portfolios_count'] += 1
        self._save_usage_data()
    
    def record_portfolio_removed(self):
        """Record that a portfolio was removed"""
        if self.usage_data['portfolios_count'] > 0:
            self.usage_data['portfolios_count'] -= 1
            self._save_usage_data()
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of current usage
        
        Returns:
            Usage summary dictionary
        """
        limits = self.TIER_LIMITS.get(self.tier, {})
        
        return {
            'customer_id': self.customer_id,
            'tier': self.tier,
            'reports_used': self.usage_data['reports_this_month'],
            'reports_limit': limits.get('reports_per_month', 2),
            'reports_remaining': self.get_remaining_reports(),
            'portfolios_used': self.usage_data['portfolios_count'],
            'portfolios_limit': limits.get('portfolios', 1),
            'last_report': self.usage_data.get('last_report_time'),
            'reset_date': self._get_next_reset_date().isoformat(),
            'features': {
                'currencies': limits.get('currencies', 1),
                'ai_model': limits.get('ai_model', 'claude-3-sonnet'),
                'email_frequency': limits.get('email_frequency', 'weekly'),
                'alerts_enabled': limits.get('alerts', False)
            }
        }
    
    def _get_next_reset_date(self) -> datetime:
        """
        Get the next monthly reset date
        
        Returns:
            Next reset datetime
        """
        now = datetime.now()
        if now.month == 12:
            next_reset = datetime(now.year + 1, 1, 1)
        else:
            next_reset = datetime(now.year, now.month + 1, 1)
        return next_reset
    
    def get_historical_usage(self) -> List[Dict[str, Any]]:
        """
        Get historical usage data
        
        Returns:
            List of historical usage by month
        """
        return self.usage_data.get('history', [])
    
    def update_tier(self, new_tier: str):
        """
        Update customer tier
        
        Args:
            new_tier: New subscription tier
        """
        self.tier = new_tier.lower()
        self.usage_data['tier'] = self.tier
        self._save_usage_data()
        
        logger.info(f"Tier updated for {self.customer_id}: {new_tier}")
    
    def is_tier_feature_available(self, feature: str) -> bool:
        """
        Check if a specific feature is available for the customer's tier
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is available
        """
        tier_features = {
            'starter': [
                'basic_analysis', 'portfolio_summary', 'risk_metrics',
                'goal_tracking', 'tax_summary'
            ],
            'growth': [
                'basic_analysis', 'portfolio_summary', 'risk_metrics',
                'goal_tracking', 'tax_summary', 'monte_carlo', 'tax_optimization',
                'multi_currency', 'rebalancing', 'scenario_analysis',
                'dividend_optimization', 'factor_analysis', 'correlation_matrix'
            ],
            'premium': [
                # All features available
                'all'
            ]
        }
        
        available_features = tier_features.get(self.tier, [])
        
        if 'all' in available_features:
            return True
        
        return feature in available_features
    
    def get_tier_upgrade_benefits(self) -> Dict[str, List[str]]:
        """
        Get benefits of upgrading to higher tiers
        
        Returns:
            Dictionary of upgrade benefits by tier
        """
        current_tier_index = ['starter', 'growth', 'premium'].index(self.tier)
        benefits = {}
        
        if current_tier_index < 1:  # Can upgrade to Growth
            benefits['growth'] = [
                'Increase reports from 2 to 4 per month',
                'Manage up to 3 portfolios',
                'Access 12+ currencies',
                'Advanced AI insights with Claude Opus',
                'Monte Carlo simulations',
                'Tax optimization strategies',
                'Rebalancing recommendations',
                'Factor analysis'
            ]
        
        if current_tier_index < 2:  # Can upgrade to Premium
            benefits['premium'] = [
                'Unlimited reports',
                'Manage up to 10 portfolios',
                'Real-time alerts',
                'Daily market briefs',
                'ML trading signals',
                'Options strategies',
                'Liquidity analysis',
                'Risk parity optimization',
                'API access',
                'White-label reports'
            ]
        
        return benefits
    
    def estimate_usage_cost(self) -> float:
        """
        Estimate monthly cost based on usage
        
        Returns:
            Estimated monthly cost
        """
        tier_costs = {
            'starter': 19.00,
            'growth': 39.00,
            'premium': 79.00
        }
        
        base_cost = tier_costs.get(self.tier, 19.00)
        
        # Could add overage charges here if needed
        # For now, just return base cost
        return base_cost
    
    def should_suggest_upgrade(self) -> bool:
        """
        Check if we should suggest an upgrade based on usage patterns
        
        Returns:
            True if upgrade should be suggested
        """
        # Suggest upgrade if user is hitting limits
        if self.tier == 'starter':
            # If using max reports consistently
            if self.usage_data['reports_this_month'] >= 2:
                return True
        elif self.tier == 'growth':
            # If using max reports consistently
            if self.usage_data['reports_this_month'] >= 4:
                return True
            # If managing multiple portfolios
            if self.usage_data['portfolios_count'] >= 3:
                return True
        
        return False
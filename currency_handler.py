"""
Multi-Currency Handler for Portfolio Intelligence
Supports 12+ currencies with real-time FX rates
Regional benchmark selection and currency analysis
"""

import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class CurrencyHandler:
    """
    Handles all currency conversions and regional considerations
    for multi-currency portfolios
    """
    
    # Supported currencies and their regional mappings
    SUPPORTED_CURRENCIES = {
        'USD': {'region': 'US', 'symbol': '$', 'name': 'US Dollar'},
        'CAD': {'region': 'CA', 'symbol': 'C$', 'name': 'Canadian Dollar'},
        'GBP': {'region': 'UK', 'symbol': '£', 'name': 'British Pound'},
        'EUR': {'region': 'EU', 'symbol': '€', 'name': 'Euro'},
        'AUD': {'region': 'AU', 'symbol': 'A$', 'name': 'Australian Dollar'},
        'JPY': {'region': 'JP', 'symbol': '¥', 'name': 'Japanese Yen'},
        'CHF': {'region': 'CH', 'symbol': 'Fr', 'name': 'Swiss Franc'},
        'NZD': {'region': 'NZ', 'symbol': 'NZ$', 'name': 'New Zealand Dollar'},
        'SGD': {'region': 'SG', 'symbol': 'S$', 'name': 'Singapore Dollar'},
        'HKD': {'region': 'HK', 'symbol': 'HK$', 'name': 'Hong Kong Dollar'},
        'INR': {'region': 'IN', 'symbol': '₹', 'name': 'Indian Rupee'},
        'CNY': {'region': 'CN', 'symbol': '¥', 'name': 'Chinese Yuan'}
    }
    
    # Regional benchmark indices
    REGIONAL_BENCHMARKS = {
        'US': {'primary': 'SPY', 'secondary': 'QQQ', 'bonds': 'AGG'},
        'CA': {'primary': 'XIC.TO', 'secondary': 'XIU.TO', 'bonds': 'XBB.TO'},
        'UK': {'primary': 'ISF.L', 'secondary': 'VUKE.L', 'bonds': 'IGLT.L'},
        'EU': {'primary': 'VEUR.AS', 'secondary': 'SX5E', 'bonds': 'IEAC.AS'},
        'AU': {'primary': 'STW.AX', 'secondary': 'VAS.AX', 'bonds': 'VAF.AX'},
        'JP': {'primary': 'EWJ', 'secondary': '1306.T', 'bonds': '2561.T'},
        'CH': {'primary': 'EWL', 'secondary': 'CHSPI.SW', 'bonds': 'SWISS.SW'},
        'NZ': {'primary': 'FNZ.NZ', 'secondary': 'NZX50', 'bonds': 'NZB.NZ'},
        'SG': {'primary': 'EWS', 'secondary': 'ES3.SI', 'bonds': 'A35.SI'},
        'HK': {'primary': 'EWH', 'secondary': '2800.HK', 'bonds': '2819.HK'},
        'IN': {'primary': 'INDA', 'secondary': 'NIFTYBEES.NS', 'bonds': 'LIQUIDBEES.NS'},
        'CN': {'primary': 'ASHR', 'secondary': '510050.SS', 'bonds': '511010.SS'}
    }
    
    # Currency pairs for FX conversion
    FX_PAIRS = {
        'USD': '',  # Base currency
        'CAD': 'CAD=X',
        'GBP': 'GBP=X',
        'EUR': 'EUR=X',
        'AUD': 'AUD=X',
        'JPY': 'JPY=X',
        'CHF': 'CHF=X',
        'NZD': 'NZD=X',
        'SGD': 'SGD=X',
        'HKD': 'HKD=X',
        'INR': 'INR=X',
        'CNY': 'CNY=X'
    }
    
    def __init__(self):
        """Initialize currency handler with rate cache"""
        self._fx_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(hours=1)  # Cache FX rates for 1 hour
        
    @lru_cache(maxsize=32)
    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get exchange rate between two currencies
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Exchange rate (from_currency/to_currency)
        """
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # Same currency
        if from_currency == to_currency:
            return 1.0
            
        # Check if currencies are supported
        if from_currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {from_currency}")
        if to_currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {to_currency}")
            
        try:
            # Try direct conversion first
            if from_currency == 'USD':
                return self._get_usd_to_currency_rate(to_currency)
            elif to_currency == 'USD':
                return 1.0 / self._get_usd_to_currency_rate(from_currency)
            else:
                # Cross rate through USD
                from_to_usd = 1.0 / self._get_usd_to_currency_rate(from_currency)
                usd_to_target = self._get_usd_to_currency_rate(to_currency)
                return from_to_usd * usd_to_target
                
        except Exception as e:
            logger.warning(f"Failed to get FX rate {from_currency}/{to_currency}: {e}")
            # Return approximate rates as fallback
            return self._get_fallback_rate(from_currency, to_currency)
    
    def _get_usd_to_currency_rate(self, currency: str) -> float:
        """Get USD to currency exchange rate"""
        if currency == 'USD':
            return 1.0
            
        # Check cache first
        if self._is_cache_valid():
            if currency in self._fx_cache:
                return self._fx_cache[currency]
        
        # Fetch from Yahoo Finance
        fx_pair = self.FX_PAIRS.get(currency)
        if not fx_pair:
            raise ValueError(f"No FX pair defined for {currency}")
            
        try:
            ticker = yf.Ticker(fx_pair)
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1]
                # Cache the rate
                self._fx_cache[currency] = rate
                self._cache_timestamp = datetime.now()
                return rate
        except Exception as e:
            logger.warning(f"Failed to fetch {fx_pair}: {e}")
            
        return self._get_fallback_rate('USD', currency)
    
    def _is_cache_valid(self) -> bool:
        """Check if FX cache is still valid"""
        if not self._cache_timestamp:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_duration
    
    def _get_fallback_rate(self, from_currency: str, to_currency: str) -> float:
        """Get approximate fallback exchange rates"""
        # Approximate rates as of 2024 (fallback only)
        fallback_usd_rates = {
            'USD': 1.0,
            'CAD': 1.35,
            'GBP': 0.79,
            'EUR': 0.92,
            'AUD': 1.52,
            'JPY': 150.0,
            'CHF': 0.88,
            'NZD': 1.62,
            'SGD': 1.34,
            'HKD': 7.82,
            'INR': 83.0,
            'CNY': 7.20
        }
        
        if from_currency == 'USD':
            return fallback_usd_rates.get(to_currency, 1.0)
        elif to_currency == 'USD':
            return 1.0 / fallback_usd_rates.get(from_currency, 1.0)
        else:
            # Cross rate
            from_usd = fallback_usd_rates.get(from_currency, 1.0)
            to_usd = fallback_usd_rates.get(to_currency, 1.0)
            return to_usd / from_usd
    
    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convert amount from one currency to another
        
        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            
        Returns:
            Converted amount
        """
        rate = self.get_exchange_rate(from_currency, to_currency)
        return amount * rate
    
    def get_regional_benchmarks(self, currency: str) -> Dict[str, str]:
        """
        Get appropriate benchmark indices for a currency/region
        
        Args:
            currency: Currency code
            
        Returns:
            Dictionary of benchmark tickers
        """
        currency = currency.upper()
        region = self.SUPPORTED_CURRENCIES.get(currency, {}).get('region', 'US')
        
        benchmarks = self.REGIONAL_BENCHMARKS.get(region, self.REGIONAL_BENCHMARKS['US'])
        
        # Add global benchmarks
        benchmarks['global'] = 'VT'  # Vanguard Total World Stock ETF
        benchmarks['sp500'] = 'SPY'  # Always include S&P 500 for comparison
        
        return benchmarks
    
    def detect_portfolio_currencies(self, holdings: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Detect all currencies in a portfolio and their weights
        
        Args:
            holdings: List of holdings with 'ticker' and 'value' keys
            
        Returns:
            Dictionary of currency weights
        """
        currency_values = {}
        total_value = 0
        
        for holding in holdings:
            ticker = holding.get('ticker', '')
            value = holding.get('value', 0)
            currency = self._detect_ticker_currency(ticker)
            
            if currency not in currency_values:
                currency_values[currency] = 0
            currency_values[currency] += value
            total_value += value
        
        # Convert to weights
        currency_weights = {}
        if total_value > 0:
            for currency, value in currency_values.items():
                currency_weights[currency] = value / total_value
                
        return currency_weights
    
    def _detect_ticker_currency(self, ticker: str) -> str:
        """
        Detect currency of a ticker based on exchange suffix
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Currency code
        """
        # Exchange suffixes to currency mapping
        exchange_currency_map = {
            '.TO': 'CAD',  # Toronto
            '.V': 'CAD',   # TSX Venture
            '.CN': 'CAD',  # Canadian Securities Exchange
            '.L': 'GBP',   # London
            '.IL': 'GBP',  # London International
            '.AS': 'EUR',  # Amsterdam
            '.PA': 'EUR',  # Paris
            '.DE': 'EUR',  # Frankfurt
            '.MI': 'EUR',  # Milan
            '.MC': 'EUR',  # Madrid
            '.AX': 'AUD',  # Australian
            '.NZ': 'NZD',  # New Zealand
            '.SI': 'SGD',  # Singapore
            '.HK': 'HKD',  # Hong Kong
            '.NS': 'INR',  # National Stock Exchange of India
            '.BO': 'INR',  # Bombay Stock Exchange
            '.SS': 'CNY',  # Shanghai
            '.SZ': 'CNY',  # Shenzhen
            '.T': 'JPY',   # Tokyo
            '.SW': 'CHF',  # Swiss
            '.VX': 'CHF',  # Swiss
        }
        
        # Check for exchange suffix
        for suffix, currency in exchange_currency_map.items():
            if ticker.upper().endswith(suffix):
                return currency
                
        # Default to USD for no suffix
        return 'USD'
    
    def calculate_currency_exposure(self, holdings: List[Dict[str, Any]], 
                                   base_currency: str) -> Dict[str, Any]:
        """
        Calculate currency exposure and risk for a portfolio
        
        Args:
            holdings: List of holdings
            base_currency: Base currency for analysis
            
        Returns:
            Currency exposure analysis
        """
        currency_weights = self.detect_portfolio_currencies(holdings)
        
        exposure_analysis = {
            'base_currency': base_currency,
            'currency_weights': currency_weights,
            'currency_count': len(currency_weights),
            'home_bias': currency_weights.get(base_currency, 0),
            'foreign_exposure': 1 - currency_weights.get(base_currency, 0),
            'concentration_risk': self._calculate_currency_concentration(currency_weights),
            'recommended_hedging': self._recommend_hedging(currency_weights, base_currency)
        }
        
        return exposure_analysis
    
    def _calculate_currency_concentration(self, currency_weights: Dict[str, float]) -> str:
        """Calculate currency concentration risk level"""
        if not currency_weights:
            return 'Unknown'
            
        max_weight = max(currency_weights.values())
        
        if max_weight > 0.8:
            return 'Very High'
        elif max_weight > 0.6:
            return 'High'
        elif max_weight > 0.4:
            return 'Moderate'
        else:
            return 'Low'
    
    def _recommend_hedging(self, currency_weights: Dict[str, float], 
                          base_currency: str) -> Dict[str, Any]:
        """Recommend currency hedging strategy"""
        foreign_exposure = 1 - currency_weights.get(base_currency, 0)
        
        recommendations = {
            'hedge_percentage': 0,
            'strategy': 'No hedging needed',
            'instruments': []
        }
        
        if foreign_exposure > 0.7:
            recommendations['hedge_percentage'] = 50
            recommendations['strategy'] = 'Consider hedging 50% of foreign exposure'
            recommendations['instruments'] = ['Currency-hedged ETFs', 'FX forwards']
        elif foreign_exposure > 0.5:
            recommendations['hedge_percentage'] = 30
            recommendations['strategy'] = 'Consider partial hedging (30%) for stability'
            recommendations['instruments'] = ['Currency-hedged ETFs']
        elif foreign_exposure > 0.3:
            recommendations['hedge_percentage'] = 0
            recommendations['strategy'] = 'Natural diversification, hedging optional'
            
        return recommendations
    
    def format_currency(self, amount: float, currency: str) -> str:
        """
        Format amount with appropriate currency symbol
        
        Args:
            amount: Amount to format
            currency: Currency code
            
        Returns:
            Formatted string
        """
        currency = currency.upper()
        currency_info = self.SUPPORTED_CURRENCIES.get(currency, {})
        symbol = currency_info.get('symbol', '$')
        
        # Special formatting for certain currencies
        if currency == 'JPY':
            return f"{symbol}{amount:,.0f}"  # No decimals for Yen
        elif currency in ['INR', 'CNY']:
            return f"{symbol}{amount:,.0f}"  # No decimals
        else:
            return f"{symbol}{amount:,.2f}"  # Standard 2 decimals
    
    def get_historical_fx_data(self, currency_pair: Tuple[str, str], 
                              period: str = '1y') -> pd.DataFrame:
        """
        Get historical FX data for analysis
        
        Args:
            currency_pair: Tuple of (from_currency, to_currency)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            
        Returns:
            DataFrame with historical FX rates
        """
        from_currency, to_currency = currency_pair
        
        # Construct the FX ticker
        if from_currency == 'USD':
            ticker_symbol = self.FX_PAIRS.get(to_currency)
        elif to_currency == 'USD':
            ticker_symbol = self.FX_PAIRS.get(from_currency)
            # We'll need to invert the rates
        else:
            # Use USD as intermediate
            return self._get_cross_rate_history(from_currency, to_currency, period)
            
        if not ticker_symbol:
            raise ValueError(f"Cannot construct FX pair for {from_currency}/{to_currency}")
            
        try:
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period=period)
            
            # Invert if needed
            if to_currency == 'USD' and from_currency != 'USD':
                data['Close'] = 1 / data['Close']
                
            return data[['Close']].rename(columns={'Close': f'{from_currency}/{to_currency}'})
            
        except Exception as e:
            logger.error(f"Failed to get historical FX data: {e}")
            return pd.DataFrame()
    
    def _get_cross_rate_history(self, from_currency: str, to_currency: str, 
                               period: str) -> pd.DataFrame:
        """Get cross rate history through USD"""
        # Get both legs
        from_usd = self.get_historical_fx_data(('USD', from_currency), period)
        to_usd = self.get_historical_fx_data(('USD', to_currency), period)
        
        if from_usd.empty or to_usd.empty:
            return pd.DataFrame()
            
        # Calculate cross rate
        cross_rate = pd.DataFrame(index=from_usd.index)
        cross_rate[f'{from_currency}/{to_currency}'] = to_usd.iloc[:, 0] / from_usd.iloc[:, 0]
        
        return cross_rate
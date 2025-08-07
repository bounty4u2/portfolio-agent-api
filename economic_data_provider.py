"""
AlphaSheet Intelligence™ - Economic & Crypto Data Provider
Integrates FRED, World Bank, and Cryptocurrency data for comprehensive analysis
Region-aware economic indicators + global macro picture
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import lru_cache
import json
import yfinance as yf
import pandas as pd
import numpy as np

# Economic data libraries
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("fredapi not installed - FRED data unavailable")

try:
    import wbgapi as wb
    WORLD_BANK_AVAILABLE = True
except ImportError:
    WORLD_BANK_AVAILABLE = False
    logging.warning("wbgapi not installed - World Bank data unavailable")

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EconomicDataProvider:
    """
    Provides economic and crypto data for portfolio analysis
    Integrates FRED, World Bank, and cryptocurrency sources
    """
    
    # FRED Series IDs for key indicators
    FRED_SERIES = {
        'US': {
            'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth rate
            'inflation': 'CPIAUCSL',           # CPI All Urban Consumers
            'unemployment': 'UNRATE',          # Unemployment Rate
            'fed_funds': 'DFF',               # Federal Funds Rate
            'treasury_10y': 'DGS10',          # 10-Year Treasury
            'treasury_2y': 'DGS2',            # 2-Year Treasury  
            'dollar_index': 'DTWEXBGS',      # Trade Weighted Dollar Index
            'vix': 'VIXCLS',                 # VIX
            'consumer_sentiment': 'UMCSENT',  # U Michigan Consumer Sentiment
            'pce_inflation': 'PCEPI',        # PCE Price Index
            'retail_sales': 'RSXFS',         # Retail Sales
            'housing_starts': 'HOUST',       # Housing Starts
            'industrial_production': 'INDPRO', # Industrial Production Index
            'job_openings': 'JTSJOL',        # JOLTS Job Openings
        },
        'EU': {
            'ecb_rate': 'ECBDFR',            # ECB Deposit Facility Rate
            'eu_unemployment': 'LRHUTTTTEZM156S', # Euro Area Unemployment
            'eu_inflation': 'CPHPTT01EZM659N',   # Euro Area CPI
        },
        'UK': {
            'boe_rate': 'BOERUKM',           # Bank of England Rate
            'uk_inflation': 'CPALTT01GBM659N', # UK CPI
        },
        'JP': {
            'boj_rate': 'IRSTCI01JPM156N',   # Japan Interest Rate
            'jp_inflation': 'CPALTT01JPM659N', # Japan CPI
        },
        'CA': {
            'boc_rate': 'IRSTCI01CAM156N',   # Canada Interest Rate
            'ca_inflation': 'CPALTT01CAM659N', # Canada CPI
        }
    }
    
    # World Bank Indicators
    WORLD_BANK_INDICATORS = {
        'gdp_growth': 'NY.GDP.MKTP.KD.ZG',     # GDP growth annual %
        'gdp_per_capita': 'NY.GDP.PCAP.CD',    # GDP per capita
        'inflation': 'FP.CPI.TOTL.ZG',         # Inflation CPI
        'unemployment': 'SL.UEM.TOTL.ZS',      # Unemployment rate
        'trade_balance': 'NE.EXP.GNFS.ZS',     # Exports % of GDP
        'debt_to_gdp': 'GC.DOD.TOTL.GD.ZS',    # Debt to GDP
        'interest_rate': 'FR.INR.RINR',        # Real interest rate
        'exchange_rate': 'PA.NUS.FCRF',        # Official exchange rate
    }
    
    # Crypto tickers for different tiers
    CRYPTO_TICKERS = {
        'starter': ['BTC-USD', 'ETH-USD'],
        'growth': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 
                  'SOL-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'AVAX-USD'],
        'premium': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
                   'SOL-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'AVAX-USD',
                   'LINK-USD', 'UNI-USD', 'ATOM-USD', 'LTC-USD', 'ETC-USD',
                   'XLM-USD', 'ALGO-USD', 'VET-USD', 'FIL-USD', 'AAVE-USD']
    }
    
    # Region to country code mapping
    REGION_MAPPING = {
        'US': 'USA', 'CA': 'CAN', 'UK': 'GBR', 'EU': 'EMU',
        'JP': 'JPN', 'AU': 'AUS', 'CN': 'CHN', 'IN': 'IND',
        'SG': 'SGP', 'HK': 'HKG', 'KR': 'KOR', 'BR': 'BRA',
        'MX': 'MEX', 'CH': 'CHE', 'SE': 'SWE', 'NO': 'NOR'
    }
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the Economic Data Provider
        
        Args:
            fred_api_key: FRED API key (or from environment)
        """
        # Initialize FRED client
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred_client = None
        
        if FRED_AVAILABLE and self.fred_api_key:
            try:
                self.fred_client = Fred(api_key=self.fred_api_key)
                logger.info("✅ FRED API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FRED: {e}")
        else:
            logger.warning("FRED API not available")
        
        # Cache for API responses (5 minute TTL)
        self._cache = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Crypto Fear & Greed Index URL
        self.crypto_sentiment_url = "https://api.alternative.me/fng/"
        
    def get_economic_data(self, region: str = 'US', 
                          customer_tier: str = 'starter') -> Dict[str, Any]:
        """
        Get comprehensive economic data for a region
        
        Args:
            region: Country/region code
            customer_tier: Customer subscription tier
            
        Returns:
            Dictionary with economic indicators
        """
        cache_key = f"econ_{region}_{datetime.now().strftime('%Y%m%d%H')}"
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return cached_data
        
        economic_data = {
            'region': region,
            'timestamp': datetime.now().isoformat(),
            'indicators': {},
            'global_context': {},
            'data_quality': 'full'
        }
        
        # Get regional data
        if region in self.FRED_SERIES and self.fred_client:
            economic_data['indicators'] = self._fetch_fred_data(region)
        else:
            economic_data['indicators'] = self._fetch_world_bank_data(region)
        
        # Get global context
        economic_data['global_context'] = self._fetch_global_indicators()
        
        # Add tier-appropriate analysis
        if customer_tier in ['growth', 'premium']:
            economic_data['advanced_metrics'] = self._calculate_advanced_metrics(
                economic_data['indicators']
            )
        
        # Cache the results
        self._cache[cache_key] = (economic_data, datetime.now())
        
        return economic_data
    
    def get_crypto_data(self, customer_tier: str = 'starter',
                       portfolio_value: float = 0) -> Dict[str, Any]:
        """
        Get cryptocurrency market data based on tier
        
        Args:
            customer_tier: Customer subscription tier
            portfolio_value: Total portfolio value for context
            
        Returns:
            Dictionary with crypto market data and analysis
        """
        cache_key = f"crypto_{customer_tier}_{datetime.now().strftime('%Y%m%d%H')}"
        
        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._cache_ttl:
                return cached_data
        
        crypto_data = {
            'timestamp': datetime.now().isoformat(),
            'tier': customer_tier,
            'market_data': {},
            'metrics': {},
            'sentiment': {},
            'recommendations': []
        }
        
        # Get crypto prices based on tier
        tickers = self.CRYPTO_TICKERS.get(customer_tier, self.CRYPTO_TICKERS['starter'])
        crypto_data['market_data'] = self._fetch_crypto_prices(tickers)
        
        # Calculate metrics
        if crypto_data['market_data']:
            crypto_data['metrics'] = self._calculate_crypto_metrics(
                crypto_data['market_data'], portfolio_value
            )
        
        # Get sentiment data
        crypto_data['sentiment'] = self._fetch_crypto_sentiment()
        
        # Generate recommendations based on tier
        crypto_data['recommendations'] = self._generate_crypto_recommendations(
            crypto_data, customer_tier
        )
        
        # Cache the results
        self._cache[cache_key] = (crypto_data, datetime.now())
        
        return crypto_data
    
    def _fetch_fred_data(self, region: str) -> Dict[str, Any]:
        """Fetch data from FRED API"""
        indicators = {}
        series_ids = self.FRED_SERIES.get(region, self.FRED_SERIES['US'])
        
        for indicator_name, series_id in series_ids.items():
            try:
                # Get latest value
                data = self.fred_client.get_series_latest_release(series_id)
                if data is not None and len(data) > 0:
                    latest_value = float(data.iloc[-1])
                    
                    # Get historical data for change calculation
                    historical = self.fred_client.get_series(
                        series_id,
                        observation_start=datetime.now() - timedelta(days=365)
                    )
                    
                    if len(historical) > 1:
                        year_ago = float(historical.iloc[-12]) if len(historical) >= 12 else float(historical.iloc[0])
                        month_ago = float(historical.iloc[-2]) if len(historical) >= 2 else latest_value
                        
                        indicators[indicator_name] = {
                            'value': latest_value,
                            'month_change': ((latest_value - month_ago) / month_ago * 100) if month_ago != 0 else 0,
                            'year_change': ((latest_value - year_ago) / year_ago * 100) if year_ago != 0 else 0,
                            'source': 'FRED'
                        }
                    else:
                        indicators[indicator_name] = {
                            'value': latest_value,
                            'month_change': 0,
                            'year_change': 0,
                            'source': 'FRED'
                        }
                        
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator_name} for {region}: {e}")
                indicators[indicator_name] = {
                    'value': None,
                    'error': str(e),
                    'source': 'FRED'
                }
        
        return indicators
    
    def _fetch_world_bank_data(self, region: str) -> Dict[str, Any]:
        """Fetch data from World Bank API"""
        indicators = {}
        
        if not WORLD_BANK_AVAILABLE:
            return indicators
        
        # Convert region code to World Bank country code
        country_code = self.REGION_MAPPING.get(region, region)
        
        for indicator_name, indicator_id in self.WORLD_BANK_INDICATORS.items():
            try:
                # Get data for last 5 years
                data = wb.data.DataFrame(
                    indicator_id,
                    country_code,
                    time=range(datetime.now().year - 5, datetime.now().year + 1)
                )
                
                if not data.empty:
                    # Get latest non-null value
                    latest_data = data.dropna()
                    if not latest_data.empty:
                        latest_value = float(latest_data.iloc[-1, 0])
                        
                        indicators[indicator_name] = {
                            'value': latest_value,
                            'year': latest_data.index[-1],
                            'source': 'World Bank'
                        }
                        
            except Exception as e:
                logger.warning(f"Failed to fetch {indicator_name} for {region}: {e}")
                indicators[indicator_name] = {
                    'value': None,
                    'error': str(e),
                    'source': 'World Bank'
                }
        
        return indicators
    
    def _fetch_global_indicators(self) -> Dict[str, Any]:
        """Fetch global economic indicators"""
        global_data = {}
        
        # Key global indicators to track
        global_series = {
            'oil_wti': 'DCOILWTICO',        # WTI Oil Price
            'oil_brent': 'DCOILBRENTEU',    # Brent Oil Price
            'gold': 'GOLDAMGBD228NLBM',     # Gold Price
            'copper': 'PCOPPUSDM',          # Copper Price
            'baltic_dry': None,        # Baltic Dry Index (shipping)
            'global_uncertainty': 'GEPUCURRENT', # Global Economic Policy Uncertainty
        }
        
        if self.fred_client:
            for indicator_name, series_id in global_series.items():
                try:
                    data = self.fred_client.get_series_latest_release(series_id)
                    if data is not None and len(data) > 0:
                        global_data[indicator_name] = {
                            'value': float(data.iloc[-1]),
                            'source': 'FRED'
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch global indicator {indicator_name}: {e}")
        
        # Add major market indices via yfinance
        market_indices = {
            'sp500': '^GSPC',
            'nasdaq': '^IXIC',
            'dow': '^DJI',
            'vix': '^VIX',
            'dollar_index': 'DX-Y.NYB',
            'euro_stoxx': '^STOXX50E',
            'nikkei': '^N225',
            'shanghai': '000001.SS',
        }
        
        for index_name, ticker in market_indices.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                if not hist.empty:
                    latest_price = float(hist['Close'].iloc[-1])
                    prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else latest_price
                    
                    global_data[index_name] = {
                        'value': latest_price,
                        'day_change': ((latest_price - prev_price) / prev_price * 100) if prev_price != 0 else 0,
                        'source': 'yfinance'
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch {index_name}: {e}")
        
        return global_data
    
    def _fetch_crypto_prices(self, tickers: List[str]) -> Dict[str, Any]:
        """Fetch cryptocurrency prices from yfinance"""
        crypto_data = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_single_crypto, ticker): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        crypto_data[ticker] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker}: {e}")
        
        return crypto_data
    
    def _fetch_single_crypto(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch single cryptocurrency data"""
        try:
            crypto = yf.Ticker(ticker)
            hist = crypto.history(period='3mo')
            info = crypto.info
            
            if hist.empty:
                return None
            
            current_price = float(hist['Close'].iloc[-1])
            day_ago = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            week_ago = float(hist['Close'].iloc[-5]) if len(hist) >= 5 else current_price
            month_ago = float(hist['Close'].iloc[-20]) if len(hist) >= 20 else float(hist['Close'].iloc[0])
            
            # Calculate volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365) if len(returns) > 0 else 0
            
            return {
                'symbol': ticker.replace('-USD', ''),
                'name': info.get('shortName', ticker),
                'price': current_price,
                'market_cap': info.get('marketCap', 0),
                'volume_24h': info.get('volume24Hr', info.get('volume', 0)),
                'day_change': ((current_price - day_ago) / day_ago * 100) if day_ago != 0 else 0,
                'week_change': ((current_price - week_ago) / week_ago * 100) if week_ago != 0 else 0,
                'month_change': ((current_price - month_ago) / month_ago * 100) if month_ago != 0 else 0,
                'volatility': volatility,
                'ath': info.get('fiftyTwoWeekHigh', 0),
                'atl': info.get('fiftyTwoWeekLow', 0)
            }
            
        except Exception as e:
            logger.warning(f"Error fetching {ticker}: {e}")
            return None
    
    def _fetch_crypto_sentiment(self) -> Dict[str, Any]:
        """Fetch crypto market sentiment indicators"""
        sentiment = {}
        
        try:
            # Fetch Fear & Greed Index
            response = requests.get(self.crypto_sentiment_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    latest = data['data'][0]
                    sentiment['fear_greed'] = {
                        'value': int(latest['value']),
                        'classification': latest['value_classification'],
                        'timestamp': latest['timestamp']
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch crypto sentiment: {e}")
        
        # Calculate Bitcoin dominance if we have the data
        if hasattr(self, '_last_crypto_data'):
            btc_mcap = self._last_crypto_data.get('BTC-USD', {}).get('market_cap', 0)
            total_mcap = sum(c.get('market_cap', 0) for c in self._last_crypto_data.values())
            
            if total_mcap > 0:
                sentiment['btc_dominance'] = (btc_mcap / total_mcap) * 100
        
        return sentiment
    
    def _calculate_crypto_metrics(self, crypto_data: Dict[str, Any],
                                 portfolio_value: float) -> Dict[str, Any]:
        """Calculate crypto market metrics"""
        metrics = {}
        
        # Store for dominance calculation
        self._last_crypto_data = crypto_data
        
        # Market statistics
        if crypto_data:
            prices = [c['price'] for c in crypto_data.values()]
            changes = [c['day_change'] for c in crypto_data.values()]
            volatilities = [c['volatility'] for c in crypto_data.values()]
            
            metrics['average_change_24h'] = np.mean(changes)
            metrics['average_volatility'] = np.mean(volatilities)
            metrics['best_performer'] = max(crypto_data.items(), 
                                          key=lambda x: x[1]['day_change'])
            metrics['worst_performer'] = min(crypto_data.items(),
                                           key=lambda x: x[1]['day_change'])
            
            # Total market cap tracked
            metrics['total_market_cap'] = sum(c.get('market_cap', 0) 
                                             for c in crypto_data.values())
        
        # Portfolio allocation recommendation (% of portfolio)
        if portfolio_value > 0:
            # Conservative allocation based on risk tolerance
            risk_allocation = {
                'conservative': 0.05,  # 5%
                'moderate': 0.10,      # 10%
                'aggressive': 0.20,    # 20%
                'very_aggressive': 0.30 # 30%
            }
            
            # Default to moderate
            metrics['suggested_allocation'] = risk_allocation.get('moderate', 0.10)
            metrics['suggested_amount'] = portfolio_value * metrics['suggested_allocation']
        
        return metrics
    
    def _calculate_advanced_metrics(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate advanced economic metrics for growth/premium tiers"""
        advanced = {}
        
        # Yield curve (if we have the data)
        if 'treasury_10y' in indicators and 'treasury_2y' in indicators:
            if indicators['treasury_10y']['value'] and indicators['treasury_2y']['value']:
                spread = indicators['treasury_10y']['value'] - indicators['treasury_2y']['value']
                advanced['yield_curve_spread'] = spread
                advanced['yield_curve_inverted'] = spread < 0
                
                if spread < 0:
                    advanced['recession_signal'] = 'WARNING: Inverted yield curve'
                elif spread < 0.5:
                    advanced['recession_signal'] = 'CAUTION: Flattening yield curve'
                else:
                    advanced['recession_signal'] = 'Normal yield curve'
        
        # Misery Index (Inflation + Unemployment)
        if 'inflation' in indicators and 'unemployment' in indicators:
            if indicators['inflation']['value'] and indicators['unemployment']['value']:
                advanced['misery_index'] = (indicators['inflation']['value'] + 
                                           indicators['unemployment']['value'])
        
        # Real Interest Rate
        if 'fed_funds' in indicators and 'inflation' in indicators:
            if indicators['fed_funds']['value'] and indicators['inflation']['value']:
                # Annualize inflation if monthly
                annual_inflation = indicators['inflation']['value']
                if annual_inflation < 10:  # Likely already annual
                    advanced['real_interest_rate'] = (indicators['fed_funds']['value'] - 
                                                     annual_inflation)
                
        # Economic heat map
        heat_score = 0
        
        # GDP growth contribution
        if 'gdp_growth' in indicators and indicators['gdp_growth'].get('value'):
            gdp = indicators['gdp_growth']['value']
            if gdp > 3: heat_score += 2
            elif gdp > 2: heat_score += 1
            elif gdp < 0: heat_score -= 2
            elif gdp < 1: heat_score -= 1
        
        # Inflation contribution
        if 'inflation' in indicators and indicators['inflation'].get('value'):
            inflation = indicators['inflation']['value']
            if 2 <= inflation <= 3: heat_score += 1
            elif inflation > 5: heat_score -= 2
            elif inflation > 4: heat_score -= 1
            elif inflation < 1: heat_score -= 1
        
        # Unemployment contribution
        if 'unemployment' in indicators and indicators['unemployment'].get('value'):
            unemployment = indicators['unemployment']['value']
            if unemployment < 4: heat_score += 1
            elif unemployment > 7: heat_score -= 2
            elif unemployment > 6: heat_score -= 1
        
        # Classify economic heat
        if heat_score >= 3:
            advanced['economic_heat'] = 'HOT - Potential overheating'
        elif heat_score >= 1:
            advanced['economic_heat'] = 'WARM - Healthy expansion'
        elif heat_score >= -1:
            advanced['economic_heat'] = 'NEUTRAL - Mixed signals'
        elif heat_score >= -3:
            advanced['economic_heat'] = 'COOL - Slowing growth'
        else:
            advanced['economic_heat'] = 'COLD - Potential recession'
        
        advanced['heat_score'] = heat_score
        
        return advanced
    
    def _generate_crypto_recommendations(self, crypto_data: Dict[str, Any],
                                        tier: str) -> List[str]:
        """Generate tier-appropriate crypto recommendations"""
        recommendations = []
        
        metrics = crypto_data.get('metrics', {})
        sentiment = crypto_data.get('sentiment', {})
        
        # Fear & Greed based recommendations
        if 'fear_greed' in sentiment:
            fg_value = sentiment['fear_greed']['value']
            if fg_value < 25:
                recommendations.append("🟢 EXTREME FEAR: Potential buying opportunity for long-term investors")
            elif fg_value > 75:
                recommendations.append("🔴 EXTREME GREED: Consider taking profits or reducing exposure")
        
        # Volatility-based recommendations
        avg_volatility = metrics.get('average_volatility', 0)
        if avg_volatility > 1.0:  # 100% annual volatility
            recommendations.append("⚠️ HIGH VOLATILITY: Only invest what you can afford to lose")
        
        # Performance-based recommendations
        avg_change = metrics.get('average_change_24h', 0)
        if avg_change < -10:
            recommendations.append("📉 MARKET SELLOFF: Dollar-cost averaging opportunity")
        elif avg_change > 10:
            recommendations.append("📈 MARKET RALLY: Avoid FOMO, stick to your plan")
        
        # Tier-specific recommendations
        if tier == 'starter':
            recommendations.append("💡 Focus on BTC and ETH for crypto exposure")
            recommendations.append("📚 Limit crypto to 5-10% of total portfolio")
            
        elif tier == 'growth':
            recommendations.append("🎯 Consider top 10 cryptos for diversification")
            if 'btc_dominance' in sentiment:
                if sentiment['btc_dominance'] < 40:
                    recommendations.append("🔄 Alt season potential - research quality altcoins")
                elif sentiment['btc_dominance'] > 60:
                    recommendations.append("₿ Bitcoin dominant - focus on BTC allocation")
                    
        elif tier == 'premium':
            recommendations.append("🏆 Explore DeFi yield opportunities with caution")
            recommendations.append("🔍 Research emerging L1/L2 platforms")
            recommendations.append("💎 Consider small allocations to promising small caps")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def get_economic_summary(self, region: str = 'US',
                            tier: str = 'starter') -> str:
        """
        Generate a text summary of economic conditions
        
        Args:
            region: Country/region code
            tier: Customer tier
            
        Returns:
            Text summary for AI analysis
        """
        econ_data = self.get_economic_data(region, tier)
        crypto_data = self.get_crypto_data(tier)
        
        # Build summary
        summary_parts = []
        
        # Regional summary
        summary_parts.append(f"ECONOMIC CONDITIONS - {region}:")
        
        indicators = econ_data.get('indicators', {})
        if 'gdp_growth' in indicators and indicators['gdp_growth'].get('value'):
            summary_parts.append(f"• GDP Growth: {indicators['gdp_growth']['value']:.1f}%")
        
        if 'inflation' in indicators and indicators['inflation'].get('value'):
            summary_parts.append(f"• Inflation: {indicators['inflation']['value']:.1f}%")
        
        if 'unemployment' in indicators and indicators['unemployment'].get('value'):
            summary_parts.append(f"• Unemployment: {indicators['unemployment']['value']:.1f}%")
        
        # Global context
        summary_parts.append("\nGLOBAL CONTEXT:")
        global_data = econ_data.get('global_context', {})
        
        if 'vix' in global_data:
            vix_level = global_data['vix']['value']
            if vix_level < 15:
                summary_parts.append(f"• Market Volatility: Low (VIX: {vix_level:.1f})")
            elif vix_level < 25:
                summary_parts.append(f"• Market Volatility: Normal (VIX: {vix_level:.1f})")
            else:
                summary_parts.append(f"• Market Volatility: High (VIX: {vix_level:.1f})")
        
        if 'oil_wti' in global_data:
            summary_parts.append(f"• Oil Price (WTI): ${global_data['oil_wti']['value']:.2f}")
        
        # Crypto summary
        summary_parts.append("\nCRYPTO MARKET:")
        
        if crypto_data['market_data']:
            btc_data = crypto_data['market_data'].get('BTC-USD', {})
            if btc_data:
                summary_parts.append(f"• Bitcoin: ${btc_data['price']:,.0f} ({btc_data['day_change']:+.1f}%)")
            
            eth_data = crypto_data['market_data'].get('ETH-USD', {})
            if eth_data:
                summary_parts.append(f"• Ethereum: ${eth_data['price']:,.0f} ({eth_data['day_change']:+.1f}%)")
        
        if 'fear_greed' in crypto_data['sentiment']:
            fg = crypto_data['sentiment']['fear_greed']
            summary_parts.append(f"• Crypto Sentiment: {fg['classification']} ({fg['value']}/100)")
        
        # Advanced metrics for higher tiers
        if tier in ['growth', 'premium'] and 'advanced_metrics' in econ_data:
            advanced = econ_data['advanced_metrics']
            summary_parts.append("\nADVANCED INDICATORS:")
            
            if 'economic_heat' in advanced:
                summary_parts.append(f"• Economic Temperature: {advanced['economic_heat']}")
            
            if 'yield_curve_inverted' in advanced:
                if advanced['yield_curve_inverted']:
                    summary_parts.append("• ⚠️ Yield Curve: INVERTED (recession signal)")
                else:
                    summary_parts.append(f"• Yield Spread: {advanced['yield_curve_spread']:.2f}%")
        
        return "\n".join(summary_parts)


# Test function
def test_economic_provider():
    """Test the economic data provider"""
    print("🧪 Testing Economic Data Provider...")
    
    # Initialize provider
    provider = EconomicDataProvider()
    
    # Test economic data
    print("\n📊 Testing Economic Data (US):")
    us_data = provider.get_economic_data('US', 'growth')
    
    if us_data['indicators']:
        for key, value in list(us_data['indicators'].items())[:3]:
            if value.get('value'):
                print(f"  • {key}: {value['value']:.2f}")
    
    # Test crypto data
    print("\n₿ Testing Crypto Data:")
    crypto_data = provider.get_crypto_data('starter')
    
    if crypto_data['market_data']:
        for ticker, data in list(crypto_data['market_data'].items())[:2]:
            print(f"  • {data['symbol']}: ${data['price']:,.2f} ({data['day_change']:+.1f}%)")
    
    # Test summary generation
    print("\n📝 Economic Summary:")
    summary = provider.get_economic_summary('US', 'growth')
    print(summary)
    
    print("\n✅ Economic Data Provider ready!")


if __name__ == '__main__':
    test_economic_provider()
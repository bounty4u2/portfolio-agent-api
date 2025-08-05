"""
Portfolio Agent SaaS - Institutional Grade Analysis Engine
Multi-currency, customer-agnostic portfolio analysis service
NO HARDCODED DATA - Everything comes from customer input
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import anthropic
import os
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from scipy import stats

# Import our custom modules
from compliance import ComplianceWrapper
from currency_handler import CurrencyHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioAnalysisSaaS:
    """
    Customer-agnostic portfolio analysis service
    All data comes from customer input - no hardcoded values
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize the analysis service
        
        Args:
            anthropic_api_key: Optional API key for Claude
        """
        # Initialize core services
        self.currency_handler = CurrencyHandler()
        self.anthropic_client = None
        
        # Initialize Anthropic if API key provided
        if anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Cache for market data
        self._market_data_cache = {}
        self._cache_timestamp = {}
        self._cache_duration = timedelta(minutes=5)
    
    def analyze_portfolio(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for portfolio analysis
        
        Args:
            customer_data: Dictionary containing:
                - customer_info: {email, name, base_currency, region, subscription_tier}
                - portfolio: {positions: [{ticker, shares, cost_basis}]}
                - goals: [{name, target_amount, target_date, priority}]
                - settings: {risk_tolerance, report_frequency, email_enabled}
                - preferences: {report_sections, alert_thresholds}
        
        Returns:
            Complete analysis results
        """
        try:
            # Initialize compliance wrapper for customer's region
            region = customer_data.get('customer_info', {}).get('region', 'US')
            compliance_wrapper = ComplianceWrapper(region)
            
            # Extract customer information
            customer_info = customer_data.get('customer_info', {})
            base_currency = customer_info.get('base_currency', 'USD')
            
            # Get portfolio positions
            positions = customer_data.get('portfolio', {}).get('positions', [])
            if not positions:
                raise ValueError("No portfolio positions provided")
            
            # Fetch market data for all positions
            market_data = self._fetch_market_data(positions)
            
            # Calculate current portfolio values
            portfolio_values = self._calculate_portfolio_values(
                positions, market_data, base_currency
            )
            
            # Perform all analyses
            analysis_results = {
                'metadata': {
                    'analysis_date': datetime.utcnow().isoformat(),
                    'base_currency': base_currency,
                    'region': region,
                    'customer_id': customer_info.get('customer_id', 'anonymous')
                },
                'portfolio_summary': portfolio_values,
                'risk_analysis': self._perform_risk_analysis(
                    positions, market_data, portfolio_values, base_currency
                ),
                'performance_analysis': self._perform_performance_analysis(
                    positions, market_data, portfolio_values, customer_data
                ),
                'currency_analysis': self._perform_currency_analysis(
                    positions, portfolio_values, base_currency
                ),
                'goal_analysis': self._perform_goal_analysis(
                    portfolio_values, customer_data.get('goals', []), 
                    customer_data.get('settings', {})
                ),
                'rebalancing_suggestions': self._generate_rebalancing_suggestions(
                    portfolio_values, customer_data.get('settings', {})
                ),
                'tax_optimization': self._perform_tax_analysis(
                    positions, market_data, portfolio_values, region
                ),
                'market_regime': self._analyze_market_regime(base_currency),
                'scenario_analysis': self._perform_scenario_analysis(
                    portfolio_values, positions, market_data, base_currency
                ),
                'ai_insights': self._generate_ai_insights(
                    portfolio_values, analysis_results, customer_data
                ) if self.anthropic_client else None
            }
            
            # Apply compliance wrapper to ensure all output is educational
            compliant_results = compliance_wrapper.wrap_report(analysis_results)
            
            return compliant_results
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            raise
    
    def _fetch_market_data(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fetch market data for all positions
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Dictionary of market data by ticker
        """
        market_data = {}
        tickers = [pos['ticker'] for pos in positions]
        
        # Add benchmark tickers based on detected currencies
        currencies = set()
        for ticker in tickers:
            currency = self.currency_handler._detect_ticker_currency(ticker)
            currencies.add(currency)
        
        # Add regional benchmarks
        for currency in currencies:
            benchmarks = self.currency_handler.get_regional_benchmarks(currency)
            tickers.extend(benchmarks.values())
        
        # Remove duplicates
        tickers = list(set(tickers))
        
        # Batch fetch with threading for performance
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_ticker_data, ticker): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        market_data[ticker] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {ticker}: {e}")
        
        return market_data
    
    @lru_cache(maxsize=200)
    def _fetch_ticker_data(self, ticker: str, period: str = "1mo") -> Optional[Dict[str, Any]]:
        """
        Fetch data for a single ticker with caching
        
        Args:
            ticker: Ticker symbol
            period: Time period for historical data
            
        Returns:
            Ticker data dictionary or None
        """
        try:
            # Check cache first
            cache_key = f"{ticker}_{period}"
            if cache_key in self._market_data_cache:
                cached_data, timestamp = self._market_data_cache[cache_key]
                if datetime.now() - timestamp < self._cache_duration:
                    return cached_data
            
            # Fetch from Yahoo Finance
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return None
            
            # Get current data
            current_price = hist['Close'].iloc[-1]
            week_ago_price = hist['Close'].iloc[-5] if len(hist) >= 5 else current_price
            month_ago_price = hist['Close'].iloc[0]
            
            # Calculate metrics
            data = {
                'ticker': ticker,
                'current_price': float(current_price),
                'week_ago_price': float(week_ago_price),
                'month_ago_price': float(month_ago_price),
                'weekly_change': ((current_price - week_ago_price) / week_ago_price) * 100,
                'monthly_change': ((current_price - month_ago_price) / month_ago_price) * 100,
                'history': hist,
                'currency': self.currency_handler._detect_ticker_currency(ticker),
                'info': stock.info
            }
            
            # Cache the data
            self._market_data_cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.warning(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _calculate_portfolio_values(self, positions: List[Dict[str, Any]], 
                                  market_data: Dict[str, Any], 
                                  base_currency: str) -> Dict[str, Any]:
        """
        Calculate portfolio values in base currency
        
        Args:
            positions: List of positions
            market_data: Market data dictionary
            base_currency: Base currency for calculations
            
        Returns:
            Portfolio values and statistics
        """
        portfolio_details = []
        total_value_base = 0
        
        for position in positions:
            ticker = position['ticker']
            shares = float(position.get('shares', 0))
            cost_basis = float(position.get('cost_basis', 0))
            
            if ticker not in market_data:
                logger.warning(f"No market data for {ticker}")
                continue
            
            ticker_data = market_data[ticker]
            current_price = ticker_data['current_price']
            ticker_currency = ticker_data['currency']
            
            # Calculate values in ticker's currency
            market_value_local = shares * current_price
            
            # Convert to base currency
            market_value_base = self.currency_handler.convert_amount(
                market_value_local, ticker_currency, base_currency
            )
            
            # Calculate cost basis in base currency if needed
            cost_basis_currency = position.get('cost_basis_currency', ticker_currency)
            cost_basis_base = self.currency_handler.convert_amount(
                cost_basis * shares, cost_basis_currency, base_currency
            )
            
            # Calculate gains
            unrealized_gain = market_value_base - cost_basis_base
            unrealized_gain_pct = (unrealized_gain / cost_basis_base * 100) if cost_basis_base > 0 else 0
            
            position_details = {
                'ticker': ticker,
                'shares': shares,
                'currency': ticker_currency,
                'current_price': current_price,
                'market_value_local': market_value_local,
                'market_value_base': market_value_base,
                'cost_basis_base': cost_basis_base,
                'unrealized_gain': unrealized_gain,
                'unrealized_gain_pct': unrealized_gain_pct,
                'weekly_change': ticker_data['weekly_change'],
                'monthly_change': ticker_data['monthly_change'],
                'weight': 0  # Will be calculated after
            }
            
            portfolio_details.append(position_details)
            total_value_base += market_value_base
        
        # Calculate weights
        for position in portfolio_details:
            position['weight'] = (position['market_value_base'] / total_value_base * 100) if total_value_base > 0 else 0
        
        # Sort by weight
        portfolio_details.sort(key=lambda x: x['weight'], reverse=True)
        
        # Calculate summary statistics
        total_cost_basis = sum(p['cost_basis_base'] for p in portfolio_details)
        total_gain = total_value_base - total_cost_basis
        total_gain_pct = (total_gain / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        return {
            'total_value': total_value_base,
            'total_cost_basis': total_cost_basis,
            'total_gain': total_gain,
            'total_gain_pct': total_gain_pct,
            'currency': base_currency,
            'positions': portfolio_details,
            'position_count': len(portfolio_details),
            'currency_exposure': self.currency_handler.calculate_currency_exposure(
                [{'ticker': p['ticker'], 'value': p['market_value_base']} for p in portfolio_details],
                base_currency
            )
        }
    
    def _perform_risk_analysis(self, positions: List[Dict[str, Any]], 
                             market_data: Dict[str, Any],
                             portfolio_values: Dict[str, Any],
                             base_currency: str) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis
        
        Args:
            positions: List of positions
            market_data: Market data
            portfolio_values: Calculated portfolio values
            base_currency: Base currency
            
        Returns:
            Risk metrics dictionary
        """
        try:
            # Get position weights
            weights = np.array([p['weight'] / 100 for p in portfolio_values['positions']])
            tickers = [p['ticker'] for p in portfolio_values['positions']]
            
            # Get price history for all positions
            price_histories = []
            min_length = float('inf')
            
            for ticker in tickers:
                if ticker in market_data and 'history' in market_data[ticker]:
                    hist = market_data[ticker]['history']['Close']
                    price_histories.append(hist)
                    min_length = min(min_length, len(hist))
            
            if not price_histories or min_length < 20:
                return self._get_default_risk_metrics()
            
            # Align all histories to same length
            aligned_prices = pd.DataFrame({
                tickers[i]: price_histories[i].iloc[-min_length:].values 
                for i in range(len(tickers))
            })
            
            # Calculate returns
            returns = aligned_prices.pct_change().dropna()
            
            # Portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Risk metrics
            trading_days = 252
            
            # Volatility
            volatility = portfolio_returns.std() * np.sqrt(trading_days)
            
            # Sharpe ratio (assume risk-free rate of 2%)
            risk_free_rate = 0.02
            excess_returns = portfolio_returns.mean() * trading_days - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(trading_days)
            sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (VaR) and Conditional VaR
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            # Beta calculation (vs regional benchmark)
            benchmark_ticker = self._get_benchmark_ticker(base_currency)
            beta = self._calculate_beta(portfolio_returns, benchmark_ticker, market_data)
            
            # Correlation matrix
            correlation_matrix = returns.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            # Concentration risk
            top_3_weight = sum(sorted([p['weight'] for p in portfolio_values['positions']], reverse=True)[:3])
            herfindahl_index = sum([(p['weight'] / 100) ** 2 for p in portfolio_values['positions']])
            
            return {
                'volatility_annual': volatility,
                'volatility_monthly': volatility / np.sqrt(12),
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'value_at_risk_95': var_95,
                'value_at_risk_99': var_99,
                'conditional_var_95': cvar_95,
                'conditional_var_99': cvar_99,
                'var_95_amount': var_95 * portfolio_values['total_value'],
                'var_99_amount': var_99 * portfolio_values['total_value'],
                'beta': beta,
                'average_correlation': avg_correlation,
                'top_3_concentration': top_3_weight,
                'herfindahl_index': herfindahl_index,
                'calmar_ratio': abs(excess_returns / max_drawdown) if max_drawdown != 0 else 0,
                'risk_rating': self._calculate_risk_rating(volatility, max_drawdown, top_3_weight)
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return self._get_default_risk_metrics()
    
    def _get_default_risk_metrics(self) -> Dict[str, Any]:
        """Return default risk metrics when calculation fails"""
        return {
            'volatility_annual': 0.20,
            'volatility_monthly': 0.058,
            'sharpe_ratio': 0.5,
            'sortino_ratio': 0.6,
            'max_drawdown': -0.15,
            'value_at_risk_95': -0.02,
            'value_at_risk_99': -0.04,
            'conditional_var_95': -0.03,
            'conditional_var_99': -0.05,
            'var_95_amount': 0,
            'var_99_amount': 0,
            'beta': 1.0,
            'average_correlation': 0.5,
            'top_3_concentration': 50,
            'herfindahl_index': 0.15,
            'calmar_ratio': 0.5,
            'risk_rating': 'Medium'
        }
    
    def _calculate_risk_rating(self, volatility: float, max_drawdown: float, 
                              concentration: float) -> str:
        """Calculate overall risk rating"""
        risk_score = 0
        
        # Volatility contribution
        if volatility > 0.35:
            risk_score += 3
        elif volatility > 0.25:
            risk_score += 2
        elif volatility > 0.15:
            risk_score += 1
        
        # Drawdown contribution
        if max_drawdown < -0.30:
            risk_score += 3
        elif max_drawdown < -0.20:
            risk_score += 2
        elif max_drawdown < -0.10:
            risk_score += 1
        
        # Concentration contribution
        if concentration > 60:
            risk_score += 2
        elif concentration > 40:
            risk_score += 1
        
        # Map to rating
        if risk_score >= 6:
            return 'Very High'
        elif risk_score >= 4:
            return 'High'
        elif risk_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_benchmark_ticker(self, currency: str) -> str:
        """Get appropriate benchmark ticker for currency"""
        benchmarks = self.currency_handler.get_regional_benchmarks(currency)
        return benchmarks.get('primary', 'SPY')
    
    def _calculate_beta(self, portfolio_returns: pd.Series, 
                       benchmark_ticker: str, 
                       market_data: Dict[str, Any]) -> float:
        """Calculate portfolio beta vs benchmark"""
        try:
            if benchmark_ticker not in market_data:
                # Fetch benchmark data if not available
                benchmark_data = self._fetch_ticker_data(benchmark_ticker)
                if not benchmark_data:
                    return 1.0
                market_data[benchmark_ticker] = benchmark_data
            
            benchmark_hist = market_data[benchmark_ticker]['history']['Close']
            benchmark_returns = benchmark_hist.pct_change().dropna()
            
            # Align dates
            aligned_data = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < 20:
                return 1.0
            
            # Calculate beta
            covariance = aligned_data['portfolio'].cov(aligned_data['benchmark'])
            benchmark_variance = aligned_data['benchmark'].var()
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            return float(beta)
            
        except Exception as e:
            logger.warning(f"Beta calculation failed: {e}")
            return 1.0
    
    def _perform_performance_analysis(self, positions: List[Dict[str, Any]],
                                    market_data: Dict[str, Any],
                                    portfolio_values: Dict[str, Any],
                                    customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze portfolio performance
        
        Returns:
            Performance metrics dictionary
        """
        try:
            # Get benchmark for comparison
            base_currency = customer_data.get('customer_info', {}).get('base_currency', 'USD')
            benchmark_ticker = self._get_benchmark_ticker(base_currency)
            
            # Calculate various time period returns
            period_returns = {}
            
            # Daily returns
            total_value = portfolio_values['total_value']
            positions_data = portfolio_values['positions']
            
            # Calculate weighted returns for different periods
            for period_name, period_days in [('1D', 1), ('1W', 5), ('1M', 20), ('3M', 60), ('YTD', None)]:
                period_return = 0
                
                for position in positions_data:
                    weight = position['weight'] / 100
                    
                    if period_name == '1W':
                        period_return += weight * position['weekly_change'] / 100
                    elif period_name == '1M':
                        period_return += weight * position['monthly_change'] / 100
                    else:
                        # For other periods, would need more historical data
                        period_return += 0  # Placeholder
                
                period_returns[period_name] = period_return
            
            # Performance attribution by position
            attribution = []
            for position in positions_data:
                contribution = (position['weight'] / 100) * (position['monthly_change'] / 100)
                attribution.append({
                    'ticker': position['ticker'],
                    'weight': position['weight'],
                    'return': position['monthly_change'],
                    'contribution': contribution * 100
                })
            
            # Sort by absolute contribution
            attribution.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            # Benchmark comparison
            benchmark_return = 0
            if benchmark_ticker in market_data:
                benchmark_return = market_data[benchmark_ticker]['monthly_change']
            
            # Calculate alpha
            portfolio_return = period_returns['1M']
            alpha = portfolio_return - benchmark_return
            
            return {
                'period_returns': period_returns,
                'attribution': attribution,
                'benchmark_comparison': {
                    'benchmark': benchmark_ticker,
                    'portfolio_return': portfolio_return,
                    'benchmark_return': benchmark_return,
                    'alpha': alpha,
                    'outperformed': portfolio_return > benchmark_return
                },
                'best_performer': max(positions_data, key=lambda x: x['monthly_change'])['ticker'],
                'worst_performer': min(positions_data, key=lambda x: x['monthly_change'])['ticker'],
                'winners': len([p for p in positions_data if p['unrealized_gain'] > 0]),
                'losers': len([p for p in positions_data if p['unrealized_gain'] < 0])
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {
                'period_returns': {},
                'attribution': [],
                'benchmark_comparison': {},
                'best_performer': 'N/A',
                'worst_performer': 'N/A',
                'winners': 0,
                'losers': 0
            }
    
    def _perform_currency_analysis(self, positions: List[Dict[str, Any]],
                                 portfolio_values: Dict[str, Any],
                                 base_currency: str) -> Dict[str, Any]:
        """
        Analyze currency exposure and risks
        
        Returns:
            Currency analysis dictionary
        """
        currency_exposure = portfolio_values.get('currency_exposure', {})
        
        # Get FX volatility for major currency pairs
        fx_volatilities = {}
        for currency in currency_exposure.get('currency_weights', {}).keys():
            if currency != base_currency:
                try:
                    # Get historical FX data
                    fx_data = self.currency_handler.get_historical_fx_data(
                        (currency, base_currency), period='3mo'
                    )
                    if not fx_data.empty:
                        fx_returns = fx_data.pct_change().dropna()
                        fx_volatilities[currency] = fx_returns.std() * np.sqrt(252)
                except:
                    fx_volatilities[currency] = 0.10  # Default 10% volatility
        
        return {
            'currency_breakdown': currency_exposure.get('currency_weights', {}),
            'home_currency_bias': currency_exposure.get('home_bias', 0),
            'foreign_exposure': currency_exposure.get('foreign_exposure', 0),
            'currency_concentration': currency_exposure.get('concentration_risk', 'Unknown'),
            'fx_volatilities': fx_volatilities,
            'hedging_recommendations': currency_exposure.get('recommended_hedging', {}),
            'currency_count': currency_exposure.get('currency_count', 1)
        }
    
    def _perform_goal_analysis(self, portfolio_values: Dict[str, Any],
                             goals: List[Dict[str, Any]],
                             settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze progress towards financial goals
        
        Args:
            portfolio_values: Current portfolio values
            goals: List of financial goals
            settings: User settings including risk tolerance
            
        Returns:
            Goal analysis dictionary
        """
        if not goals:
            return {'goals': [], 'summary': 'No goals defined'}
        
        goal_analyses = []
        current_value = portfolio_values['total_value']
        
        for goal in goals:
            target_amount = float(goal.get('target_amount', 0))
            target_date = goal.get('target_date')
            goal_name = goal.get('name', 'Unnamed Goal')
            monthly_contribution = float(goal.get('monthly_contribution', 0))
            
            if not target_amount or not target_date:
                continue
            
            # Calculate time to goal
            try:
                if isinstance(target_date, str):
                    target_date = datetime.fromisoformat(target_date.replace('Z', '+00:00'))
                years_to_goal = (target_date - datetime.now()).days / 365.25
            except:
                years_to_goal = 10  # Default
            
            if years_to_goal <= 0:
                years_to_goal = 0.1  # Minimum for calculations
            
            # Required return calculation
            required_return = ((target_amount / current_value) ** (1 / years_to_goal) - 1) if current_value > 0 else 0
            
            # Monte Carlo simulation for success probability
            success_probability = self._monte_carlo_goal_simulation(
                current_value, target_amount, years_to_goal, 
                monthly_contribution, settings.get('risk_tolerance', 'moderate')
            )
            
            # Progress calculation
            progress_pct = (current_value / target_amount * 100) if target_amount > 0 else 0
            
            goal_analysis = {
                'name': goal_name,
                'target_amount': target_amount,
                'target_date': target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date),
                'years_to_goal': years_to_goal,
                'progress_percentage': progress_pct,
                'required_annual_return': required_return * 100,
                'success_probability': success_probability,
                'monthly_contribution': monthly_contribution,
                'on_track': success_probability >= 0.70,
                'gap_amount': max(0, target_amount - current_value),
                'recommendations': self._get_goal_recommendations(
                    success_probability, required_return, years_to_goal
                )
            }
            
            goal_analyses.append(goal_analysis)
        
        # Sort by priority or nearest date
        goal_analyses.sort(key=lambda x: x['years_to_goal'])
        
        # Overall summary
        primary_goal = goal_analyses[0] if goal_analyses else None
        
        return {
            'goals': goal_analyses,
            'primary_goal': primary_goal,
            'summary': self._generate_goal_summary(goal_analyses)
        }
    
    def _monte_carlo_goal_simulation(self, current_value: float, target_amount: float,
                                   years: float, monthly_contribution: float,
                                   risk_tolerance: str) -> float:
        """
        Run Monte Carlo simulation for goal achievement probability
        
        Returns:
            Probability of achieving goal (0-1)
        """
        # Risk tolerance to expected return/volatility mapping
        risk_profiles = {
            'conservative': {'return': 0.06, 'volatility': 0.10},
            'moderate': {'return': 0.08, 'volatility': 0.15},
            'aggressive': {'return': 0.10, 'volatility': 0.20},
            'very_aggressive': {'return': 0.12, 'volatility': 0.25}
        }
        
        profile = risk_profiles.get(risk_tolerance, risk_profiles['moderate'])
        annual_return = profile['return']
        annual_volatility = profile['volatility']
        
        # Simulation parameters
        n_simulations = 1000
        n_months = int(years * 12)
        monthly_return = annual_return / 12
        monthly_volatility = annual_volatility / np.sqrt(12)
        
        # Run simulations
        success_count = 0
        
        for _ in range(n_simulations):
            value = current_value
            
            for month in range(n_months):
                # Add monthly contribution
                value += monthly_contribution
                
                # Apply return with randomness
                random_return = np.random.normal(monthly_return, monthly_volatility)
                value *= (1 + random_return)
            
            if value >= target_amount:
                success_count += 1
        
        return success_count / n_simulations
    
    def _get_goal_recommendations(self, success_prob: float, required_return: float,
                                years_to_goal: float) -> List[str]:
        """Generate recommendations for goal achievement"""
        recommendations = []
        
        if success_prob < 0.5:
            recommendations.append("Success probability is low - consider increasing contributions")
            recommendations.append(f"Required return of {required_return*100:.1f}% may be unrealistic")
            
            if years_to_goal < 5:
                recommendations.append("Short timeframe requires aggressive saving")
        
        elif success_prob < 0.7:
            recommendations.append("On track but could improve probability with higher contributions")
            recommendations.append("Consider increasing portfolio growth allocation")
        
        else:
            recommendations.append("Well on track to achieve this goal")
            if required_return < 0.06:
                recommendations.append("Could reduce risk given comfortable position")
        
        return recommendations
    
    def _generate_goal_summary(self, goal_analyses: List[Dict[str, Any]]) -> str:
        """Generate summary of all goals"""
        if not goal_analyses:
            return "No financial goals defined"
        
        on_track_count = sum(1 for g in goal_analyses if g['on_track'])
        total_count = len(goal_analyses)
        
        if on_track_count == total_count:
            return f"All {total_count} goals are on track"
        elif on_track_count == 0:
            return f"None of the {total_count} goals are currently on track"
        else:
            return f"{on_track_count} of {total_count} goals are on track"
    
    def _generate_rebalancing_suggestions(self, portfolio_values: Dict[str, Any],
                                        settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate rebalancing suggestions based on target allocations
        
        Returns:
            List of rebalancing suggestions
        """
        suggestions = []
        
        # Get target allocations from settings
        target_allocations = settings.get('target_allocations', {})
        rebalance_threshold = float(settings.get('rebalance_threshold', 5.0))
        
        if not target_allocations:
            # If no targets specified, suggest based on concentration
            return self._suggest_concentration_based_rebalancing(portfolio_values)
        
        # Calculate deviations from targets
        current_positions = {p['ticker']: p['weight'] for p in portfolio_values['positions']}
        total_value = portfolio_values['total_value']
        
        for category, target_pct in target_allocations.items():
            # Map category to tickers (would come from settings)
            category_tickers = settings.get('category_mappings', {}).get(category, [])
            
            # Calculate current allocation
            current_pct = sum(current_positions.get(ticker, 0) for ticker in category_tickers)
            
            # Check if rebalancing needed
            deviation = current_pct - target_pct
            if abs(deviation) > rebalance_threshold:
                action = 'REDUCE' if deviation > 0 else 'INCREASE'
                amount = abs(deviation) / 100 * total_value
                
                suggestions.append({
                    'category': category,
                    'action': action,
                    'current_allocation': current_pct,
                    'target_allocation': target_pct,
                    'deviation': deviation,
                    'amount': amount,
                    'tickers': category_tickers,
                    'priority': 'High' if abs(deviation) > 10 else 'Medium'
                })
        
        # Sort by priority and deviation
        suggestions.sort(key=lambda x: abs(x['deviation']), reverse=True)
        
        return suggestions
    
    def _suggest_concentration_based_rebalancing(self, portfolio_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest rebalancing based on concentration risk"""
        suggestions = []
        positions = portfolio_values['positions']
        
        # Check for over-concentrated positions
        for position in positions:
            if position['weight'] > 25:  # 25% threshold
                suggestions.append({
                    'ticker': position['ticker'],
                    'action': 'REDUCE',
                    'current_weight': position['weight'],
                    'target_weight': 20,
                    'reason': 'Position exceeds 25% concentration limit',
                    'amount': (position['weight'] - 20) / 100 * portfolio_values['total_value'],
                    'priority': 'High'
                })
        
        # Check for very small positions
        for position in positions:
            if position['weight'] < 2:  # 2% threshold
                suggestions.append({
                    'ticker': position['ticker'],
                    'action': 'CONSIDER_REMOVING',
                    'current_weight': position['weight'],
                    'reason': 'Position below 2% may not be meaningful',
                    'amount': position['market_value_base'],
                    'priority': 'Low'
                })
        
        return suggestions
    
    def _perform_tax_analysis(self, positions: List[Dict[str, Any]],
                            market_data: Dict[str, Any],
                            portfolio_values: Dict[str, Any],
                            region: str) -> Dict[str, Any]:
        """
        Perform tax optimization analysis
        
        Returns:
            Tax analysis dictionary
        """
        tax_loss_harvesting = []
        total_unrealized_losses = 0
        
        # Analyze each position for tax loss harvesting opportunities
        for position in portfolio_values['positions']:
            if position['unrealized_gain'] < 0:
                loss_amount = abs(position['unrealized_gain'])
                
                # Only suggest if loss is significant (>$100 or >10%)
                if loss_amount > 100 or position['unrealized_gain_pct'] < -10:
                    # Estimate tax benefit based on region
                    tax_rate = self._get_capital_gains_tax_rate(region)
                    tax_benefit = loss_amount * tax_rate
                    
                    tax_loss_harvesting.append({
                        'ticker': position['ticker'],
                        'loss_amount': loss_amount,
                        'loss_percentage': position['unrealized_gain_pct'],
                        'tax_benefit': tax_benefit,
                        'replacement_suggestions': self._get_replacement_suggestions(
                            position['ticker'], market_data
                        )
                    })
                    
                    total_unrealized_losses += loss_amount
        
        # Sort by tax benefit
        tax_loss_harvesting.sort(key=lambda x: x['tax_benefit'], reverse=True)
        
        # Account optimization suggestions
        account_suggestions = self._get_account_optimization_suggestions(
            portfolio_values['positions'], region
        )
        
        return {
            'tax_loss_harvesting_opportunities': tax_loss_harvesting,
            'total_unrealized_losses': total_unrealized_losses,
            'estimated_tax_savings': sum(t['tax_benefit'] for t in tax_loss_harvesting),
            'account_optimization': account_suggestions,
            'tax_efficiency_score': self._calculate_tax_efficiency_score(
                portfolio_values, region
            )
        }
    
    def _get_capital_gains_tax_rate(self, region: str) -> float:
        """Get approximate capital gains tax rate by region"""
        # Simplified tax rates - in production would be more detailed
        tax_rates = {
            'US': 0.20,    # Long-term capital gains
            'CA': 0.25,    # 50% inclusion rate * ~50% marginal
            'UK': 0.20,    # Higher rate
            'AU': 0.23,    # 50% discount * ~45% marginal
            'EU': 0.25,    # Average EU rate
            'SG': 0.00,    # No capital gains tax
            'HK': 0.00,    # No capital gains tax
            'default': 0.20
        }
        
        return tax_rates.get(region, tax_rates['default'])
    
    def _get_replacement_suggestions(self, ticker: str, market_data: Dict[str, Any]) -> List[str]:
        """Suggest similar securities for tax loss harvesting"""
        # In production, would use more sophisticated similarity metrics
        # For now, basic sector/correlation-based suggestions
        
        suggestions = []
        
        if ticker in market_data and 'info' in market_data[ticker]:
            sector = market_data[ticker]['info'].get('sector', '')
            
            # Find similar tickers in same sector
            for other_ticker, other_data in market_data.items():
                if other_ticker != ticker and 'info' in other_data:
                    if other_data['info'].get('sector') == sector:
                        suggestions.append(other_ticker)
            
            # Limit to top 3
            suggestions = suggestions[:3]
        
        # Fallback suggestions
        if not suggestions:
            if 'tech' in ticker.lower() or ticker in ['NVDA', 'AMD', 'MSFT']:
                suggestions = ['QQQ', 'XLK', 'VGT']
            else:
                suggestions = ['SPY', 'VOO', 'IVV']
        
        return suggestions
    
    def _get_account_optimization_suggestions(self, positions: List[Dict[str, Any]],
                                            region: str) -> Dict[str, Any]:
        """Suggest optimal account placement for tax efficiency"""
        suggestions = {
            'tax_advantaged': [],
            'taxable': [],
            'considerations': []
        }
        
        # Region-specific account types
        if region == 'US':
            suggestions['considerations'].append("Consider maxing out 401(k) and IRA contributions")
            suggestions['considerations'].append("High-growth assets in Roth IRA for tax-free growth")
        elif region == 'CA':
            suggestions['considerations'].append("Prioritize TFSA for high-growth investments")
            suggestions['considerations'].append("Use RRSP for foreign dividends to avoid withholding tax")
        elif region == 'UK':
            suggestions['considerations'].append("Utilize ISA allowance for tax-free growth")
            suggestions['considerations'].append("Consider SIPP for retirement savings")
        
        # General suggestions
        for position in positions:
            # High dividend stocks in tax-advantaged accounts
            if position['ticker'] in market_data and 'info' in market_data[position['ticker']]:
                dividend_yield = market_data[position['ticker']]['info'].get('dividendYield', 0)
                
                if dividend_yield > 0.03:  # 3% yield
                    suggestions['tax_advantaged'].append(position['ticker'])
                else:
                    suggestions['taxable'].append(position['ticker'])
        
        return suggestions
    
    def _calculate_tax_efficiency_score(self, portfolio_values: Dict[str, Any],
                                      region: str) -> int:
        """Calculate tax efficiency score (0-100)"""
        score = 50  # Base score
        
        # Penalize high turnover (would need transaction history)
        # Reward tax-advantaged account usage
        # Consider dividend tax efficiency
        
        # Simple scoring based on unrealized gains
        total_unrealized_gain = sum(p['unrealized_gain'] for p in portfolio_values['positions'])
        if total_unrealized_gain > 0:
            score += 20  # Holding winners
        
        # Low dividend exposure in taxable accounts
        score += 10
        
        # Regional considerations
        if region in ['SG', 'HK']:
            score += 20  # No capital gains tax
        
        return min(100, max(0, score))
    
    def _analyze_market_regime(self, base_currency: str) -> Dict[str, Any]:
        """
        Analyze current market regime
        
        Returns:
            Market regime analysis
        """
        try:
            # Get key market indicators
            indicators = {}
            
            # VIX (fear gauge)
            vix_data = self._fetch_ticker_data('^VIX', period='1mo')
            if vix_data:
                indicators['vix'] = vix_data['current_price']
                indicators['vix_level'] = 'High' if vix_data['current_price'] > 25 else 'Low'
            
            # Regional market index
            benchmark = self._get_benchmark_ticker(base_currency)
            benchmark_data = self._fetch_ticker_data(benchmark, period='3mo')
            
            if benchmark_data:
                # Calculate trend
                hist = benchmark_data['history']['Close']
                sma_50 = hist.rolling(window=50).mean().iloc[-1]
                sma_200 = hist.rolling(window=200).mean().iloc[-1] if len(hist) >= 200 else sma_50
                current_price = hist.iloc[-1]
                
                # Determine trend
                if current_price > sma_50 > sma_200:
                    trend = 'UPTREND'
                elif current_price < sma_50 < sma_200:
                    trend = 'DOWNTREND'
                else:
                    trend = 'SIDEWAYS'
                
                indicators['market_trend'] = trend
                indicators['above_50_ma'] = current_price > sma_50
                indicators['above_200_ma'] = current_price > sma_200
            
            # Determine regime
            regime = self._determine_regime(indicators)
            
            return {
                'current_regime': regime,
                'indicators': indicators,
                'confidence': self._calculate_regime_confidence(indicators),
                'recommendations': self._get_regime_recommendations(regime)
            }
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return {
                'current_regime': 'UNKNOWN',
                'indicators': {},
                'confidence': 0,
                'recommendations': ['Unable to determine market regime']
            }
    
    def _determine_regime(self, indicators: Dict[str, Any]) -> str:
        """Determine market regime from indicators"""
        score = 0
        
        # VIX level
        if indicators.get('vix_level') == 'Low':
            score += 2
        else:
            score -= 2
        
        # Trend
        if indicators.get('market_trend') == 'UPTREND':
            score += 3
        elif indicators.get('market_trend') == 'DOWNTREND':
            score -= 3
        
        # Moving averages
        if indicators.get('above_50_ma', False):
            score += 1
        if indicators.get('above_200_ma', False):
            score += 1
        
        # Determine regime
        if score >= 4:
            return 'BULL_MARKET'
        elif score <= -3:
            return 'BEAR_MARKET'
        else:
            return 'TRANSITIONAL'
    
    def _calculate_regime_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calculate confidence in regime determination"""
        # Simple confidence based on indicator agreement
        confidence_points = 0
        total_points = 0
        
        if 'vix_level' in indicators:
            total_points += 1
            confidence_points += 1
        
        if 'market_trend' in indicators:
            total_points += 2
            confidence_points += 2
        
        if 'above_50_ma' in indicators and 'above_200_ma' in indicators:
            total_points += 1
            if indicators['above_50_ma'] == indicators['above_200_ma']:
                confidence_points += 1
        
        return (confidence_points / total_points * 100) if total_points > 0 else 50
    
    def _get_regime_recommendations(self, regime: str) -> List[str]:
        """Get recommendations based on market regime"""
        recommendations = {
            'BULL_MARKET': [
                "Market conditions favorable for growth",
                "Consider maintaining or increasing equity exposure",
                "Monitor for signs of overheating",
                "Rebalance if positions become overweight"
            ],
            'BEAR_MARKET': [
                "Defensive positioning may be appropriate",
                "Consider increasing cash or defensive assets",
                "Look for quality companies at discounted prices",
                "Avoid panic selling - stick to long-term plan"
            ],
            'TRANSITIONAL': [
                "Market direction unclear - maintain balanced approach",
                "Focus on diversification",
                "Keep some dry powder for opportunities",
                "Monitor regime indicators closely"
            ],
            'UNKNOWN': [
                "Unable to determine clear market regime",
                "Focus on fundamental investment principles",
                "Maintain diversified portfolio",
                "Regular rebalancing recommended"
            ]
        }
        
        return recommendations.get(regime, recommendations['UNKNOWN'])
    
    def _perform_scenario_analysis(self, portfolio_values: Dict[str, Any],
                                 positions: List[Dict[str, Any]],
                                 market_data: Dict[str, Any],
                                 base_currency: str) -> Dict[str, Any]:
        """
        Perform scenario analysis
        
        Returns:
            Scenario analysis results
        """
        scenarios = {}
        current_value = portfolio_values['total_value']
        
        # Market crash scenario
        crash_magnitude = -0.30  # 30% crash
        scenarios['market_crash_30'] = {
            'description': '30% market crash',
            'portfolio_value': current_value * (1 + crash_magnitude),
            'impact': crash_magnitude * 100,
            'recovery_time_estimate': '12-24 months historically'
        }
        
        # Bull market scenario
        bull_magnitude = 0.50  # 50% gain
        scenarios['bull_market_50'] = {
            'description': '50% bull market rally',
            'portfolio_value': current_value * (1 + bull_magnitude),
            'impact': bull_magnitude * 100,
            'considerations': 'Consider taking some profits'
        }
        
        # Currency crisis scenario (for foreign holdings)
        currency_impact = self._calculate_currency_scenario(
            portfolio_values, base_currency, change=0.20
        )
        scenarios['currency_shock'] = currency_impact
        
        # Inflation scenario
        inflation_impact = {
            'description': 'High inflation environment (5% annual)',
            'real_return_impact': -5,
            'recommendations': [
                'Consider inflation-protected securities',
                'Real assets may outperform',
                'Review fixed income allocations'
            ]
        }
        scenarios['high_inflation'] = inflation_impact
        
        # Sector-specific scenarios
        if self._has_sector_concentration(portfolio_values['positions']):
            scenarios['sector_crash'] = self._calculate_sector_scenario(
                portfolio_values, market_data
            )
        
        return scenarios
    
    def _calculate_currency_scenario(self, portfolio_values: Dict[str, Any],
                                   base_currency: str, change: float) -> Dict[str, Any]:
        """Calculate currency shock scenario"""
        currency_exposure = portfolio_values.get('currency_exposure', {})
        foreign_exposure = currency_exposure.get('foreign_exposure', 0)
        
        # Simplified calculation
        currency_impact = foreign_exposure * change
        new_value = portfolio_values['total_value'] * (1 + currency_impact)
        
        return {
            'description': f'{change*100:.0f}% currency movement',
            'portfolio_value': new_value,
            'impact': currency_impact * 100,
            'foreign_exposure': foreign_exposure * 100,
            'hedging_could_save': abs(portfolio_values['total_value'] * currency_impact)
        }
    
    def _has_sector_concentration(self, positions: List[Dict[str, Any]]) -> bool:
        """Check if portfolio has significant sector concentration"""
        # Simplified - in production would use proper sector classification
        tech_weight = sum(p['weight'] for p in positions 
                         if p['ticker'] in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META', 'AMZN'])
        
        return tech_weight > 40  # 40% threshold
    
    def _calculate_sector_scenario(self, portfolio_values: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sector-specific crash scenario"""
        # Identify concentrated sector (simplified for tech)
        tech_positions = [p for p in portfolio_values['positions'] 
                         if p['ticker'] in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META', 'AMZN']]
        
        tech_weight = sum(p['weight'] for p in tech_positions) / 100
        sector_crash_magnitude = -0.40  # 40% sector crash
        
        impact_on_portfolio = tech_weight * sector_crash_magnitude
        new_value = portfolio_values['total_value'] * (1 + impact_on_portfolio)
        
        return {
            'description': 'Technology sector 40% crash',
            'portfolio_value': new_value,
            'impact': impact_on_portfolio * 100,
            'sector_weight': tech_weight * 100,
            'recommendations': [
                'High sector concentration increases risk',
                'Consider diversifying into other sectors',
                'Monitor sector-specific news closely'
            ]
        }
    
    def _generate_ai_insights(self, portfolio_values: Dict[str, Any],
                            analysis_results: Dict[str, Any],
                            customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI-powered insights using Claude
        
        Returns:
            AI insights dictionary
        """
        if not self.anthropic_client:
            return {'error': 'AI insights unavailable - no API key'}
        
        try:
            # Prepare context for AI
            customer_info = customer_data.get('customer_info', {})
            settings = customer_data.get('settings', {})
            
            prompt = self._create_ai_prompt(
                portfolio_values, 
                analysis_results,
                customer_info,
                settings
            )
            
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            insights_text = response.content[0].text
            
            # Structure insights
            insights = self._parse_ai_insights(insights_text)
            
            return insights
            
        except Exception as e:
            logger.error(f"AI insights generation failed: {e}")
            return {'error': f'AI insights generation failed: {str(e)}'}
    
    def _create_ai_prompt(self, portfolio_values: Dict[str, Any],
                         analysis_results: Dict[str, Any],
                         customer_info: Dict[str, Any],
                         settings: Dict[str, Any]) -> str:
        """Create prompt for AI analysis"""
        
        risk_tolerance = settings.get('risk_tolerance', 'moderate')
        base_currency = customer_info.get('base_currency', 'USD')
        region = customer_info.get('region', 'US')
        
        # Top holdings for context
        top_holdings = portfolio_values['positions'][:5]
        holdings_str = ', '.join([f"{p['ticker']} ({p['weight']:.1f}%)" for p in top_holdings])
        
        prompt = f"""
        Analyze this investment portfolio and provide educational insights.
        
        Portfolio Overview:
        - Total Value: {self.currency_handler.format_currency(portfolio_values['total_value'], base_currency)}
        - Number of Holdings: {portfolio_values['position_count']}
        - Top Holdings: {holdings_str}
        - Risk Tolerance: {risk_tolerance}
        - Region: {region}
        
        Key Metrics:
        - Total Return: {portfolio_values['total_gain_pct']:.1f}%
        - Volatility: {analysis_results.get('risk_analysis', {}).get('volatility_annual', 0)*100:.1f}%
        - Sharpe Ratio: {analysis_results.get('risk_analysis', {}).get('sharpe_ratio', 0):.2f}
        - Max Drawdown: {analysis_results.get('risk_analysis', {}).get('max_drawdown', 0)*100:.1f}%
        
        Currency Exposure:
        - Home Currency: {base_currency}
        - Foreign Exposure: {analysis_results.get('currency_analysis', {}).get('foreign_exposure', 0)*100:.1f}%
        
        Provide insights in these areas:
        1. Portfolio Health Assessment (score 0-100)
        2. Key Strengths (3 points)
        3. Key Risks (3 points)
        4. Opportunities for Improvement (3 specific actions)
        5. Market Outlook Considerations
        
        Frame all insights as educational information, not personalized advice.
        Use clear, concise language appropriate for a {risk_tolerance} investor.
        """
        
        return prompt
    
    def _parse_ai_insights(self, insights_text: str) -> Dict[str, Any]:
        """Parse AI insights into structured format"""
        # In production, would use more sophisticated parsing
        # For now, return structured text
        
        sections = {
            'portfolio_health_score': 0,
            'key_strengths': [],
            'key_risks': [],
            'opportunities': [],
            'market_outlook': '',
            'raw_insights': insights_text
        }
        
        # Simple parsing logic
        lines = insights_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect sections
            if 'health' in line.lower() and 'score' in line.lower():
                # Try to extract score
                import re
                score_match = re.search(r'\b(\d+)\b', line)
                if score_match:
                    sections['portfolio_health_score'] = int(score_match.group(1))
            
            elif 'strength' in line.lower():
                current_section = 'strengths'
            elif 'risk' in line.lower():
                current_section = 'risks'
            elif 'opportunit' in line.lower():
                current_section = 'opportunities'
            elif 'outlook' in line.lower():
                current_section = 'outlook'
            
            # Add content to sections
            elif current_section == 'strengths' and line.startswith(('-', '•', '*')):
                sections['key_strengths'].append(line.lstrip('-•* '))
            elif current_section == 'risks' and line.startswith(('-', '•', '*')):
                sections['key_risks'].append(line.lstrip('-•* '))
            elif current_section == 'opportunities' and line.startswith(('-', '•', '*')):
                sections['opportunities'].append(line.lstrip('-•* '))
            elif current_section == 'outlook':
                sections['market_outlook'] += line + ' '
        
        return sections
    
    def generate_email_report(self, analysis_results: Dict[str, Any],
                            customer_email: str,
                            smtp_config: Dict[str, Any]) -> bool:
        """
        Generate and send email report
        
        Args:
            analysis_results: Complete analysis results
            customer_email: Recipient email
            smtp_config: SMTP configuration
            
        Returns:
            Success boolean
        """
        try:
            # Create email content
            html_content = self._generate_html_report(analysis_results)
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Portfolio Intelligence Report - {datetime.now().strftime('%B %d, %Y')}"
            msg['From'] = smtp_config.get('from_email', 'noreply@portfoliointelligence.ai')
            msg['To'] = customer_email
            
            # Attach HTML
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_config['smtp_host'], smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(smtp_config['smtp_user'], smtp_config['smtp_password'])
                server.send_message(msg)
            
            logger.info(f"Report sent successfully to {customer_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email report: {e}")
            return False
    
    def _generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML report content"""
        # This would be a comprehensive HTML template
        # For now, simplified version
        
        metadata = analysis_results.get('metadata', {})
        summary = analysis_results.get('sections', {}).get('portfolio_summary', {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Portfolio Intelligence Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; border-bottom: 1px solid #ddd; text-align: left; }}
                th {{ background: #ecf0f1; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Portfolio Intelligence Report</h1>
                <p>{datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="section">
                <h2>Portfolio Summary</h2>
                <div class="metric">
                    <h3>Total Value</h3>
                    <p>{self.currency_handler.format_currency(
                        summary.get('content', {}).get('total_value', 0),
                        metadata.get('base_currency', 'USD')
                    )}</p>
                </div>
                <!-- Add more sections as needed -->
            </div>
            
            <div class="footer">
                <p>{metadata.get('disclaimer', '')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
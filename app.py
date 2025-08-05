"""
Portfolio Intelligence API - SaaS Version
Multi-currency, multi-region portfolio analysis service
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from datetime import datetime, timedelta
import logging
from functools import wraps
import json
import hashlib
from typing import Dict, Any, Optional

# Import our modules
from portfolio_agent_saas import PortfolioAnalysisSaaS
from compliance import ComplianceWrapper
from currency_handler import CurrencyHandler

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['*'])  # Configure appropriately for production

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
portfolio_service = PortfolioAnalysisSaaS(anthropic_api_key)
currency_handler = CurrencyHandler()

# In-memory storage (replace with database in production)
CUSTOMERS = {}
ANALYSIS_JOBS = {}
API_KEYS = {
    'test-key-123': {
        'customer_id': 'test-customer',
        'email': 'test@example.com',
        'subscription_tier': 'free',
        'credits_remaining': 3,
        'created_at': datetime.utcnow().isoformat()
    }
}

# Subscription tiers
SUBSCRIPTION_TIERS = {
    'free': {
        'reports_per_month': 3,
        'features': ['basic_analysis', 'email_report'],
        'priority': False,
        'ai_model': 'claude-3-sonnet-20240229'
    },
    'starter': {
        'reports_per_month': 30,
        'features': ['full_analysis', 'email_report', 'basic_alerts'],
        'priority': False,
        'ai_model': 'claude-3-opus-20240229',
        'price': 39
    },
    'professional': {
        'reports_per_month': 'unlimited',
        'features': ['full_analysis', 'email_report', 'advanced_alerts', 'api_access'],
        'priority': True,
        'ai_model': 'claude-3-opus-20240229',
        'price': 99
    },
    'enterprise': {
        'reports_per_month': 'unlimited',
        'features': ['full_analysis', 'email_report', 'real_time_alerts', 'api_access', 'white_label'],
        'priority': True,
        'ai_model': 'claude-3-opus-20240229',
        'price': 299
    }
}


def validate_api_key(f):
    """Decorator to validate API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'No API key provided'}), 401
        
        if api_key not in API_KEYS:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Check credits for free tier
        key_info = API_KEYS[api_key]
        if key_info['subscription_tier'] == 'free' and key_info['credits_remaining'] <= 0:
            return jsonify({'error': 'No credits remaining. Please upgrade your subscription.'}), 402
        
        # Add customer info to request
        request.customer_info = key_info
        
        return f(*args, **kwargs)
    
    return decorated_function


def validate_portfolio_data(portfolio_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate incoming portfolio data
    
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    required_fields = ['customer_info', 'portfolio', 'settings']
    for field in required_fields:
        if field not in portfolio_data:
            return False, f"Missing required field: {field}"
    
    # Validate customer info
    customer_info = portfolio_data.get('customer_info', {})
    if not customer_info.get('email'):
        return False, "Customer email is required"
    
    if not customer_info.get('base_currency'):
        return False, "Base currency is required"
    
    # Validate currency is supported
    base_currency = customer_info.get('base_currency', '').upper()
    if base_currency not in currency_handler.SUPPORTED_CURRENCIES:
        return False, f"Unsupported currency: {base_currency}"
    
    # Validate portfolio positions
    portfolio = portfolio_data.get('portfolio', {})
    positions = portfolio.get('positions', [])
    
    if not positions:
        return False, "Portfolio must contain at least one position"
    
    for position in positions:
        if not position.get('ticker'):
            return False, "Each position must have a ticker"
        
        if not isinstance(position.get('shares', 0), (int, float)) or position.get('shares', 0) <= 0:
            return False, f"Invalid shares for {position.get('ticker')}"
    
    return True, None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'services': {
            'portfolio_analysis': 'operational',
            'currency_handler': 'operational',
            'ai_insights': 'operational' if anthropic_api_key else 'unavailable'
        }
    })


@app.route('/api/analyze', methods=['POST'])
@validate_api_key
def analyze_portfolio():
    """
    Main endpoint for portfolio analysis
    
    Expected JSON payload:
    {
        "customer_info": {
            "email": "customer@example.com",
            "name": "John Doe",
            "base_currency": "USD",
            "region": "US"
        },
        "portfolio": {
            "positions": [
                {
                    "ticker": "AAPL",
                    "shares": 100,
                    "cost_basis": 150.00,
                    "cost_basis_currency": "USD"
                }
            ]
        },
        "goals": [
            {
                "name": "Retirement",
                "target_amount": 1000000,
                "target_date": "2040-01-01",
                "monthly_contribution": 500
            }
        ],
        "settings": {
            "risk_tolerance": "moderate",
            "rebalance_threshold": 5,
            "report_frequency": "monthly",
            "target_allocations": {
                "stocks": 70,
                "bonds": 30
            }
        }
    }
    """
    try:
        # Get request data
        portfolio_data = request.json
        
        # Validate data
        is_valid, error_message = validate_portfolio_data(portfolio_data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Add customer ID from API key
        portfolio_data['customer_info']['customer_id'] = request.customer_info['customer_id']
        portfolio_data['customer_info']['subscription_tier'] = request.customer_info['subscription_tier']
        
        # Create analysis job
        job_id = str(uuid.uuid4())
        ANALYSIS_JOBS[job_id] = {
            'status': 'processing',
            'created_at': datetime.utcnow().isoformat(),
            'customer_id': request.customer_info['customer_id']
        }
        
        # Log request
        logger.info(f"Analysis requested for customer {request.customer_info['customer_id']}")
        
        # Process analysis (in production, this would be async)
        try:
            # Run analysis
            analysis_results = portfolio_service.analyze_portfolio(portfolio_data)
            
            # Update job status
            ANALYSIS_JOBS[job_id]['status'] = 'completed'
            ANALYSIS_JOBS[job_id]['completed_at'] = datetime.utcnow().isoformat()
            
            # Send email report if enabled
            if portfolio_data.get('settings', {}).get('email_report', True):
                email_sent = send_email_report(
                    analysis_results,
                    portfolio_data['customer_info']['email'],
                    portfolio_data['customer_info'].get('name', 'Valued Investor')
                )
                ANALYSIS_JOBS[job_id]['email_sent'] = email_sent
            
            # Deduct credit for free tier
            if request.customer_info['subscription_tier'] == 'free':
                API_KEYS[request.headers.get('X-API-Key')]['credits_remaining'] -= 1
            
            # Return results
            return jsonify({
                'job_id': job_id,
                'status': 'completed',
                'results': analysis_results,
                'email_sent': ANALYSIS_JOBS[job_id].get('email_sent', False)
            })
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            ANALYSIS_JOBS[job_id]['status'] = 'failed'
            ANALYSIS_JOBS[job_id]['error'] = str(e)
            
            return jsonify({
                'job_id': job_id,
                'status': 'failed',
                'error': 'Analysis failed. Please try again.'
            }), 500
            
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        return jsonify({'error': 'Invalid request format'}), 400


@app.route('/api/status/<job_id>', methods=['GET'])
@validate_api_key
def get_job_status(job_id):
    """Get analysis job status"""
    if job_id not in ANALYSIS_JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    job = ANALYSIS_JOBS[job_id]
    
    # Verify customer owns this job
    if job['customer_id'] != request.customer_info['customer_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify(job)


@app.route('/api/register', methods=['POST'])
def register_customer():
    """Register a new customer"""
    try:
        data = request.json
        
        # Validate required fields
        required = ['email', 'name', 'base_currency', 'region']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Generate customer ID and API key
        customer_id = f"cust_{uuid.uuid4().hex[:12]}"
        api_key = f"pk_{uuid.uuid4().hex}"
        
        # Create customer record
        customer = {
            'customer_id': customer_id,
            'email': data['email'],
            'name': data['name'],
            'base_currency': data['base_currency'].upper(),
            'region': data['region'].upper(),
            'subscription_tier': 'free',
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Store customer
        CUSTOMERS[customer_id] = customer
        
        # Create API key
        API_KEYS[api_key] = {
            'customer_id': customer_id,
            'email': data['email'],
            'subscription_tier': 'free',
            'credits_remaining': 3,
            'created_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"New customer registered: {customer_id}")
        
        return jsonify({
            'customer_id': customer_id,
            'api_key': api_key,
            'subscription_tier': 'free',
            'credits_remaining': 3
        })
        
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/subscription/status', methods=['GET'])
@validate_api_key
def get_subscription_status():
    """Get current subscription status"""
    customer_info = request.customer_info
    tier_info = SUBSCRIPTION_TIERS.get(customer_info['subscription_tier'], {})
    
    return jsonify({
        'subscription_tier': customer_info['subscription_tier'],
        'credits_remaining': customer_info.get('credits_remaining', 'unlimited'),
        'features': tier_info.get('features', []),
        'reports_per_month': tier_info.get('reports_per_month', 0),
        'next_reset': calculate_next_reset_date()
    })


@app.route('/api/subscription/upgrade', methods=['POST'])
@validate_api_key
def upgrade_subscription():
    """Upgrade subscription tier"""
    try:
        data = request.json
        new_tier = data.get('tier')
        
        if new_tier not in SUBSCRIPTION_TIERS:
            return jsonify({'error': 'Invalid subscription tier'}), 400
        
        # In production, this would integrate with Stripe
        # For now, just update the tier
        api_key = request.headers.get('X-API-Key')
        API_KEYS[api_key]['subscription_tier'] = new_tier
        
        # Reset credits for paid tiers
        if new_tier != 'free':
            API_KEYS[api_key]['credits_remaining'] = 'unlimited'
        
        logger.info(f"Customer {request.customer_info['customer_id']} upgraded to {new_tier}")
        
        return jsonify({
            'status': 'success',
            'new_tier': new_tier,
            'message': f'Successfully upgraded to {new_tier} tier'
        })
        
    except Exception as e:
        logger.error(f"Upgrade failed: {str(e)}")
        return jsonify({'error': 'Upgrade failed'}), 500


@app.route('/api/currencies', methods=['GET'])
def get_supported_currencies():
    """Get list of supported currencies"""
    currencies = []
    
    for code, info in currency_handler.SUPPORTED_CURRENCIES.items():
        currencies.append({
            'code': code,
            'name': info['name'],
            'symbol': info['symbol'],
            'region': info['region']
        })
    
    return jsonify({
        'currencies': currencies,
        'total': len(currencies)
    })


@app.route('/api/benchmarks/<currency>', methods=['GET'])
def get_currency_benchmarks(currency):
    """Get benchmark indices for a currency"""
    currency = currency.upper()
    
    if currency not in currency_handler.SUPPORTED_CURRENCIES:
        return jsonify({'error': 'Unsupported currency'}), 400
    
    benchmarks = currency_handler.get_regional_benchmarks(currency)
    
    return jsonify({
        'currency': currency,
        'region': currency_handler.SUPPORTED_CURRENCIES[currency]['region'],
        'benchmarks': benchmarks
    })


def send_email_report(analysis_results: Dict[str, Any], email: str, name: str) -> bool:
    """
    Send email report to customer
    
    Args:
        analysis_results: Analysis results
        email: Customer email
        name: Customer name
        
    Returns:
        Success boolean
    """
    try:
        smtp_config = {
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'smtp_user': os.getenv('YOUR_EMAIL'),
            'smtp_password': os.getenv('EMAIL_PASSWORD'),
            'from_email': os.getenv('YOUR_EMAIL', 'noreply@portfoliointelligence.ai')
        }
        
        # Generate and send email
        success = portfolio_service.generate_email_report(
            analysis_results,
            email,
            smtp_config
        )
        
        return success
        
    except Exception as e:
        logger.error(f"Email sending failed: {str(e)}")
        return False


def calculate_next_reset_date() -> str:
    """Calculate next credit reset date"""
    now = datetime.utcnow()
    
    # First day of next month
    if now.month == 12:
        next_reset = datetime(now.year + 1, 1, 1)
    else:
        next_reset = datetime(now.year, now.month + 1, 1)
    
    return next_reset.isoformat()


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
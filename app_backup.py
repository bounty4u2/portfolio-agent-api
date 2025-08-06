"""
AlphaSheet Intelligence™ - Main API Application
Complete integration with tier management, usage tracking, and branding
"""

from flask import Flask, request, jsonify, render_template_string
import os
import logging
from datetime import datetime
from typing import Dict, Any

# Import our main components
from portfolio_agent_saas import PortfolioAgentSaaS
from tier_configuration import TierConfiguration, TierEnforcer
from regional_tax_systems import EnhancedPortfolioAgentSaaS

# Import utilities (FIXED IMPORTS)
from utils.usage_tracker import UsageTracker
from utils.email_sender import EmailScheduler
from utils.alert_system import AlertSystem

# Import branding (BOTH branding configs for compatibility)
try:
    from visual_branding import AlphaSheetVisualBranding
except ImportError:
    print("Warning: visual_branding.py not found")
    AlphaSheetVisualBranding = None

try:
    from branding_config import AlphaSheetBranding, BRANDING_CONFIG
except ImportError:
    print("Warning: branding_config.py not found")
    AlphaSheetBranding = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration from environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'your-api-key-here')
TIER_ENFORCEMENT = os.getenv('TIER_ENFORCEMENT', 'true').lower() == 'true'
DEFAULT_TIER = os.getenv('DEFAULT_TIER', 'starter')

# In-memory storage for demo (replace with database in production)
USAGE_CACHE = {}

@app.route('/')
def home():
    """Landing page with AlphaSheet Intelligence™ branding"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AlphaSheet Intelligence™ API</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .container {{
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
                max-width: 600px;
                text-align: center;
            }}
            .logo {{
                width: 80px;
                height: 80px;
                background: {AlphaSheetVisualBranding.GRADIENT_CSS};
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 48px;
                margin: 0 auto 20px;
            }}
            h1 {{
                color: #2c3e50;
                margin: 10px 0;
            }}
            .status {{
                background: #d4edda;
                color: #155724;
                padding: 10px 20px;
                border-radius: 20px;
                display: inline-block;
                margin: 20px 0;
            }}
            .endpoints {{
                text-align: left;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .endpoint {{
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 5px;
                border-left: 3px solid #4A90E2;
            }}
            code {{
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">α</div>
            <h1>AlphaSheet Intelligence™</h1>
            <p style="color: #666;">Institutional-Grade Portfolio Analysis API</p>
            <div class="status">✓ API Online</div>
            
            <div class="endpoints">
                <h3>Available Endpoints:</h3>
                <div class="endpoint">
                    <strong>POST</strong> <code>/analyze</code> - Analyze portfolio
                </div>
                <div class="endpoint">
                    <strong>GET</strong> <code>/check-usage</code> - Check usage limits
                </div>
                <div class="endpoint">
                    <strong>GET</strong> <code>/tier-features</code> - View tier features
                </div>
                <div class="endpoint">
                    <strong>GET</strong> <code>/health</code> - Health check
                </div>
            </div>
            
            <p style="color: #999; font-size: 12px; margin-top: 30px;">
                Part of the AlphaSheet AI™ Suite<br>
                © 2024 AlphaSheet. All rights reserved.
            </p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/analyze', methods=['POST'])
def analyze_portfolio():
    """
    Main portfolio analysis endpoint with tier enforcement
    """
    try:
        # Parse request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': AlphaSheetVisualBranding.format_error_message('invalid_request')
            }), 400
        
        # Extract customer information
        customer_info = data.get('customer_info', {})
        customer_id = customer_info.get('customer_id', 'anonymous')
        customer_tier = customer_info.get('tier', DEFAULT_TIER).lower()
        region = customer_info.get('region', 'US')
        customer_email = customer_info.get('email', '')
        
        logger.info(f"Analysis request from {customer_id} (Tier: {customer_tier})")
        
        # Initialize usage tracker
        tracker = UsageTracker(customer_id, customer_tier)
        
        # Check usage limits if tier enforcement is enabled
        if TIER_ENFORCEMENT:
            if not tracker.can_generate_report():
                usage_summary = tracker.get_usage_summary()
                return jsonify({
                    'success': False,
                    'error': AlphaSheetVisualBranding.format_error_message('rate_limit', user_friendly=True),
                    'usage': usage_summary,
                    'upgrade_benefits': tracker.get_tier_upgrade_benefits()
                }), 429
        
        # Validate portfolio data
        portfolio = data.get('portfolio', {})
        if not portfolio or not portfolio.get('positions'):
            return jsonify({
                'success': False,
                'error': 'Portfolio data is required with at least one position'
            }), 400
        
        # Initialize the appropriate agent based on tier
        logger.info(f"Initializing {AlphaSheetVisualBranding.TIER_NAMES[customer_tier]} agent...")
        
        # Base agent for all tiers
        agent = PortfolioAgentSaaS(
            customer_tier=customer_tier,
            api_key=ANTHROPIC_API_KEY
        )
        
        # For premium tier, also initialize enhanced analytics
        enhanced_results = None
        if customer_tier == 'premium':
            logger.info("Initializing enhanced analytics for Premium tier...")
            enhanced_agent = EnhancedPortfolioAgentSaaS(
                customer_tier=customer_tier,
                region=region,
                api_key=ANTHROPIC_API_KEY
            )
        
        # Perform analysis
        logger.info("Starting portfolio analysis...")
        results = agent.analyze_portfolio(data)
        
        # Add enhanced analytics for premium tier
        if customer_tier == 'premium' and enhanced_agent:
            logger.info("Adding enhanced analytics...")
            enhanced_results = enhanced_agent.analyze_portfolio_enhanced(data)
            results['enhanced_analytics'] = enhanced_results
        
        # Generate HTML report with branding
        logger.info("Generating branded HTML report...")
        html_report = agent.generate_html_report(results)
        
        # Record usage
        if TIER_ENFORCEMENT:
            tracker.record_report_generation()
        
        # Get usage summary
        usage_summary = tracker.get_usage_summary()
        
        # Check for alerts (Premium only)
        alerts = []
        if customer_tier == 'premium':
            alert_system = AlertSystem()
            alerts = alert_system.check_all_alerts(results)
            
            # Send alert emails if any critical alerts
            if alerts and customer_email:
                high_priority_alerts = [a for a in alerts if a['severity'] == 'high']
                if high_priority_alerts:
                    scheduler = EmailScheduler('premium')
                    subject, content = alert_system.format_alert_email(high_priority_alerts)
                    scheduler.send_email(customer_email, subject, content, is_html=True)
        
        # Wrap response with branding (use appropriate branding module)
        response_data = {
            'analysis': results,
            'html_report': html_report,
            'alerts': alerts,
            'usage': usage_summary
        }
        
        # Use AlphaSheetBranding if available (from your current setup)
        if AlphaSheetBranding:
            response = AlphaSheetBranding.get_api_response_wrapper(
                response_data,
                customer_tier
            )
        # Otherwise use AlphaSheetVisualBranding
        elif AlphaSheetVisualBranding:
            response = AlphaSheetVisualBranding.get_api_response_wrapper(
                response_data,
                customer_tier
            )
        else:
            # Fallback if no branding module available
            response = {
                'success': True,
                'data': response_data,
                'metadata': {
                    'powered_by': 'AlphaSheet Intelligence™',
                    'tier': customer_tier,
                    'timestamp': datetime.now().isoformat()
                }
            }
        
        logger.info(f"Analysis complete for {customer_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {str(e)}")
        return jsonify({
            'success': False,
            'error': AlphaSheetVisualBranding.format_error_message('general', user_friendly=True),
            'details': str(e) if app.debug else None
        }), 500

@app.route('/check-usage', methods=['GET'])
def check_usage():
    """Check usage limits for a customer"""
    try:
        customer_id = request.args.get('customer_id')
        tier = request.args.get('tier', DEFAULT_TIER)
        
        if not customer_id:
            return jsonify({
                'success': False,
                'error': 'customer_id is required'
            }), 400
        
        tracker = UsageTracker(customer_id, tier)
        usage_summary = tracker.get_usage_summary()
        
        # Add upgrade suggestions if approaching limits
        should_upgrade = tracker.should_suggest_upgrade()
        if should_upgrade:
            usage_summary['upgrade_suggestion'] = True
            usage_summary['upgrade_benefits'] = tracker.get_tier_upgrade_benefits()
        
        response = AlphaSheetVisualBranding.get_api_response_wrapper(
            usage_summary,
            tier
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in check_usage: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/tier-features', methods=['GET'])
def get_tier_features():
    """Get feature comparison across all tiers"""
    try:
        comparison = TierConfiguration.get_tier_comparison()
        
        # Add visual branding to the comparison
        for tier_key in comparison:
            tier_data = comparison[tier_key]
            tier_data['brand_name'] = AlphaSheetVisualBranding.TIER_NAMES.get(tier_key, tier_key)
            tier_data['color'] = AlphaSheetVisualBranding.TIER_COLORS[tier_key]['primary']
        
        response = {
            'success': True,
            'product': AlphaSheetVisualBranding.PRODUCT_NAME,
            'tagline': AlphaSheetVisualBranding.TAGLINE,
            'tiers': comparison
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in get_tier_features: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if API key is configured
        api_key_configured = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != 'your-api-key-here')
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'product': AlphaSheetVisualBranding.PRODUCT_NAME,
            'version': '1.0.0',
            'services': {
                'api': 'online',
                'anthropic_api': 'configured' if api_key_configured else 'not_configured',
                'tier_enforcement': 'enabled' if TIER_ENFORCEMENT else 'disabled'
            }
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/send-weekly-email', methods=['POST'])
def send_weekly_email():
    """
    Endpoint to trigger weekly email (for cron jobs or manual trigger)
    """
    try:
        data = request.json
        customer_id = data.get('customer_id')
        customer_tier = data.get('tier', DEFAULT_TIER)
        customer_email = data.get('email')
        portfolio_data = data.get('portfolio_data', {})
        
        if not customer_email:
            return jsonify({
                'success': False,
                'error': 'Email address is required'
            }), 400
        
        # Initialize email scheduler
        scheduler = EmailScheduler(customer_tier)
        
        # Format email based on tier
        if customer_tier == 'starter':
            content = scheduler.format_weekly_summary(portfolio_data)
            is_html = False
        elif customer_tier == 'growth':
            content = scheduler.format_intelligence_brief(portfolio_data)
            is_html = True
        else:  # premium
            content = scheduler.format_market_brief(portfolio_data)
            is_html = True
        
        # Get branded subject
        subject = AlphaSheetVisualBranding.get_email_subject('weekly_summary', customer_tier)
        
        # Send email
        success = scheduler.send_email(customer_email, subject, content, is_html)
        
        return jsonify({
            'success': success,
            'message': 'Email sent successfully' if success else 'Failed to send email'
        })
        
    except Exception as e:
        logger.error(f"Error in send_weekly_email: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with branding"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'product': AlphaSheetVisualBranding.PRODUCT_NAME
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with branding"""
    return jsonify({
        'success': False,
        'error': AlphaSheetVisualBranding.format_error_message('general', user_friendly=True)
    }), 500

if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(os.getenv('PORT', 5000))
    
    # Print startup message
    print(f"""
    ╔══════════════════════════════════════════════╗
    ║                                              ║
    ║     AlphaSheet Intelligence™ API            ║
    ║     Version 1.0.0                            ║
    ║                                              ║
    ║     Starting on port {port}...              ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=port)
"""
Portfolio Agent API with Tier Management
"""
from flask import Flask, request, jsonify
import os
from portfolio_agent_saas import PortfolioAgentSaaS
from tier_configuration import TierConfiguration, UsageTracker
from regional_tax_systems import EnhancedPortfolioAgentSaaS

app = Flask(__name__)

# Initialize with environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

@app.route('/analyze', methods=['POST'])
def analyze_portfolio():
    """Main analysis endpoint with tier enforcement"""
    try:
        data = request.json
        
        # Extract customer info
        customer_info = data.get('customer_info', {})
        customer_id = customer_info.get('customer_id', 'anonymous')
        customer_tier = customer_info.get('tier', 'starter')
        region = customer_info.get('region', 'US')
        
        # Check usage limits
        tracker = UsageTracker(customer_id, customer_tier)
        if not tracker.can_generate_report():
            return jsonify({
                'error': 'Monthly report limit reached',
                'usage': tracker.get_usage_summary()
            }), 429
        
        # Initialize agent with tier
        agent = PortfolioAgentSaaS(
            customer_tier=customer_tier,
            api_key=ANTHROPIC_API_KEY
        )
        
        # For premium tier, also initialize enhanced analytics
        if customer_tier == 'premium':
            enhanced_agent = EnhancedPortfolioAgentSaaS(
                customer_tier=customer_tier,
                region=region,
                api_key=ANTHROPIC_API_KEY
            )
        
        # Perform analysis
        results = agent.analyze_portfolio(data)
        
        # Add enhanced analytics for premium
        if customer_tier == 'premium':
            enhanced_results = enhanced_agent.analyze_portfolio_enhanced(data)
            results['enhanced_analytics'] = enhanced_results
        
        # Generate HTML report
        html_report = agent.generate_html_report(results)
        
        # Record usage
        tracker.record_report_generation()
        
        # Add usage info to response
        results['usage'] = tracker.get_usage_summary()
        
        return jsonify({
            'success': True,
            'results': results,
            'html_report': html_report,
            'usage': tracker.get_usage_summary()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-usage', methods=['GET'])
def check_usage():
    """Check usage limits for a customer"""
    customer_id = request.args.get('customer_id')
    tier = request.args.get('tier', 'starter')
    
    tracker = UsageTracker(customer_id, tier)
    return jsonify(tracker.get_usage_summary())

@app.route('/tier-features', methods=['GET'])
def get_tier_features():
    """Get feature comparison across tiers"""
    return jsonify(TierConfiguration.get_tier_comparison())

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
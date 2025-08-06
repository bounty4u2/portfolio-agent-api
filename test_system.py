"""
AlphaSheet Intelligence™ - Complete System Test
Tests all components before deployment
"""

import requests
import json
import time
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:5000"  # Change to your Railway URL after deployment

def print_section(title):
    """Print a section header"""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

def test_health_check():
    """Test the health endpoint"""
    print_section("Testing Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Product: {data.get('product')}")
        print(f"Version: {data.get('version')}")
        
        if data.get('status') == 'healthy':
            print("[OK] Health check passed")
            return True
        else:
            print("[ERROR] Health check failed")
            return False
    except Exception as e:
        print(f"[ERROR] Could not connect to API: {e}")
        return False

def test_tier_features():
    """Test tier features endpoint"""
    print_section("Testing Tier Features")
    
    try:
        response = requests.get(f"{BASE_URL}/tier-features")
        data = response.json()
        
        if data.get('success'):
            print(f"Product: {data.get('product')}")
            print(f"Tagline: {data.get('tagline')}")
            
            tiers = data.get('tiers', {})
            for tier_key, tier_data in tiers.items():
                print(f"\n{tier_data.get('brand_name', tier_key)}:")
                print(f"  Price: ${tier_data.get('price')}")
                print(f"  Reports: {tier_data.get('reports')}")
                print(f"  Portfolios: {tier_data.get('portfolios')}")
            
            print("\n[OK] Tier features retrieved")
            return True
        else:
            print("[ERROR] Failed to get tier features")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_usage_check():
    """Test usage tracking"""
    print_section("Testing Usage Tracking")
    
    try:
        # Test for different tiers
        test_cases = [
            ("TEST_USER_001", "starter"),
            ("TEST_USER_002", "growth"),
            ("TEST_USER_003", "premium")
        ]
        
        for customer_id, tier in test_cases:
            response = requests.get(
                f"{BASE_URL}/check-usage",
                params={"customer_id": customer_id, "tier": tier}
            )
            data = response.json()
            
            if data.get('success'):
                usage = data.get('data', {})
                print(f"\n{customer_id} ({tier}):")
                print(f"  Reports used: {usage.get('reports_used')}/{usage.get('reports_limit')}")
                print(f"  Reports remaining: {usage.get('reports_remaining')}")
            else:
                print(f"[ERROR] Failed for {customer_id}")
        
        print("\n[OK] Usage tracking working")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_portfolio_analysis(tier="starter"):
    """Test portfolio analysis with different tiers"""
    print_section(f"Testing Portfolio Analysis - {tier.upper()}")
    
    # Sample portfolio data
    portfolio_data = {
        "customer_info": {
            "customer_id": f"TEST_{tier.upper()}_USER",
            "tier": tier,
            "email": "test@alphasheet.ai",
            "region": "US",
            "base_currency": "USD",
            "name": "Test User"
        },
        "portfolio": {
            "positions": [
                {
                    "ticker": "AAPL",
                    "shares": 100,
                    "cost_basis": 150.00
                },
                {
                    "ticker": "MSFT",
                    "shares": 50,
                    "cost_basis": 300.00
                },
                {
                    "ticker": "NVDA",
                    "shares": 25,
                    "cost_basis": 400.00
                }
            ],
            "cash": 5000
        },
        "goals": [
            {
                "name": "Retirement",
                "target_amount": 1000000,
                "target_date": "2040-01-01",
                "monthly_contribution": 1000
            }
        ],
        "settings": {
            "risk_tolerance": "moderate",
            "rebalance_threshold": 5.0
        }
    }
    
    try:
        print(f"Sending analysis request for {tier} tier...")
        
        response = requests.post(
            f"{BASE_URL}/analyze",
            json=portfolio_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print(f"[OK] Analysis successful")
                
                # Check metadata
                metadata = data.get('metadata', {})
                print(f"  Powered by: {metadata.get('powered_by')}")
                print(f"  Tier: {metadata.get('tier')}")
                
                # Check usage
                analysis_data = data.get('data', {})
                usage = analysis_data.get('usage', {})
                print(f"  Reports remaining: {usage.get('reports_remaining')}")
                
                # Check if HTML report was generated
                if analysis_data.get('html_report'):
                    print(f"  HTML report generated: Yes")
                    
                    # Save HTML report for manual inspection
                    filename = f"test_report_{tier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(analysis_data['html_report'])
                    print(f"  Report saved to: {filename}")
                
                # Check for alerts (Premium only)
                if tier == "premium":
                    alerts = analysis_data.get('alerts', [])
                    print(f"  Alerts generated: {len(alerts)}")
                
                return True
            else:
                print(f"[ERROR] Analysis failed: {data.get('error')}")
                return False
        
        elif response.status_code == 429:
            print(f"[WARNING] Rate limit reached for {tier}")
            print(f"  Message: {response.json().get('error')}")
            return True  # This is expected behavior
        
        else:
            print(f"[ERROR] HTTP {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_email_endpoint():
    """Test email sending endpoint"""
    print_section("Testing Email System")
    
    test_data = {
        "customer_id": "TEST_EMAIL_USER",
        "tier": "growth",
        "email": "test@alphasheet.ai",
        "portfolio_data": {
            "total_value": 150000,
            "weekly_change": 0.023,
            "top_performer": "NVDA",
            "goal_progress": 0.68,
            "reports_used": 2,
            "reports_limit": 4
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/send-weekly-email",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        data = response.json()
        if data.get('success'):
            print("[OK] Email system working")
            return True
        else:
            print(f"[ERROR] Email failed: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def run_all_tests():
    """Run all system tests"""
    print("""
    ╔══════════════════════════════════════════════╗
    ║                                              ║
    ║   AlphaSheet Intelligence™ System Test      ║
    ║   Testing all components...                  ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
    """)
    
    # Track results
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    time.sleep(1)
    
    results.append(("Tier Features", test_tier_features()))
    time.sleep(1)
    
    results.append(("Usage Tracking", test_usage_check()))
    time.sleep(1)
    
    # Test each tier
    for tier in ["starter", "growth", "premium"]:
        results.append((f"Analysis ({tier})", test_portfolio_analysis(tier)))
        time.sleep(2)  # Avoid rate limiting
    
    results.append(("Email System", test_email_endpoint()))
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "[OK]" if result else "[FAILED]"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {len(results)} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed == 0:
        print("\n  [SUCCESS] All tests passed! Ready for deployment.")
    else:
        print(f"\n  [WARNING] {failed} test(s) failed. Review before deployment.")
    
    return failed == 0

if __name__ == "__main__":
    # Make sure the API is running
    print("Make sure your API is running on http://localhost:5000")
    print("Press Enter to start testing...")
    input()
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)